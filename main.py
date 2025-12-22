import os
import uuid
import shutil
import zipfile
import traceback
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, Form, HTTPException, File
from fastapi.middleware.cors import CORSMiddleware
from llama_parse import LlamaParse
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models
from flashrank import Ranker, RerankRequest

# --- 1. 环境变量读取 ---
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# 集合名称 (保持不变)
COLLECTION_NAME = "telecom_collection_v2"

print(f"DEBUG CONFIG: URL={QDRANT_URL}")

# --- 2. 初始化 Re-ranker ---
print("⏳ Initializing FlashRank Reranker...")
reranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/tmp/flashrank_cache")
print("✅ Reranker initialized!")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if not QDRANT_URL:
    raise ValueError("❌ Fatal Error: QDRANT_URL is missing!")

# 初始化 Qdrant
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, prefer_grpc=False)

@app.on_event("startup")
def startup_event():
    print(f"🚀 Connecting to Qdrant at: {QDRANT_URL} ...")
    try:
        collections = client.get_collections()
        print(f"✅ Connected! Found {len(collections.collections)} collections.")
    except Exception as e:
        print(f"❌ Connection Failed! Error: {e}")

@app.get("/")
def health_check():
    return {"status": "ok", "service": "Telecom Complex Ingest API"}

# --- 辅助函数：解压 ZIP ---
def extract_zip(zip_path: str, extract_to: str):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

# --- 辅助函数：判断文件是主件还是附件 ---
def guess_doc_type(filename: str) -> str:
    # 简单的启发式规则，可根据电信业务习惯修改
    main_keywords = ["通知", "公告", "管理办法", "规定", "主件", "正文"]
    if any(k in filename for k in main_keywords):
        return "main"
    return "attachment"

@app.post("/ingest")
async def ingest_package(file: UploadFile = File(...), package_id: str = Form(None)):
    """
    高级入库接口：
    支持上传 .zip 压缩包（包含主件+附件）或 单个文件。
    """
    if not LLAMA_CLOUD_API_KEY:
         raise HTTPException(status_code=500, detail="LLAMA_CLOUD_API_KEY not set.")

    # 如果没传 ID，生成一个新的 Group ID (案卷ID)
    group_id = package_id if package_id else str(uuid.uuid4())
    
    # 临时目录
    base_tmp_dir = f"/tmp/ingest_{group_id}"
    os.makedirs(base_tmp_dir, exist_ok=True)
    
    upload_path = f"{base_tmp_dir}/{file.filename}"
    
    try:
        # 1. 保存上传的文件
        content = await file.read()
        with open(upload_path, "wb") as f:
            f.write(content)
        
        files_to_process = []

        # 2. 判断是否为 ZIP
        if file.filename.lower().endswith(".zip"):
            print(f"📦 Detected ZIP package: {file.filename}, extracting...")
            extract_dir = f"{base_tmp_dir}/extracted"
            extract_zip(upload_path, extract_dir)
            
            # 遍历解压后的所有文件
            for root, dirs, files in os.walk(extract_dir):
                for fname in files:
                    if fname.startswith(".") or "__MACOSX" in root: continue # 跳过系统隐藏文件
                    files_to_process.append(os.path.join(root, fname))
        else:
            # 单文件
            files_to_process.append(upload_path)

        print(f"task: Processing {len(files_to_process)} files in Group: {group_id}")

        # 3. 初始化 LlamaParse (开启高级模式以处理复杂表格)
        parser = LlamaParse(
            api_key=LLAMA_CLOUD_API_KEY,
            result_type="markdown",
            premium_mode=True,  # ⚠️ 开启高级模式，解析表格更准 (会消耗 Credit)
            verbose=True,
            language="zh"       # 强制中文识别
        )

        total_chunks = 0
        all_points = [] # 暂时存放所有切片，最后一起入库

        # 4. 循环处理每个文件
        for file_path in files_to_process:
            fname = os.path.basename(file_path)
            doc_type = guess_doc_type(fname) # 识别是主件还是附件
            
            print(f"📄 Parsing ({doc_type}): {fname} ...")
            
            # LlamaParse 解析
            documents = await parser.aload_data(file_path)
            if not documents:
                print(f"⚠️ Warning: No text found in {fname}")
                continue
                
            markdown_text = documents[0].text
            
            # 切片
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_text(markdown_text)
            
            # 准备入库数据 (携带 Group ID 和 类型)
            for i, chunk_text in enumerate(chunks):
                all_points.append({
                    "content": chunk_text,
                    "metadata": {
                        "group_id": group_id,     # 核心：关联ID
                        "filename": fname,
                        "doc_type": doc_type,     # main 或 attachment
                        "chunk_index": i,
                        "source_package": file.filename
                    }
                })
            
            total_chunks += len(chunks)

        # 5. 批量入库
        if all_points:
            print(f"💾 Upserting {len(all_points)} total chunks to Qdrant...")
            
            # 提取文本列表用于向量化
            texts = [p["content"] for p in all_points]
            metadatas = [p["metadata"] for p in all_points]
            ids = [str(uuid.uuid4()) for _ in all_points]

            client.add(
                collection_name=COLLECTION_NAME,
                documents=texts,
                metadata=metadatas,
                ids=ids
            )
        
        return {
            "status": "success", 
            "group_id": group_id, 
            "files_processed": len(files_to_process),
            "total_chunks": total_chunks
        }
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 清理临时目录
        if os.path.exists(base_tmp_dir):
            shutil.rmtree(base_tmp_dir)

@app.post("/delete")
async def delete_package(group_id: str = Form(...)):
    """按 Group ID 删除整套文档"""
    try:
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="group_id",
                            match=models.MatchValue(value=group_id)
                        )
                    ]
                )
            )
        )
        return {"status": "deleted", "group_id": group_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search_docs(query: str = Form(...), limit: int = 5):
    """
    检索接口 (保持 FlashRank 重排序逻辑)
    """
    try:
        print(f"🔎 Searching for: {query}")
        
        search_result = client.query(
            collection_name=COLLECTION_NAME,
            query_text=query,
            limit=50 
        )
        
        if not search_result:
            return []

        passages = [
            {"id": str(res.id), "text": res.document, "meta": res.metadata}
            for res in search_result
        ]

        # Rerank
        rerank_request = RerankRequest(query=query, passages=passages)
        ranked_results = reranker.rerank(rerank_request)

        top_results = ranked_results[:limit]
        
        # 返回结果 (现在包含了 filename, group_id 等丰富信息)
        return [
            {
                "content": res["text"],
                "score": float(res["score"]),
                "metadata": res["meta"]
            } 
            for res in top_results
        ]

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))