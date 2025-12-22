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

COLLECTION_NAME = "telecom_collection_v2"

print(f"DEBUG CONFIG: URL={QDRANT_URL}")

# --- 2. 初始化 Re-ranker ---
print("⏳ Initializing FlashRank Reranker...")
# 依然使用这个速度快且效果好的轻量模型
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
    return {"status": "ok", "service": "Telecom Ingest API Optimized"}

# --- 辅助函数 ---
def extract_zip(zip_path: str, extract_to: str):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def guess_doc_type(filename: str) -> str:
    main_keywords = ["通知", "公告", "管理办法", "规定", "主件", "正文"]
    if any(k in filename for k in main_keywords):
        return "main"
    return "attachment"

@app.post("/ingest")
async def ingest_package(file: UploadFile = File(...), package_id: str = Form(None)):
    """
    入库接口：支持 ZIP 包，针对电信文档优化了解析指令和切片大小
    """
    if not LLAMA_CLOUD_API_KEY:
         raise HTTPException(status_code=500, detail="LLAMA_CLOUD_API_KEY not set.")

    group_id = package_id if package_id else str(uuid.uuid4())
    base_tmp_dir = f"/tmp/ingest_{group_id}"
    os.makedirs(base_tmp_dir, exist_ok=True)
    upload_path = f"{base_tmp_dir}/{file.filename}"
    
    try:
        content = await file.read()
        with open(upload_path, "wb") as f:
            f.write(content)
        
        files_to_process = []
        if file.filename.lower().endswith(".zip"):
            print(f"📦 Detected ZIP package: {file.filename}")
            extract_dir = f"{base_tmp_dir}/extracted"
            extract_zip(upload_path, extract_dir)
            for root, dirs, files in os.walk(extract_dir):
                for fname in files:
                    if fname.startswith(".") or "__MACOSX" in root: continue
                    files_to_process.append(os.path.join(root, fname))
        else:
            files_to_process.append(upload_path)

        # 🟢 优化点 1：移除 language="zh" 防止报错，增加解析指令优化表格
        parser = LlamaParse(
            api_key=LLAMA_CLOUD_API_KEY,
            result_type="markdown",
            premium_mode=True, 
            verbose=True,
            parsing_instruction="这是一个电信运营商的政策文档，包含大量复杂的嵌套表格。请尽可能保留表格的结构，不要遗漏任何数字。如果表格跨页，请将其合并。"
        )

        total_chunks = 0
        all_points = [] 

        for file_path in files_to_process:
            fname = os.path.basename(file_path)
            doc_type = guess_doc_type(fname)
            print(f"📄 Parsing ({doc_type}): {fname}")
            
            try:
                documents = await parser.aload_data(file_path)
            except Exception as parse_error:
                print(f"❌ Parse Error on {fname}: {parse_error}")
                continue

            if not documents: 
                print(f"⚠️ Warning: No text found in {fname}")
                continue
                
            markdown_text = documents[0].text
            
            # 🟢 优化点 2：增大 chunk_size 到 2000，overlap 到 500
            # 这样能保证上下文连贯，解决"不知道是哪个月份"的问题
            splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
            chunks = splitter.split_text(markdown_text)
            
            for i, chunk_text in enumerate(chunks):
                all_points.append({
                    "content": chunk_text,
                    "metadata": {
                        "group_id": group_id,
                        "filename": fname,
                        "doc_type": doc_type,
                        "chunk_index": i,
                        "source_package": file.filename
                    }
                })
            total_chunks += len(chunks)

        if total_chunks == 0:
            return {"status": "error", "msg": "No documents parsed."}

        if all_points:
            print(f"💾 Upserting {len(all_points)} chunks...")
            texts = [p["content"] for p in all_points]
            metadatas = [p["metadata"] for p in all_points]
            ids = [str(uuid.uuid4()) for _ in all_points]

            client.add(
                collection_name=COLLECTION_NAME,
                documents=texts,
                metadata=metadatas,
                ids=ids
            )
        
        return {"status": "success", "group_id": group_id, "chunks": total_chunks}
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(base_tmp_dir):
            shutil.rmtree(base_tmp_dir)

@app.post("/delete")
async def delete_package(target_id: str = Form(..., description="填入 group_id 或 file_id")):
    try:
        if not client.collection_exists(COLLECTION_NAME):
             return {"status": "skipped", "msg": "Collection does not exist."}

        # 删除 group_id 匹配的
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(key="group_id", match=models.MatchValue(value=target_id))]
                )
            )
        )
        # 删除 file_id 匹配的 (兼容旧数据)
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(key="file_id", match=models.MatchValue(value=target_id))]
                )
            )
        )
        return {"status": "deleted", "target_id": target_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset")
async def reset_database():
    """仅删除集合，不重新创建，由 ingest 自动处理重建"""
    try:
        client.delete_collection(COLLECTION_NAME)
        return {"status": "success", "msg": "Collection deleted."}
    except Exception as e:
        return {"status": "success", "msg": "Collection already clear."}

@app.post("/search")
async def search_docs(query: str = Form(...), limit: int = 5):
    try:
        # 防止刚 reset 完报错
        if not client.collection_exists(COLLECTION_NAME):
            return []

        print(f"🔎 Searching for: {query}")
        
        # 🟢 优化点 3：扩大初筛范围到 100 条，宁滥勿缺
        search_result = client.query(
            collection_name=COLLECTION_NAME,
            query_text=query,
            limit=100 
        )
        
        if not search_result:
            return []

        passages = [
            {"id": str(res.id), "text": res.document, "meta": res.metadata}
            for res in search_result
        ]

        print(f"⚖️ Reranking {len(passages)} documents...")
        rerank_request = RerankRequest(query=query, passages=passages)
        ranked_results = reranker.rerank(rerank_request)

        top_results = ranked_results[:limit]
        
        # 🟢 优化点 4：强制 float 转换，修复 500 报错
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