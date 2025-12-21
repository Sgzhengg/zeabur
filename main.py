import os
import uuid
import traceback
from fastapi import FastAPI, UploadFile, Form, HTTPException, File
from fastapi.middleware.cors import CORSMiddleware
from llama_parse import LlamaParse
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models

# --- 1. 环境变量配置 (关键修改) ---
# Zeabur 部署必须通过环境变量设置这些值，否则启动报错
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
COLLECTION_NAME = "telecom_collection"

# 检查必要变量，如果没有则报错，避免盲目启动
if not QDRANT_URL:
    print("❌ ERROR: QDRANT_URL is missing. Please set it in Zeabur Variables.")
    # 这里的默认值仅用于本地测试，Zeabur 上请务必设置环境变量
    QDRANT_URL = "http://qdrant:6333" 

if not LLAMA_CLOUD_API_KEY:
    print("⚠️ WARNING: LLAMA_CLOUD_API_KEY is missing. Ingest will fail.")

app = FastAPI()

# --- 2. CORS 设置 ---
app.add_middleware(
    CORSMiddleware,
    # 生产环境建议将 "*" 改为你的前端域名 (如 https://xxx.zeabur.app)
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. 初始化 Qdrant 客户端 ---
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, prefer_grpc=False)

@app.on_event("startup")
def startup_event():
    """启动时检查并创建集合"""
    print(f"Connecting to Qdrant at: {QDRANT_URL} ...")
    try:
        if not client.collection_exists(COLLECTION_NAME):
            print(f"Collection {COLLECTION_NAME} not found, creating...")
            # 注意：这里我们使用 FastEmbed (BAAI/bge-small-en-v1.5)，它的固定维度是 384
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
            )
            print(f"Collection {COLLECTION_NAME} created successfully.")
        else:
            print(f"Collection {COLLECTION_NAME} exists. Ready to go.")
    except Exception as e:
        print(f"⚠️ Connection Warning: Could not connect to Qdrant at startup.")
        print(f"Details: {e}")
        print("Service will start, but database operations might fail.")

@app.get("/")
def health_check():
    return {"status": "ok", "service": "Telecom Ingest API"}

@app.post("/ingest")
# 关键修复：这里加了 File(...) 确保 Swagger 显示文件上传按钮
async def ingest_file(file: UploadFile = File(...), file_id: str = Form(...)):
    """接收文件 -> 解析 -> 切片 -> 向量化 -> 入库"""
    
    # 再次检查 Key
    if not LLAMA_CLOUD_API_KEY:
        raise HTTPException(status_code=500, detail="LLAMA_CLOUD_API_KEY not set on server.")

    temp_filename = f"/tmp/{uuid.uuid4()}_{file.filename}"
    try:
        content = await file.read()
        with open(temp_filename, "wb") as f:
            f.write(content)
        
        print(f"Parsing file: {file.filename}...")
        parser = LlamaParse(api_key=LLAMA_CLOUD_API_KEY, result_type="markdown")
        documents = await parser.aload_data(temp_filename)
        
        if not documents:
            raise HTTPException(status_code=400, detail="LlamaParse returned empty content.")
            
        markdown_text = documents[0].text
        
        print("Splitting text...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(markdown_text)
        
        print(f"Upserting {len(chunks)} chunks...")
        # FastEmbed 会自动下载模型并生成向量
        client.add(
            collection_name=COLLECTION_NAME,
            documents=chunks,
            metadata=[{"file_id": file_id, "chunk_index": i, "source": file.filename} for i in range(len(chunks))],
            ids=[str(uuid.uuid4()) for _ in range(len(chunks))]
        )
        
        return {"status": "success", "chunks_count": len(chunks)}
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

@app.post("/delete")
async def delete_file(file_id: str = Form(...)):
    """根据 file_id 删除数据"""
    try:
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="file_id",
                            match=models.MatchValue(value=file_id)
                        )
                    ]
                )
            )
        )
        return {"status": "deleted", "file_id": file_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search_docs(query: str = Form(...), limit: int = 5):
    """搜索接口"""
    try:
        results = client.query(
            collection_name=COLLECTION_NAME,
            query_text=query,
            limit=limit
        )
        return [{"content": res.document, "score": res.score, "metadata": res.metadata} for res in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 注意：这里删除了 if __name__ == "__main__"，因为我们现在用 Procfile 启动