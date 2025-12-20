import os
import uuid
import traceback
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from llama_parse import LlamaParse
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models

# --- 配置读取 ---
# Zeabur 会自动注入 PORT，但其他变量需要在 Dashboard 设置
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333") 
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY", "")
COLLECTION_NAME = "telecom_collection"

app = FastAPI()

# --- CORS 设置 (允许前端跨域调用) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 初始化 Qdrant 客户端 ---
# prefer_grpc=False 能减少一些网络依赖问题
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, prefer_grpc=False)

@app.on_event("startup")
def startup_event():
    """应用启动时尝试创建集合，如果失败不仅不崩溃，只打印警告"""
    print("Application starting up...")
    try:
        # 尝试连接数据库
        if not client.collection_exists(COLLECTION_NAME):
            print(f"Collection {COLLECTION_NAME} not found, creating...")
            # 使用默认的 fastembed 模型 (BAAI/bge-small-en-v1.5) 维度为 384
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
            )
            print(f"Collection {COLLECTION_NAME} created successfully.")
        else:
            print(f"Collection {COLLECTION_NAME} exists.")
    except Exception as e:
        print(f"⚠️ WARNING: Failed to connect to Qdrant on startup. Check URL: {QDRANT_URL}")
        print(f"Error details: {e}")
        print("The app will continue running, but requests relying on Qdrant will fail until it is ready.")

@app.get("/")
def health_check():
    """健康检查接口，用于确认服务活著"""
    return {"status": "ok", "service": "Telecom Ingest API"}

@app.post("/ingest")
async def ingest_file(file: UploadFile, file_id: str = Form(...)):
    """接收文件 -> LlamaParse -> 切片 -> 本地向量化 -> 存入 Qdrant"""
    temp_filename = f"/tmp/{uuid.uuid4()}_{file.filename}"
    try:
        # 1. 保存临时文件
        content = await file.read()
        with open(temp_filename, "wb") as f:
            f.write(content)
        
        # 2. 调用 LlamaParse
        print(f"Parsing file: {file.filename}...")
        parser = LlamaParse(api_key=LLAMA_CLOUD_API_KEY, result_type="markdown")
        documents = await parser.aload_data(temp_filename)
        
        if not documents:
            raise HTTPException(status_code=400, detail="LlamaParse returned no content.")
            
        markdown_text = documents[0].text
        
        # 3. 切片
        print("Splitting text...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(markdown_text)
        
        if not chunks:
            return {"status": "warning", "message": "File parsed but no chunks created (text too short?)"}

        print(f"Upserting {len(chunks)} chunks to Qdrant...")

        # 4. 向量化 + 入库 (使用 FastEmbed，自动下载模型并运行 CPU 推理)
        # 注意：第一次运行会下载模型（约500MB），可能会稍慢
        client.add(
            collection_name=COLLECTION_NAME,
            documents=chunks, 
            metadata=[{"file_id": file_id, "chunk_index": i, "source": file.filename} for i in range(len(chunks))],
            ids=[str(uuid.uuid4()) for _ in range(len(chunks))]
        )
        
        return {"status": "success", "chunks_count": len(chunks)}
        
    except Exception as e:
        print("Error during ingestion:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 清理临时文件
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

@app.post("/delete")
async def delete_file(file_id: str = Form(...)):
    """根据 file_id 删除文档"""
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
    """
    搜索接口：n8n Agent 可以直接调这个接口，不用自己做 Embedding
    """
    try:
        # client.query 也会自动使用 FastEmbed 将 query 转为向量
        results = client.query(
            collection_name=COLLECTION_NAME,
            query_text=query,
            limit=limit
        )
        
        # 格式化返回给 LLM
        return [
            {
                "content": res.document, # FastEmbed 模式下 document 存放在这里
                "score": res.score,
                "metadata": res.metadata
            } 
            for res in results
        ]
    except Exception as e:
        print(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- 启动入口 (关键修正) ---
if __name__ == "__main__":
    import uvicorn
    # 关键：从环境变量获取 PORT，如果不存在则默认 8000
    # Zeabur 会设置这个环境变量
    port = int(os.getenv("PORT", 8000))
    print(f"Starting server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)