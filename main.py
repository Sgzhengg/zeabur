import os
import uuid
import traceback
from fastapi import FastAPI, UploadFile, Form, HTTPException, File
from fastapi.middleware.cors import CORSMiddleware
from llama_parse import LlamaParse
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models

# --- 恢复为读取环境变量 ---
# 这样代码就通用了，Key 都在 Zeabur 界面里管理
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") 

COLLECTION_NAME = "telecom_collection"

# 增加一个启动前的打印检查，方便看日志调试
print(f"DEBUG CONFIG: URL={QDRANT_URL}, LLAMA_KEY_LEN={len(LLAMA_CLOUD_API_KEY) if LLAMA_CLOUD_API_KEY else 0}")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化 Qdrant
# 如果环境变量没读到，这里会报错，正好帮我们发现问题
if not QDRANT_URL:
    raise ValueError("❌ Fatal Error: QDRANT_URL is missing in environment variables!")

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, prefer_grpc=False)

@app.on_event("startup")
def startup_event():
    print(f"🚀 Connecting to Qdrant at: {QDRANT_URL} ...")
    try:
        if not client.collection_exists(COLLECTION_NAME):
            print(f"Collection {COLLECTION_NAME} not found, creating...")
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
            )
            print(f"✅ Collection {COLLECTION_NAME} created successfully.")
        else:
            print(f"✅ Collection {COLLECTION_NAME} exists. Ready.")
    except Exception as e:
        print(f"❌ Connection Failed! Error: {e}")

@app.get("/")
def health_check():
    return {"status": "ok", "service": "Telecom Ingest API"}

@app.post("/ingest")
async def ingest_file(file: UploadFile = File(...), file_id: str = Form(...)):
    # 双重检查
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
    try:
        results = client.query(
            collection_name=COLLECTION_NAME,
            query_text=query,
            limit=limit
        )
        return [{"content": res.document, "score": res.score, "metadata": res.metadata} for res in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))