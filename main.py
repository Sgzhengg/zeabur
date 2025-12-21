import os
import uuid
import traceback
from fastapi import FastAPI, UploadFile, Form, HTTPException, File
from fastapi.middleware.cors import CORSMiddleware
from llama_parse import LlamaParse
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models
# 引入重排序库
from flashrank import Ranker, RerankRequest

# --- 1. 环境变量读取 ---
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# 集合名称
COLLECTION_NAME = "telecom_collection_v2"

print(f"DEBUG CONFIG: URL={QDRANT_URL}, LLAMA_KEY_LEN={len(LLAMA_CLOUD_API_KEY) if LLAMA_CLOUD_API_KEY else 0}")

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
    return {"status": "ok", "service": "Telecom Ingest API with Rerank"}

@app.post("/ingest")
async def ingest_file(file: UploadFile = File(...), file_id: str = Form(...)):
    """入库接口"""
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
    """高级检索接口"""
    try:
        print(f"🔎 Searching for: {query}")
        
        # 1. 扩大召回
        search_result = client.query(
            collection_name=COLLECTION_NAME,
            query_text=query,
            limit=50 
        )
        
        if not search_result:
            return []

        # 2. 格式化
        passages = [
            {
                "id": str(res.id), 
                "text": res.document, 
                "meta": res.metadata
            }
            for res in search_result
        ]

        # 3. 重排序 (修正了这里的方法名)
        print(f"⚖️ Reranking {len(passages)} documents...")
        rerank_request = RerankRequest(query=query, passages=passages)
        
        # 🔴 关键修正：从 .rank() 改为 .rerank()
        ranked_results = reranker.rerank(rerank_request)

        # 4. 截取
        top_results = ranked_results[:limit]
        
        print(f"✅ Return top {len(top_results)} results.")
        
        return [
            {
                "content": res["text"],
                "score": res["score"],
                "metadata": res["meta"]
            } 
            for res in top_results
        ]

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
        