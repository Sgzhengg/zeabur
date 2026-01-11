import os
import uuid
import shutil
import zipfile
import traceback
from pathlib import Path
from typing import List

# 🟢 引入 Redis 库
import redis

from fastapi import FastAPI, UploadFile, Form, HTTPException, File
from fastapi.middleware.cors import CORSMiddleware
from llama_parse import LlamaParse
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models
from flashrank import Ranker, RerankRequest
from pydantic import BaseModel

# --- 1. 环境变量读取 ---
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# 🟢 Redis 配置 (根据你的截图，默认 Host 改为 "redis")
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None) # 如果有密码，请在 Zeabur 变量里设置

COLLECTION_NAME = "telecom_collection_v2"

print(f"DEBUG CONFIG: QDRANT_URL={QDRANT_URL}, REDIS_HOST={REDIS_HOST}")

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
        print(f"✅ Connected to Qdrant! Found {len(collections.collections)} collections.")
    except Exception as e:
        print(f"❌ Qdrant Connection Failed! Error: {e}")

@app.get("/")
def health_check():
    return {"status": "ok", "service": "Telecom Ingest API (With Agentic RAG Endpoints)"}

# ========== Pydantic 数据模型（用于新端点） ==========

class QueryAnalysisRequest(BaseModel):
    query: str

class ExtractTableRequest(BaseModel):
    document_id: str

class CompareDocumentsRequest(BaseModel):
    doc_ids: List[str]

# ========== 辅助函数 ==========

def extract_zip(zip_path: str, extract_to: str):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def guess_doc_type(filename: str) -> str:
    main_keywords = ["通知", "公告", "管理办法", "规定", "主件", "正文"]
    if any(k in filename for k in main_keywords):
        return "main"
    return "attachment"

# ========== 核心业务端点 ==========

@app.post("/ingest")
async def ingest_package(file: UploadFile = File(...), package_id: str = Form(None)):
    """入库接口"""
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

        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(key="group_id", match=models.MatchValue(value=target_id))]
                )
            )
        )
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
    """
    一键重置：同时清空 Qdrant 和 Redis
    """
    report = []

    # 1. 清空 Qdrant
    try:
        client.delete_collection(COLLECTION_NAME)
        report.append("Qdrant collection deleted")
    except Exception as e:
        # 如果集合本来就不存在，不算错
        report.append(f"Qdrant skipped ({str(e)})")

    # 2. 🟢 清空 Redis (记忆)
    try:
        # 连接到 Redis
        r = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD,
            decode_responses=True,
            socket_timeout=3 # 设置超时防止卡死
        )
        # 执行清空指令
        r.flushdb()
        report.append("Redis memory flushed")
    except Exception as e:
        print(f"❌ Redis Reset Failed: {e}")
        report.append(f"Redis failed: {str(e)}")

    return {"status": "success", "details": " | ".join(report)}

@app.post("/search")
async def search_docs(query: str = Form(...), limit: int = 5):
    try:
        if not client.collection_exists(COLLECTION_NAME):
            return []

        print(f"🔎 Searching for: {query}")

        search_result = client.query(
            collection_name=COLLECTION_NAME,
            query_text=query,
            limit=300
        )

        if not search_result:
            return []

        passages = [
            {"id": str(res.id), "text": res.document, "meta": res.metadata}
            for res in search_result
        ]

        rerank_request = RerankRequest(query=query, passages=passages)
        ranked_results = reranker.rerank(rerank_request)

        top_results = ranked_results[:limit]

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

# ========== 🆕 Agentic RAG 增强端点 ==========

@app.post("/analyze_query")
async def analyze_query(request: QueryAnalysisRequest):
    """
    分析查询复杂度，返回执行计划
    帮助 AI Agent 决定检索策略
    """
    query = request.query.lower()

    # 默认简单查询
    analysis = {
        "query_type": "simple",              # simple | complex | table | aggregation
        "sub_queries": [],                    # 分解后的子查询
        "required_tools": ["search"],         # 需要的工具
        "reasoning": "直接检索",              # 推理说明
        "suggested_approach": "single_step"   # single_step | multi_step | parallel
    }

    # 检测关键词
    comparison_keywords = ["对比", "差异", "变化", "vs", "区别"]
    aggregation_keywords = ["总计", "统计", "汇总", "平均", "求和"]
    multi_year_keywords = ["2023", "2024", "2022", "2025", "历年", "逐年"]
    table_keywords = ["表格", "excel", "附件", "sheet", "明细"]
    calculation_keywords = ["计算", "激励", "提成", "金额", "费用", "合计"]

    has_comparison = any(kw in query for kw in comparison_keywords)
    has_aggregation = any(kw in query for kw in aggregation_keywords)
    has_multi_year = any(kw in query for kw in multi_year_keywords)
    has_table = any(kw in query for kw in table_keywords)
    has_calculation = any(kw in query for kw in calculation_keywords)

    # 分类逻辑
    if has_comparison and has_multi_year:
        # 复杂跨年度对比查询
        analysis["query_type"] = "complex"
        analysis["required_tools"] = ["search", "compare"]
        analysis["suggested_approach"] = "parallel"
        analysis["reasoning"] = "检测到跨年度对比查询，需要分别检索各年度文档"

        # 提取年份并分解查询
        years_found = []
        for year in ["2022", "2023", "2024", "2025"]:
            if year in query:
                years_found.append(year)

        if years_found:
            # 移除年份，保留核心问题
            base_query = request.query
            for yr in years_found:
                base_query = base_query.replace(yr, "").replace("历年", "").replace("逐年", "")

            # 生成子查询
            analysis["sub_queries"] = [
                f"{yr}年{base_query.strip()}".replace("  ", " ")
                for yr in years_found
            ]

    elif has_table:
        # 表格数据提取
        analysis["query_type"] = "table"
        analysis["required_tools"] = ["search", "extract_table"]
        analysis["reasoning"] = "检测到表格数据查询，建议优先提取 Excel 附件"

    elif has_aggregation or (has_calculation and "、" in query):
        # 数据聚合或复杂计算
        analysis["query_type"] = "aggregation"
        analysis["required_tools"] = ["search", "calculate"]
        analysis["suggested_approach"] = "multi_step"
        analysis["reasoning"] = "检测到数据聚合或复杂计算需求，建议分步检索"

        # 如果包含多个问题（顿号分隔）
        if "、" in request.query:
            sub_questions = [q.strip() for q in request.query.split("、") if q.strip()]
            analysis["sub_queries"] = sub_questions

    else:
        # 简单查询
        analysis["reasoning"] = "简单查询，可直接检索"

    return analysis

@app.post("/extract_tables")
async def extract_tables(request: ExtractTableRequest):
    """
    从文档中提取表格数据
    识别 Markdown 格式的表格并返回结构化数据
    """
    doc_id = request.document_id

    try:
        if not client.collection_exists(COLLECTION_NAME):
            return {
                "document_id": doc_id,
                "table_count": 0,
                "tables": [],
                "error": "Collection not found"
            }

        # 搜索该文档的所有片段
        search_result = client.query(
            collection_name=COLLECTION_NAME,
            query_text=doc_id,  # 用文档名/ID作为查询
            limit=100
        )

        if not search_result:
            return {
                "document_id": doc_id,
                "table_count": 0,
                "tables": [],
                "message": "No content found for this document"
            }

        # 过滤并提取表格内容
        tables = []
        for res in search_result:
            content = res.document

            # 简单检测 Markdown 表格：包含 | 和分隔线
            if "|" in content and ("|---" in content or "| ===" in content):
                tables.append({
                    "content": content,
                    "source": res.metadata.get("filename", "unknown"),
                    "chunk_id": str(res.id),
                    "doc_type": res.metadata.get("doc_type", "unknown"),
                    "row_count": content.count("\n") + 1  # 估算行数
                })

        return {
            "document_id": doc_id,
            "total_chunks": len(search_result),
            "table_count": len(tables),
            "tables": tables[:10]  # 最多返回10个表格，避免过大
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compare_documents")
async def compare_documents(request: CompareDocumentsRequest):
    """
    跨文档对比
    提取多个文档的关键信息，便于 Agent 进行对比分析
    """
    doc_ids = request.doc_ids
    results = {}

    try:
        if not client.collection_exists(COLLECTION_NAME):
            return {
                "comparison_result": {},
                "error": "Collection not found"
            }

        for doc_id in doc_ids:
            # 搜索每个文档
            search_result = client.query(
                collection_name=COLLECTION_NAME,
                query_text=doc_id,
                limit=50
            )

            if not search_result:
                results[doc_id] = {
                    "found": False,
                    "message": "No content found"
                }
                continue

            # 提取关键信息
            # 1. 文件名
            filenames = set(res.metadata.get("filename", "") for res in search_result)

            # 2. 关键片段（取前3个相关度最高的）
            key_points = [res.document for res in search_result[:3]]

            # 3. 文档类型
            doc_types = set(res.metadata.get("doc_type", "") for res in search_result)

            results[doc_id] = {
                "found": True,
                "filenames": list(filenames),
                "doc_types": list(doc_types),
                "total_chunks": len(search_result),
                "key_points": key_points,
                "sample_metadata": search_result[0].metadata if search_result else {}
            }

        return {
            "comparison_result": results,
            "summary": {
                "documents_compared": len(doc_ids),
                "successful": sum(1 for r in results.values() if r.get("found", False)),
                "failed": sum(1 for r in results.values() if not r.get("found", False))
            }
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ========== 端点总结 ==========
# /ingest       - 文档入库（ZIP/单文件）
# /search       - 向量搜索 + 重排序
# /delete       - 删除文档
# /reset        - 重置数据库（Qdrant + Redis）
# /analyze_query - 🆕 分析查询复杂度
# /extract_tables - 🆕 提取表格数据
# /compare_documents - 🆕 跨文档对比
