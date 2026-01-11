import os
import uuid
import shutil
import zipfile
import traceback
from pathlib import Path
from typing import List, Optional

# 🟢 引入 Redis 库
import redis

from fastapi import FastAPI, UploadFile, Form, HTTPException, File
from fastapi.middleware.cors import CORSMiddleware
from llama_parse import LlamaParse
from qdrant_client import QdrantClient, models
from flashrank import Ranker, RerankRequest
from pydantic import BaseModel

# 🆕 LlamaIndex 相关导入
try:
    from llama_index.core import Document
    from llama_index.core.node_parser import MarkdownElementNodeParser
    print("✅ LlamaIndex modules imported successfully")
    HAS_LLAMAINDEX = True
except ImportError as e:
    print(f"⚠️ Warning: LlamaIndex import error: {e}")
    print("   Will use fallback mode (optimized chunking)")
    HAS_LLAMAINDEX = False
    MarkdownElementNodeParser = None
    Document = None

# --- 1. 环境变量读取 ---
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# 🟢 Redis 配置
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

COLLECTION_NAME = "telecom_collection_v2"
TABLES_COLLECTION_NAME = "telecom_tables_v2"  # 🆕 专门存储表格

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
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    prefer_grpc=False
)

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
    return {
        "status": "ok",
        "service": "Telecom Ingest API (With MarkdownElementNodeParser)",
        "features": ["LlamaParse", "MarkdownElementNodeParser", "Table Extraction", "Qdrant+FlashRank"]
    }

# ========== Pydantic 数据模型 ==========

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

# ========== 🆕 核心：使用 MarkdownElementNodeParser 处理文档 ==========

async def process_document_with_element_parser(
    file_path: str,
    filename: str,
    group_id: str,
    source_package: str
) -> dict:
    """
    使用 MarkdownElementNodeParser 处理文档
    分别处理文本节点和表格对象
    """
    print(f"📄 Processing: {filename}")

    # 1. 使用 LlamaParse 解析文档
    parser = LlamaParse(
        api_key=LLAMA_CLOUD_API_KEY,
        result_type="markdown",
        premium_mode=True,
        verbose=True,
        parsing_instruction="""
这是一个电信运营商的渠道政策文档，请按以下要求解析：

【表格处理 - 最高优先级】
1. **必须保留所有表格的完整结构**，包括嵌套表格、合并单元格
2. **跨页表格必须合并**成一个完整的表格
3. 表格输出为 Markdown 格式，使用标准语法
4. **不要遗漏任何数字、金额、百分比**
5. 保留表格标题和说明文字

【文本处理】
1. 保留所有业务名称、产品名称、活动名称
2. 保留关键条款、条件说明、注意事项
3. 分级标题用 # ## ### 等 Markdown 语法标注

关键原则：宁可保留多余信息，也不要遗漏任何业务规则和数字！
        """.strip()
    )

    try:
        documents = await parser.aload_data(file_path)
        if not documents:
            print(f"⚠️ Warning: No text found in {filename}")
            return {"success": False, "error": "No documents parsed"}

        markdown_text = documents[0].text
        doc_type = guess_doc_type(filename)

        # 🆕 2. 检查是否可用 MarkdownElementNodeParser
        if HAS_LLAMAINDEX:
            print("  ✨ Using MarkdownElementNodeParser (table extraction mode)")
            return await _process_with_element_parser(
                markdown_text, filename, group_id, source_package, doc_type
            )
        else:
            print("  ⚠️ Using fallback mode (optimized for tables)")
            return await _process_with_fallback(
                markdown_text, filename, group_id, source_package, doc_type
            )

    except Exception as e:
        print(f"❌ Error processing {filename}: {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}


async def _process_with_element_parser(
    markdown_text: str,
    filename: str,
    group_id: str,
    source_package: str,
    doc_type: str
) -> dict:
    """使用 MarkdownElementNodeParser 处理（推荐模式）"""
    try:
        # 使用 MarkdownElementNodeParser 解析
        node_parser = MarkdownElementNodeParser(
            num_workers=4,  # 并发处理
        )

        # 创建 LlamaIndex Document 对象
        llama_doc = Document(text=markdown_text, metadata={"filename": filename})

        # 获取节点和对象
        nodes = node_parser.get_nodes_from_documents([llama_doc])
        base_nodes, objects = node_parser.get_nodes_and_objects(nodes)

        print(f"  📊 Extracted {len(base_nodes)} text nodes")
        print(f"  📋 Extracted {len(objects)} table objects")

        total_stored = 0

        # 📌 存储文本节点
        from qdrant_client.models import PointStruct
        points_to_upload = []

        for i, node in enumerate(base_nodes):
            if node.text.strip():
                points_to_upload.append(
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector={},  # Qdrant 会自动生成向量
                        payload={
                            "document": node.text,
                            "group_id": group_id,
                            "filename": filename,
                            "doc_type": doc_type,
                            "chunk_type": "text",
                            "node_index": i,
                            "source_package": source_package
                        }
                    )
                )

        # 📌 存储表格对象（完整表格，不被切断！）
        for i, obj in enumerate(objects):
            if obj.text.strip():
                points_to_upload.append(
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector={},  # Qdrant 会自动生成向量
                        payload={
                            "document": obj.text,
                            "group_id": group_id,
                            "filename": filename,
                            "doc_type": doc_type,
                            "chunk_type": "table",
                            "table_index": i,
                            "source_package": source_package,
                            "is_table": True
                        }
                    )
                )

        # 批量上传
        if points_to_upload:
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points_to_upload
            )
            total_stored = len(points_to_upload)

        print(f"  ✅ Stored {total_stored} chunks (text + tables)")

        return {
            "success": True,
            "text_nodes": len(base_nodes),
            "table_objects": len(objects),
            "total_chunks": total_stored,
            "mode": "element_parser"
        }

    except Exception as e:
        print(f"❌ Element Parser failed, falling back: {e}")
        traceback.print_exc()
        return await _process_with_fallback(
            markdown_text, filename, group_id, source_package, doc_type
        )


async def _process_with_fallback(
    markdown_text: str,
    filename: str,
    group_id: str,
    source_package: str,
    doc_type: str
) -> dict:
    """回退模式：使用大 chunk_size 保留表格完整性"""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    print("  🔄 Using fallback mode (large chunk size)")

    # 使用更大的 chunk_size 减少切断表格的概率
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,  # 增大到4000
        chunk_overlap=800,  # 增大 overlap
        separators=[
            "\n\n##",
            "\n\n###",
            "\n\n",
            "\n| ",  # 尝试在表格行前切分
            "\n",
            "。",
            " ",
            ""
        ],
    )

    chunks = splitter.split_text(markdown_text)
    print(f"  📊 Split into {len(chunks)} chunks")

    # 🆕 使用批量上传
    from qdrant_client.models import PointStruct
    points_to_upload = []

    for i, chunk in enumerate(chunks):
        if chunk.strip():
            # 检测是否包含表格
            is_table = "|" in chunk and ("|---" in chunk or "| ===" in chunk)

            points_to_upload.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector={},
                    payload={
                        "document": chunk,
                        "group_id": group_id,
                        "filename": filename,
                        "doc_type": doc_type,
                        "chunk_type": "table" if is_table else "text",
                        "chunk_index": i,
                        "source_package": source_package,
                        "is_table": is_table
                    }
                )
            )

    # 批量上传
    if points_to_upload:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points_to_upload
        )
        total_stored = len(points_to_upload)
    else:
        total_stored = 0

    print(f"  ✅ Stored {total_stored} chunks (fallback mode)")

    return {
        "success": True,
        "text_nodes": len([c for c in chunks if "|" not in c]),
        "table_objects": len([c for c in chunks if "|" in c]),
        "total_chunks": total_stored,
        "mode": "fallback"
    }

# ========== 核心业务端点 ==========

@app.post("/ingest")
async def ingest_package(file: UploadFile = File(...), package_id: str = Form(None)):
    """
    文档入库接口 - 🆕 使用 MarkdownElementNodeParser
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

        # 🆕 统计信息
        total_text_nodes = 0
        total_table_objects = 0
        processed_files = []

        # 🆕 使用新的 Element Parser 处理每个文件
        for file_path in files_to_process:
            fname = os.path.basename(file_path)
            result = await process_document_with_element_parser(
                file_path=file_path,
                filename=fname,
                group_id=group_id,
                source_package=file.filename
            )

            if result["success"]:
                total_text_nodes += result.get("text_nodes", 0)
                total_table_objects += result.get("table_objects", 0)
                processed_files.append({
                    "filename": fname,
                    "status": "success",
                    "text_nodes": result.get("text_nodes", 0),
                    "table_objects": result.get("table_objects", 0)
                })
            else:
                processed_files.append({
                    "filename": fname,
                    "status": "failed",
                    "error": result.get("error", "Unknown error")
                })

        total_chunks = total_text_nodes + total_table_objects

        if total_chunks == 0:
            return {
                "status": "error",
                "msg": "No documents parsed successfully.",
                "processed_files": processed_files
            }

        return {
            "status": "success",
            "group_id": group_id,
            "total_text_nodes": total_text_nodes,
            "total_table_objects": total_table_objects,
            "total_chunks": total_chunks,
            "processed_files": processed_files
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(base_tmp_dir):
            shutil.rmtree(base_tmp_dir)

@app.post("/delete")
async def delete_package(target_id: str = Form(..., description="填入 group_id 或 file_id")):
    """删除文档 - 🆕 同时删除文本和表格"""
    try:
        # 删除主集合
        if client.collection_exists(COLLECTION_NAME):
            client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[models.FieldCondition(key="group_id", match=models.MatchValue(value=target_id))]
                    )
                )
            )

        # 🆕 删除表格集合
        if client.collection_exists(TABLES_COLLECTION_NAME):
            client.delete(
                collection_name=TABLES_COLLECTION_NAME,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[models.FieldCondition(key="group_id", match=models.MatchValue(value=target_id))]
                    )
                )
            )

        return {"status": "deleted", "target_id": target_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset")
async def reset_database():
    """
    一键重置：同时清空 Qdrant（文本+表格）和 Redis
    """
    report = []

    # 1. 清空主集合
    try:
        client.delete_collection(COLLECTION_NAME)
        report.append("Qdrant text collection deleted")
    except Exception as e:
        report.append(f"Qdrant text skipped ({str(e)})")

    # 🆕 2. 清空表格集合
    try:
        client.delete_collection(TABLES_COLLECTION_NAME)
        report.append("Qdrant tables collection deleted")
    except Exception as e:
        report.append(f"Qdrant tables skipped ({str(e)})")

    # 3. 清空 Redis
    try:
        r = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD,
            decode_responses=True,
            socket_timeout=3
        )
        r.flushdb()
        report.append("Redis memory flushed")
    except Exception as e:
        print(f"❌ Redis Reset Failed: {e}")
        report.append(f"Redis failed: {str(e)}")

    return {"status": "success", "details": " | ".join(report)}

@app.post("/search")
async def search_docs(query: str = Form(...), limit: int = 5):
    """
    🆕 搜索接口 - 同时搜索文本和表格
    使用 query_points 替代已弃用的 query 方法
    """
    try:
        all_results = []

        # 🆕 使用中文 embedding 模型
        from qdrant_client.models import Document, QueryType

        # 1. 搜索文本集合
        if client.collection_exists(COLLECTION_NAME):
            print(f"🔎 Searching text collection for: {query}")
            text_results = client.query_points(
                collection_name=COLLECTION_NAME,
                query=Document(text=query, model="BAAI/bge-small-zh-v1.5"),  # 🆕 中文模型
                limit=200,
                with_payload=True,
            )

            for res in text_results.points:
                # 🆕 从 payload 中提取数据
                all_results.append({
                    "id": str(res.id),
                    "text": res.payload.get("document", ""),
                    "meta": res.payload,
                    "source": "text",
                    "score": res.score  # 🆕 直接使用返回的 score
                })

        # 2. 🆕 搜索表格集合（重点！）
        if client.collection_exists(TABLES_COLLECTION_NAME):
            print(f"📋 Searching tables collection for: {query}")
            table_results = client.query_points(
                collection_name=TABLES_COLLECTION_NAME,
                query=Document(text=query, model="BAAI/bge-small-zh-v1.5"),  # 🆕 中文模型
                limit=100,
                with_payload=True,
            )

            for res in table_results.points:
                # 🆕 从 payload 中提取数据
                all_results.append({
                    "id": str(res.id),
                    "text": res.payload.get("document", ""),
                    "meta": res.payload,
                    "source": "table",
                    "score": res.score
                })

        if not all_results:
            return []

        print(f"  📊 Found {len(all_results)} results (text + tables)")

        # 3. 重排序（FlashRank）- 仍然有用，可以进一步优化结果
        passages = [
            {"id": r["id"], "text": r["text"], "meta": r["meta"]}
            for r in all_results
        ]

        rerank_request = RerankRequest(query=query, passages=passages)
        ranked_results = reranker.rerank(rerank_request)

        top_results = ranked_results[:limit]

        # 4. 🆕 在结果中标注来源
        return [
            {
                "content": res["text"],
                "score": float(res["score"]),
                "metadata": res["meta"],
                "content_type": "table" if res["meta"].get("is_table") else "text"
            }
            for res in top_results
        ]

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ========== 🆕 Agentic RAG 增强端点 ==========

@app.post("/analyze_query")
async def analyze_query(request: QueryAnalysisRequest):
    """分析查询复杂度，返回执行计划"""
    query = request.query.lower()

    analysis = {
        "query_type": "simple",
        "sub_queries": [],
        "required_tools": ["search"],
        "reasoning": "直接检索",
        "suggested_approach": "single_step"
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
        analysis["query_type"] = "complex"
        analysis["required_tools"] = ["search", "compare"]
        analysis["suggested_approach"] = "parallel"
        analysis["reasoning"] = "检测到跨年度对比查询，需要分别检索各年度文档"

        years_found = []
        for year in ["2022", "2023", "2024", "2025"]:
            if year in query:
                years_found.append(year)

        if years_found:
            base_query = request.query
            for yr in years_found:
                base_query = base_query.replace(yr, "").replace("历年", "").replace("逐年", "")

            analysis["sub_queries"] = [
                f"{yr}年{base_query.strip()}".replace("  ", " ")
                for yr in years_found
            ]

    elif has_table:
        analysis["query_type"] = "table"
        analysis["required_tools"] = ["search", "extract_table"]
        analysis["reasoning"] = "检测到表格数据查询，会优先从表格集合检索"

    elif has_aggregation or (has_calculation and "、" in query):
        analysis["query_type"] = "aggregation"
        analysis["required_tools"] = ["search", "calculate"]
        analysis["suggested_approach"] = "multi_step"
        analysis["reasoning"] = "检测到数据聚合或复杂计算需求，建议分步检索"

        if "、" in request.query:
            sub_questions = [q.strip() for q in request.query.split("、") if q.strip()]
            analysis["sub_queries"] = sub_questions

    else:
        analysis["reasoning"] = "简单查询，将同时搜索文本和表格"

    return analysis

@app.post("/extract_tables")
async def extract_tables(request: ExtractTableRequest):
    """
    🆕 从表格集合中提取表格数据
    """
    doc_id = request.document_id

    try:
        if not client.collection_exists(TABLES_COLLECTION_NAME):
            return {
                "document_id": doc_id,
                "table_count": 0,
                "tables": [],
                "error": "Tables collection not found"
            }

        # 搜索表格集合
        search_result = client.query(
            collection_name=TABLES_COLLECTION_NAME,
            query_text=doc_id,
            limit=100
        )

        if not search_result:
            return {
                "document_id": doc_id,
                "table_count": 0,
                "tables": [],
                "message": "No tables found for this document"
            }

        tables = []
        for res in search_result:
            tables.append({
                "content": res.document,
                "source": res.metadata.get("filename", "unknown"),
                "chunk_id": str(res.id),
                "table_index": res.metadata.get("table_index", 0),
                "row_count": res.document.count("\n") + 1
            })

        return {
            "document_id": doc_id,
            "total_chunks": len(search_result),
            "table_count": len(tables),
            "tables": tables[:10]
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compare_documents")
async def compare_documents(request: CompareDocumentsRequest):
    """🆕 跨文档对比 - 同时搜索文本和表格"""
    doc_ids = request.doc_ids
    results = {}

    try:
        for doc_id in doc_ids:
            # 搜索主集合
            text_results = []
            if client.collection_exists(COLLECTION_NAME):
                text_search = client.query(
                    collection_name=COLLECTION_NAME,
                    query_text=doc_id,
                    limit=30
                )
                text_results = [res.document for res in text_search[:3]]

            # 搜索表格集合
            table_results = []
            if client.collection_exists(TABLES_COLLECTION_NAME):
                table_search = client.query(
                    collection_name=TABLES_COLLECTION_NAME,
                    query_text=doc_id,
                    limit=30
                )
                table_results = [res.document for res in table_search[:3]]

            results[doc_id] = {
                "text_chunks": len(text_results),
                "table_chunks": len(table_results),
                "text_samples": text_results,
                "table_samples": table_results
            }

        return {
            "comparison_result": results,
            "summary": f"对比了 {len(doc_ids)} 个文档"
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ========== 🆕 统计信息端点 ==========

@app.get("/stats")
async def get_stats():
    """获取知识库统计信息"""
    stats = {
        "collections": {}
    }

    # 主集合统计
    if client.collection_exists(COLLECTION_NAME):
        collection_info = client.get_collection(COLLECTION_NAME)
        stats["collections"]["text"] = {
            "name": COLLECTION_NAME,
            "points_count": collection_info.points_count,
            "status": "active"
        }
    else:
        stats["collections"]["text"] = {"status": "not_created"}

    # 表格集合统计
    if client.collection_exists(TABLES_COLLECTION_NAME):
        collection_info = client.get_collection(TABLES_COLLECTION_NAME)
        stats["collections"]["tables"] = {
            "name": TABLES_COLLECTION_NAME,
            "points_count": collection_info.points_count,
            "status": "active"
        }
    else:
        stats["collections"]["tables"] = {"status": "not_created"}

    return stats

# ========== 端点总结 ==========
# /ingest       - 🆕 文档入库（使用 MarkdownElementNodeParser）
# /search       - 🆕 搜索（同时搜索文本和表格）
# /delete       - 删除文档（同时删除文本和表格）
# /reset        - 重置数据库（文本+表格+Redis）
# /stats        - 🆕 统计信息
# /analyze_query - 分析查询复杂度
# /extract_tables - 提取表格数据
# /compare_documents - 跨文档对比
