# 🔧 Zeabur 部署修复指南

## ✅ 已修复的问题

### 问题1：包不存在
- ❌ `llama-index-readers-llamaparse` - 删除
- ❌ `llama-index-vector-stores-qdrant` - 删除
- ✅ 只保留 `llama-index-core` - **MarkdownElementNodeParser 已包含**

### 问题2：Python 版本
- Zeabur 可能使用 Python 3.12+
- 简化依赖，避免版本冲突

### 问题3：回退机制
- ✅ 添加了 `HAS_LLAMAINDEX` 标志
- ✅ 自动检测并切换模式

---

## 🚀 立即部署

### 步骤1：本地测试（可选）

```bash
cd C:\Users\ASUS\OneDrive\Desktop\elecom-ingest-api

# 运行快速测试
python quick-test.py
```

**预期输出：**
```
✅ FastAPI
✅ Qdrant
✅ FlashRank
✅ LlamaIndex Core
✅ LangChain

🎯 测试 MarkdownElementNodeParser
✅ MarkdownElementNodeParser 可用（或 ⚠️ 不可用，也OK）
```

### 步骤2：提交代码

```bash
cd C:\Users\ASUS\OneDrive\Desktop\elecom-ingest-api

git add .
git commit -m "Fix deployment: simplify requirements

- Remove llama-index-readers-llamaparse (package doesn't exist)
- Keep only llama-index-core (MarkdownElementNodeParser included)
- Add HAS_LLAMAINDEX flag for auto fallback
- Remove unnecessary embed_model initialization"

git push
```

### 步骤3：等待 Zeabur 部署

**查看日志，期望看到：**
```
✅ LlamaIndex modules imported successfully
⏳ Initializing FlashRank Reranker...
✅ Reranker initialized!
🚀 Connecting to Qdrant at: ...
✅ Connected to Qdrant!
```

**或者看到警告（也OK）：**
```
⚠️ Warning: LlamaIndex import error: ...
   Will use fallback mode (optimized chunking)
```

---

## 📊 两种模式对比

| 模式 | 依赖 | 表格处理 | 效果 |
|------|------|----------|------|
| **理想模式** | llama-index-core | 单独存储到 `telecom_tables_v2` | ⭐⭐⭐⭐⭐ |
| **回退模式** | 只需 langchain | 大chunk_size=4000 | ⭐⭐⭐⭐ |

**两种模式都比原来的 chunk_size=2000 好得多！**

---

## 🧪 验证部署

### 1. 检查服务状态
```bash
curl http://elecom-ingest-api:8080/
```

**返回：**
```json
{
  "status": "ok",
  "service": "Telecom Ingest API (With MarkdownElementNodeParser)"
}
```

### 2. 查看统计
```bash
curl http://elecom-ingest-api:8080/stats
```

### 3. 重置并重新导入
```bash
# 重置
curl -X POST http://elecom-ingest-api:8080/reset

# 导入测试文档
curl -X POST http://elecom-ingest-api:8080/ingest \
  -F "file=@C:\Users\ASUS\OneDrive\Desktop\广州移动12月份营销方案\2025年12月渠道产品政策（1128版）(1).docx"
```

---

## ❓ 如果还是失败

### 方案A：完全移除 llama-index

如果 `llama-index-core` 还是导致问题，可以完全移除：

**requirements.txt**
```txt
fastapi
uvicorn[standard]
python-multipart
llama-parse
qdrant-client
langchain-text-splitters
fastembed
flashrank
redis
pydantic>=2.0.0
```

这样会强制使用**回退模式**（chunk_size=4000），效果也很好！

### 方案B：检查 Zeabur Python 版本

在 Zeabur 项目设置中，指定 Python 版本：

**创建 `runtime.txt`**
```
python-3.11.9
```

---

## ✅ 成功标志

1. ✅ 服务正常启动
2. ✅ `/ingest` 能导入文档
3. ✅ `/search` 能搜索到内容
4. ✅ 日志显示使用的模式（element_parser 或 fallback）

---

## 🎯 下一步

部署成功后：
1. 测试"新入网自动充业务"查询
2. 查看是否返回完整表格
3. 对比之前的搜索结果

---

**预计部署时间：** 2-3分钟
**首次构建：** 3-5分钟（安装依赖）
**后续构建：** 1-2分钟
