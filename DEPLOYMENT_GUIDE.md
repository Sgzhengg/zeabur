# 🚀 MarkdownElementNodeParser 部署指南

## 📋 更新内容

### 1. 新增依赖
- `llama-index-core` - LlamaIndex 核心库
- `llama-index-readers-llama-parse` - LlamaParse 集成
- `llama-index-node-parser` - MarkdownElementNodeParser
- `llama-index-embeddings-fastembed` - Embedding 模型
- `llama-index-vector-stores-qdrant` - Qdrant 向量存储

### 2. 核心改进
- ✅ 使用 `MarkdownElementNodeParser` 自动识别表格边界
- ✅ 文本节点和表格对象**分别存储**到不同集合
- ✅ 搜索时同时检索文本和表格
- ✅ 表格**不会被切断**，保持完整性

### 3. 新增端点
- `GET /stats` - 查看知识库统计（文本/表格数量）

---

## 🔧 部署步骤

### 步骤1：提交代码

```bash
cd C:\Users\ASUS\OneDrive\Desktop\elecom-ingest-api

# 查看变更
git diff

# 提交
git add .
git commit -m "Upgrade to MarkdownElementNodeParser for better table extraction

- Add llama-index dependencies
- Separate text and tables into different collections
- Use MarkdownElementNodeParser to preserve table integrity
- Search both text and tables collections
- Add /stats endpoint"

# 推送
git push
```

### 步骤2：等待 Zeabur 自动部署

- Zeabur 检测到推送后会自动重新构建
- 首次构建可能需要 2-3 分钟（安装新依赖）
- 查看部署日志确认成功

### 步骤3：重置数据库（重要！）

```bash
# 清空旧数据
curl -X POST http://elecom-ingest-api:8080/reset

# 预期返回：
# {"status": "success", "details": "Qdrant text collection deleted | Qdrant tables collection deleted | Redis memory flushed"}
```

### 步骤4：重新导入文档

使用你的测试文档重新导入：
```bash
curl -X POST http://elecom-ingest-api:8080/ingest \
  -F "file=@C:\Users\ASUS\OneDrive\Desktop\广州移动12月份营销方案\2025年12月渠道产品政策（1128版）(1).docx"
```

---

## 🧪 测试验证

### 测试1：查看统计信息

```bash
curl http://elecom-ingest-api:8080/stats
```

**预期返回：**
```json
{
  "collections": {
    "text": {
      "name": "telecom_collection_v2",
      "points_count": 150,
      "status": "active"
    },
    "tables": {
      "name": "telecom_tables_v2",
      "points_count": 25,
      "status": "active"
    }
  }
}
```

### 测试2：搜索"新入网自动充"

```bash
curl -X POST http://elecom-ingest-api:8080/search \
  -d "query=新入网自动充业务" \
  -d "limit=5"
```

**关键指标：**
- 返回结果中应该有 `"content_type": "table"` 的结果
- 表格内容应该是**完整的**，不被切断

### 测试3：在 n8n 中测试

1. 打开 n8n 工作流
2. 执行测试：
   ```
   办理"新入网自动充业务"后，用户每月如何获得流量券奖励？
   一个号码最多可以获得多少次？
   ```
3. 查看返回结果中是否包含 `content_type: table`

---

## 🎯 预期效果对比

### 之前（直接 chunk 分割）
```
文档 → LlamaParse → Markdown → chunk_size=2000 → Qdrant
                                      ↓
                            表格可能被切断 ❌
```

### 现在（MarkdownElementNodeParser）
```
文档 → LlamaParse → Markdown → MarkdownElementNodeParser
                                              ↓
                            ┌─────────────────┴─────────────────┐
                            ↓                                   ↓
                      文本节点                            表格对象
                            ↓                                   ↓
                    telecom_collection_v2              telecom_tables_v2
                            ↓                                   ↓
                      完整文本块                        完整表格 ✅
```

---

## 🔍 问题排查

### 问题1：部署失败 - 依赖安装错误

**症状：** Zeabur 部署日志显示 `ModuleNotFoundError: No module named 'llama_index'`

**解决：**
```bash
# 检查 requirements.txt 格式
cat requirements.txt

# 确保没有版本冲突
# 如果使用 Python 3.8+，可能需要：
pip install llama-index-core --upgrade
```

### 问题2：搜索结果为空

**症状：** `/search` 返回 `[]`

**排查步骤：**
```bash
# 1. 检查集合是否存在
curl http://elecom-ingest-api:8080/stats

# 2. 查看日志
# 在 Zeabur 控制台查看实时日志

# 3. 重新导入文档
curl -X POST http://elecom-ingest-api:8080/reset
curl -X POST http://elecom-ingest-api:8080/ingest -F "file=@..."
```

### 问题3：表格还是找不到

**可能原因：**
1. 文档中没有"新入网自动充"这个词，而是"新入网即充"或其他变体
2. LlamaParse 解析时丢失了表格

**解决：**
```bash
# 测试不同的查询词
curl -X POST http://elecom-ingest-api:8080/search \
  -d "query=新入网即充" \
  -d "limit=10"

curl -X POST http://elecom-ingest-api:8080/search \
  -d "query=流量券" \
  -d "limit=10"

curl -X POST http://elecom-ingest-api:8080/search \
  -d "query=入网 自动充" \
  -d "limit=10"
```

---

## 📊 性能对比

| 指标 | 旧版本 | 新版本 |
|------|--------|--------|
| **表格完整性** | ❌ 可能被切断 | ✅ 100%完整 |
| **检索精度** | ⚠️ 取决于chunk大小 | ✅ 表格单独索引 |
| **存储空间** | 1个集合 | 2个集合（文本+表格） |
| **搜索速度** | 单集合搜索 | 双集合并行（稍慢但更准） |
| **维护成本** | 低 | 低（自动化） |

---

## ✅ 成功标志

部署成功的标志：
1. ✅ `/stats` 显示两个集合都有数据
2. ✅ 搜索结果包含 `"content_type": "table"`
3. ✅ 表格内容完整，包含所有列和行
4. ✅ "新入网自动充"能找到相关信息

---

## 🎓 参考资料

- [LlamaParse 官方文档](https://docs.llamaindex.ai/en/stable/examples/data_connectors/llama_parse/)
- [MarkdownElementNodeParser 说明](https://docs.llamaindex.ai/en/stable/examples/node_parser/markdown_element_node_parser/)
- [腾讯云文章：使用 LlamaParse 从文档创建知识图谱](https://cloud.tencent.com/developer/article/2429392)

---

## 💡 下一步优化（可选）

如果效果还不够理想，可以考虑：

1. **调整 MarkdownElementNodeParser 参数**
   ```python
   node_parser = MarkdownElementNodeParser(
       num_workers=8,  # 增加并发
       llm=your_llm,   # 使用LLM提取表格摘要
   )
   ```

2. **为表格单独生成摘要**
   ```python
   # 提取表格后，用LLM生成简短摘要
   table_summary = llm.complete(f"总结这个表格的内容：{table_content}")
   ```

3. **添加表格标题索引**
   ```python
   metadata = {
       "table_title": extract_title(table),
       "table_columns": extract_columns(table),
   }
   ```

---

**部署时间：** 约5分钟
**首次构建：** 约3-5分钟（安装新依赖）
**后续部署：** 约2分钟
