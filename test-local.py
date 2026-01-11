#!/usr/bin/env python3
"""
本地测试脚本 - 验证导入是否正常
"""
import sys

print("=" * 60)
print("🧪 本地依赖测试")
print("=" * 60)
print()

# 测试1: 基础依赖
print("📦 测试 1: 基础依赖")
print("-" * 60)
try:
    import fastapi
    print("✅ FastAPI:", fastapi.__version__)
except ImportError as e:
    print("❌ FastAPI:", e)
    sys.exit(1)

try:
    import qdrant_client
    print("✅ Qdrant Client: OK")
except ImportError as e:
    print("❌ Qdrant Client:", e)

try:
    import flashrank
    print("✅ FlashRank: OK")
except ImportError as e:
    print("❌ FlashRank:", e)

print()

# 测试2: LlamaIndex 依赖
print("📦 测试 2: LlamaIndex 依赖")
print("-" * 60)
try:
    from llama_index.core import Document
    print("✅ llama-index.core: OK")
except ImportError as e:
    print("⚠️  llama-index.core:", e)
    print("   → 将使用回退模式")

try:
    from llama_index.core.node_parser import MarkdownElementNodeParser
    print("✅ MarkdownElementNodeParser: 可用")
except ImportError as e:
    print("❌ MarkdownElementNodeParser:", e)
    print("   → 将使用回退模式（大 chunk_size）")

try:
    from llama_index.embeddings.fastembed import FastEmbedEmbedding
    print("✅ FastEmbedEmbedding: OK")
except ImportError as e:
    print("⚠️  FastEmbedEmbedding:", e)

print()

# 测试3: 旧依赖
print("📦 测试 3: 旧版依赖（LangChain）")
print("-" * 60)
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    print("✅ RecursiveCharacterTextSplitter: OK")
except ImportError as e:
    print("❌ RecursiveCharacterTextSplitter:", e)

print()
print("=" * 60)
print("✅ 测试完成！")
print("=" * 60)
print()
print("💡 提示：")
print("  - 如果所有依赖都正常，可以推送到 Zeabur")
print("  - 如果 MarkdownElementNodeParser 不可用，会自动使用回退模式")
print("  - 回退模式使用 chunk_size=4000，也能较好地保留表格")
print()
