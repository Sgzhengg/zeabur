#!/usr/bin/env python3
"""
快速测试：验证关键依赖是否可用
"""
import sys

print("🔍 快速依赖检查")
print("=" * 50)

# 测试关键依赖
tests = [
    ("FastAPI", "fastapi"),
    ("Qdrant", "qdrant_client"),
    ("FlashRank", "flashrank"),
    ("LlamaIndex Core", "llama_index.core"),
    ("LangChain", "langchain_text_splitters"),
]

all_ok = True
for name, module in tests:
    try:
        __import__(module)
        print(f"✅ {name}")
    except ImportError as e:
        print(f"❌ {name}: {e}")
        all_ok = False

print("=" * 50)

# 测试 MarkdownElementNodeParser
print("\n🎯 测试 MarkdownElementNodeParser")
try:
    from llama_index.core.node_parser import MarkdownElementNodeParser
    print("✅ MarkdownElementNodeParser 可用")
    print("   → 将使用理想模式（表格单独存储）")
except ImportError as e:
    print(f"⚠️  MarkdownElementNodeParser 不可用: {e}")
    print("   → 将使用回退模式（chunk_size=4000）")
    print("   → 回退模式也能很好地保留表格！")

print("\n" + "=" * 50)
if all_ok:
    print("✅ 所有关键依赖正常！")
    print("🚀 可以推送到 Zeabur 了")
else:
    print("❌ 有依赖缺失，请检查")
    sys.exit(1)
