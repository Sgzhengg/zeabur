#!/usr/bin/env python3
"""
测试脚本：检查文档是否在知识库中
"""

import os
import requests
import json

# 配置
API_URL = "http://elecom-ingest-api:8080"  # 如果在本地测试，改为实际地址
SEARCH_ENDPOINT = f"{API_URL}/search"

# 测试查询
test_queries = [
    "新入网自动充业务",
    "新入网即充",
    "入网自动充",
    "流量券奖励",
    "潮玩青春卡",
    "渠道产品政策"
]

print("=" * 60)
print("🔍 知识库检索测试")
print("=" * 60)
print()

for query in test_queries:
    print(f"📝 查询: {query}")
    print("-" * 60)

    try:
        response = requests.post(
            SEARCH_ENDPOINT,
            data={"query": query, "limit": 5},
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )

        if response.status_code == 200:
            results = response.json()

            if len(results) == 0:
                print("❌ 未找到相关内容")
            else:
                print(f"✅ 找到 {len(results)} 条结果")
                print()

                for i, result in enumerate(results[:2], 1):  # 只显示前2条
                    print(f"结果 {i}:")
                    print(f"  相关度: {result.get('score', 0):.4f}")
                    print(f"  来源: {result.get('metadata', {}).get('filename', 'unknown')}")
                    print(f"  内容预览: {result.get('content', '')[:150]}...")
                    print()

        else:
            print(f"❌ 请求失败: {response.status_code}")
            print(response.text)

    except Exception as e:
        print(f"❌ 错误: {e}")

    print("=" * 60)
    print()
