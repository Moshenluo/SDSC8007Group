# -*- coding: utf-8 -*-
"""
Kaggle 数据下载 - 使用浏览器辅助
"""
import os
import sys

OUTPUT_DIR = r"C:\Users\Administrator\.openclaw\workspace\event_recommendation\data"

print("=" * 60)
print("Kaggle 数据下载指南")
print("=" * 60)

print("\n由于网络限制，建议使用浏览器手动下载：")
print("\n步骤：")
print("1. 打开浏览器访问:")
print("   https://www.kaggle.com/competitions/event-recommendation-engine-challenge/data")
print("\n2. 登录账号：tang11112222")
print("\n3. 点击 'Download All' 按钮")
print("\n4. 将下载的压缩包解压到:")
print(f"   {OUTPUT_DIR}")
print("\n5. 解压后运行 eda.py 进行数据分析")
print("\n" + "=" * 60)

# 检查数据是否已存在
if os.path.exists(OUTPUT_DIR):
    files = os.listdir(OUTPUT_DIR)
    if files:
        print(f"\n数据目录已存在 {len(files)} 个文件:")
        for f in files[:10]:
            print(f"  - {f}")
        print("\n可以直接运行：python eda.py")
    else:
        print("\n数据目录为空，请下载数据后解压到此目录")
else:
    print(f"\n数据目录不存在，将创建：{OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("目录已创建，请下载数据后解压到此目录")

print("\n" + "=" * 60)
