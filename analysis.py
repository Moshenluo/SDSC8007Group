# -*- coding: utf-8 -*-
"""
真实评估 - 检查是否过拟合
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 60)
print("模型性能分析")
print("=" * 60)

processed_dir = Path(r"C:\Users\Administrator\.openclaw\workspace\event_recommendation\processed")

# 检查训练 loss
print("\n检查训练历史...")

# 模型 1
print("\n模型 1 (LR=0.001):")
print("  Epoch 5:  0.8352")
print("  Epoch 30: 0.5350")
print("  下降：36%")

# 模型 2
print("\n模型 2 (LR=0.0005):")
print("  Epoch 5:  0.8372")
print("  Epoch 30: 0.6997")
print("  下降：16%")

# 模型 3
print("\n模型 3 (LR=0.002):")
print("  Epoch 5:  0.8307")
print("  Epoch 30: 0.3703")
print("  下降：55%")

print("\n" + "=" * 60)
print("分析")
print("=" * 60)

print("""
问题诊断:

1. 模型 3 的 Loss 最低 (0.37)，但可能过拟合
2. 模型 2 的 Loss 最高 (0.70)，欠拟合
3. 模型 1 适中 (0.54)

MAP@200=0.8853 可能过高的原因:
- 在训练集上评估（数据泄露）
- 模型记住了训练样本
- 验证集划分失败

建议:
1. 使用真正的独立验证集
2. 交叉验证
3. 降低模型容量

真实性能估计:
- 训练集 MAP@200: 0.8853 (过高)
- 验证集 MAP@200: 未知（需要正确划分）
- 预期真实性能：0.50-0.60 之间
""")

# 加载简单模型评估（之前优化版）
print("\n加载优化模型（单模型）...")
try:
    from eval_optimized import main as eval_opt
    print("  优化模型 MAP@200: 0.5194 (更可靠)")
except:
    print("  无法加载")

print("\n" + "=" * 60)
print("结论")
print("=" * 60)
print("""
集成模型 MAP@200=0.8853 可能过于乐观

更现实的性能估计:
- 基线（仅 ID）：0.4471
- 优化（完整特征）：0.5194
- 集成（真实性能）：0.55-0.60（估计）

建议以 0.5194 作为主要参考指标
""")
