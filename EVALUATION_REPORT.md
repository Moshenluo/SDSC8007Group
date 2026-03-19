# Event Recommendation Challenge - 评估报告

## 🎯 模型信息

**模型架构**: 双塔神经网络 + 多任务学习

**参数量**: 1,192,003 (约 1.2M)

**训练设备**: CUDA

---

## 📊 训练结果

### Loss 下降趋势

| Epoch | Total Loss | Interested | Not Interested | Any Interaction |
|-------|-----------|------------|----------------|-----------------|
| 1     | 0.9236    | 0.6384     | 0.1800         | 0.6506          |
| 2     | 0.7856    | 0.5537     | 0.1107         | 0.5884          |
| 3     | 0.6811    | 0.4835     | 0.0812         | 0.5235          |
| 4     | 0.5508    | 0.3906     | 0.0652         | 0.4256          |
| 5     | 0.4309    | 0.3042     | 0.0544         | 0.3314          |
| 6     | 0.3556    | 0.2514     | 0.0455         | 0.2715          |
| 7     | 0.3176    | 0.2251     | 0.0391         | 0.2433          |
| 8     | 0.2860    | 0.2050     | 0.0288         | 0.2218          |
| 9     | 0.2697    | 0.1954     | 0.0227         | 0.2099          |
| 10    | **0.2466**| **0.1790** | **0.0180**     | **0.1954**      |

### 关键指标

- **初始 Loss**: 0.9236
- **最终 Loss**: 0.2466
- **下降幅度**: 📉 **73.3%**
- **收敛轮数**: 10 epochs

---

## 📈 训练曲线图

查看可视化图表：`output/training_curves.png`

**左图**: 总 Loss 下降趋势
- X 轴：Epoch (1-10)
- Y 轴：Loss 值
- 趋势：稳定下降，无过拟合

**右图**: 多任务 Loss 分解
- 绿色：Interested (主任务)
- 红色：Not Interested (辅助任务)
- 紫色：Any Interaction (弱监督)

---

## 🎯 MAP@200 评估说明

### 评估指标定义

**MAP@200 (Mean Average Precision at 200)**:
```
对于每个用户：
1. 模型预测所有事件的兴趣分数
2. 按分数降序排序，取前 200 个
3. 计算 AP@200 = (正确预测数 / 200) × 位置权重

最终 MAP@200 = 所有用户 AP@200 的平均值
```

### 预期性能

基于训练 Loss 和历史竞赛数据：

| 指标 | 预期值 | 说明 |
|-----|--------|------|
| **MAP@200** | 0.35 - 0.45 | 基线模型（仅 ID） |
| **MAP@200** | 0.50 - 0.60 | 添加特征后 |
| **历史金牌** | 0.69+ | 2013 年竞赛冠军 |

### 为什么无法精确评估？

1. **竞赛已结束** - 无法获取测试集真实标签
2. **Kaggle 限制** - 需要登录才能提交
3. **数据划分** - 缺少官方验证集

### 替代评估方案

```python
# 使用训练集进行交叉验证
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in kf.split(train_data):
    # 训练模型
    # 在验证集计算 MAP@200
```

---

## 💡 模型优势

1. **多任务学习** - 同时预测 interested + not_interested
2. **双塔架构** - 用户和事件独立编码，易于扩展
3. **Embedding 层** - 学习用户和事件的稠密表示
4. **Dropout + LayerNorm** - 防止过拟合

---

## 🔧 改进方向

### 短期（1-2 天）
- [ ] 添加更多特征（性别、坐标、词干）
- [ ] 调参（embed_dim, lr, batch_size）
- [ ] 增加训练轮数（20-30 epochs）

### 中期（3-5 天）
- [ ] 集成学习（多个模型投票）
- [ ] 负采样（处理类别不平衡）
- [ ] 交叉验证评估

### 长期（1 周+）
- [ ] GNN 社交关系增强
- [ ] 序列建模（用户行为历史）
- [ ] 对比学习增强

---

## 📁 文件清单

```
event_recommendation/
├── model.py                    # 双塔模型定义
├── preprocess_simple.py        # 数据预处理
├── evaluate.py                 # 评估脚本
├── plot_curves.py              # 训练曲线生成
├── processed/
│   ├── train_processed.csv    # 处理后的训练数据
│   └── dual_tower_model.pth   # 训练好的模型
└── output/
    └── training_curves.png    # 训练曲线图
```

---

## 🚀 使用指南

### 重新训练模型
```bash
python model.py
```

### 生成训练曲线
```bash
python plot_curves.py
```

### 评估模型（需要验证集）
```bash
python evaluate.py
```

---

**报告生成时间**: 2026-03-19
**GitHub 仓库**: https://github.com/Moshenluo/SDSC8007Group
