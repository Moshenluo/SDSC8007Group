# Event Recommendation Challenge - 深度学习方案

## 📁 项目结构

```
event_recommendation/
├── data/                    # 放置下载的数据文件
│   ├── train.csv
│   ├── test.csv
│   ├── users.csv
│   ├── user_friends.csv
│   ├── events.csv
│   └── event_attendees.csv
├── eda.py                   # 数据探索脚本
├── requirements.txt         # 依赖
└── README.md               # 本文件
```

---

## 🚀 快速开始

### 1. 下载数据

**方法 A: Kaggle CLI (推荐)**
```bash
# 安装 Kaggle CLI
pip install kaggle

# 配置 API key (从 https://www.kaggle.com/account 下载 kaggle.json)
# 将 kaggle.json 放到 ~/.kaggle/kaggle.json (Linux/Mac) 或 C:\Users\<你的用户名>\.kaggle\kaggle.json (Windows)

# 下载数据
kaggle competitions download -c event-recommendation-engine-challenge

# 解压到 data/ 目录
mkdir data
move event-recommendation-engine-challenge.zip data/
cd data
unzip event-recommendation-engine-challenge.zip
```

**方法 B: 手动下载**
1. 访问 https://www.kaggle.com/competitions/event-recommendation-engine-challenge/data
2. 点击 "Download All"
3. 解压到 `data/` 目录

---

### 2. 安装依赖

```bash
# 使用 uv (推荐)
uv pip install -r requirements.txt

# 或使用 pip
pip install -r requirements.txt
```

---

### 3. 运行数据探索

```bash
# 使用 uv
uv run python eda.py

# 或直接运行
python eda.py
```

运行后会生成：
- 控制台输出：各数据集的详细统计
- `eda_output/` 目录：可视化图表

---

## 📊 数据概览

| 文件 | 说明 | 大小 |
|-----|------|------|
| train.csv | 训练集 (user, event, invited, timestamp, interested, not_interested) | ~50MB |
| test.csv | 测试集 (无标签) | ~10MB |
| users.csv | 用户信息 (性别、年龄、位置、时区) | ~5MB |
| user_friends.csv | 好友关系 | ~20MB |
| events.csv | 事件信息 (地点、时间、词干计数) | ~200MB |
| event_attendees.csv | 事件参与者 | ~10MB |

---

## 🎯 模型方案：双塔 + 多任务学习

### 模型架构
```
用户塔：user_id + gender + birthyear + timezone → MLP → 用户向量
事件塔：event_id + city + country + lat/lng + 词干 → MLP → 事件向量
                              ↓
                        点积 → Sigmoid → 概率
```

### 多任务设计
- **任务 1**: 预测 `interested` (主任务)
- **任务 2**: 预测 `not_interested` (辅助任务)
- **任务 3**: 预测是否参加 (从 event_attendees 弱监督)

### 预期效果
- 基线 (单任务): MAP@200 ≈ 0.50
- 双塔 + 多任务：MAP@200 ≈ 0.55-0.65

---

## 📝 下一步

1. ✅ 运行 `eda.py` 了解数据分布
2. ⏳ 等待确认数据特征后，生成完整模型代码
3. ⏳ 训练 + 调优 + 评估

---

## 📚 参考资料

- [Kaggle 竞赛页面](https://www.kaggle.com/competitions/event-recommendation-engine-challenge)
- [评估指标说明](https://www.kaggle.com/wiki/MeanAveragePrecision)
- [双塔模型论文](https://www.researchgate.net/publication/323120134_Deep_Neural_Networks_for_YouTube_Recommendations)
