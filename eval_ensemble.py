# -*- coding: utf-8 -*-
"""
评估集成模型
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 60)
print("集成模型评估")
print("=" * 60)

processed_dir = Path(r"C:\Users\Administrator\.openclaw\workspace\event_recommendation\processed")
data_dir = Path(r"C:\Users\Administrator\.openclaw\workspace\event_recommendation\data")

# 加载数据
print("\n加载数据...")
train_df = pd.read_csv(data_dir / "train.csv")
train_full = pd.read_csv(processed_dir / "train_full.csv")

# 特征列
user_cols = ['user_encoded'] + ['gender_encoded', 'birthyear_scaled', 'timezone_scaled', 'country_encoded_x']
event_cols = ['event_encoded'] + ['country_encoded_y', 'city_encoded', 'lat_scaled', 'lng_scaled'] + [f'c_{i}_scaled' for i in range(1, 21)]

# 模型
class EnsembleModel(torch.nn.Module):
    def __init__(self, seed=42):
        super().__init__()
        self.user_emb = torch.nn.Embedding(3391, 64)
        self.event_emb = torch.nn.Embedding(13418, 64)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(64*2 + len(user_cols)-1 + len(event_cols)-1, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(256, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
        )
        self.head1 = torch.nn.Linear(64, 1)
        self.head2 = torch.nn.Linear(64, 1)
        self.head3 = torch.nn.Linear(64, 1)
    
    def forward(self, u, e):
        uid = u[:, 0].long()
        eid = e[:, 0].long()
        ue = self.user_emb(uid)
        ee = self.event_emb(eid)
        x = torch.cat([ue, ee, u[:, 1:], e[:, 1:]], dim=1)
        x = self.fc(x)
        return self.head1(x)

# 加载 3 个模型
print("加载集成模型...")
models = []
for i in range(3):
    model = EnsembleModel(seed=42+i*100)
    model.load_state_dict(torch.load(processed_dir / f"ensemble_model_{i}.pth", map_location='cpu'))
    model.eval()
    models.append(model)
print(f"  已加载 3 个模型")

# MAP@200 计算
def apk(actual, predicted, k=200):
    if len(predicted) > k: predicted = predicted[:k]
    if len(actual) == 0: return 0.0
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    return score / min(len(actual), k)

# 评估
print("\n评估 MAP@200 (集成投票)...")
user_groups = train_full.groupby('user')
actual = []
predicted = []

with torch.no_grad():
    for idx, (user, group) in enumerate(user_groups):
        if (idx + 1) % 500 == 0:
            print(f"  {idx + 1}/{len(user_groups)}")
        
        interested = group[group['label_interested'] == 1]['event'].tolist()
        if len(interested) == 0: continue
        
        # 集成预测（3 个模型平均）
        scores = []
        for _, row in group.iterrows():
            u = torch.tensor([[row[c] for c in user_cols]], dtype=torch.float32)
            e = torch.tensor([[row[c] for c in event_cols]], dtype=torch.float32)
            
            # 3 个模型预测并平均
            ensemble_score = 0
            for model in models:
                o1 = model(u, e)
                ensemble_score += torch.sigmoid(o1).item()
            ensemble_score /= 3.0
            
            scores.append((row['event'], ensemble_score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        pred_events = [s[0] for s in scores]
        
        actual.append(interested)
        predicted.append(pred_events)

map_score = np.mean([apk(a, p, 200) for a, p in zip(actual, predicted)])

print(f"\n{'='*60}")
print("集成模型评估结果")
print(f"{'='*60}")
print(f"MAP@200: {map_score:.4f}")
print(f"用户数：{len(actual):,}")
print(f"{'='*60}")

# 对比
print(f"\n性能对比:")
print(f"  基线模型：0.4471")
print(f"  优化模型：0.5194")
print(f"  集成模型：{map_score:.4f}")
print(f"  提升：{(map_score - 0.4471) / 0.4471 * 100:.1f}%")

# 评级
if map_score >= 0.69:
    rating = "[GOLD] 金牌水平！"
elif map_score >= 0.59:
    rating = "[BRONZE] 铜牌水平"
elif map_score >= 0.50:
    rating = "[HONOR] 荣誉奖"
else:
    rating = "[BASELINE] 基线"

print(f"\nRating: {rating}")

# 保存结果
import json
with open(processed_dir / "map200_ensemble.json", 'w') as f:
    json.dump({
        'model': 'Ensemble (3 models)',
        'map_at_200': float(map_score),
        'improvement_vs_baseline': f"+{(map_score - 0.4471) / 0.4471 * 100:.1f}%",
        'num_users': len(actual)
    }, f, indent=2)

print(f"\n结果已保存！")
