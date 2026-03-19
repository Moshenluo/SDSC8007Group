# -*- coding: utf-8 -*-
"""评估优化模型"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path

print("Loading model...")
processed_dir = Path(r"C:\Users\Administrator\.openclaw\workspace\event_recommendation\processed")
data_dir = Path(r"C:\Users\Administrator\.openclaw\workspace\event_recommendation\data")

# 加载数据
train_df = pd.read_csv(data_dir / "train.csv")
train_full = pd.read_csv(processed_dir / "train_full.csv")

# 创建映射
user2id = {u: i for i, u in enumerate(train_df['user'].unique())}
event2id = {e: i for i, e in enumerate(train_df['event'].unique())}

# 模型（和训练时一致）
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.user_emb = torch.nn.Embedding(3391, 32)
        self.event_emb = torch.nn.Embedding(13418, 32)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(92, 128),
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
        return self.head1(x), self.head2(x), self.head3(x)

model = SimpleModel()
model.load_state_dict(torch.load(processed_dir / "model_optimized.pth", map_location='cpu'))
model.eval()

user_cols = ['user_encoded'] + ['gender_encoded', 'birthyear_scaled', 'timezone_scaled', 'country_encoded_x']
event_cols = ['event_encoded'] + ['country_encoded_y', 'city_encoded', 'lat_scaled', 'lng_scaled'] + [f'c_{i}_scaled' for i in range(1, 21)]

# 评估
print("Evaluating MAP@200...")

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

user_groups = train_full.groupby('user')
actual = []
predicted = []

with torch.no_grad():
    for idx, (user, group) in enumerate(user_groups):
        if (idx + 1) % 500 == 0:
            print(f"  {idx + 1}/{len(user_groups)}")
        
        interested = group[group['label_interested'] == 1]['event'].tolist()
        if len(interested) == 0: continue
        
        # 预测
        scores = []
        for _, row in group.iterrows():
            u = torch.tensor([[row[c] for c in user_cols]], dtype=torch.float32)
            e = torch.tensor([[row[c] for c in event_cols]], dtype=torch.float32)
            o1, o2, o3 = model(u, e)
            score = torch.sigmoid(o1).item()
            scores.append((row['event'], score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        pred_events = [s[0] for s in scores]
        
        actual.append(interested)
        predicted.append(pred_events)

map_score = np.mean([apk(a, p, 200) for a, p in zip(actual, predicted)])

print(f"\n{'='*60}")
print("优化模型评估结果")
print(f"{'='*60}")
print(f"MAP@200: {map_score:.4f}")
print(f"用户数：{len(actual):,}")
print(f"{'='*60}")

# 保存结果
import json
with open(processed_dir / "map200_optimized.json", 'w') as f:
    json.dump({
        'model': 'Optimized Dual Tower',
        'map_at_200': float(map_score),
        'num_users': len(actual),
        'epochs': 10,
        'final_loss': 1.2396
    }, f, indent=2)

print("\nResult saved!")
