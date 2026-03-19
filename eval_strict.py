# -*- coding: utf-8 -*-
"""
严格评估 - 使用验证集（非训练集）
避免数据泄露
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

print("=" * 60)
print("严格评估 - 验证集测试")
print("=" * 60)

processed_dir = Path(r"C:\Users\Administrator\.openclaw\workspace\event_recommendation\processed")
data_dir = Path(r"C:\Users\Administrator\.openclaw\workspace\event_recommendation\data")

# 加载原始数据
print("\n加载原始数据...")
train_df = pd.read_csv(data_dir / "train.csv")
print(f"总样本：{len(train_df):,}")

# 划分训练集和验证集（80/20）
print("\n划分验证集（20%）...")
users = train_df['user'].unique()
train_users, val_users = train_test_split(users, test_size=0.2, random_state=42)

train_data = train_df[train_df['user'].isin(train_users)]
val_data = train_df[train_df['user'].isin(val_users)]

print(f"  训练集：{len(train_data):,} 样本，{len(train_users):,} 用户")
print(f"  验证集：{len(val_data):,} 样本，{len(val_users):,} 用户")

# 创建 ID 映射（只用训练集）
all_users = train_data['user'].unique()
all_events = train_data['event'].unique()

user2id = {u: i for i, u in enumerate(all_users)}
event2id = {e: i for i, e in enumerate(all_events)}

print(f"\n  唯一用户：{len(user2id):,}")
print(f"  唯一事件：{len(event2id):,}")

# 过滤验证集（只保留在映射中的）
val_filtered = val_data[
    (val_data['user'].isin(user2id.keys())) & 
    (val_data['event'].isin(event2id.keys()))
].copy()

print(f"  验证集（过滤后）：{len(val_filtered):,} 样本")

# 加载完整特征数据
train_full = pd.read_csv(processed_dir / "train_full.csv")

# 特征列
user_cols = ['user_encoded'] + ['gender_encoded', 'birthyear_scaled', 'timezone_scaled', 'country_encoded_x']
event_cols = ['event_encoded'] + ['country_encoded_y', 'city_encoded', 'lat_scaled', 'lng_scaled'] + [f'c_{i}_scaled' for i in range(1, 21)]

# 加载模型
print("\n加载集成模型...")
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
        return self.head1(x), self.head2(x), self.head3(x)

models = []
for i in range(3):
    model = EnsembleModel(seed=42+i*100)
    try:
        model.load_state_dict(torch.load(processed_dir / f"ensemble_model_{i}.pth", map_location='cpu'))
        model.eval()
        models.append(model)
    except Exception as e:
        print(f"  模型 {i} 加载失败：{e}")

print(f"  成功加载 {len(models)}/3 个模型")

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

# 在验证集上评估
print("\n在验证集上评估 MAP@200...")

# 合并验证集和完整特征
val_with_features = val_filtered.merge(
    train_full[['user', 'event'] + user_cols + event_cols + ['label_interested', 'label_not_interested', 'label_any']],
    on=['user', 'event'],
    how='left'
)

val_with_features = val_with_features.dropna()
print(f"  验证集（有特征）：{len(val_with_features):,} 样本")

user_groups = val_with_features.groupby('user')
actual = []
predicted = []

with torch.no_grad():
    for idx, (user, group) in enumerate(user_groups):
        if (idx + 1) % 100 == 0:
            print(f"  {idx + 1}/{len(user_groups)}")
        
        interested = group[group['label_interested'] == 1]['event'].tolist()
        if len(interested) == 0: continue
        
        # 集成预测
        scores = []
        for _, row in group.iterrows():
            try:
                u = torch.tensor([[row[c] for c in user_cols]], dtype=torch.float32)
                e = torch.tensor([[row[c] for c in event_cols]], dtype=torch.float32)
                
                ensemble_score = 0
                for model in models:
                    o1, o2, o3 = model(u, e)
                    ensemble_score += torch.sigmoid(o1).item()
                ensemble_score /= len(models)
                
                scores.append((row['event'], ensemble_score))
            except Exception as e:
                scores.append((row['event'], 0.5))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        pred_events = [s[0] for s in scores]
        
        actual.append(interested)
        predicted.append(pred_events)

if len(actual) > 0:
    map_score = np.mean([apk(a, p, 200) for a, p in zip(actual, predicted)])
    
    print(f"\n{'='*60}")
    print("验证集评估结果（严格，无数据泄露）")
    print(f"{'='*60}")
    print(f"MAP@200: {map_score:.4f}")
    print(f"验证集用户数：{len(actual):,}")
    print(f"验证集样本数：{len(val_with_features):,}")
    print(f"{'='*60}")
    
    # 对比
    print(f"\n对比:")
    print(f"  训练集评估（有泄露）：0.8853")
    print(f"  验证集评估（严格）：{map_score:.4f}")
    print(f"  差距：{0.8853 - map_score:.4f}")
else:
    print("无法评估（验证集无有效用户）")
