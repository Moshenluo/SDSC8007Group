# -*- coding: utf-8 -*-
"""
5 折交叉验证 - 正确评估
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader
import json

print("=" * 60)
print("5 折交叉验证 - 严格评估")
print("=" * 60)

data_dir = Path(r"C:\Users\Administrator\.openclaw\workspace\event_recommendation\data")
processed_dir = Path(r"C:\Users\Administrator\.openclaw\workspace\event_recommendation\processed")

# 加载数据
print("\n加载数据...")
train_df = pd.read_csv(data_dir / "train.csv")
train_full = pd.read_csv(processed_dir / "train_full.csv")
print(f"总样本：{len(train_df):,}")

# 特征列
user_cols = ['user_encoded'] + ['gender_encoded', 'birthyear_scaled', 'timezone_scaled', 'country_encoded_x']
event_cols = ['event_encoded'] + ['country_encoded_y', 'city_encoded', 'lat_scaled', 'lng_scaled'] + [f'c_{i}_scaled' for i in range(1, 21)]

class EventDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        u = torch.tensor([row[c] for c in user_cols], dtype=torch.float32)
        e = torch.tensor([row[c] for c in event_cols], dtype=torch.float32)
        labels = torch.tensor([row['label_interested'], row['label_not_interested'], row['label_any']], dtype=torch.float32)
        return u, e, labels, row['user'], row['event']

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

# 模型
class SimpleModel(torch.nn.Module):
    def __init__(self):
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

# 交叉验证
print("\n开始 5 折交叉验证...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)

fold_scores = []
all_users_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(train_full)):
    print(f"\n{'='*60}")
    print(f"Fold {fold + 1}/5")
    print(f"{'='*60}")
    
    # 划分数据
    train_data = train_full.iloc[train_idx]
    val_data = train_full.iloc[val_idx]
    
    print(f"  训练样本：{len(train_data):,}")
    print(f"  验证样本：{len(val_data):,}")
    
    # 创建 DataLoader
    train_dataset = EventDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, drop_last=True)
    
    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleModel().to(device)
    
    # 训练
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-6)
    
    print(f"  训练 15 epochs...")
    for epoch in range(15):
        model.train()
        total_loss = 0
        for u, e, labels, _, _ in train_loader:
            u, e, labels = u.to(device), e.to(device), labels.to(device)
            optimizer.zero_grad()
            o1, o2, o3 = model(u, e)
            loss = criterion(o1, labels[:, 0:1]) + 0.5*criterion(o2, labels[:, 1:2]) + 0.3*criterion(o3, labels[:, 2:3])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        
        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1}/15: Loss={total_loss/len(train_loader):.4f}")
    
    # 在验证集上评估
    print(f"  评估 MAP@200...")
    model.eval()
    
    val_groups = val_data.groupby('user')
    actual = []
    predicted = []
    
    with torch.no_grad():
        for user, group in val_groups:
            interested = group[group['label_interested'] == 1]['event'].tolist()
            if len(interested) == 0: continue
            
            # 预测
            scores = []
            for _, row in group.iterrows():
                try:
                    u = torch.tensor([[row[c] for c in user_cols]], dtype=torch.float32).to(device)
                    e = torch.tensor([[row[c] for c in event_cols]], dtype=torch.float32).to(device)
                    o1, o2, o3 = model(u, e)
                    score = torch.sigmoid(o1).cpu().item()
                    scores.append((row['event'], score))
                except:
                    scores.append((row['event'], 0.5))
            
            scores.sort(key=lambda x: x[1], reverse=True)
            pred_events = [s[0] for s in scores]
            
            actual.append(interested)
            predicted.append(pred_events)
    
    if len(actual) > 0:
        map_score = np.mean([apk(a, p, 200) for a, p in zip(actual, predicted)])
        fold_scores.append(map_score)
        all_users_scores.append(len(actual))
        print(f"  Fold {fold + 1} MAP@200: {map_score:.4f} ({len(actual)} 用户)")
    else:
        print(f"  Fold {fold + 1}: 无有效用户")

# 汇总结果
print(f"\n{'='*60}")
print("5 折交叉验证结果")
print(f"{'='*60}")

if len(fold_scores) > 0:
    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    
    print(f"Fold 数：{len(fold_scores)}")
    print(f"平均 MAP@200: {mean_score:.4f} (+/- {std_score:.4f})")
    print(f"评估用户总数：{sum(all_users_scores):,}")
    print(f"{'='*60}")
    
    # 对比
    print(f"\n性能对比:")
    print(f"  基线模型（仅 ID）：0.4471")
    print(f"  优化模型（训练集）：0.5194")
    print(f"  交叉验证（真实）：{mean_score:.4f}")
    
    # 评级
    if mean_score >= 0.59:
        rating = "[BRONZE] 铜牌水平"
    elif mean_score >= 0.50:
        rating = "[HONOR] 荣誉奖"
    elif mean_score >= 0.35:
        rating = "[BASELINE] 基线"
    else:
        rating = "[NEEDS_WORK] 需改进"
    
    print(f"\n评级：{rating}")
    
    # 保存结果
    with open(processed_dir / "cv5_result.json", 'w') as f:
        json.dump({
            'method': '5-Fold Cross Validation',
            'map_at_200': float(mean_score),
            'std': float(std_score),
            'folds': fold_scores,
            'total_users': sum(all_users_scores),
            'rating': rating
        }, f, indent=2)
    
    print(f"\n结果已保存！")
else:
    print("无有效结果")
