# -*- coding: utf-8 -*-
"""
集成学习训练 - 3 个模型投票
目标：MAP@200 > 0.55
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import json

print("=" * 60)
print("集成学习训练 (3 Models Ensemble)")
print("=" * 60)

# 配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

processed_dir = Path(r"C:\Users\Administrator\.openclaw\workspace\event_recommendation\processed")

# 加载数据
print("\n加载数据...")
train_full = pd.read_csv(processed_dir / "train_full.csv")
print(f"Samples: {len(train_full):,}")

# 特征列
user_cols = ['user_encoded'] + ['gender_encoded', 'birthyear_scaled', 'timezone_scaled', 'country_encoded_x']
event_cols = ['event_encoded'] + ['country_encoded_y', 'city_encoded', 'lat_scaled', 'lng_scaled'] + [f'c_{i}_scaled' for i in range(1, 21)]

class EventDataset(Dataset):
    def __init__(self, df, mode="train"):
        self.df = df.reset_index(drop=True)
        self.mode = mode
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        u = torch.tensor([row[c] for c in user_cols], dtype=torch.float32)
        e = torch.tensor([row[c] for c in event_cols], dtype=torch.float32)
        if self.mode == "train":
            labels = torch.tensor([row['label_interested'], row['label_not_interested'], row['label_any']], dtype=torch.float32)
            return u, e, labels
        return u, e, row['event']

# 模型（更大容量）
class EnsembleModel(nn.Module):
    def __init__(self, seed=42):
        super().__init__()
        torch.manual_seed(seed)
        self.user_emb = nn.Embedding(3391, 64)
        self.event_emb = nn.Embedding(13418, 64)
        self.fc = nn.Sequential(
            nn.Linear(64*2 + len(user_cols)-1 + len(event_cols)-1, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.head1 = nn.Linear(64, 1)
        self.head2 = nn.Linear(64, 1)
        self.head3 = nn.Linear(64, 1)
    
    def forward(self, u, e):
        uid = u[:, 0].long()
        eid = e[:, 0].long()
        ue = self.user_emb(uid)
        ee = self.event_emb(eid)
        x = torch.cat([ue, ee, u[:, 1:], e[:, 1:]], dim=1)
        x = self.fc(x)
        return self.head1(x), self.head2(x), self.head3(x)

# 训练函数
def train_model(model_idx, seed):
    print(f"\n{'='*60}")
    print(f"训练模型 {model_idx + 1}/3 (seed={seed})")
    print(f"{'='*60}")
    
    # 数据（不同 shuffle）
    dataset = EventDataset(train_full.sample(frac=1, random_state=seed))
    loader = DataLoader(dataset, batch_size=256, shuffle=True, drop_last=True)
    
    # 模型
    model = EnsembleModel(seed=seed).to(device)
    
    # 优化器（不同学习率）
    lrs = [0.001, 0.0005, 0.002]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lrs[model_idx], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)
    
    criterion = nn.BCEWithLogitsLoss()
    
    # 训练 30 epochs
    best_loss = float('inf')
    for epoch in range(30):
        model.train()
        total_loss = 0
        
        for u, e, labels in loader:
            u, e, labels = u.to(device), e.to(device), labels.to(device)
            optimizer.zero_grad()
            
            o1, o2, o3 = model(u, e)
            loss = criterion(o1, labels[:, 0:1]) + 0.5*criterion(o2, labels[:, 1:2]) + 0.3*criterion(o3, labels[:, 2:3])
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        scheduler.step()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/30: Loss={avg_loss:.4f} (LR={scheduler.get_last_lr()[0]:.6f})")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
    
    # 保存模型
    save_path = processed_dir / f"ensemble_model_{model_idx}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"模型 {model_idx + 1} 已保存：{save_path}")
    
    return model

# 训练 3 个模型
models = []
for i in range(3):
    model = train_model(i, seed=42+i*100)
    models.append(model)

print("\n" + "=" * 60)
print("集成学习训练完成！")
print("=" * 60)
