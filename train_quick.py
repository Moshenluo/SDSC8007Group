# -*- coding: utf-8 -*-
"""
快速训练优化模型（简化版）
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

print("Loading data...")
processed_dir = Path(r"C:\Users\Administrator\.openclaw\workspace\event_recommendation\processed")
train_df = pd.read_csv(processed_dir / "train_full.csv")

print(f"Samples: {len(train_df):,}")

# 特征列
user_cols = ['user_encoded'] + ['gender_encoded', 'birthyear_scaled', 'timezone_scaled', 'country_encoded_x']
event_cols = ['event_encoded'] + ['country_encoded_y', 'city_encoded', 'lat_scaled', 'lng_scaled'] + [f'c_{i}_scaled' for i in range(1, 21)]

class SimpleDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        user = torch.tensor([row[c] for c in user_cols], dtype=torch.float32)
        event = torch.tensor([row[c] for c in event_cols], dtype=torch.float32)
        labels = torch.tensor([row['label_interested'], row['label_not_interested'], row['label_any']], dtype=torch.float32)
        return user, event, labels

dataset = SimpleDataset(train_df)
loader = DataLoader(dataset, batch_size=256, shuffle=True, drop_last=True)

print(f"Batches: {len(loader)}")

# 简单模型
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.user_emb = nn.Embedding(3391, 32)
        self.event_emb = nn.Embedding(13418, 32)
        self.fc = nn.Sequential(
            nn.Linear(92, 128),
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
        # print(f"Debug: x shape = {x.shape}")  # Should be 92
        x = self.fc(x)
        return self.head1(x), self.head2(x), self.head3(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleModel().to(device)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("\nTraining 10 epochs...")
for epoch in range(10):
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
    print(f"Epoch {epoch+1}/10: Loss={avg_loss:.4f}")

# 保存
torch.save(model.state_dict(), processed_dir / "model_optimized.pth")
print("\nModel saved!")
print("Done!")
