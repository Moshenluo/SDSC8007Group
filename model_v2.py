# -*- coding: utf-8 -*-
"""
双塔模型 v2（优化版）
- 添加用户特征（性别、年龄、时区、国家）
- 添加事件特征（城市、坐标、词干）
- 更多训练轮数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import json


class DualTowerNetV2(nn.Module):
    """双塔神经网络 v2（带完整特征）"""
    
    def __init__(self, 
                 user_feat_dim=5,      # user_id + gender + birthyear + timezone + country
                 event_feat_dim=24,    # event_id + country + city + lat + lng + 20 counts
                 user_embed_dim=32,
                 event_embed_dim=32,
                 hidden_dim=128,
                 num_users=3391,
                 num_events=13418
                 ):
        super().__init__()
        
        # ========== 用户塔 ==========
        # ID Embedding
        self.user_embedding = nn.Embedding(num_users, user_embed_dim)
        
        # 数值特征 MLP（去掉 user_id）
        self.user_mlp = nn.Sequential(
            nn.Linear(4, hidden_dim),  # user_feat_dim - 1 = 5 - 1 = 4
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, user_embed_dim),
            nn.BatchNorm1d(user_embed_dim),
            nn.ReLU(),
        )
        
        # 用户塔输出
        self.user_output = nn.Sequential(
            nn.Linear(user_embed_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        # ========== 事件塔 ==========
        # ID Embedding
        self.event_embedding = nn.Embedding(num_events, event_embed_dim)
        
        # 数值特征 MLP（去掉 event_id）
        self.event_mlp = nn.Sequential(
            nn.Linear(23, hidden_dim),  # event_feat_dim - 1 = 24 - 1 = 23
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, event_embed_dim),
            nn.BatchNorm1d(event_embed_dim),
            nn.ReLU(),
        )
        
        # 事件塔输出
        self.event_output = nn.Sequential(
            nn.Linear(event_embed_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        # ========== 多任务输出头 ==========
        # 任务 1: interested
        self.head_interested = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # 任务 2: not_interested
        self.head_not_interested = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # 任务 3: any_interaction
        self.head_any = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, user_features, event_features):
        # ID 已经单独传入
        user_ids = user_features[:, 0].long()
        event_ids = event_features[:, 0].long()
        
        # 数值特征（去掉 ID）
        user_numeric = user_features[:, 1:]
        event_numeric = event_features[:, 1:]
        
        # 用户塔
        user_emb = self.user_embedding(user_ids)
        user_mlp_out = self.user_mlp(user_numeric)
        user_out = self.user_output(torch.cat([user_emb, user_mlp_out], dim=1))
        
        # 事件塔
        event_emb = self.event_embedding(event_ids)
        event_mlp_out = self.event_mlp(event_numeric)
        event_out = self.event_output(torch.cat([event_emb, event_mlp_out], dim=1))
        
        # 多任务预测
        combined = torch.cat([user_out, event_out], dim=1)
        
        out_interested = self.head_interested(combined)
        out_not_interested = self.head_not_interested(combined)
        out_any = self.head_any(combined)
        
        return out_interested, out_not_interested, out_any


class EventDatasetV2(Dataset):
    """Dataset v2（完整特征）"""
    
    def __init__(self, df, mode="train"):
        self.df = df.reset_index(drop=True)
        self.mode = mode
        
        # 用户特征列（4 列：去掉 user_encoded）
        self.user_cols = ['gender_encoded', 'birthyear_scaled', 
                          'timezone_scaled', 'country_encoded_x']
        
        # 事件特征列（23 列：去掉 event_encoded）
        self.event_cols = ['country_encoded_y', 'city_encoded',
                           'lat_scaled', 'lng_scaled'] + [f'c_{i}_scaled' for i in range(1, 21)]
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        user_feat = torch.tensor([row[col] for col in self.user_cols], dtype=torch.float32)
        event_feat = torch.tensor([row[col] for col in self.event_cols], dtype=torch.float32)
        
        if self.mode == "train":
            labels = torch.tensor([
                float(row.get('label_interested', 0)),
                float(row.get('label_not_interested', 0)),
                float(row.get('label_any', 0))
            ], dtype=torch.float32)
            return user_feat, event_feat, labels
        else:
            return user_feat, event_feat, row.get('user', 0), row.get('event', 0)


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    loss_dict = {'loss_interested': 0, 'loss_not_interested': 0, 'loss_any': 0}
    
    for user_feat, event_feat, labels in dataloader:
        user_feat = user_feat.to(device)
        event_feat = event_feat.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(user_feat, event_feat)
        loss, losses = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        for k in loss_dict:
            loss_dict[k] += losses[k]
    
    n = len(dataloader)
    return total_loss / n, {k: v / n for k, v in loss_dict.items()}


def main():
    print("=" * 60)
    print("双塔模型 v2 训练（优化版）")
    print("=" * 60)
    
    # 配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 路径
    processed_dir = Path(r"C:\Users\Administrator\.openclaw\workspace\event_recommendation\processed")
    
    # 加载元数据
    with open(processed_dir / "metadata.json") as f:
        metadata = json.load(f)
    
    # 加载数据
    print("\n加载数据...")
    train_df = pd.read_csv(processed_dir / "train_full.csv")
    print(f"  训练样本：{len(train_df):,}")
    
    # 创建 Dataset
    print("\n创建 Dataset...")
    train_dataset = EventDatasetV2(train_df, mode="train")
    
    # DataLoader
    batch_size = 256
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    print(f"  Batch size: {batch_size}")
    print(f"  Batches: {len(train_loader)}")
    
    # 创建模型
    print("\n创建模型...")
    model = DualTowerNetV2(
        user_feat_dim=5,   # user_encoded + gender + birthyear + timezone + country_x
        event_feat_dim=24, # event_encoded + country_y + city + lat + lng + 20 counts
        user_embed_dim=32,
        event_embed_dim=32,
        hidden_dim=128,
        num_users=metadata['num_users'],
        num_events=metadata['num_events']
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量：{total_params:,}")
    
    # 损失 + 优化器
    from model import MultiTaskLoss
    criterion = MultiTaskLoss(weights=[1.0, 0.5, 0.3])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5)
    
    # 训练
    print("\n开始训练...")
    num_epochs = 20
    
    history = []
    for epoch in range(num_epochs):
        train_loss, losses = train_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        
        history.append({
            'epoch': epoch + 1,
            'loss': train_loss,
            'loss_interested': losses['loss_interested'],
            'loss_not_interested': losses['loss_not_interested'],
            'loss_any': losses['loss_any']
        })
        
        print(f"Epoch {epoch+1}/{num_epochs}: Loss={train_loss:.4f} (LR={scheduler.get_last_lr()[0]:.6f})")
    
    # 保存模型
    print("\n保存模型...")
    save_path = processed_dir / "dual_tower_v2_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'metadata': metadata
    }, save_path)
    print(f"  模型已保存：{save_path}")
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
