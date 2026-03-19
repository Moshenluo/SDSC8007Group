# -*- coding: utf-8 -*-
"""
双塔模型 + 多任务学习
- 用户塔 + 事件塔
- 多任务输出头 (interested + not_interested)
- 训练循环 + 评估
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path


class DualTowerNet(nn.Module):
    """双塔神经网络"""
    
    def __init__(self, 
                 user_feat_dim=5,      # 用户特征维度
                 event_feat_dim=24,    # 事件特征维度
                 embed_dim=64,         # Embedding 维度
                 hidden_dim=128,       # 隐藏层维度
                 num_users=3391,       # 用户数量
                 num_events=13418      # 事件数量
                 ):
        super().__init__()
        
        # ========== 用户塔 ==========
        # ID Embedding
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        
        # 数值特征 MLP（如果 feat_dim > 1）
        if user_feat_dim > 1:
            self.user_mlp = nn.Sequential(
                nn.Linear(user_feat_dim - 1, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, embed_dim),
                nn.LayerNorm(embed_dim),
            )
        else:
            self.user_mlp = None
        
        # 用户塔输出层
        tower_input_dim = embed_dim * 2 if (user_feat_dim > 1) else embed_dim
        self.user_tower = nn.Sequential(
            nn.Linear(tower_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        # ========== 事件塔 ==========
        # ID Embedding
        self.event_embedding = nn.Embedding(num_events, embed_dim)
        
        # 数值特征 MLP（如果 feat_dim > 1）
        if event_feat_dim > 1:
            self.event_mlp = nn.Sequential(
                nn.Linear(event_feat_dim - 1, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, embed_dim),
                nn.LayerNorm(embed_dim),
            )
        else:
            self.event_mlp = None
        
        # 事件塔输出层
        event_tower_input_dim = embed_dim * 2 if (event_feat_dim > 1) else embed_dim
        self.event_tower = nn.Sequential(
            nn.Linear(event_tower_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        # ========== 多任务输出头 ==========
        # 任务 1: interested (主任务)
        self.head_interested = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
        )
        
        # 任务 2: not_interested (辅助任务)
        self.head_not_interested = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
        )
        
        # 任务 3: any_interaction (弱监督)
        self.head_any = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
        )
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, user_features, event_features):
        """
        前向传播
        
        Args:
            user_features: (batch, user_feat_dim) - 包含 user_id
            event_features: (batch, event_feat_dim) - 包含 event_id
        """
        # 提取 ID
        user_ids = user_features[:, 0].long()
        event_ids = event_features[:, 0].long()
        
        # ========== 用户塔 ==========
        user_emb = self.user_embedding(user_ids)
        if self.user_mlp is not None:
            user_numeric = user_features[:, 1:]
            user_mlp_out = self.user_mlp(user_numeric)
            user_out = self.user_tower(torch.cat([user_emb, user_mlp_out], dim=1))
        else:
            user_out = self.user_tower(user_emb)
        
        # ========== 事件塔 ==========
        event_emb = self.event_embedding(event_ids)
        if self.event_mlp is not None:
            event_numeric = event_features[:, 1:]
            event_mlp_out = self.event_mlp(event_numeric)
            event_out = self.event_tower(torch.cat([event_emb, event_mlp_out], dim=1))
        else:
            event_out = self.event_tower(event_emb)
        
        # ========== 拼接 + 多任务预测 ==========
        combined = torch.cat([user_out, event_out], dim=1)
        # combined dim = hidden_dim * 2
        
        out_interested = self.head_interested(combined)
        out_not_interested = self.head_not_interested(combined)
        out_any = self.head_any(combined)
        
        return out_interested, out_not_interested, out_any


class MultiTaskLoss(nn.Module):
    """多任务损失函数"""
    
    def __init__(self, weights=[1.0, 0.5, 0.3]):
        super().__init__()
        self.weights = weights
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, outputs, targets):
        """
        Args:
            outputs: (out_interested, out_not_interested, out_any)
            targets: (batch, 3) - [interested, not_interested, any]
        """
        out_int, out_not, out_any = outputs
        
        loss_int = self.bce(out_int, targets[:, 0:1])
        loss_not = self.bce(out_not, targets[:, 1:2])
        loss_any = self.bce(out_any, targets[:, 2:3])
        
        total_loss = (
            self.weights[0] * loss_int +
            self.weights[1] * loss_not +
            self.weights[2] * loss_any
        )
        
        return total_loss, {
            'loss_interested': loss_int.item(),
            'loss_not_interested': loss_not.item(),
            'loss_any': loss_any.item()
        }


class EventRecommendationDataset(Dataset):
    """PyTorch Dataset (简化版 - 只用 ID)"""
    
    def __init__(self, df, mode="train"):
        self.df = df.reset_index(drop=True)
        self.mode = mode
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 只用 ID（其他特征后续添加）
        user_feat = torch.tensor([row['user_id']], dtype=torch.float32)
        event_feat = torch.tensor([row['event_id']], dtype=torch.float32)
        
        if self.mode == "train":
            labels = torch.tensor([
                float(row.get('label_interested', 0)),
                float(row.get('label_not_interested', 0)),
                float(row.get('label_interested', 0)) + float(row.get('label_not_interested', 0))  # any
            ], dtype=torch.float32)
            return user_feat, event_feat, labels
        else:
            return user_feat, event_feat, row.get('user', 0), row.get('event', 0)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    loss_dict = {'loss_interested': 0, 'loss_not_interested': 0, 'loss_any': 0}
    
    for batch_idx, (user_feat, event_feat, labels) in enumerate(dataloader):
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
    
    n_batches = len(dataloader)
    return total_loss / n_batches, {k: v / n_batches for k, v in loss_dict.items()}


def evaluate(model, dataloader, device):
    """评估模型"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for user_feat, event_feat, labels in dataloader:
            user_feat = user_feat.to(device)
            event_feat = event_feat.to(device)
            
            outputs = model(user_feat, event_feat)
            preds = torch.sigmoid(outputs[0]).cpu().numpy()
            
            all_preds.extend(preds.flatten())
            all_labels.extend(labels[:, 0].numpy())
    
    return np.array(all_preds), np.array(all_labels)


def main():
    """主函数"""
    print("=" * 60)
    print("双塔模型训练 - 阶段 2")
    print("=" * 60)
    
    # 配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 路径
    processed_dir = Path(r"C:\Users\Administrator\.openclaw\workspace\event_recommendation\processed")
    
    # 加载数据
    print("\n加载数据...")
    train_df = pd.read_csv(processed_dir / "train_processed.csv")
    print(f"  训练样本：{len(train_df):,}")
    
    # 创建 Dataset
    print("\n创建 Dataset...")
    train_dataset = EventRecommendationDataset(train_df, mode="train")
    
    # 创建 DataLoader
    batch_size = 256
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    print(f"  Batch size: {batch_size}")
    print(f"  Batches per epoch: {len(train_loader)}")
    
    # 创建模型（简化版 - 只用 ID）
    print("\n创建模型...")
    model = DualTowerNet(
        user_feat_dim=1,
        event_feat_dim=1,
        embed_dim=64,
        hidden_dim=128,
        num_users=3391,
        num_events=13418
    ).to(device)
    
    # 打印参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量：{total_params:,}")
    print(f"  可训练参数：{trainable_params:,}")
    
    # 损失函数 + 优化器
    criterion = MultiTaskLoss(weights=[1.0, 0.5, 0.3])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    
    # 训练
    print("\n开始训练...")
    num_epochs = 10
    
    for epoch in range(num_epochs):
        train_loss, losses = train_epoch(model, train_loader, criterion, optimizer, device)
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"    - Interested: {losses['loss_interested']:.4f}")
        print(f"    - Not Interested: {losses['loss_not_interested']:.4f}")
        print(f"    - Any: {losses['loss_any']:.4f}")
        
        # 学习率调整
        scheduler.step(train_loss)
    
    # 保存模型
    print("\n保存模型...")
    save_path = processed_dir / "dual_tower_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, save_path)
    print(f"  模型已保存到：{save_path}")
    
    print("\n" + "=" * 60)
    print("阶段 2 完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
