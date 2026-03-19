# -*- coding: utf-8 -*-
"""
数据预处理模块
- 用户/事件 ID 映射
- 类别特征编码
- 数值特征归一化
- 构建 PyTorch Dataset
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path
import pickle
import json


class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.user_encoder = LabelEncoder()
        self.event_encoder = LabelEncoder()
        self.gender_encoder = LabelEncoder()
        self.country_encoder = LabelEncoder()
        self.user_scaler = StandardScaler()
        self.event_scaler = StandardScaler()
        
        # 映射字典（保存用）
        self.user2id = {}
        self.event2id = {}
        self.id2user = {}
        self.id2event = {}
        
    def load_data(self):
        """加载所有数据文件"""
        print("加载数据...")
        
        self.train = pd.read_csv(self.data_dir / "train.csv")
        self.test = pd.read_csv(self.data_dir / "test.csv")
        self.users = pd.read_csv(self.data_dir / "users.csv")
        self.events = pd.read_csv(self.data_dir / "events.csv")
        self.friends = pd.read_csv(self.data_dir / "user_friends.csv")
        self.attendees = pd.read_csv(self.data_dir / "event_attendees.csv")
        
        print(f"  train: {self.train.shape}")
        print(f"  test: {self.test.shape}")
        print(f"  users: {self.users.shape}")
        print(f"  events: {self.events.shape}")
        print(f"  friends: {self.friends.shape}")
        print(f"  attendees: {self.attendees.shape}")
        
    def process_users(self):
        """处理用户数据"""
        print("\n处理用户数据...")
        
        users = self.users.copy()
        
        # 处理 birthyear（转换为数值）
        users['birthyear'] = pd.to_numeric(users['birthyear'], errors='coerce')
        users['birthyear'] = users['birthyear'].fillna(users['birthyear'].median())
        
        # 处理 timezone（转换为数值）
        users['timezone'] = pd.to_numeric(users['timezone'], errors='coerce')
        users['timezone'] = users['timezone'].fillna(0)
        
        # 处理 gender（类别编码）
        users['gender'] = users['gender'].fillna('unknown')
        self.gender_encoder.fit(users['gender'])
        users['gender_encoded'] = self.gender_encoder.transform(users['gender'])
        
        # 处理 location（提取国家）
        users['country'] = users['location'].apply(
            lambda x: x.split()[-1] if pd.notna(x) and len(x.split()) > 1 else 'Unknown'
        )
        self.country_encoder.fit(users['country'])
        users['country_encoded'] = self.country_encoder.transform(users['country'])
        
        # 归一化数值特征
        user_numeric = users[['birthyear', 'timezone']].values
        user_numeric_scaled = self.user_scaler.fit_transform(user_numeric)
        users['birthyear_scaled'] = user_numeric_scaled[:, 0]
        users['timezone_scaled'] = user_numeric_scaled[:, 1]
        
        self.users_processed = users
        print(f"  处理完成：{users.shape}")
        
    def process_events(self):
        """处理事件数据"""
        print("\n处理事件数据...")
        
        events = self.events.copy()
        
        # 只保留前 9 列基础特征 + 词干计数
        base_cols = ['event_id', 'user_id', 'start_time', 'city', 'state', 
                     'zip', 'country', 'lat', 'lng']
        count_cols = [col for col in events.columns if col.startswith('count_')]
        
        # 选择需要的列
        event_cols = base_cols + count_cols[:20]  # 只用前 20 个词干特征（降维）
        events = events[event_cols].copy()
        
        # 处理坐标
        events['lat'] = events['lat'].fillna(0)
        events['lng'] = events['lng'].fillna(0)
        
        # 处理国家
        events['country'] = events['country'].fillna('Unknown')
        # 复用用户的国家编码器（如果有重叠）
        events['country_encoded'] = events['country'].apply(
            lambda x: self.country_encoder.transform([x])[0] 
            if x in self.country_encoder.classes_ 
            else 0
        )
        
        # 归一化数值特征
        event_numeric = events[['lat', 'lng'] + count_cols[:20]].values
        event_numeric_scaled = self.event_scaler.fit_transform(event_numeric)
        
        for i, col in enumerate(['lat_scaled', 'lng_scaled'] + [f'count_{i}_scaled' for i in range(20)]):
            events[col] = event_numeric_scaled[:, i]
        
        self.events_processed = events
        print(f"  处理完成：{events.shape}")
        
    def process_interactions(self):
        """处理交互数据（train/test）"""
        print("\n处理交互数据...")
        
        # 编码用户和事件 ID
        all_users = pd.concat([self.train['user'], self.test['user']]).unique()
        all_events = pd.concat([self.train['event'], self.test['event']]).unique()
        
        self.user_encoder.fit(all_users)
        self.event_encoder.fit(all_events)
        
        # 创建映射字典
        self.user2id = {user: idx for idx, user in enumerate(self.user_encoder.classes_)}
        self.event2id = {event: idx for idx, event in enumerate(self.event_encoder.classes_)}
        self.id2user = {idx: user for user, idx in self.user2id.items()}
        self.id2event = {idx: event for event, idx in self.event2id.items()}
        
        # 转换 train
        self.train['user_id'] = self.user_encoder.transform(self.train['user'])
        self.train['event_id'] = self.event_encoder.transform(self.train['event'])
        
        # 转换 test
        self.test['user_id'] = self.user_encoder.transform(self.test['user'])
        self.test['event_id'] = self.event_encoder.transform(self.test['event'])
        
        # 创建标签（多任务）
        # 任务 1: interested (主任务)
        self.train['label_interested'] = self.train['interested']
        # 任务 2: not_interested (辅助任务)
        self.train['label_not_interested'] = self.train['not_interested']
        # 任务 3: 是否有任何交互（弱监督）
        self.train['label_any_interaction'] = (
            (self.train['interested'] == 1) | (self.train['not_interested'] == 1)
        ).astype(int)
        
        print(f"  用户数：{len(all_users):,}")
        print(f"  事件数：{len(all_events):,}")
        print(f"  train 样本：{len(self.train):,}")
        print(f"  test 样本：{len(self.test):,}")
        
    def merge_features(self):
        """合并用户和事件特征到交互数据"""
        print("\n合并特征...")
        
        # 合并用户特征
        user_features = self.users_processed[['user_id', 'gender_encoded', 
                                               'birthyear_scaled', 'timezone_scaled',
                                               'country_encoded']].copy()
        self.train = self.train.merge(user_features, on='user_id', how='left')
        self.test = self.test.merge(user_features, on='user_id', how='left')
        
        # 合并事件特征
        event_features = self.events_processed[['event_id', 'country_encoded',
                                                 'lat_scaled', 'lng_scaled'] + 
                                                [f'count_{i}_scaled' for i in range(20)]].copy()
        self.train = self.train.merge(event_features, on='event_id', how='left')
        self.test = self.test.merge(event_features, on='event_id', how='left')
        
        # 填充缺失值
        self.train = self.train.fillna(0)
        self.test = self.test.fillna(0)
        
        print(f"  train 特征列：{len(self.train.columns)}")
        print(f"  test 特征列：{len(self.test.columns)}")
        
    def save_preprocessor(self, save_dir: Path):
        """保存预处理器"""
        print("\n保存预处理器...")
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存映射字典
        with open(save_dir / "mappings.json", "w", encoding="utf-8") as f:
            json.dump({
                "user2id": {str(k): v for k, v in self.user2id.items()},
                "event2id": {str(k): v for k, v in self.event2id.items()},
                "id2user": {str(k): v for k, v in self.id2user.items()},
                "id2event": {str(k): v for k, v in self.id2event.items()}
            }, f, ensure_ascii=False)
        
        # 保存编码器
        with open(save_dir / "encoders.pkl", "wb") as f:
            pickle.dump({
                "user_encoder": self.user_encoder,
                "event_encoder": self.event_encoder,
                "gender_encoder": self.gender_encoder,
                "country_encoder": self.country_encoder,
                "user_scaler": self.user_scaler,
                "event_scaler": self.event_scaler
            }, f)
        
        # 保存处理后的数据
        self.train.to_csv(save_dir / "train_processed.csv", index=False)
        self.test.to_csv(save_dir / "test_processed.csv", index=False)
        
        print(f"  保存到：{save_dir}")
        
    def run(self, save_dir: Path = None):
        """运行完整预处理流程"""
        self.load_data()
        self.process_users()
        self.process_events()
        self.process_interactions()
        self.merge_features()
        
        if save_dir:
            self.save_preprocessor(save_dir)
        
        print("\n✅ 预处理完成!")
        return self


class EventRecommendationDataset(Dataset):
    """PyTorch Dataset"""
    
    def __init__(self, df: pd.DataFrame, mode: str = "train"):
        self.df = df
        self.mode = mode
        
        # 用户特征列
        self.user_feature_cols = ['user_id', 'gender_encoded', 'birthyear_scaled', 
                                   'timezone_scaled', 'country_encoded']
        
        # 事件特征列
        self.event_feature_cols = ['event_id', 'country_encoded', 'lat_scaled', 'lng_scaled'] + \
                                   [f'count_{i}_scaled' for i in range(20)]
        
        # 标签列
        self.label_cols = ['label_interested', 'label_not_interested', 'label_any_interaction']
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 用户特征
        user_features = torch.tensor(
            [row[col] for col in self.user_feature_cols], 
            dtype=torch.float32
        )
        
        # 事件特征
        event_features = torch.tensor(
            [row[col] for col in self.event_feature_cols], 
            dtype=torch.float32
        )
        
        if self.mode == "train":
            # 多任务标签
            labels = torch.tensor(
                [row[col] for col in self.label_cols], 
                dtype=torch.float32
            )
            return user_features, event_features, labels
        else:
            # 测试集返回原始 ID 用于提交
            return user_features, event_features, row['user'], row['event']


def main():
    """主函数"""
    print("Start preprocessing...")
    
    # 路径
    data_dir = Path(r"C:\Users\Administrator\.openclaw\workspace\event_recommendation\data")
    save_dir = Path(r"C:\Users\Administrator\.openclaw\workspace\event_recommendation\processed")
    
    # 预处理
    preprocessor = DataPreprocessor(data_dir)
    preprocessor.run(save_dir)
    
    # 创建 Dataset
    print("Creating Dataset...")
    train_dataset = EventRecommendationDataset(preprocessor.train, mode="train")
    test_dataset = EventRecommendationDataset(preprocessor.test, mode="test")
    
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Test dataset: {len(test_dataset)} samples")
    
    # 测试数据加载
    print("Testing data loading...")
    user_feat, event_feat, labels = train_dataset[0]
    print(f"  User feat dim: {user_feat.shape}")
    print(f"  Event feat dim: {event_feat.shape}")
    print(f"  Labels dim: {labels.shape}")
    
    print("Done!")


if __name__ == "__main__":
    main()
