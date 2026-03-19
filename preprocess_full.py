# -*- coding: utf-8 -*-
"""
完整数据预处理（包含所有特征）
- 用户特征：gender, birthyear, timezone, location
- 事件特征：country, city, lat, lng, 词干计数
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pathlib import Path
import pickle
import json


def load_and_process_data(data_dir):
    """加载并处理所有数据"""
    print("加载数据...")
    
    # 加载数据
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    users = pd.read_csv(data_dir / "users.csv")
    events = pd.read_csv(data_dir / "events.csv")
    
    print(f"  train: {train.shape}")
    print(f"  users: {users.shape}")
    print(f"  events: {events.shape}")
    
    return train, test, users, events


def process_users(users):
    """处理用户特征"""
    print("\n处理用户特征...")
    
    # 处理 birthyear
    users['birthyear_num'] = pd.to_numeric(users['birthyear'], errors='coerce')
    median_birthyear = users['birthyear_num'].median()
    users['birthyear_filled'] = users['birthyear_num'].fillna(median_birthyear)
    
    # 处理 timezone
    users['timezone_num'] = pd.to_numeric(users['timezone'], errors='coerce')
    users['timezone_filled'] = users['timezone_num'].fillna(0)
    
    # 处理 gender
    users['gender_filled'] = users['gender'].fillna('unknown')
    gender_enc = LabelEncoder()
    users['gender_encoded'] = gender_enc.fit_transform(users['gender_filled'])
    
    # 处理 location（提取国家）
    def extract_country(loc):
        if pd.isna(loc) or len(str(loc).split()) < 2:
            return 'Unknown'
        return str(loc).split()[-1]
    
    users['country'] = users['location'].apply(extract_country)
    country_enc = LabelEncoder()
    users['country_encoded'] = country_enc.fit_transform(users['country'])
    
    # 归一化数值特征
    scaler = StandardScaler()
    user_numeric = users[['birthyear_filled', 'timezone_filled']].values
    user_scaled = scaler.fit_transform(user_numeric)
    users['birthyear_scaled'] = user_scaled[:, 0]
    users['timezone_scaled'] = user_scaled[:, 1]
    
    print(f"  用户特征处理完成")
    print(f"  性别类别：{list(gender_enc.classes_)}")
    print(f"  国家类别数：{len(country_enc.classes_)}")
    
    return users, gender_enc, country_enc, scaler


def process_events(events):
    """处理事件特征"""
    print("\n处理事件特征...")
    
    # 处理坐标
    events['lat_filled'] = events['lat'].fillna(0)
    events['lng_filled'] = events['lng'].fillna(0)
    
    # 处理国家
    events['country_filled'] = events['country'].fillna('Unknown')
    country_enc = LabelEncoder()
    events['country_encoded'] = country_enc.fit_transform(events['country_filled'])
    
    # 处理城市
    events['city_filled'] = events['city'].fillna('Unknown')
    city_enc = LabelEncoder()
    events['city_encoded'] = city_enc.fit_transform(events['city_filled'])
    
    # 选择前 20 个词干特征（实际列名是 c_1, c_2, ...）
    count_cols = [f'c_{i}' for i in range(1, 21)]
    
    # 归一化数值特征
    scaler = StandardScaler()
    event_numeric = events[['lat_filled', 'lng_filled'] + count_cols].values
    event_scaled = scaler.fit_transform(event_numeric)
    
    for i, col in enumerate(['lat_scaled', 'lng_scaled'] + [f'c_{i}_scaled' for i in range(1, 21)]):
        events[col] = event_scaled[:, i]
    
    print(f"  事件特征处理完成")
    print(f"  国家类别数：{len(country_enc.classes_)}")
    print(f"  城市类别数：{len(city_enc.classes_)}")
    
    return events, country_enc, city_enc, scaler


def merge_features(train, test, users, events):
    """合并特征到训练/测试数据"""
    print("\n合并特征...")
    
    # 用户 ID 编码
    all_users = pd.concat([train['user'], test['user']]).unique()
    user_enc = LabelEncoder()
    user_enc.fit(all_users)
    
    # 事件 ID 编码
    all_events = pd.concat([train['event'], test['event']]).unique()
    event_enc = LabelEncoder()
    event_enc.fit(all_events)
    
    # 转换 ID
    train['user_encoded'] = user_enc.transform(train['user'])
    train['event_encoded'] = event_enc.transform(train['event'])
    test['user_encoded'] = user_enc.transform(test['user'])
    test['event_encoded'] = event_enc.transform(test['event'])
    
    # 准备用户特征
    user_features = users[['user_id', 'gender_encoded', 'birthyear_scaled', 
                           'timezone_scaled', 'country_encoded']].copy()
    user_features = user_features.rename(columns={'user_id': 'user'})
    
    # 准备事件特征
    event_features = events[['event_id', 'country_encoded', 'city_encoded',
                             'lat_scaled', 'lng_scaled'] + 
                            [f'c_{i}_scaled' for i in range(1, 21)]].copy()
    event_features = event_features.rename(columns={'event_id': 'event'})
    
    # 合并
    train = train.merge(user_features, on='user', how='left')
    train = train.merge(event_features, on='event', how='left')
    
    test = test.merge(user_features, on='user', how='left')
    test = test.merge(event_features, on='event', how='left')
    
    # 填充缺失值
    train = train.fillna(0)
    test = test.fillna(0)
    
    # 创建多任务标签
    train['label_interested'] = train['interested']
    train['label_not_interested'] = train['not_interested']
    train['label_any'] = ((train['interested'] == 1) | (train['not_interested'] == 1)).astype(int)
    
    print(f"  训练集特征列数：{len(train.columns)}")
    print(f"  测试集特征列数：{len(test.columns)}")
    
    return train, test, user_enc, event_enc


def save_processed_data(train, test, encoders, save_dir):
    """保存处理后的数据"""
    print("\n保存处理后的数据...")
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存数据
    train.to_csv(save_dir / "train_full.csv", index=False)
    test.to_csv(save_dir / "test_full.csv", index=False)
    
    # 保存编码器
    with open(save_dir / "encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)
    
    # 保存元数据
    metadata = {
        'num_users': len(encoders['user_enc'].classes_),
        'num_events': len(encoders['event_enc'].classes_),
        'user_feature_dim': 5,  # user_id + gender + birthyear + timezone + country
        'event_feature_dim': 24,  # event_id + country + city + lat + lng + 20 counts
        'train_samples': len(train),
        'test_samples': len(test)
    }
    
    with open(save_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  数据已保存到：{save_dir}")
    print(f"  元数据：{metadata}")


def main():
    """主函数"""
    print("=" * 60)
    print("完整数据预处理（优化版）")
    print("=" * 60)
    
    data_dir = Path(r"C:\Users\Administrator\.openclaw\workspace\event_recommendation\data")
    save_dir = Path(r"C:\Users\Administrator\.openclaw\workspace\event_recommendation\processed")
    
    # 加载数据
    train, test, users, events = load_and_process_data(data_dir)
    
    # 处理用户
    users, gender_enc, country_enc_user, user_scaler = process_users(users)
    
    # 处理事件
    events, country_enc_event, city_enc, event_scaler = process_events(events)
    
    # 合并特征
    train, test, user_enc, event_enc = merge_features(train, test, users, events)
    
    # 保存
    encoders = {
        'user_enc': user_enc,
        'event_enc': event_enc,
        'gender_enc': gender_enc,
        'country_enc_user': country_enc_user,
        'country_enc_event': country_enc_event,
        'city_enc': city_enc,
        'user_scaler': user_scaler,
        'event_scaler': event_scaler
    }
    
    save_processed_data(train, test, encoders, save_dir)
    
    print("\n" + "=" * 60)
    print("预处理完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
