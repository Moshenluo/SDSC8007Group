# -*- coding: utf-8 -*-
"""
MAP@200 评估 - 使用验证集模拟比赛评分
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split


def apk(actual, predicted, k=200):
    """Average Precision at k"""
    if len(predicted) > k:
        predicted = predicted[:k]
    
    if len(actual) == 0:
        return 0.0
    
    score = 0.0
    num_hits = 0.0
    
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    
    return score / min(len(actual), k)


def mapk(actual, predicted, k=200):
    """Mean Average Precision at k"""
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def create_validation_split(data_dir, val_ratio=0.2):
    """从训练集划分验证集"""
    print("划分验证集...")
    
    train_df = pd.read_csv(data_dir / "train.csv")
    
    # 按用户分组，确保同一用户的数据在同一组
    users = train_df['user'].unique()
    train_users, val_users = train_test_split(users, test_size=val_ratio, random_state=42)
    
    train_data = train_df[train_df['user'].isin(train_users)]
    val_data = train_df[train_df['user'].isin(val_users)]
    
    print(f"  训练集用户：{len(train_users):,}, 样本：{len(train_data):,}")
    print(f"  验证集用户：{len(val_users):,}, 样本：{len(val_data):,}")
    
    return train_data, val_data


def prepare_data(train_data, val_data):
    """准备数据（编码 ID）"""
    print("准备数据...")
    
    # 创建 ID 映射
    all_users = pd.concat([train_data['user'], val_data['user']]).unique()
    all_events = pd.concat([train_data['event'], val_data['event']]).unique()
    
    user2id = {user: idx for idx, user in enumerate(all_users)}
    event2id = {event: idx for idx, event in enumerate(all_events)}
    
    # 转换 ID
    train_data = train_data.copy()
    val_data = val_data.copy()
    
    train_data['user_id'] = train_data['user'].map(user2id)
    train_data['event_id'] = train_data['event'].map(event2id)
    
    val_data['user_id'] = val_data['user'].map(user2id)
    val_data['event_id'] = val_data['event'].map(event2id)
    
    # 过滤无效数据（确保在模型 Embedding 范围内）
    train_data = train_data.dropna(subset=['user_id', 'event_id'])
    val_data = val_data.dropna(subset=['user_id', 'event_id'])
    
    # 过滤超出模型范围的用户和事件
    train_data = train_data[(train_data['user_id'] < 3391) & (train_data['user_id'] >= 0)]
    train_data = train_data[(train_data['event_id'] < 13418) & (train_data['event_id'] >= 0)]
    
    val_data = val_data[(val_data['user_id'] < 3391) & (val_data['user_id'] >= 0)]
    val_data = val_data[(val_data['event_id'] < 13418) & (val_data['event_id'] >= 0)]
    
    train_data['user_id'] = train_data['user_id'].astype(int)
    train_data['event_id'] = train_data['event_id'].astype(int)
    val_data['user_id'] = val_data['user_id'].astype(int)
    val_data['event_id'] = val_data['event_id'].astype(int)
    
    print(f"  训练样本：{len(train_data):,}")
    print(f"  验证样本：{len(val_data):,}")
    
    return train_data, val_data, user2id, event2id


def evaluate_on_validation(model, val_data, device, k=200):
    """在验证集上计算 MAP@200"""
    print(f"\n在验证集上评估 MAP@{k}...")
    
    model.eval()
    
    # 按用户分组
    val_groups = val_data.groupby('user')
    
    actual = []  # 真实感兴趣的事件
    predicted = []  # 预测排序的事件
    
    with torch.no_grad():
        for user_id, group in val_groups:
            # 真实感兴趣的事件（interested=1）
            interested_events = group[group['interested'] == 1]['event'].tolist()
            
            if len(interested_events) == 0:
                continue  # 跳过没有正样本的用户
            
            # 为该用户的所有事件生成预测分数
            events = group['event'].values
            event_ids = group['event_id'].values
            
            scores = []
            for event, event_idx in zip(events, event_ids):
                user_feat = torch.tensor([[user_id]], dtype=torch.float32).to(device)
                event_feat = torch.tensor([[event_idx]], dtype=torch.float32).to(device)
                
                out_int, out_not, out_any = model(user_feat, event_feat)
                score = torch.sigmoid(out_int).cpu().item()
                scores.append((event, score))
            
            # 按分数降序排序
            scores.sort(key=lambda x: x[1], reverse=True)
            pred_events = [e[0] for e in scores]
            
            actual.append(interested_events)
            predicted.append(pred_events)
    
    # 计算 MAP@200
    map_score = mapk(actual, predicted, k=k)
    
    print(f"  验证集用户数：{len(actual):,}")
    print(f"  MAP@{k}: {map_score:.4f}")
    
    return map_score, actual, predicted


def main():
    """主函数"""
    print("=" * 60)
    print("MAP@200 评估 - 验证集模拟")
    print("=" * 60)
    
    # 路径
    data_dir = Path(r"C:\Users\Administrator\.openclaw\workspace\event_recommendation\data")
    processed_dir = Path(r"C:\Users\Administrator\.openclaw\workspace\event_recommendation\processed")
    
    # 加载模型
    print("\n加载模型...")
    from model import DualTowerNet
    
    device = torch.device('cpu')
    
    model = DualTowerNet(
        user_feat_dim=1,
        event_feat_dim=1,
        embed_dim=64,
        hidden_dim=128,
        num_users=3391,
        num_events=13418
    ).to(device)
    
    checkpoint = torch.load(processed_dir / "dual_tower_model.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  模型已加载，设备：{device}")
    
    # 划分验证集
    train_data, val_data = create_validation_split(data_dir, val_ratio=0.2)
    
    # 准备数据
    train_data, val_data, user2id, event2id = prepare_data(train_data, val_data)
    
    # 在验证集上评估
    map_score, actual, predicted = evaluate_on_validation(model, val_data, device, k=200)
    
    # 保存评估结果
    print("\n保存评估结果...")
    result_path = processed_dir / "map200_evaluation.json"
    
    import json
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump({
            'map_at_200': float(map_score),
            'num_val_users': len(actual),
            'num_val_samples': len(val_data),
            'model_info': {
                'embed_dim': 64,
                'hidden_dim': 128,
                'num_users': 3391,
                'num_events': 13418,
                'total_params': 1192003
            }
        }, f, indent=2, ensure_ascii=False)
    
    print(f"  评估结果已保存：{result_path}")
    
    # 性能分析
    print("\n" + "=" * 60)
    print("性能分析")
    print("=" * 60)
    
    if map_score >= 0.60:
        print(f"\n✅ 优秀！MAP@200 = {map_score:.4f}")
        print("   达到历史银牌水平（0.60+）")
    elif map_score >= 0.50:
        print(f"\n👍 良好！MAP@200 = {map_score:.4f}")
        print("   达到良好水平，可以继续优化")
    elif map_score >= 0.35:
        print(f"\n📊 基线水平 MAP@200 = {map_score:.4f}")
        print("   建议添加更多特征提升性能")
    else:
        print(f"\n⚠️  需要改进 MAP@200 = {map_score:.4f}")
        print("   建议检查模型或增加训练")
    
    print("\n" + "=" * 60)
    print("评估完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
