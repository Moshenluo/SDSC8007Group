# -*- coding: utf-8 -*-
"""
MAP@200 评估 - 使用训练集进行自评估
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict


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


def main():
    """主函数"""
    print("=" * 60)
    print("MAP@200 评估 - 训练集自评估")
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
    print(f"  模型已加载")
    
    # 加载训练数据
    print("\n加载训练数据...")
    train_df = pd.read_csv(data_dir / "train.csv")
    print(f"  总样本：{len(train_df):,}")
    
    # 创建 ID 映射（和训练时一致）
    all_users = train_df['user'].unique()
    all_events = train_df['event'].unique()
    
    user2id = {user: idx for idx, user in enumerate(all_users)}
    event2id = {event: idx for idx, event in enumerate(all_events)}
    
    # 只保留在模型范围内的用户
    train_df = train_df[train_df['user'].isin(list(user2id.keys())[:3391])]
    
    # 按用户分组评估
    print("\n计算 MAP@200...")
    user_groups = train_df.groupby('user')
    
    actual = []
    predicted = []
    
    model.eval()
    
    with torch.no_grad():
        for idx, (user, group) in enumerate(user_groups):
            if idx % 500 == 0:
                print(f"  处理用户：{idx}/{len(user_groups)}")
            
            # 真实感兴趣的事件
            interested_events = group[group['interested'] == 1]['event'].tolist()
            
            if len(interested_events) == 0:
                continue
            
            # 为该用户的所有事件生成预测分数
            events = group['event'].values
            event_ids = group['event'].values
            
            scores = []
            for event, event_id in zip(events, event_ids):
                # 使用原始用户 ID 和事件 ID（需要映射）
                user_idx = user2id.get(user, -1)
                event_idx = event2id.get(event, -1)
                
                if user_idx < 0 or event_idx < 0 or user_idx >= 3391 or event_idx >= 13418:
                    scores.append((event, 0.5))  # 默认分数
                    continue
                
                user_feat = torch.tensor([[user_idx]], dtype=torch.float32).to(device)
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
    map_score = mapk(actual, predicted, k=200)
    
    print(f"\n  评估用户数：{len(actual):,}")
    print(f"  MAP@200: {map_score:.4f}")
    
    # 保存结果
    print("\n保存评估结果...")
    import json
    result_path = processed_dir / "map200_result.json"
    
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump({
            'map_at_200': float(map_score),
            'num_users_evaluated': len(actual),
            'total_samples': len(train_df),
            'model_type': 'Dual Tower + Multi-task',
            'epochs_trained': 10,
            'final_loss': 0.2466
        }, f, indent=2, ensure_ascii=False)
    
    print(f"  结果已保存：{result_path}")
    
    # 性能评级
    print("\n" + "=" * 60)
    print("性能评级")
    print("=" * 60)
    
    if map_score >= 0.65:
        rating = "[GOLD] Excellent (接近金牌水平)"
    elif map_score >= 0.55:
        rating = "[SILVER] Good (银牌水平)"
    elif map_score >= 0.45:
        rating = "[BRONZE] Fair (铜牌水平)"
    elif map_score >= 0.35:
        rating = "[BASELINE] Needs optimization"
    else:
        rating = "[WARNING] Needs improvement"
    
    print(f"\n  MAP@200 = {map_score:.4f}")
    print(f"  评级：{rating}")
    
    print("\n" + "=" * 60)
    print("评估完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
