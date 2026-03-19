# -*- coding: utf-8 -*-
"""
评估与提交生成
- MAP@200 评估函数
- 生成 Kaggle 提交文件
- 训练曲线可视化
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


def apk(actual, predicted, k=200):
    """
    Average Precision at k
    
    Args:
        actual: 真实感兴趣的事件列表
        predicted: 预测的事件列表（已排序）
        k: 取前 k 个
    """
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
    """
    Mean Average Precision at k
    
    Args:
        actual: 真实列表的列表 [[user1_events], [user2_events], ...]
        predicted: 预测列表的列表 [[user1_pred], [user2_pred], ...]
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def generate_submission(model, test_df, device, save_path):
    """生成 Kaggle 提交文件"""
    print("生成提交文件...")
    
    model.eval()
    
    # 按用户分组
    user_groups = test_df.groupby('user')
    
    submissions = []
    
    with torch.no_grad():
        for user_id, group in user_groups:
            events = group['event'].values
            event_ids = group['event_id'].values
            
            # 为每个事件生成预测分数
            scores = []
            for event_id, event_idx in zip(events, event_ids):
                user_feat = torch.tensor([[user_id]], dtype=torch.float32).to(device)
                event_feat = torch.tensor([[event_idx]], dtype=torch.float32).to(device)
                
                out_int, out_not, out_any = model(user_feat, event_feat)
                score = torch.sigmoid(out_int).cpu().item()
                scores.append((event_id, score))
            
            # 按分数降序排序
            scores.sort(key=lambda x: x[1], reverse=True)
            event_list = ' '.join([str(e[0]) for e in scores])
            
            submissions.append({
                'User': user_id,
                'Events': event_list
            })
    
    # 保存提交文件
    sub_df = pd.DataFrame(submissions)
    sub_df = sub_df.sort_values('User')  # 按用户 ID 排序
    sub_df.to_csv(save_path, index=False)
    
    print(f"  提交文件已保存：{save_path}")
    print(f"  用户数：{len(sub_df):,}")
    
    return sub_df


def plot_training_history(history, save_path):
    """绘制训练曲线"""
    print("绘制训练曲线...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：总 Loss
    axes[0].plot(history['epochs'], history['total_loss'], 'b-', linewidth=2, label='Total Loss')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Loss Over Epochs', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 右图：各任务 Loss
    axes[1].plot(history['epochs'], history['loss_interested'], 'g-', linewidth=2, label='Interested')
    axes[1].plot(history['epochs'], history['loss_not_interested'], 'r-', linewidth=2, label='Not Interested')
    axes[1].plot(history['epochs'], history['loss_any'], 'm-', linewidth=2, label='Any Interaction')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title('Multi-Task Loss Breakdown', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  训练曲线已保存：{save_path}")


def main():
    """主函数"""
    print("=" * 60)
    print("评估与提交生成 - 阶段 3")
    print("=" * 60)
    
    # 路径
    processed_dir = Path(r"C:\Users\Administrator\.openclaw\workspace\event_recommendation\processed")
    data_dir = Path(r"C:\Users\Administrator\.openclaw\workspace\event_recommendation\data")
    output_dir = Path(r"C:\Users\Administrator\.openclaw\workspace\event_recommendation\output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载模型
    print("\n加载模型...")
    from model import DualTowerNet
    
    device = torch.device('cpu')  # 使用 CPU 避免内存问题
    
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
    
    # 加载测试数据
    print("\n加载测试数据...")
    test_df = pd.read_csv(data_dir / "test.csv")
    
    # 编码 ID（需要和训练时一致）
    train_df = pd.read_csv(data_dir / "train.csv")
    all_users = pd.concat([train_df['user'], test_df['user']]).unique()
    all_events = pd.concat([train_df['event'], test_df['event']]).unique()
    
    user2id = {user: idx for idx, user in enumerate(all_users)}
    event2id = {event: idx for idx, event in enumerate(all_events)}
    
    test_df['user_id'] = test_df['user'].map(user2id)
    test_df['event_id'] = test_df['event'].map(event2id)
    
    # 过滤掉未知用户/事件（确保在 Embedding 范围内）
    test_df = test_df.dropna(subset=['user_id', 'event_id'])
    test_df = test_df[test_df['user_id'] < 3391]  # num_users
    test_df = test_df[test_df['event_id'] < 13418]  # num_events
    test_df['user_id'] = test_df['user_id'].astype(int)
    test_df['event_id'] = test_df['event_id'].astype(int)
    
    print(f"  测试样本：{len(test_df):,}")
    
    # 生成提交文件
    print("\n生成提交文件...")
    submission_path = output_dir / "submission.csv"
    generate_submission(model, test_df, device, submission_path)
    
    # 绘制训练曲线
    print("\n绘制训练曲线...")
    history = {
        'epochs': list(range(1, 11)),
        'total_loss': [0.9236, 0.7856, 0.6811, 0.5508, 0.4309, 
                       0.3556, 0.3176, 0.2860, 0.2697, 0.2466],
        'loss_interested': [0.6384, 0.5537, 0.4835, 0.3906, 0.3042,
                            0.2514, 0.2251, 0.2050, 0.1954, 0.1790],
        'loss_not_interested': [0.1800, 0.1107, 0.0812, 0.0652, 0.0544,
                                 0.0455, 0.0391, 0.0288, 0.0227, 0.0180],
        'loss_any': [0.6506, 0.5884, 0.5235, 0.4256, 0.3314,
                     0.2715, 0.2433, 0.2218, 0.2099, 0.1954]
    }
    
    curve_path = output_dir / "training_curves.png"
    plot_training_history(history, curve_path)
    
    # 示例 MAP@200 计算（使用训练集模拟）
    print("\nMAP@200 评估（示例）...")
    # 这里只是演示，实际需要验证集
    print("  注意：由于竞赛已结束，无法获取真实标签")
    print("  实际使用时需要验证集计算 MAP@200")
    
    print("\n" + "=" * 60)
    print("阶段 3 完成！")
    print("=" * 60)
    
    print("\n输出文件:")
    print(f"  1. 提交文件：{submission_path}")
    print(f"  2. 训练曲线：{curve_path}")


if __name__ == "__main__":
    main()
