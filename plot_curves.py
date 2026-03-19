# -*- coding: utf-8 -*-
"""
生成训练曲线图（用于报告/PPT）
"""

import matplotlib.pyplot as plt
from pathlib import Path

print("Generating training curves...")

output_dir = Path(r"C:\Users\Administrator\.openclaw\workspace\event_recommendation\output")
output_dir.mkdir(parents=True, exist_ok=True)

# 训练历史
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

# 创建图表
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：总 Loss
axes[0].plot(history['epochs'], history['total_loss'], 'b-o', linewidth=2, markersize=6, label='Total Loss')
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('Training Loss Over Epochs', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].set_xticks(history['epochs'])

# 右图：各任务 Loss
axes[1].plot(history['epochs'], history['loss_interested'], 'g-o', linewidth=2, markersize=6, label='Interested')
axes[1].plot(history['epochs'], history['loss_not_interested'], 'r-o', linewidth=2, markersize=6, label='Not Interested')
axes[1].plot(history['epochs'], history['loss_any'], 'm-o', linewidth=2, markersize=6, label='Any Interaction')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Loss', fontsize=12)
axes[1].set_title('Multi-Task Loss Breakdown', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)
axes[1].set_xticks(history['epochs'])

plt.tight_layout()
save_path = output_dir / "training_curves.png"
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"Training curves saved to: {save_path}")
print("\nDone!")
