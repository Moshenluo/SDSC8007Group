"""
Event Recommendation Challenge - 数据探索分析 (EDA)
运行前请确保已下载数据并解压到 data/ 目录
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 数据路径
DATA_DIR = Path("./data")

def load_data():
    """加载所有数据文件"""
    print("=" * 60)
    print("加载数据文件...")
    print("=" * 60)
    
    data = {}
    
    # 训练集
    print("\n[1] 加载 train.csv...")
    data['train'] = pd.read_csv(DATA_DIR / "train.csv")
    print(f"    形状：{data['train'].shape}")
    print(f"    列：{list(data['train'].columns)}")
    
    # 测试集
    print("\n[2] 加载 test.csv...")
    data['test'] = pd.read_csv(DATA_DIR / "test.csv")
    print(f"    形状：{data['test'].shape}")
    
    # 用户数据
    print("\n[3] 加载 users.csv...")
    data['users'] = pd.read_csv(DATA_DIR / "users.csv")
    print(f"    形状：{data['users'].shape}")
    print(f"    列：{list(data['users'].columns)}")
    
    # 用户好友关系
    print("\n[4] 加载 user_friends.csv...")
    data['friends'] = pd.read_csv(DATA_DIR / "user_friends.csv")
    print(f"    形状：{data['friends'].shape}")
    
    # 事件数据
    print("\n[5] 加载 events.csv...")
    data['events'] = pd.read_csv(DATA_DIR / "events.csv")
    print(f"    形状：{data['events'].shape}")
    print(f"    列数：{len(data['events'].columns)}")
    
    # 事件参与者
    print("\n[6] 加载 event_attendees.csv...")
    data['attendees'] = pd.read_csv(DATA_DIR / "event_attendees.csv")
    print(f"    形状：{data['attendees'].shape}")
    
    return data


def analyze_train(data):
    """分析训练集"""
    print("\n" + "=" * 60)
    print("训练集分析")
    print("=" * 60)
    
    train = data['train']
    
    print("\n【基本统计】")
    print(f"总样本数：{len(train):,}")
    print(f"唯一用户数：{train['user'].nunique():,}")
    print(f"唯一事件数：{train['event'].nunique():,}")
    
    print("\n【标签分布】")
    interested = (train['interested'] == 1).sum()
    not_interested = (train['not_interested'] == 1).sum()
    no_action = len(train) - interested - not_interested
    
    print(f"感兴趣 (interested=1): {interested:,} ({interested/len(train)*100:.2f}%)")
    print(f"不感兴趣 (not_interested=1): {not_interested:,} ({not_interested/len(train)*100:.2f}%)")
    print(f"无操作 (两者都为 0): {no_action:,} ({no_action/len(train)*100:.2f}%)")
    
    print("\n【是否被邀请分布】")
    invited = (train['invited'] == 1).sum()
    print(f"被邀请：{invited:,} ({invited/len(train)*100:.2f}%)")
    print(f"未被邀请：{len(train) - invited:,} ({(len(train)-invited)/len(train)*100:.2f}%)")


def analyze_users(data):
    """分析用户数据"""
    print("\n" + "=" * 60)
    print("用户数据分析")
    print("=" * 60)
    
    users = data['users']
    
    print("\n【基本统计】")
    print(f"总用户数：{len(users):,}")
    
    print("\n【性别分布】")
    gender_dist = users['gender'].value_counts()
    for gender, count in gender_dist.items():
        print(f"  {gender}: {count:,} ({count/len(users)*100:.2f}%)")
    
    print("\n【地理位置分布 (Top 10)】")
    location_dist = users['location'].value_counts().head(10)
    for loc, count in location_dist.items():
        print(f"  {loc}: {count:,}")
    
    print("\n【时区分布 (Top 5)】")
    timezone_dist = users['timezone'].value_counts().head(5)
    for tz, count in timezone_dist.items():
        print(f"  UTC{int(tz):+d}分钟：{count:,}")
    
    print("\n【出生年份分布】")
    # 转换为数值型，忽略非数值
    birthyear_numeric = pd.to_numeric(users['birthyear'], errors='coerce')
    print(f"  最早：{birthyear_numeric.min()}")
    print(f"  最晚：{birthyear_numeric.max()}")
    print(f"  中位数：{birthyear_numeric.median()}")


def analyze_events(data):
    """分析事件数据"""
    print("\n" + "=" * 60)
    print("事件数据分析")
    print("=" * 60)
    
    events = data['events']
    
    print("\n【基本统计】")
    print(f"总事件数：{len(events):,}")
    
    print("\n【国家分布 (Top 10)】")
    country_dist = events['country'].value_counts().head(10)
    for country, count in country_dist.items():
        print(f"  {country}: {count:,} ({count/len(events)*100:.2f}%)")
    
    print("\n【词干特征】")
    count_cols = [col for col in events.columns if col.startswith('count_')]
    print(f"  词干计数列数：{len(count_cols)}")
    print(f"  平均非零词干数：{(events[count_cols] != 0).sum(axis=1).mean():.2f}")


def analyze_friends(data):
    """分析好友关系"""
    print("\n" + "=" * 60)
    print("好友关系分析")
    print("=" * 60)
    
    friends = data['friends']
    
    print(f"\n总用户 - 好友记录：{len(friends):,}")
    
    # 计算好友数量
    friends['friend_count'] = friends['friends'].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
    
    print(f"\n好友数量统计:")
    print(f"  平均值：{friends['friend_count'].mean():.2f}")
    print(f"  中位数：{friends['friend_count'].median():.2f}")
    print(f"  最大值：{friends['friend_count'].max()}")
    print(f"  无好友用户：{(friends['friend_count'] == 0).sum():,}")


def analyze_attendees(data):
    """分析事件参与者"""
    print("\n" + "=" * 60)
    print("事件参与者分析")
    print("=" * 60)
    
    attendees = data['attendees']
    
    # 计算参与人数
    attendees['yes_count'] = attendees['yes'].apply(lambda x: len(str(x).split()) if pd.notna(x) and x != '' else 0)
    attendees['maybe_count'] = attendees['maybe'].apply(lambda x: len(str(x).split()) if pd.notna(x) and x != '' else 0)
    attendees['no_count'] = attendees['no'].apply(lambda x: len(str(x).split()) if pd.notna(x) and x != '' else 0)
    
    print(f"\n总事件数：{len(attendees):,}")
    print(f"\n参与情况统计:")
    print(f"  平均确定参加人数：{attendees['yes_count'].mean():.2f}")
    print(f"  平均可能参加人数：{attendees['maybe_count'].mean():.2f}")
    print(f"  平均不参加人数：{attendees['no_count'].mean():.2f}")


def generate_visualizations(data):
    """生成可视化图表"""
    print("\n" + "=" * 60)
    print("生成可视化图表...")
    print("=" * 60)
    
    # 创建输出目录
    output_dir = Path("./eda_output")
    output_dir.mkdir(exist_ok=True)
    
    # 图 1: 训练集标签分布
    plt.figure(figsize=(10, 6))
    train = data['train']
    labels = ['感兴趣', '不感兴趣', '无操作']
    sizes = [
        (train['interested'] == 1).sum(),
        (train['not_interested'] == 1).sum(),
        len(train) - (train['interested'] == 1).sum() - (train['not_interested'] == 1).sum()
    ]
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c', '#95a5a6'])
    plt.title('训练集标签分布')
    plt.savefig(output_dir / '01_label_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [OK] 保存：01_label_distribution.png")
    
    # 图 2: 用户性别分布
    plt.figure(figsize=(8, 6))
    users = data['users']
    users['gender'].value_counts().plot(kind='bar', color=['#3498db', '#e91e63', '#95a5a6'])
    plt.title('用户性别分布')
    plt.xlabel('性别')
    plt.ylabel('用户数')
    plt.xticks(rotation=0)
    plt.savefig(output_dir / '02_gender_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [OK] 保存：02_gender_distribution.png")
    
    # 图 3: 事件国家分布 (Top 10)
    plt.figure(figsize=(12, 6))
    events = data['events']
    top_countries = events['country'].value_counts().head(10)
    plt.bar(range(len(top_countries)), top_countries.values, color='#9b59b6')
    plt.xticks(range(len(top_countries)), top_countries.index, rotation=45)
    plt.title('事件国家分布 (Top 10)')
    plt.xlabel('国家')
    plt.ylabel('事件数')
    plt.savefig(output_dir / '03_country_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [OK] 保存：03_country_distribution.png")
    
    # 图 4: 好友数量分布
    plt.figure(figsize=(10, 6))
    friends = data['friends']
    friends['friend_count'] = friends['friends'].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
    plt.hist(friends['friend_count'], bins=50, color='#1abc9c', edgecolor='black')
    plt.title('用户好友数量分布')
    plt.xlabel('好友数量')
    plt.ylabel('用户数')
    plt.xlim(0, 200)
    plt.savefig(output_dir / '04_friend_count_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [OK] 保存：04_friend_count_distribution.png")
    
    print(f"\n所有图表已保存到：{output_dir.absolute()}")


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("Event Recommendation Challenge - 数据探索分析")
    print("=" * 60)
    
    # 加载数据
    data = load_data()
    
    # 分析各数据集
    analyze_train(data)
    analyze_users(data)
    analyze_events(data)
    analyze_friends(data)
    analyze_attendees(data)
    
    # 生成可视化
    generate_visualizations(data)
    
    print("\n" + "=" * 60)
    print("数据探索完成!")
    print("=" * 60)
    
    # 返回数据供进一步分析
    return data


if __name__ == "__main__":
    data = main()
