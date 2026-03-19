import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pathlib import Path
import pickle
import json
import traceback

print("Start...")

try:
    data_dir = Path(r"C:\Users\Administrator\.openclaw\workspace\event_recommendation\data")
    save_dir = Path(r"C:\Users\Administrator\.openclaw\workspace\event_recommendation\processed")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    train = pd.read_csv(data_dir / "train.csv")
    users = pd.read_csv(data_dir / "users.csv")
    events = pd.read_csv(data_dir / "events.csv")
    print(f"  train: {train.shape}, users: {users.shape}, events: {events.shape}")
    
    # Process users
    print("Processing users...")
    users['birthyear_num'] = pd.to_numeric(users['birthyear'], errors='coerce')
    users['birthyear'] = users['birthyear_num'].fillna(users['birthyear_num'].median())
    users['timezone'] = pd.to_numeric(users['timezone'], errors='coerce').fillna(0)
    users['gender'] = users['gender'].fillna('unknown')
    
    gender_enc = LabelEncoder()
    users['gender_encoded'] = gender_enc.fit_transform(users['gender'])
    print(f"  Gender classes: {gender_enc.classes_}")
    
    # Process events (simplified)
    print("Processing events...")
    events['lat'] = events['lat'].fillna(0)
    events['lng'] = events['lng'].fillna(0)
    events['country'] = events['country'].fillna('Unknown')
    
    # Process interactions
    print("Processing interactions...")
    all_users = pd.concat([train['user'], pd.read_csv(data_dir / "test.csv")['user']]).unique()
    all_events = pd.concat([train['event'], pd.read_csv(data_dir / "test.csv")['event']]).unique()
    
    user_enc = LabelEncoder()
    event_enc = LabelEncoder()
    user_enc.fit(all_users)
    event_enc.fit(all_events)
    
    train['user_id'] = user_enc.transform(train['user'])
    train['event_id'] = event_enc.transform(train['event'])
    
    # Create labels
    train['label_interested'] = train['interested']
    train['label_not_interested'] = train['not_interested']
    
    print(f"  Users: {len(all_users):,}, Events: {len(all_events):,}")
    print(f"  Train samples: {len(train):,}")
    
    # Save processed data
    print("Saving processed data...")
    train[['user_id', 'event_id', 'label_interested', 'label_not_interested']].to_csv(
        save_dir / "train_processed.csv", index=False
    )
    
    # Save mappings
    with open(save_dir / "mappings.json", "w") as f:
        json.dump({
            "num_users": len(all_users),
            "num_events": len(all_events)
        }, f)
    
    print("Done!")
    
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()
