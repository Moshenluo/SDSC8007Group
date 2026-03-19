import pandas as pd
import numpy as np
from pathlib import Path

print("Start preprocessing...")

data_dir = Path(r"C:\Users\Administrator\.openclaw\workspace\event_recommendation\data")

# Load data
print("Loading train.csv...")
train = pd.read_csv(data_dir / "train.csv")
print(f"Train shape: {train.shape}")

print("Loading users.csv...")
users = pd.read_csv(data_dir / "users.csv")
print(f"Users shape: {users.shape}")

print("Loading events.csv...")
events = pd.read_csv(data_dir / "events.csv")
print(f"Events shape: {events.shape}")

print("\nDone!")
