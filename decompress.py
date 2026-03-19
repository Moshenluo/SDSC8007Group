# -*- coding: utf-8 -*-
import gzip
import shutil
from pathlib import Path

DATA_DIR = Path(r"C:\Users\Administrator\.openclaw\workspace\event_recommendation\data")

files_to_decompress = [
    "user_friends.csv.gz",
    "events.csv.gz",
    "event_attendees.csv.gz"
]

print("开始解压 .gz 文件...")

for filename in files_to_decompress:
    gz_path = DATA_DIR / filename
    output_path = DATA_DIR / filename.replace(".gz", "")
    
    if gz_path.exists():
        print(f"解压：{filename}")
        with gzip.open(gz_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f"  -> {output_path.name}")
    else:
        print(f"跳过（文件不存在）: {filename}")

print("\n解压完成!")
