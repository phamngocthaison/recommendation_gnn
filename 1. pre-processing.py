import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict
import json

# Load raw data
df = pd.read_csv('ml-1m/ratings.dat', sep='::', engine='python',
                 names=['user_id', 'item_id', 'rating', 'timestamp'])

# Chuyển về implicit feedback: chỉ giữ rating >= 4
df = df[df['rating'] >= 4].copy()

# Gán lại user_id và item_id thành chỉ số liên tục
user2id = {u: idx for idx, u in enumerate(df['user_id'].unique())}
item2id = {i: idx for idx, i in enumerate(df['item_id'].unique())}

df['user_id'] = df['user_id'].map(user2id)
df['item_id'] = df['item_id'].map(item2id)

num_users = len(user2id)
num_items = len(item2id)

print(f"Số user: {num_users}, Số item: {num_items}, Số tương tác: {len(df)}")

# Sắp xếp theo thời gian để chọn tương tác cuối cùng cho test
df = df.sort_values(by=['user_id', 'timestamp'])

# Chia train/test: giữ lại 1 tương tác cuối cùng của mỗi user cho test
train_rows = []
test_rows = []

for u in df['user_id'].unique():
    user_data = df[df['user_id'] == u]
    if len(user_data) < 2:
        continue
    test_rows.append(user_data.iloc[-1])
    train_rows.extend(user_data.iloc[:-1].to_dict('records'))

train_df = pd.DataFrame(train_rows)
test_df = pd.DataFrame(test_rows)

print(f"Train: {len(train_df)}, Test: {len(test_df)}")

train_df.to_csv("movielens_train.csv", index=False)
test_df.to_csv("movielens_test.csv", index=False)
user2id = {int(u): int(idx) for idx, u in enumerate(df['user_id'].unique())}
item2id = {int(i): int(idx) for idx, i in enumerate(df['item_id'].unique())}
# Nếu cần lưu map ID gốc:
with open('user2id.json', 'w') as f:
    json.dump(user2id, f)
with open('item2id.json', 'w') as f:
    json.dump(item2id, f)