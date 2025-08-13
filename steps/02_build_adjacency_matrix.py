#!/usr/bin/env python3
"""
BƯỚC 2: XÂY DỰNG ADJACENCY MATRIX
====================================

Mục đích:
- Load dữ liệu đã xử lý từ Bước 1
- Xây dựng adjacency matrix cho user-item graph
- Chuẩn hóa adjacency matrix theo LightGCN paper
- Tạo data loaders cho training

Output:
- Adjacency matrix đã chuẩn hóa
- Data loaders cho training và validation

"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict
import json
import os
import sys
import scipy.sparse as sp

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import utility functions
from utils import get_input_from_previous_step, get_step_output_path

class BPRDataset(Dataset):
    """Dataset cho Bayesian Personalized Ranking (BPR) loss"""
    
    def __init__(self, user_item_pairs, num_users, num_items, num_negatives=1):
        self.user_item_pairs = user_item_pairs
        self.num_users = num_users
        self.num_items = num_items
        self.num_negatives = num_negatives
        
        # Tạo user_items dictionary để negative sampling hiệu quả
        self.user_items = defaultdict(set)
        for user, item in user_item_pairs:
            self.user_items[user].add(item)
    
    def __len__(self):
        return len(self.user_item_pairs)
    
    def __getitem__(self, idx):
        user, pos_item = self.user_item_pairs[idx]
        
        # Negative sampling
        neg_items = []
        for _ in range(self.num_negatives):
            while True:
                neg_item = np.random.randint(0, self.num_items)
                if neg_item not in self.user_items[user]:
                    neg_items.append(neg_item)
                    break
        
        return torch.LongTensor([user]), torch.LongTensor([pos_item]), torch.LongTensor(neg_items)

def load_processed_data():
    """Load dữ liệu đã xử lý từ Bước 1"""
    print("📁 Loading processed data from Step 1...")
    
    # Load data từ thư mục output của step 01
    train_df = pd.read_csv(get_input_from_previous_step("01_preprocessing", 'movielens_train.csv'))
    test_df = pd.read_csv(get_input_from_previous_step("01_preprocessing", 'movielens_test.csv'))
    
    with open(get_input_from_previous_step("01_preprocessing", 'user2id.json'), 'r') as f:
        user2id = json.load(f)
    with open(get_input_from_previous_step("01_preprocessing", 'item2id.json'), 'r') as f:
        item2id = json.load(f)
    
    num_users = len(user2id)
    num_items = len(item2id)
    
    print(f"✅ Loaded {len(train_df):,} training interactions")
    print(f"✅ Loaded {len(test_df):,} testing interactions")
    print(f"✅ Loaded {num_users:,} users and {num_items:,} movies")
    
    return train_df, test_df, user2id, item2id, num_users, num_items

def build_adjacency_matrix(train_df, num_users, num_items):
    """Xây dựng adjacency matrix cho user-item bipartite graph"""
    print("\n🔗 Building adjacency matrix...")
    
    # Lấy danh sách users và items thực sự có trong training data
    actual_users = sorted(train_df['user_id'].unique())
    actual_items = sorted(train_df['item_id'].unique())
    
    print(f"📊 Actual users in training data: {len(actual_users)}")
    print(f"📊 Actual items in training data: {len(actual_items)}")
    
    # Tạo user-item interaction matrix chỉ cho users và items thực sự có tương tác
    user_item_matrix = np.zeros((len(actual_users), len(actual_items)))
    
    # Tạo mapping từ global ID sang local ID
    user_local_id = {u: idx for idx, u in enumerate(actual_users)}
    item_local_id = {i: idx for idx, i in enumerate(actual_items)}
    
    for _, row in train_df.iterrows():
        user_id = row['user_id']
        item_id = row['item_id']
        local_user_id = user_local_id[user_id]
        local_item_id = item_local_id[item_id]
        user_item_matrix[local_user_id, local_item_id] = 1
    
    print(f"✅ Created interaction matrix: {user_item_matrix.shape}")
    print(f"📊 Matrix density: {np.sum(user_item_matrix) / (user_item_matrix.shape[0] * user_item_matrix.shape[1]):.6f}")
    
    # Xây dựng adjacency matrix theo LightGCN paper
    # A = [0, R; R^T, 0] với R là user-item matrix
    total_nodes = len(actual_users) + len(actual_items)
    adj_matrix = np.zeros((total_nodes, total_nodes))
    
    # R (user-item interactions)
    adj_matrix[:len(actual_users), len(actual_users):] = user_item_matrix
    # R^T (item-user interactions)
    adj_matrix[len(actual_users):, :len(actual_users)] = user_item_matrix.T
    
    print(f"✅ Built adjacency matrix: {adj_matrix.shape}")
    print(f"📊 Adjacency matrix density: {np.sum(adj_matrix) / (adj_matrix.shape[0] * adj_matrix.shape[1]):.6f}")
    
    # Lưu mapping để sử dụng sau này
    mapping_info = {
        'user_local_id': user_local_id,
        'item_local_id': item_local_id,
        'actual_users': actual_users,
        'actual_items': actual_items,
        'total_nodes': total_nodes
    }
    
    return adj_matrix, user_item_matrix, mapping_info

def normalize_adjacency_matrix(adj_matrix):
    """Chuẩn hóa adjacency matrix theo LightGCN paper"""
    print("\n⚖️ Normalizing adjacency matrix...")
    
    # Tính degree matrix
    degree_matrix = np.sum(adj_matrix, axis=1)
    
    # Kiểm tra isolated nodes (degree = 0)
    isolated_nodes = np.sum(degree_matrix == 0)
    if isolated_nodes > 0:
        print(f"⚠️ Warning: Found {isolated_nodes} isolated nodes (degree = 0)")
    
    # Chuẩn hóa: D^(-1/2) * A * D^(-1/2)
    degree_matrix_safe = degree_matrix.copy()
    degree_matrix_safe[degree_matrix_safe == 0] = 1  # Thay thế 0 bằng 1 để tránh division by zero
    
    degree_matrix_inv_sqrt = np.power(degree_matrix_safe, -0.5)
    
    # Tạo diagonal matrix
    degree_matrix_inv_sqrt = np.diag(degree_matrix_inv_sqrt)
    
    # Chuẩn hóa
    norm_adj = degree_matrix_inv_sqrt @ adj_matrix @ degree_matrix_inv_sqrt
    
    print(f"✅ Normalized adjacency matrix")
    print(f"📊 Normalized matrix stats:")
    print(f"   Min value: {np.min(norm_adj):.6f}")
    print(f"   Max value: {np.max(norm_adj):.6f}")
    print(f"   Mean value: {np.mean(norm_adj):.6f}")
    print(f"   Std value: {np.std(norm_adj):.6f}")
    
    return norm_adj

def create_data_loaders(train_df, num_users, num_items, batch_size=1024, num_negatives=1):
    """Tạo data loaders cho training"""
    print("\n📦 Creating data loaders...")
    
    # Chuyển đổi DataFrame thành list of tuples
    user_item_pairs = list(zip(train_df['user_id'], train_df['item_id']))
    
    # Chia train/validation (80/20)
    np.random.shuffle(user_item_pairs)
    split_idx = int(len(user_item_pairs) * 0.8)
    
    train_pairs = user_item_pairs[:split_idx]
    val_pairs = user_item_pairs[split_idx:]
    
    print(f"✅ Train pairs: {len(train_pairs):,}")
    print(f"✅ Validation pairs: {len(val_pairs):,}")
    
    # Tạo datasets
    train_dataset = BPRDataset(train_pairs, num_users, num_items, num_negatives)
    val_dataset = BPRDataset(val_pairs, num_users, num_items, num_negatives)
    
    # Tạo data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"✅ Created data loaders with batch_size={batch_size}")
    print(f"✅ Train batches: {len(train_loader)}")
    print(f"✅ Validation batches: {len(val_loader)}")
    
    return train_loader, val_loader

def get_user_item_interactions(train_df):
    """Tạo dictionaries cho user-item và item-user interactions"""
    print("\n🔍 Creating interaction dictionaries...")
    
    user_items = defaultdict(set)
    item_users = defaultdict(set)
    
    for _, row in train_df.iterrows():
        user_id = row['user_id']
        item_id = row['item_id']
        user_items[user_id].add(item_id)
        item_users[item_id].add(user_id)
    
    print(f"✅ Created user_items dict: {len(user_items)} users")
    print(f"✅ Created item_users dict: {len(item_users)} items")
    
    # Thống kê interactions
    user_interaction_counts = [len(items) for items in user_items.values()]
    item_interaction_counts = [len(users) for users in item_users.values()]
    
    print(f"📊 User interaction stats:")
    print(f"   Min: {min(user_interaction_counts)}")
    print(f"   Max: {max(user_interaction_counts)}")
    print(f"   Mean: {np.mean(user_interaction_counts):.1f}")
    print(f"   Median: {np.median(user_interaction_counts):.1f}")
    
    print(f"📊 Item interaction stats:")
    print(f"   Min: {min(item_interaction_counts)}")
    print(f"   Max: {max(item_interaction_counts)}")
    print(f"   Mean: {np.mean(item_interaction_counts):.1f}")
    print(f"   Median: {np.median(item_interaction_counts):.1f}")
    
    return user_items, item_users

def save_adjacency_matrix(norm_adj, save_path="norm_adj_matrix.npz"):
    """Lưu adjacency matrix đã chuẩn hóa"""
    print(f"\n💾 Saving normalized adjacency matrix...")
    
    # Lưu vào thư mục output của step 02
    save_path = get_step_output_path("02_adjacency", "norm_adj_matrix.npz")
    
    # Lưu dưới dạng sparse matrix để tiết kiệm bộ nhớ
    sparse_adj = sp.csr_matrix(norm_adj)
    sp.save_npz(save_path, sparse_adj)
    
    file_size = os.path.getsize(save_path) / (1024 * 1024)  # MB
    print(f"✅ Saved sparse adjacency matrix to {save_path}: {file_size:.2f} MB")
    
    return save_path

def save_mapping_info(mapping_info, save_path="mapping_info.json"):
    """Lưu thông tin mapping để sử dụng trong các bước tiếp theo"""
    print(f"\n💾 Saving mapping information...")
    
    # Lưu vào thư mục output của step 02
    save_path = get_step_output_path("02_adjacency", "mapping_info.json")
    
    # Chuyển đổi numpy arrays thành lists để có thể serialize
    serializable_mapping = {
        'user_local_id': {int(k): int(v) for k, v in mapping_info['user_local_id'].items()},
        'item_local_id': {int(k): int(v) for k, v in mapping_info['item_local_id'].items()},
        'actual_users': [int(u) for u in mapping_info['actual_users']],
        'actual_items': [int(i) for i in mapping_info['actual_items']],
        'total_nodes': int(mapping_info['total_nodes'])
    }
    
    with open(save_path, 'w') as f:
        json.dump(serializable_mapping, f, indent=2)
    
    file_size = os.path.getsize(save_path) / 1024  # KB
    print(f"✅ Saved mapping information to {save_path}: {file_size:.2f} KB")
    
    return save_path

def main():
    """Main function"""
    print("🚀 BƯỚC 2: XÂY DỰNG ADJACENCY MATRIX")
    print("=" * 60)
    
    try:
        # Bước 1: Load dữ liệu đã xử lý
        train_df, test_df, user2id, item2id, num_users, num_items = load_processed_data()
        
        # Bước 2: Xây dựng adjacency matrix
        adj_matrix, user_item_matrix, mapping_info = build_adjacency_matrix(train_df, num_users, num_items)
        
        # Bước 3: Chuẩn hóa adjacency matrix
        norm_adj = normalize_adjacency_matrix(adj_matrix)
        
        # Bước 4: Tạo data loaders
        train_loader, val_loader = create_data_loaders(train_df, num_users, num_items)
        
        # Bước 5: Tạo interaction dictionaries
        user_items, item_users = get_user_item_interactions(train_df)
        
        # Bước 6: Lưu adjacency matrix
        adj_file = save_adjacency_matrix(norm_adj)
        
        # Bước 7: Lưu mapping information
        mapping_file = save_mapping_info(mapping_info)
        
        print("\n🎉 XÂY DỰNG ADJACENCY MATRIX HOÀN THÀNH!")
        print("=" * 60)
        print("📁 Files created:")
        print(f"   - {adj_file} (normalized adjacency matrix)")
        print(f"   - {mapping_file} (mapping information)")
        print("\n📊 Graph Statistics:")
        print(f"   - Nodes: {mapping_info['total_nodes']:,} (users + movies)")
        print(f"   - Edges: {np.sum(adj_matrix):,}")
        print(f"   - Density: {np.sum(adj_matrix) / (mapping_info['total_nodes'] ** 2):.6f}")
        print("\n➡️ Next step: Chạy 03_train_lightgcn.py")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Hãy kiểm tra lại dữ liệu và thử lại.")

if __name__ == "__main__":
    main()
