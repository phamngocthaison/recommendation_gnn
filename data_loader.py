import pandas as pd
import torch
import scipy.sparse as sp
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import defaultdict
import random


class BPRDataset(Dataset):
    def __init__(self, interactions, num_users, num_items, num_negatives=1):
        self.interactions = interactions
        self.user_item_set = set((u, i) for u, i in interactions)
        self.num_users = num_users
        self.num_items = num_items
        self.num_negatives = num_negatives
        
        # Create user-item dictionary for faster negative sampling
        self.user_items = defaultdict(set)
        for u, i in interactions:
            self.user_items[u].add(i)

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        user, pos_item = self.interactions[idx]
        
        # Sample negative items
        neg_items = []
        for _ in range(self.num_negatives):
            while True:
                neg_item = random.randint(0, self.num_items - 1)
                if neg_item not in self.user_items[user]:
                    neg_items.append(neg_item)
                    break
        
        return user, pos_item, neg_items[0] if len(neg_items) == 1 else neg_items


def load_movielens(path="movielens_train.csv"):
    """Load MovieLens dataset"""
    df = pd.read_csv(path)
    user_item_pairs = list(zip(df['user_id'], df['item_id']))
    num_users = df['user_id'].max() + 1
    num_items = df['item_id'].max() + 1
    return user_item_pairs, num_users, num_items


def build_adj_matrix(user_item_pairs, num_users, num_items):
    """Build normalized adjacency matrix for LightGCN"""
    # Build user-item interaction matrix
    rows = []
    cols = []
    for u, i in user_item_pairs:
        rows.append(u)
        cols.append(i + num_users)  # item index offset
        rows.append(i + num_users)
        cols.append(u)

    data = [1.0] * len(rows)
    mat = sp.coo_matrix((data, (rows, cols)), shape=(num_users + num_items, num_users + num_items))

    # Normalize adjacency matrix
    deg_sum = np.array(mat.sum(1)).flatten()
    # Handle zero degree nodes to avoid divide by zero
    deg_sum[deg_sum == 0] = 1.0
    deg = sp.diags(np.power(deg_sum, -0.5))
    norm_adj = deg @ mat @ deg
    return norm_adj.tocoo()


def create_data_loaders(user_item_pairs, num_users, num_items, batch_size=1024, num_negatives=1):
    """Create train and validation data loaders"""
    # Split data into train and validation
    random.shuffle(user_item_pairs)
    split_idx = int(0.9 * len(user_item_pairs))
    train_pairs = user_item_pairs[:split_idx]
    val_pairs = user_item_pairs[split_idx:]
    
    # Create datasets
    train_dataset = BPRDataset(train_pairs, num_users, num_items, num_negatives)
    val_dataset = BPRDataset(val_pairs, num_users, num_items, num_negatives)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader


def get_user_item_interactions(user_item_pairs):
    """Get user-item interaction dictionary"""
    user_items = defaultdict(set)
    item_users = defaultdict(set)
    
    for u, i in user_item_pairs:
        user_items[u].add(i)
        item_users[i].add(u)
    
    return user_items, item_users