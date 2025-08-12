#!/usr/bin/env python3
"""
BƯỚC 3: TRAINING LIGHTGCN MODEL
=================================

Mục đích:
- Load adjacency matrix và data loaders từ Bước 2
- Khởi tạo và training LightGCN model
- Sử dụng BPR loss và negative sampling
- Early stopping và model checkpointing
- Lưu model weights và training curves

Output:
- lightgcn_movielens.pt: Model weights
- training_curves.png: Training curves
- training_log.json: Training logs

Paper reference: LightGCN: Simplifying and Powering Graph Convolution Network 
for Recommendation (He et al., SIGIR 2020)
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import json
import time
from datetime import datetime
import scipy.sparse as sp
import sys

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import utility functions
from utils import get_input_from_previous_step, get_step_output_path

# Import từ các bước trước
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lightgcn import LightGCN

def load_adjacency_matrix(adj_path="norm_adj_matrix.npz"):
    """Load adjacency matrix đã chuẩn hóa từ Bước 2"""
    print("📁 Loading normalized adjacency matrix...")
    
    # Load từ thư mục output của step 02
    adj_path = get_input_from_previous_step("02_adjacency", "norm_adj_matrix.npz")
    
    if not os.path.exists(adj_path):
        raise FileNotFoundError(f"File {adj_path} không tồn tại. Hãy chạy 02_build_adjacency_matrix.py trước.")
    
    # Load sparse matrix
    sparse_adj = sp.load_npz(adj_path)
    print(f"✅ Loaded adjacency matrix: {sparse_adj.shape}")
    print(f"📊 Matrix density: {sparse_adj.nnz / (sparse_adj.shape[0] * sparse_adj.shape[1]):.6f}")
    
    return sparse_adj

def load_mapping_info():
    """Load mapping information từ Bước 2"""
    print("📁 Loading mapping information...")
    
    # Load từ thư mục output của step 02
    mapping_path = get_input_from_previous_step("02_adjacency", "mapping_info.json")
    
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"File {mapping_path} không tồn tại. Hãy chạy 02_build_adjacency_matrix.py trước.")
    
    with open(mapping_path, 'r') as f:
        mapping_info = json.load(f)
    
    print(f"✅ Loaded mapping information")
    print(f"📊 Actual users: {len(mapping_info['actual_users'])}")
    print(f"📊 Actual items: {len(mapping_info['actual_items'])}")
    
    return mapping_info

def convert_to_local_ids(pairs, mapping_info):
    """Chuyển đổi global IDs sang local IDs sử dụng mapping"""
    print("🔄 Converting global IDs to local IDs...")
    
    user_local_id = mapping_info['user_local_id']
    item_local_id = mapping_info['item_local_id']
    
    local_pairs = []
    for user_id, item_id in pairs:
        if user_id in user_local_id and item_id in item_local_id:
            local_user_id = user_local_id[user_id]
            local_item_id = item_local_id[item_id]
            local_pairs.append((local_user_id, local_item_id))
        else:
            # Skip pairs that don't exist in the filtered data
            continue
    
    print(f"✅ Converted {len(pairs)} pairs to {len(local_pairs)} local pairs")
    return local_pairs

def load_training_data():
    """Load training data và tạo data loaders"""
    print("\n📦 Loading training data...")
    
    # Load data từ thư mục output của step 01
    import pandas as pd
    train_df = pd.read_csv(get_input_from_previous_step("01_preprocessing", 'movielens_train.csv'))
    
    with open(get_input_from_previous_step("01_preprocessing", 'user2id.json'), 'r') as f:
        user2id = json.load(f)
    with open(get_input_from_previous_step("01_preprocessing", 'item2id.json'), 'r') as f:
        item2id = json.load(f)
    
    num_users = len(user2id)
    num_items = len(item2id)
    
    print(f"✅ Loaded {len(train_df):,} training interactions")
    print(f"✅ {num_users:,} users, {num_items:,} movies")
    
    # Tạo user-item pairs
    user_item_pairs = list(zip(train_df['user_id'], train_df['item_id']))
    
    # Chia train/validation (80/20)
    np.random.shuffle(user_item_pairs)
    split_idx = int(len(user_item_pairs) * 0.8)
    
    train_pairs = user_item_pairs[:split_idx]
    val_pairs = user_item_pairs[split_idx:]
    
    print(f"✅ Train pairs: {len(train_pairs):,}")
    print(f"✅ Validation pairs: {len(val_pairs):,}")
    
    return train_df, train_pairs, val_pairs, num_users, num_items

class BPRDataset(torch.utils.data.Dataset):
    """Dataset cho Bayesian Personalized Ranking (BPR) loss"""
    
    def __init__(self, user_item_pairs, num_users, num_items, num_negatives=1):
        self.user_item_pairs = user_item_pairs
        self.num_users = num_users
        self.num_items = num_items
        self.num_negatives = num_negatives
        
        # Tạo user_items dictionary để negative sampling hiệu quả
        self.user_items = {}
        for user, item in user_item_pairs:
            if user not in self.user_items:
                self.user_items[user] = set()
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

def create_data_loaders(train_pairs, val_pairs, num_users, num_items, batch_size=1024, num_negatives=1):
    """Tạo data loaders cho training và validation"""
    print(f"\n📦 Creating data loaders (batch_size={batch_size}, negatives={num_negatives})...")
    
    # Tạo datasets
    train_dataset = BPRDataset(train_pairs, num_users, num_items, num_negatives)
    val_dataset = BPRDataset(val_pairs, num_users, num_items, num_negatives)
    
    # Tạo data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"✅ Train batches: {len(train_loader)}")
    print(f"✅ Validation batches: {len(val_loader)}")
    
    return train_loader, val_loader

def initialize_model(num_users, num_items, norm_adj, hyperparams):
    """Khởi tạo LightGCN model"""
    print(f"\n🤖 Initializing LightGCN model...")
    print(f"   - Embedding dimension: {hyperparams['embed_dim']}")
    print(f"   - Number of layers: {hyperparams['n_layers']}")
    print(f"   - Dropout: {hyperparams['dropout']}")
    
    model = LightGCN(
        num_users=num_users,
        num_items=num_items,
        norm_adj=norm_adj,
        embed_dim=hyperparams['embed_dim'],
        n_layers=hyperparams['n_layers'],
        dropout=hyperparams['dropout']
    )
    
    # Đếm parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✅ Model initialized")
    print(f"   - Total parameters: {total_params:,}")
    print(f"   - Trainable parameters: {trainable_params:,}")
    
    return model

def train_epoch(model, train_loader, optimizer, device, epoch):
    """Train cho một epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
    
    for batch in progress_bar:
        users, pos_items, neg_items = batch
        users = users.squeeze().to(device)
        pos_items = pos_items.squeeze().to(device)
        neg_items = neg_items.squeeze().to(device)
        
        optimizer.zero_grad()
        loss = model.compute_loss(users, pos_items, neg_items)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches

def validate_model(model, val_loader, device, epoch):
    """Validate model"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]", leave=False)
    
    with torch.no_grad():
        for batch in progress_bar:
            users, pos_items, neg_items = batch
            users = users.squeeze().to(device)
            pos_items = pos_items.squeeze().to(device)
            neg_items = neg_items.squeeze().to(device)
            
            loss = model.compute_loss(users, pos_items, neg_items)
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches

def plot_training_curves(train_losses, val_losses, save_path="training_curves.png"):
    """Vẽ training curves"""
    print(f"\n📊 Plotting training curves...")
    
    # Lưu vào thư mục output của step 03
    save_path = get_step_output_path("03_training", "training_curves.png")
    
    plt.figure(figsize=(12, 8))
    
    # Training curves
    plt.subplot(2, 1, 1)
    plt.plot(train_losses, label='Training Loss', color='#2E86AB', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', color='#A23B72', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Loss difference
    plt.subplot(2, 1, 2)
    loss_diff = [t - v for t, v in zip(train_losses, val_losses)]
    plt.plot(loss_diff, label='Train - Val Loss', color='#F18F01', linewidth=2)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss Difference', fontsize=12)
    plt.title('Training vs Validation Loss Difference', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved training curves to {save_path}")

def save_training_log(train_losses, val_losses, hyperparams, training_time, save_path="training_log.json"):
    """Lưu training log"""
    print(f"\n💾 Saving training log...")
    
    # Lưu vào thư mục output của step 03
    save_path = get_step_output_path("03_training", "training_log.json")
    
    log_data = {
        "hyperparameters": hyperparams,
        "training_time_seconds": training_time,
        "total_epochs": len(train_losses),
        "final_train_loss": train_losses[-1] if train_losses else None,
        "final_val_loss": val_losses[-1] if val_losses else None,
        "best_val_loss": min(val_losses) if val_losses else None,
        "best_epoch": val_losses.index(min(val_losses)) + 1 if val_losses else None,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "timestamp": datetime.now().isoformat()
    }
    
    with open(save_path, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    print(f"✅ Saved training log to {save_path}")

def main():
    """Main function"""
    print("🚀 BƯỚC 3: TRAINING LIGHTGCN MODEL")
    print("=" * 60)
    
    # Hyperparameters
    hyperparams = {
        'embed_dim': 64,
        'n_layers': 3,
        'dropout': 0.1,
        'lr': 0.001,
        'batch_size': 1024,
        'num_epochs': 100,
        'num_negatives': 1,
        'early_stopping_patience': 10,
        'weight_decay': 1e-5
    }
    
    print("⚙️ Hyperparameters:")
    for key, value in hyperparams.items():
        print(f"   - {key}: {value}")
    
    try:
        # Bước 1: Load adjacency matrix
        norm_adj = load_adjacency_matrix()
        

        
        # Bước 3: Load training data
        train_df, train_pairs, val_pairs, num_users, num_items = load_training_data()
        
        # Bước 4: Sử dụng số nodes từ adjacency matrix
        total_nodes = norm_adj.shape[0]
        # Giả sử số users và items bằng nhau (hoặc gần bằng)
        adj_num_users = total_nodes // 2
        adj_num_items = total_nodes - adj_num_users
        
        print(f"📊 Adjacency matrix has {total_nodes} nodes")
        print(f"📊 Training data has {num_users} users and {num_items} items")
        print(f"📊 Using {adj_num_users} users and {adj_num_items} items for model")
        
        # Bước 5: Lọc training data để chỉ giữ pairs phù hợp với model dimensions
        filtered_train_pairs = [(u, i) for u, i in train_pairs if u < adj_num_users and i < adj_num_items]
        filtered_val_pairs = [(u, i) for u, i in val_pairs if u < adj_num_users and i < adj_num_items]
        
        print(f"✅ Filtered training pairs: {len(train_pairs)} -> {len(filtered_train_pairs)}")
        print(f"✅ Filtered validation pairs: {len(val_pairs)} -> {len(filtered_val_pairs)}")
        
        # Bước 6: Tạo data loaders với số nodes từ adjacency matrix
        train_loader, val_loader = create_data_loaders(
            filtered_train_pairs, filtered_val_pairs, adj_num_users, adj_num_items, 
            batch_size=hyperparams['batch_size'], 
            num_negatives=hyperparams['num_negatives']
        )
        
        # Bước 7: Khởi tạo model
        model = initialize_model(adj_num_users, adj_num_items, norm_adj, hyperparams)
        
        # Bước 8: Setup device và optimizer
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\n🖥️ Using device: {device}")
        
        model = model.to(device)
        optimizer = optim.Adam(
            model.parameters(), 
            lr=hyperparams['lr'], 
            weight_decay=hyperparams['weight_decay']
        )
        
        # Bước 9: Training loop
        print(f"\n🔥 Starting training for {hyperparams['num_epochs']} epochs...")
        print("=" * 60)
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        start_time = time.time()
        
        for epoch in range(hyperparams['num_epochs']):
            epoch_start = time.time()
            
            # Train
            train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
            train_losses.append(train_loss)
            
            # Validate
            val_loss = validate_model(model, val_loader, device, epoch)
            val_losses.append(val_loss)
            
            epoch_time = time.time() - epoch_start
            
            # Print progress
            print(f"Epoch {epoch+1:3d}/{hyperparams['num_epochs']} | "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                  f"Time: {epoch_time:.1f}s")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                model_save_path = get_step_output_path("03_training", "lightgcn_movielens.pt")
                torch.save(model.state_dict(), model_save_path)
                print(f"   🎯 New best model! Val Loss: {val_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= hyperparams['early_stopping_patience']:
                    print(f"\n⏹️ Early stopping after {epoch+1} epochs")
                    break
        
        total_training_time = time.time() - start_time
        
        # Bước 10: Lưu kết quả
        print(f"\n💾 Saving results...")
        
        # Plot training curves
        plot_training_curves(train_losses, val_losses)
        
        # Save training log
        save_training_log(train_losses, val_losses, hyperparams, total_training_time)
        
        print("\n🎉 TRAINING HOÀN THÀNH!")
        print("=" * 60)
        print("📁 Files created:")
        print("   - lightgcn_movielens.pt (model weights)")
        print("   - training_curves.png (training curves)")
        print("   - training_log.json (training logs)")
        print(f"\n⏱️ Total training time: {total_training_time/60:.1f} minutes")
        print(f"🏆 Best validation loss: {best_val_loss:.4f}")
        print(f"📊 Final train/val loss: {train_losses[-1]:.4f}/{val_losses[-1]:.4f}")
        print("\n➡️ Next step: Chạy 04_evaluate_model.py")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print("Hãy kiểm tra lại dữ liệu và thử lại.")

if __name__ == "__main__":
    main()
