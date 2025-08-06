import torch
import numpy as np
from data_loader import load_movielens, build_adj_matrix, create_data_loaders
from lightgcn import LightGCN
import torch.optim as optim

def test_fixes():
    """Test if the fixes work correctly"""
    print("=== Testing Fixes ===")
    
    # Test 1: Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Test 2: Load data without divide by zero warning
    print("\nLoading data...")
    try:
        pairs, num_users, num_items = load_movielens()
        print(f"✓ Data loaded successfully: {num_users} users, {num_items} items, {len(pairs)} interactions")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return
    
    # Test 3: Build adjacency matrix without warning
    print("\nBuilding adjacency matrix...")
    try:
        norm_adj = build_adj_matrix(pairs, num_users, num_items)
        print(f"✓ Adjacency matrix built successfully: {norm_adj.shape}")
    except Exception as e:
        print(f"✗ Error building adjacency matrix: {e}")
        return
    
    # Test 4: Create data loaders
    print("\nCreating data loaders...")
    try:
        train_loader, val_loader = create_data_loaders(pairs, num_users, num_items, batch_size=1024)
        print(f"✓ Data loaders created: {len(train_loader)} train batches, {len(val_loader)} val batches")
    except Exception as e:
        print(f"✗ Error creating data loaders: {e}")
        return
    
    # Test 5: Create model
    print("\nCreating model...")
    try:
        model = LightGCN(
            num_users=num_users,
            num_items=num_items,
            norm_adj=norm_adj,
            embed_dim=64,
            n_layers=3,
            dropout=0.1
        )
        model = model.to(device)
        print(f"✓ Model created and moved to {device}")
    except Exception as e:
        print(f"✗ Error creating model: {e}")
        return
    
    # Test 6: Test optimizer and scheduler
    print("\nTesting optimizer and scheduler...")
    try:
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        print("✓ Optimizer and scheduler created successfully")
    except Exception as e:
        print(f"✗ Error creating optimizer/scheduler: {e}")
        return
    
    # Test 7: Test forward pass
    print("\nTesting forward pass...")
    try:
        with torch.no_grad():
            user_emb, item_emb = model.forward()
            print(f"✓ Forward pass successful: user_emb {user_emb.shape}, item_emb {item_emb.shape}")
    except Exception as e:
        print(f"✗ Error in forward pass: {e}")
        return
    
    # Test 8: Test one batch
    print("\nTesting one training batch...")
    try:
        model.train()
        batch = next(iter(train_loader))
        users, pos_items, neg_items = batch
        users = users.to(device)
        pos_items = pos_items.to(device)
        neg_items = neg_items.to(device)
        
        optimizer.zero_grad()
        loss = model.compute_loss(users, pos_items, neg_items)
        loss.backward()
        optimizer.step()
        
        print(f"✓ Training batch successful: loss = {loss.item():.4f}")
    except Exception as e:
        print(f"✗ Error in training batch: {e}")
        return
    
    print("\n=== All tests passed! ===")
    print("You can now run train_gpu.py successfully.")

if __name__ == "__main__":
    test_fixes() 