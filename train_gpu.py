import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import json
import pandas as pd

from data_loader import load_movielens, build_adj_matrix, create_data_loaders
from lightgcn import LightGCN
from evaluate import evaluate_model, get_user_item_interactions


def train_epoch(model, train_loader, optimizer, device):
    """Train for one epoch with GPU optimization"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch in tqdm(train_loader, desc="Training"):
        users, pos_items, neg_items = batch
        # Move to GPU
        users = users.to(device, non_blocking=True)
        pos_items = pos_items.to(device, non_blocking=True)
        neg_items = neg_items.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        loss = model.compute_loss(users, pos_items, neg_items)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def validate_model(model, val_loader, device):
    """Validate model with GPU"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            users, pos_items, neg_items = batch
            # Move to GPU
            users = users.to(device, non_blocking=True)
            pos_items = pos_items.to(device, non_blocking=True)
            neg_items = neg_items.to(device, non_blocking=True)
            
            loss = model.compute_loss(users, pos_items, neg_items)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def plot_training_curves(train_losses, val_losses, save_path="training_curves_gpu.png"):
    """Plot training and validation loss curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss (GPU)')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    # Set device with GPU optimization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        # Set memory fraction to avoid OOM
        torch.cuda.set_per_process_memory_fraction(0.8)
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        # Force CUDA device selection
        torch.cuda.set_device(0)
        print(f"CUDA device set to: {torch.cuda.current_device()}")
    else:
        print("CUDA not available, using CPU")
        print("If you have a GPU, make sure:")
        print("1. NVIDIA drivers are installed")
        print("2. PyTorch with CUDA support is installed")
        print("3. GPU is not being used by other processes")
    
    # Hyperparameters optimized for GPU
    embed_dim = 64
    n_layers = 3
    dropout = 0.1
    lr = 0.001
    batch_size = 2048  # Increased for GPU
    num_epochs = 100
    num_negatives = 1
    early_stopping_patience = 10
    
    # Load data
    print("Loading data...")
    pairs, num_users, num_items = load_movielens()
    print(f"Number of users: {num_users}")
    print(f"Number of items: {num_items}")
    print(f"Number of interactions: {len(pairs)}")
    
    # Build adjacency matrix
    print("Building adjacency matrix...")
    norm_adj = build_adj_matrix(pairs, num_users, num_items)
    
    # Create data loaders with GPU optimization
    print("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(
        pairs, num_users, num_items, batch_size=batch_size, num_negatives=num_negatives
    )
    
    # Initialize model
    print("Initializing model...")
    model = LightGCN(
        num_users=num_users,
        num_items=num_items,
        norm_adj=norm_adj,
        embed_dim=embed_dim,
        n_layers=n_layers,
        dropout=dropout
    )
    
    # Move model to GPU
    model = model.to(device)
    print(f"Model moved to {device}")
    
    # Verify model is on correct device
    print(f"Model device check:")
    print(f"  User embedding device: {model.embedding_user.weight.device}")
    print(f"  Item embedding device: {model.embedding_item.weight.device}")
    print(f"  Adjacency matrix device: {model.adj.device}")
    
    # Test forward pass to ensure everything works
    with torch.no_grad():
        test_user_emb, test_item_emb = model.forward()
        print(f"  Forward pass test - User emb device: {test_user_emb.device}")
        print(f"  Forward pass test - Item emb device: {test_item_emb.device}")
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # Learning rate scheduler for better convergence
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training loop
    print("Starting training...")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate_model(model, val_loader, device)
        val_losses.append(val_loss)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), "lightgcn_movielens_gpu.pt")
            print("Model saved!")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses)
    
    # Load best model for evaluation
    model.load_state_dict(torch.load("lightgcn_movielens_gpu.pt"))
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_df = pd.read_csv("movielens_test.csv")
    user_items, item_users = get_user_item_interactions(pairs)
    
    results = evaluate_model(model, test_df, user_items, K_list=[5, 10, 20])
    
    print("\n=== Final Test Results (GPU) ===")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
    
    # Save results
    with open("evaluation_results_gpu.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print("\nTraining completed!")
    print("Files saved:")
    print("- lightgcn_movielens_gpu.pt (model weights)")
    print("- training_curves_gpu.png (training curves)")
    print("- evaluation_results_gpu.json (test results)")


if __name__ == "__main__":
    main() 