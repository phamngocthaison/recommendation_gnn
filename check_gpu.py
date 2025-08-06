import torch
import numpy as np
from lightgcn import LightGCN
from scipy.sparse import csr_matrix

def check_gpu():
    """Check GPU availability and test model"""
    print("=== GPU Check ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
        print(f"Device memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("No CUDA device available, will use CPU")
    
    # Test model creation
    print("\n=== Model Test ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create a small test adjacency matrix
    num_users, num_items = 100, 200
    test_data = np.random.randint(0, 100, (50, 2))  # 50 random interactions
    test_matrix = csr_matrix((np.ones(50), (test_data[:, 0], test_data[:, 1])), 
                           shape=(num_users + num_items, num_users + num_items))
    
    # Create model
    model = LightGCN(
        num_users=num_users,
        num_items=num_items,
        norm_adj=test_matrix,
        embed_dim=64,
        n_layers=3,
        dropout=0.1
    )
    
    # Move to device
    model = model.to(device)
    print(f"Model moved to {device}")
    
    # Test forward pass
    with torch.no_grad():
        user_emb, item_emb = model.forward()
        print(f"User embeddings shape: {user_emb.shape}")
        print(f"Item embeddings shape: {item_emb.shape}")
        print(f"User embeddings device: {user_emb.device}")
        print(f"Item embeddings device: {item_emb.device}")
    
    print("\n=== Test completed successfully! ===")

if __name__ == "__main__":
    check_gpu() 