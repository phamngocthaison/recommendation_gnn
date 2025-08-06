import torch
from data_loader import load_movielens, build_adj_matrix, get_user_item_interactions
from lightgcn import LightGCN
import pandas as pd
import numpy as np
from collections import defaultdict
import json


def evaluate_model(model, test_df, user_items, K_list=[5, 10, 20]):
    """Evaluate model with multiple metrics"""
    model.eval()
    user_emb, item_emb = model.get_embeddings()
    
    # Convert to numpy for faster computation
    user_emb = user_emb.detach().cpu().numpy()
    item_emb = item_emb.detach().cpu().numpy()
    
    metrics = {}
    for k in K_list:
        metrics[f'recall@{k}'] = []
        metrics[f'ndcg@{k}'] = []
        metrics[f'precision@{k}'] = []
    
    for idx, row in test_df.iterrows():
        user = int(row['user_id'])
        true_item = int(row['item_id'])
        
        # Get user's training items to exclude from recommendations
        train_items = user_items.get(user, set())
        
        # Calculate scores for all items
        scores = np.dot(user_emb[user], item_emb.T)
        
        # Set scores of training items to -inf to exclude them
        for item in train_items:
            scores[item] = -np.inf
        
        # Get top-K recommendations
        for K in K_list:
            top_k_indices = np.argsort(scores)[-K:][::-1]
            
            # Calculate metrics
            if true_item in top_k_indices:
                rank = np.where(top_k_indices == true_item)[0][0]
                
                # Recall@K
                metrics[f'recall@{K}'].append(1.0)
                
                # NDCG@K
                dcg = 1.0 / np.log2(rank + 2)
                idcg = 1.0 / np.log2(1 + 1)  # ideal DCG for rank 0
                metrics[f'ndcg@{K}'].append(dcg / idcg)
                
                # Precision@K
                metrics[f'precision@{K}'].append(1.0 / K)
            else:
                metrics[f'recall@{K}'].append(0.0)
                metrics[f'ndcg@{K}'].append(0.0)
                metrics[f'precision@{K}'].append(0.0)
    
    # Calculate average metrics
    results = {}
    for metric_name, values in metrics.items():
        results[metric_name] = np.mean(values)
    
    return results


def get_recommendations(model, user_id, user_items, K=10):
    """Get top-K recommendations for a specific user"""
    model.eval()
    user_emb, item_emb = model.get_embeddings()
    
    # Calculate scores
    scores = torch.matmul(user_emb[user_id], item_emb.T)
    
    # Exclude items user has already interacted with
    train_items = user_items.get(user_id, set())
    for item in train_items:
        scores[item] = -float('inf')
    
    # Get top-K items
    top_k_scores, top_k_indices = torch.topk(scores, K)
    
    return top_k_indices.cpu().numpy(), top_k_scores.cpu().numpy()


def main():
    # Load data
    print("Loading data...")
    pairs, num_users, num_items = load_movielens()
    user_items, item_users = get_user_item_interactions(pairs)
    
    # Build adjacency matrix
    print("Building adjacency matrix...")
    norm_adj = build_adj_matrix(pairs, num_users, num_items)
    
    # Load model
    print("Loading model...")
    model = LightGCN(num_users, num_items, norm_adj, embed_dim=64, n_layers=3)
    
    try:
        model.load_state_dict(torch.load("lightgcn_movielens.pt"))
        print("Model loaded successfully!")
    except FileNotFoundError:
        print("Model file not found. Please train the model first.")
        return
    
    # Load test data
    test_df = pd.read_csv("movielens_test.csv")
    
    # Evaluate model
    print("Evaluating model...")
    results = evaluate_model(model, test_df, user_items, K_list=[5, 10, 20])
    
    # Print results
    print("\n=== Evaluation Results ===")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
    
    # Example recommendations for a few users
    print("\n=== Example Recommendations ===")
    sample_users = test_df['user_id'].unique()[:3]
    
    for user in sample_users:
        recommendations, scores = get_recommendations(model, user, user_items, K=5)
        print(f"User {user} recommendations: {recommendations.tolist()}")
        print(f"Scores: {scores.tolist()}")
        print()


if __name__ == "__main__":
    main()