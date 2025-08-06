import torch
import pandas as pd
import numpy as np
import json
from data_loader import load_movielens, build_adj_matrix, get_user_item_interactions
from lightgcn import LightGCN
from evaluate import get_recommendations


def load_movie_info():
    """Load movie information from MovieLens dataset"""
    try:
        movies_df = pd.read_csv('ml-1m/movies.dat', sep='::', engine='python',
                               names=['movie_id', 'title', 'genres'], encoding='latin-1')
        return movies_df
    except:
        print("Warning: Could not load movie information")
        return None


def get_movie_title(movie_id, movies_df, item2id):
    """Get movie title from movie ID"""
    if movies_df is None:
        return f"Movie_{movie_id}"
    
    # Convert internal ID back to original movie ID
    original_movie_id = None
    for orig_id, internal_id in item2id.items():
        if internal_id == movie_id:
            original_movie_id = orig_id
            break
    
    if original_movie_id is None:
        return f"Movie_{movie_id}"
    
    movie_info = movies_df[movies_df['movie_id'] == original_movie_id]
    if len(movie_info) > 0:
        return movie_info.iloc[0]['title']
    else:
        return f"Movie_{movie_id}"


def demo_recommendations():
    """Demo function to show recommendations for sample users"""
    print("=== LightGCN Movie Recommendation Demo ===\n")
    
    # Load data
    print("Loading data and model...")
    pairs, num_users, num_items = load_movielens()
    user_items, item_users = get_user_item_interactions(pairs)
    
    # Load ID mappings
    with open('user2id.json', 'r') as f:
        user2id = json.load(f)
    with open('item2id.json', 'r') as f:
        item2id = json.load(f)
    
    # Build adjacency matrix
    norm_adj = build_adj_matrix(pairs, num_users, num_items)
    
    # Load model
    model = LightGCN(num_users, num_items, norm_adj, embed_dim=64, n_layers=3)
    try:
        model.load_state_dict(torch.load("lightgcn_movielens.pt"))
        print("Model loaded successfully!")
    except FileNotFoundError:
        print("Error: Model file not found. Please run training first.")
        return
    
    # Load movie information
    movies_df = load_movie_info()
    
    # Load test data for sample users
    test_df = pd.read_csv("movielens_test.csv")
    sample_users = test_df['user_id'].unique()[:5]
    
    print(f"\nShowing recommendations for {len(sample_users)} sample users:")
    print("=" * 60)
    
    for i, user in enumerate(sample_users, 1):
        print(f"\nUser {i} (Internal ID: {user}):")
        
        # Get user's training items
        train_items = user_items.get(user, set())
        print(f"  Training items: {len(train_items)} movies")
        
        # Get recommendations
        recommendations, scores = get_recommendations(model, user, user_items, K=10)
        
        print("  Top 10 Recommendations:")
        for j, (item_id, score) in enumerate(zip(recommendations, scores), 1):
            movie_title = get_movie_title(item_id, movies_df, item2id)
            print(f"    {j:2d}. {movie_title} (Score: {score:.3f})")
        
        # Show test item
        test_item = test_df[test_df['user_id'] == user]['item_id'].iloc[0]
        test_movie_title = get_movie_title(test_item, movies_df, item2id)
        print(f"  Test item: {test_movie_title}")
        
        if i < len(sample_users):
            print("-" * 60)


def interactive_recommendations():
    """Interactive mode for user input"""
    print("\n=== Interactive Recommendation Mode ===")
    print("Enter a user ID to get recommendations (or 'quit' to exit)")
    
    # Load data
    pairs, num_users, num_items = load_movielens()
    user_items, item_users = get_user_item_interactions(pairs)
    
    # Load ID mappings
    with open('user2id.json', 'r') as f:
        user2id = json.load(f)
    with open('item2id.json', 'r') as f:
        item2id = json.load(f)
    
    # Build adjacency matrix
    norm_adj = build_adj_matrix(pairs, num_users, num_items)
    
    # Load model
    model = LightGCN(num_users, num_items, norm_adj, embed_dim=64, n_layers=3)
    try:
        model.load_state_dict(torch.load("lightgcn_movielens.pt"))
    except FileNotFoundError:
        print("Error: Model file not found. Please run training first.")
        return
    
    # Load movie information
    movies_df = load_movie_info()
    
    while True:
        user_input = input("\nEnter user ID (0-{}): ".format(num_users-1))
        
        if user_input.lower() == 'quit':
            break
        
        try:
            user_id = int(user_input)
            if user_id < 0 or user_id >= num_users:
                print(f"Invalid user ID. Please enter a number between 0 and {num_users-1}")
                continue
            
            # Get user's training items
            train_items = user_items.get(user_id, set())
            print(f"\nUser {user_id} has {len(train_items)} training interactions")
            
            # Get recommendations
            recommendations, scores = get_recommendations(model, user_id, user_items, K=10)
            
            print("\nTop 10 Recommendations:")
            for i, (item_id, score) in enumerate(zip(recommendations, scores), 1):
                movie_title = get_movie_title(item_id, movies_df, item2id)
                print(f"  {i:2d}. {movie_title} (Score: {score:.3f})")
                
        except ValueError:
            print("Please enter a valid number")


def main():
    """Main function"""
    print("LightGCN Movie Recommendation System")
    print("====================================")
    
    # Check if model exists
    try:
        torch.load("lightgcn_movielens.pt")
        model_exists = True
    except FileNotFoundError:
        model_exists = False
        print("Warning: No trained model found. Please run training first.")
        return
    
    # Demo mode
    demo_recommendations()
    
    # Interactive mode
    try:
        interactive_recommendations()
    except KeyboardInterrupt:
        print("\n\nExiting...")
    except EOFError:
        print("\n\nExiting...")


if __name__ == "__main__":
    main() 