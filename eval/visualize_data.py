import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
import json
import random
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

# Set style
plt.style.use('default')
sns.set_palette("husl")

def load_data():
    """Load MovieLens data and movie information"""
    # Load processed data
    train_df = pd.read_csv('../movielens_train.csv')
    
    # Load movie information
    try:
        movies_df = pd.read_csv('../ml-1m/movies.dat', sep='::', engine='python',
                                names=['movie_id', 'title', 'genres'], encoding='latin-1')
    except:
        print("Warning: Could not load movie information")
        movies_df = None
    
    # Load ID mappings
    with open('../user2id.json', 'r') as f:
        user2id = json.load(f)
    with open('../item2id.json', 'r') as f:
        item2id = json.load(f)
    
    return train_df, movies_df, user2id, item2id

def get_movie_title(movie_id, movies_df, item2id):
    """Get movie title from internal movie ID"""
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

def create_sample_graph(train_df, movies_df, user2id, item2id, max_users=50, max_items=100):
    """Create a sample graph for visualization"""
    # Get user interaction counts
    user_counts = train_df['user_id'].value_counts()
    item_counts = train_df['item_id'].value_counts()
    
    # Select top users and items for visualization
    top_users = user_counts.head(max_users).index.tolist()
    top_items = item_counts.head(max_items).index.tolist()
    
    # Filter interactions for selected users and items
    sample_df = train_df[
        (train_df['user_id'].isin(top_users)) & 
        (train_df['item_id'].isin(top_items))
    ]
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes
    for user_id in top_users:
        G.add_node(f"user_{user_id}", type='user', id=user_id)
    
    for item_id in top_items:
        title = get_movie_title(item_id, movies_df, item2id)
        G.add_node(f"item_{item_id}", type='item', id=item_id, title=title)
    
    # Add edges
    for _, row in sample_df.iterrows():
        user_id = row['user_id']
        item_id = row['item_id']
        G.add_edge(f"user_{user_id}", f"item_{item_id}")
    
    return G, top_users, top_items

def visualize_graph(G, top_users, top_items, movies_df, item2id, save_path="movie_network.png"):
    """Create the network visualization"""
    plt.figure(figsize=(20, 16))
    
    # Use spring layout for better visualization
    pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
    
    # Separate user and item nodes
    user_nodes = [n for n in G.nodes() if G.nodes[n]['type'] == 'user']
    item_nodes = [n for n in G.nodes() if G.nodes[n]['type'] == 'item']
    
    # Calculate node sizes based on degree (number of connections)
    user_sizes = [G.degree(n) * 50 for n in user_nodes]
    item_sizes = [G.degree(n) * 30 for n in item_nodes]
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.2, edge_color='gray', width=0.5)
    
    # Draw user nodes (red)
    nx.draw_networkx_nodes(G, pos, nodelist=user_nodes, 
                          node_color='red', node_size=user_sizes, 
                          alpha=0.7, edgecolors='darkred', linewidths=1)
    
    # Draw item nodes (blue)
    nx.draw_networkx_nodes(G, pos, nodelist=item_nodes, 
                          node_color='blue', node_size=item_sizes, 
                          alpha=0.6, edgecolors='darkblue', linewidths=1)
    
    # Add labels for popular items
    popular_items = []
    for item_node in item_nodes:
        item_id = G.nodes[item_node]['id']
        degree = G.degree(item_node)
        if degree > np.percentile([G.degree(n) for n in item_nodes], 75):
            title = G.nodes[item_node]['title']
            # Truncate long titles
            if len(title) > 30:
                title = title[:27] + "..."
            popular_items.append((item_node, title))
    
    # Draw labels for popular items
    item_labels = {node: title for node, title in popular_items}
    nx.draw_networkx_labels(G, pos, labels=item_labels, 
                           font_size=8, font_color='darkblue', 
                           font_weight='bold')
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                  markersize=10, label='Users'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                  markersize=8, label='Movies')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    # Add title and statistics
    plt.title('MovieLens-1M User-Movie Interaction Network\n' + 
              f'Users: {len(user_nodes)}, Movies: {len(item_nodes)}, Interactions: {G.number_of_edges()}', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Add statistics text
    stats_text = f"""
    Network Statistics:
    • Total Users: {len(user_nodes)}
    • Total Movies: {len(item_nodes)}
    • Total Interactions: {G.number_of_edges()}
    • Average User Degree: {np.mean([G.degree(n) for n in user_nodes]):.1f}
    • Average Movie Degree: {np.mean([G.degree(n) for n in item_nodes]):.1f}
    • Most Popular Movie: {max(popular_items, key=lambda x: G.degree(x[0]))[1] if popular_items else 'N/A'}
    """
    
    plt.figtext(0.02, 0.02, stats_text, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    return popular_items

def create_interaction_heatmap(train_df, top_users, top_items, save_path="interaction_heatmap.png"):
    """Create a heatmap of user-item interactions"""
    # Create interaction matrix
    interaction_matrix = np.zeros((len(top_users), len(top_items)))
    
    # Fill the matrix
    for i, user_id in enumerate(top_users):
        for j, item_id in enumerate(top_items):
            if len(train_df[(train_df['user_id'] == user_id) & (train_df['item_id'] == item_id)]) > 0:
                interaction_matrix[i, j] = 1
    
    plt.figure(figsize=(15, 10))
    sns.heatmap(interaction_matrix, cmap='Blues', cbar_kws={'label': 'Interaction'})
    plt.title('User-Movie Interaction Heatmap\n(Top Users vs Top Movies)', fontsize=16, fontweight='bold')
    plt.xlabel('Movies', fontsize=12)
    plt.ylabel('Users', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def create_statistics_plots(train_df, save_path_prefix="statistics"):
    """Create various statistics plots"""
    # User interaction distribution
    user_counts = train_df['user_id'].value_counts()
    item_counts = train_df['item_id'].value_counts()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # User interaction distribution
    axes[0, 0].hist(user_counts.values, bins=50, alpha=0.7, color='red', edgecolor='black')
    axes[0, 0].set_title('User Interaction Distribution')
    axes[0, 0].set_xlabel('Number of Movies Rated')
    axes[0, 0].set_ylabel('Number of Users')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Item interaction distribution
    axes[0, 1].hist(item_counts.values, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 1].set_title('Movie Interaction Distribution')
    axes[0, 1].set_xlabel('Number of Users Rating')
    axes[0, 1].set_ylabel('Number of Movies')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Top users
    top_users = user_counts.head(20)
    axes[1, 0].bar(range(len(top_users)), top_users.values, color='red', alpha=0.7)
    axes[1, 0].set_title('Top 20 Most Active Users')
    axes[1, 0].set_xlabel('User Rank')
    axes[1, 0].set_ylabel('Number of Movies Rated')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Top items
    top_items = item_counts.head(20)
    axes[1, 1].bar(range(len(top_items)), top_items.values, color='blue', alpha=0.7)
    axes[1, 1].set_title('Top 20 Most Popular Movies')
    axes[1, 1].set_xlabel('Movie Rank')
    axes[1, 1].set_ylabel('Number of Users Rating')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_path_prefix}_plots.png", dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to create all visualizations"""
    print("Loading MovieLens-1M data...")
    train_df, movies_df, user2id, item2id = load_data()
    
    print(f"Loaded {len(train_df)} interactions")
    print(f"Number of users: {train_df['user_id'].nunique()}")
    print(f"Number of movies: {train_df['item_id'].nunique()}")
    
    print("\nCreating network visualization...")
    G, top_users, top_items = create_sample_graph(train_df, movies_df, user2id, item2id)
    
    print("Generating network graph...")
    popular_items = visualize_graph(G, top_users, top_items, movies_df, item2id)
    
    print("Creating interaction heatmap...")
    create_interaction_heatmap(train_df, top_users, top_items)
    
    print("Creating statistics plots...")
    create_statistics_plots(train_df)
    
    print("\nVisualization completed!")
    print("Generated files:")
    print("- movie_network.png (Network graph)")
    print("- interaction_heatmap.png (Interaction heatmap)")
    print("- statistics_plots.png (Statistics plots)")
    
    # Print some popular movies
    print("\nTop popular movies in the network:")
    for i, (item_node, title) in enumerate(popular_items[:10], 1):
        degree = G.degree(item_node)
        print(f"{i:2d}. {title} (Degree: {degree})")

if __name__ == "__main__":
    main() 