import streamlit as st
import torch
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from data_loader import load_movielens, build_adj_matrix, get_user_item_interactions
from lightgcn import LightGCN
from evaluate import get_recommendations
import time
import re

# Page config
st.set_page_config(
    page_title="LightGCN Movie Recommendations",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .movie-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        position: relative;
    }
    .user-profile-container {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        margin-bottom: 1rem;
    }
    .score-badge {
        position: absolute;
        top: 0.5rem;
        right: 0.5rem;
        background-color: #ff6b6b;
        color: white;
        padding: 0.3rem 0.6rem;
        border-radius: 0.5rem;
        font-size: 0.9rem;
        font-weight: bold;
    }
    .year-badge {
        background-color: rgba(255,255,255,0.2);
        padding: 0.2rem 0.5rem;
        border-radius: 0.5rem;
        font-size: 0.8rem;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load all data and model"""
    with st.spinner("Loading data and model..."):
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
        
        # Try to load GPU model first, then CPU model
        model_files = ["lightgcn_movielens_gpu.pt", "lightgcn_movielens.pt"]
        model_loaded = False
        
        for model_file in model_files:
            try:
                model.load_state_dict(torch.load(model_file, map_location='cpu'))
                st.success(f"Model loaded from {model_file}")
                model_loaded = True
                break
            except FileNotFoundError:
                continue
        
        if not model_loaded:
            st.error("No model file found. Please run training first.")
            return None, None, None, None, None, None
        
        # Load movie information with better parsing
        try:
            movies_df = pd.read_csv('ml-1m/movies.dat', sep='::', engine='python',
                                   names=['movie_id', 'title', 'genres'], encoding='latin-1')
            
            # Parse title and year
            def extract_title_year(title):
                # Pattern: "Title (Year)" or "Title, The (Year)"
                match = re.match(r'(.+?)\s*\((\d{4})\)', title)
                if match:
                    return match.group(1).strip(), int(match.group(2))
                return title, None
            
            movies_df[['title_clean', 'year']] = movies_df['title'].apply(
                lambda x: pd.Series(extract_title_year(x))
            )
            
            # Create a mapping from movie_id to movie info
            movie_info = {}
            for _, row in movies_df.iterrows():
                movie_info[int(row['movie_id'])] = {
                    'title': row['title_clean'],
                    'year': row['year'],
                    'genres': row['genres'],
                    'full_title': row['title']
                }
            
        except Exception as e:
            st.warning(f"Could not load movie information: {e}")
            movie_info = {}
        
        return model, user_items, item_users, user2id, item2id, movie_info

def get_movie_info(movie_id, movie_info, item2id):
    """Get detailed movie information from movie ID"""
    if not movie_info:
        return {
            'title': f"Movie_{movie_id}",
            'year': None,
            'genres': "Unknown",
            'full_title': f"Movie_{movie_id}"
        }
    
    # Convert internal ID back to original movie ID
    original_movie_id = None
    for orig_id, internal_id in item2id.items():
        if internal_id == movie_id:
            original_movie_id = orig_id
            break
    
    if original_movie_id is None:
        return {
            'title': f"Movie_{movie_id}",
            'year': None,
            'genres': "Unknown",
            'full_title': f"Movie_{movie_id}"
        }
    
    # Convert to int for lookup
    original_movie_id_int = int(original_movie_id)
    
    if original_movie_id_int not in movie_info:
        return {
            'title': f"Movie_{movie_id}",
            'year': None,
            'genres': "Unknown",
            'full_title': f"Movie_{movie_id}"
        }
    
    movie_data = movie_info[original_movie_id_int]
    return movie_data

def format_genres(genres_str):
    """Format genres string for better display"""
    if not genres_str or genres_str == "Unknown":
        return "Unknown"
    
    genres = genres_str.split('|')
    # Limit to 3 genres for better display
    if len(genres) > 3:
        genres = genres[:2] + ['...']
    
    return ' • '.join(genres)

def display_movie_card(movie_data, rank=None, score=None):
    """Display a movie card using Streamlit components"""
    # Create a container with custom styling
    with st.container():
        # Movie title
        if rank:
            st.markdown(f"**#{rank} - {movie_data['title']}**")
        else:
            st.markdown(f"**{movie_data['title']}**")
        
        # Year and genres in columns
        col1, col2 = st.columns([1, 2])
        with col1:
            if movie_data['year']:
                st.markdown(f"**{movie_data['year']}**")
        with col2:
            genres = format_genres(movie_data['genres'])
            st.markdown(f"*{genres}*")
        
        # Score
        if score is not None:
            st.markdown(f"**Score: {score:.3f}**")
        
        st.divider()

def main():
    # Header
    st.markdown('<h1 class="main-header">🎬 LightGCN Movie Recommendations</h1>', unsafe_allow_html=True)
    
    # Load data
    data = load_data()
    if data[0] is None:
        st.stop()
    
    model, user_items, item_users, user2id, item2id, movie_info = data
    
    # Sidebar
    st.sidebar.title("🎯 Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Dashboard", "🔍 User Recommendations", "📊 Model Analysis", "🎲 Random Recommendations"]
    )
    
    if page == "🏠 Dashboard":
        show_dashboard(model, user_items, item_users, user2id, item2id, movie_info)
    elif page == "🔍 User Recommendations":
        show_user_recommendations(model, user_items, item_users, user2id, item2id, movie_info)
    elif page == "📊 Model Analysis":
        show_model_analysis(model, user_items, item_users, user2id, item2id, movie_info)
    elif page == "🎲 Random Recommendations":
        show_random_recommendations(model, user_items, item_users, user2id, item2id, movie_info)

def show_dashboard(model, user_items, item_users, user2id, item2id, movie_info):
    """Show main dashboard"""
    st.header("📊 Model Overview")
    
    # Model info with better styling
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("👥 Total Users", len(user_items))
    
    with col2:
        st.metric("🎬 Total Movies", len(item_users))
    
    with col3:
        total_interactions = sum(len(items) for items in user_items.values())
        st.metric("💫 Total Interactions", f"{total_interactions:,}")
    
    with col4:
        avg_interactions = total_interactions / len(user_items) if len(user_items) > 0 else 0
        st.metric("📈 Avg/User", f"{avg_interactions:.1f}")
    
    # User activity distribution
    st.subheader("📈 User Activity Distribution")
    user_activity = [len(items) for items in user_items.values()]
    
    fig = px.histogram(
        x=user_activity,
        nbins=50,
        title="Distribution of User Interactions",
        labels={'x': 'Number of Interactions', 'y': 'Number of Users'},
        color_discrete_sequence=['#667eea']
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Top genres (if available)
    if movie_info:
        st.subheader("🎭 Popular Genres")
        all_genres = []
        for movie_data in movie_info.values():
            if movie_data['genres'] and movie_data['genres'] != "Unknown":
                all_genres.extend(movie_data['genres'].split('|'))
        
        if all_genres:
            genre_counts = pd.Series(all_genres).value_counts().head(10)
            
            fig = px.bar(
                x=genre_counts.values,
                y=genre_counts.index,
                orientation='h',
                title="Top 10 Movie Genres",
                labels={'x': 'Number of Movies', 'y': 'Genre'},
                color_discrete_sequence=['#764ba2']
            )
            st.plotly_chart(fig, use_container_width=True)

def show_user_recommendations(model, user_items, item_users, user2id, item2id, movie_info):
    """Show user-specific recommendations"""
    st.header("🔍 User Recommendations")
    
    # User selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        user_id = st.number_input(
            "Enter User ID:",
            min_value=0,
            max_value=len(user_items)-1,
            value=0,
            step=1
        )
    
    with col2:
        k_recommendations = st.slider("Number of recommendations:", 5, 20, 10)
    
    if st.button("Get Recommendations", type="primary"):
        with st.spinner("Generating recommendations..."):
            # Get user info
            train_items = user_items.get(user_id, set())
            
            # Get recommendations
            recommendations, scores = get_recommendations(model, user_id, user_items, K=k_recommendations)
            
            # Display results
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # User profile
                st.markdown("""
                <div class="user-profile-container">
                    <h3>👤 User Profile</h3>
                </div>
                """, unsafe_allow_html=True)
                
                st.metric("Training Interactions", len(train_items))
                
                if len(train_items) > 0:
                    st.write("**Recently watched movies:**")
                    recent_items = list(train_items)[:5]  # Show last 5
                    for item in recent_items:
                        movie_data = get_movie_info(item, movie_info, item2id)
                        display_movie_card(movie_data)
            
            with col2:
                st.subheader(f"🎯 Top {k_recommendations} Recommendations")
                
                for i, (item_id, score) in enumerate(zip(recommendations, scores), 1):
                    movie_data = get_movie_info(item_id, movie_info, item2id)
                    display_movie_card(movie_data, rank=i, score=score)
            
            # Score distribution
            st.subheader("📊 Recommendation Scores")
            fig = px.bar(
                x=list(range(1, len(scores)+1)),
                y=scores,
                title="Recommendation Scores",
                labels={'x': 'Rank', 'y': 'Score'},
                color_discrete_sequence=['#667eea']
            )
            st.plotly_chart(fig, use_container_width=True)

def show_model_analysis(model, user_items, item_users, user2id, item2id, movie_info):
    """Show model analysis and insights"""
    st.header("📊 Model Analysis")
    
    # Model architecture
    st.subheader("🏗️ Model Architecture")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Embedding Dimension", "64")
    
    with col2:
        st.metric("GCN Layers", "3")
    
    with col3:
        total_params = sum(p.numel() for p in model.parameters())
        st.metric("Model Parameters", f"{total_params:,}")
    
    # User embedding visualization
    st.subheader("👥 User Embeddings")
    
    with st.spinner("Computing user embeddings..."):
        model.eval()
        with torch.no_grad():
            user_emb, item_emb = model.get_embeddings()
            user_emb_np = user_emb.detach().cpu().numpy()
    
    # PCA for visualization
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=2)
    user_emb_2d = pca.fit_transform(user_emb_np)
    
    # Create scatter plot
    fig = px.scatter(
        x=user_emb_2d[:, 0],
        y=user_emb_2d[:, 1],
        title="User Embeddings (PCA 2D)",
        labels={'x': 'PC1', 'y': 'PC2'},
        opacity=0.6,
        color_discrete_sequence=['#667eea']
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Similar users
    st.subheader("👥 Find Similar Users")
    
    col1, col2 = st.columns(2)
    
    with col1:
        target_user = st.number_input(
            "Enter User ID to find similar users:",
            min_value=0,
            max_value=len(user_items)-1,
            value=0
        )
    
    with col2:
        n_similar = st.slider("Number of similar users:", 5, 20, 10)
    
    if st.button("Find Similar Users"):
        with st.spinner("Finding similar users..."):
            # Calculate cosine similarity
            target_emb = user_emb_np[target_user]
            similarities = []
            
            for user_id in range(len(user_emb_np)):
                if user_id != target_user:
                    user_emb = user_emb_np[user_id]
                    similarity = np.dot(target_emb, user_emb) / (np.linalg.norm(target_emb) * np.linalg.norm(user_emb))
                    similarities.append((user_id, similarity))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_similar = similarities[:n_similar]
            
            st.write(f"**Top {n_similar} similar users to User {target_user}:**")
            
            for i, (user_id, similarity) in enumerate(top_similar, 1):
                interactions = len(user_items.get(user_id, set()))
                st.write(f"{i}. User {user_id} (Similarity: {similarity:.3f}, Interactions: {interactions})")

def show_random_recommendations(model, user_items, item_users, user2id, item2id, movie_info):
    """Show random user recommendations"""
    st.header("🎲 Random User Recommendations")
    
    if st.button("Get Random User Recommendations", type="primary"):
        # Select random users
        import random
        random_users = random.sample(list(user_items.keys()), 3)
        
        for i, user_id in enumerate(random_users, 1):
            st.subheader(f"🎲 Random User {i}: User {user_id}")
            
            # Get recommendations
            recommendations, scores = get_recommendations(model, user_id, user_items, K=5)
            
            # Display recommendations
            cols = st.columns(5)
            for j, (item_id, score) in enumerate(zip(recommendations, scores)):
                with cols[j]:
                    movie_data = get_movie_info(item_id, movie_info, item2id)
                    display_movie_card(movie_data, rank=j+1, score=score)
            
            st.divider()

if __name__ == "__main__":
    main() 