#!/usr/bin/env python3
"""
BƯỚC 1: TIỀN XỬ LÝ DỮ LIỆU MOVIELENS-1M
============================================

Mục đích: 
- Load dữ liệu thô từ MovieLens-1M
- Chuyển đổi sang implicit feedback (rating >= 4)
- Tạo mapping user_id và item_id liên tục
- Chia train/test split
- Lưu dữ liệu đã xử lý

Output:
- movielens_train.csv: Dữ liệu training
- movielens_test.csv: Dữ liệu testing  
- user2id.json: Mapping user ID
- item2id.json: Mapping item ID
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict
import json
import os
import sys

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import utility functions
from utils import get_ml1m_path, get_step_output_path

def load_raw_data():
    """Load dữ liệu thô từ MovieLens-1M"""
    print("📁 Loading raw data from MovieLens-1M...")
    
    # Load ratings data
    ratings_path = get_ml1m_path('ratings.dat')
    df = pd.read_csv(ratings_path, sep='::', engine='python',
                     names=['user_id', 'item_id', 'rating', 'timestamp'])
    
    print(f"✅ Loaded {len(df):,} ratings from {df['user_id'].nunique():,} users and {df['item_id'].nunique():,} movies")
    print(f"📊 Rating distribution:")
    print(df['rating'].value_counts().sort_index())
    
    return df

def convert_to_implicit_feedback(df):
    """Chuyển đổi sang implicit feedback (rating >= 4)"""
    print("\n🔄 Converting to implicit feedback...")
    
    # Chỉ giữ ratings >= 4 (positive interactions)
    df_implicit = df[df['rating'] >= 4].copy()
    
    print(f"✅ Kept {len(df_implicit):,} positive interactions (rating >= 4)")
    print(f"📉 Removed {len(df) - len(df_implicit):,} negative interactions")
    
    return df_implicit

def create_id_mappings(df):
    """Tạo mapping user_id và item_id liên tục"""
    print("\n🆔 Creating ID mappings...")
    
    # Tạo mapping liên tục
    user2id = {u: idx for idx, u in enumerate(df['user_id'].unique())}
    item2id = {i: idx for idx, i in enumerate(df['item_id'].unique())}
    
    # Áp dụng mapping
    df['user_id'] = df['user_id'].map(user2id)
    df['item_id'] = df['item_id'].map(item2id)
    
    num_users = len(user2id)
    num_items = len(item2id)
    
    print(f"✅ Mapped {num_users:,} users to indices 0-{num_users-1}")
    print(f"✅ Mapped {num_items:,} movies to indices 0-{num_items-1}")
    
    return df, user2id, item2id, num_users, num_items

def create_train_test_split(df):
    """Chia dữ liệu thành train/test split"""
    print("\n✂️ Creating train/test split...")
    
    # Sắp xếp theo thời gian để chọn tương tác cuối cùng cho test
    df = df.sort_values(by=['user_id', 'timestamp'])
    
    train_rows = []
    test_rows = []
    
    for u in df['user_id'].unique():
        user_data = df[df['user_id'] == u]
        if len(user_data) < 2:  # Bỏ qua users có ít hơn 2 tương tác
            continue
        test_rows.append(user_data.iloc[-1])  # Tương tác cuối cùng
        train_rows.extend(user_data.iloc[:-1].to_dict('records'))  # Các tương tác còn lại
    
    train_df = pd.DataFrame(train_rows)
    test_df = pd.DataFrame(test_rows)
    
    print(f"✅ Train set: {len(train_df):,} interactions")
    print(f"✅ Test set: {len(test_df):,} interactions")
    print(f"📊 Train/Test ratio: {len(train_df)/len(test_df):.2f}")
    
    return train_df, test_df

def save_processed_data(train_df, test_df, user2id, item2id):
    """Lưu dữ liệu đã xử lý"""
    print("\n💾 Saving processed data...")
    
    # Lưu CSV files vào thư mục output của step 01
    train_df.to_csv(get_step_output_path("01_preprocessing", "movielens_train.csv"), index=False)
    test_df.to_csv(get_step_output_path("01_preprocessing", "movielens_test.csv"), index=False)
    
    # Lưu ID mappings
    user2id = {int(u): int(idx) for u, idx in user2id.items()}
    item2id = {int(i): int(idx) for i, idx in item2id.items()}
    
    with open(get_step_output_path("01_preprocessing", 'user2id.json'), 'w') as f:
        json.dump(user2id, f, indent=2)
    with open(get_step_output_path("01_preprocessing", 'item2id.json'), 'w') as f:
        json.dump(item2id, f, indent=2)
    
    print("✅ Saved files to steps/outputs/01_preprocessing/:")
    print("   - movielens_train.csv")
    print("   - movielens_test.csv")
    print("   - user2id.json")
    print("   - item2id.json")

def print_summary_stats(train_df, test_df, num_users, num_items):
    """In thống kê tổng quan"""
    print("\n📈 SUMMARY STATISTICS")
    print("=" * 50)
    print(f"Total Users: {num_users:,}")
    print(f"Total Movies: {num_items:,}")
    print(f"Total Interactions: {len(train_df) + len(test_df):,}")
    print(f"Training Interactions: {len(train_df):,}")
    print(f"Testing Interactions: {len(test_df):,}")
    print(f"Sparsity: {1 - (len(train_df) + len(test_df))/(num_users * num_items):.4f}")
    
    # User interaction statistics
    user_counts = train_df['user_id'].value_counts()
    print(f"\nUser Interaction Stats:")
    print(f"  Min interactions per user: {user_counts.min()}")
    print(f"  Max interactions per user: {user_counts.max()}")
    print(f"  Mean interactions per user: {user_counts.mean():.1f}")
    print(f"  Median interactions per user: {user_counts.median():.1f}")
    
    # Movie interaction statistics
    item_counts = train_df['item_id'].value_counts()
    print(f"\nMovie Interaction Stats:")
    print(f"  Min interactions per movie: {item_counts.min()}")
    print(f"  Max interactions per movie: {item_counts.max()}")
    print(f"  Mean interactions per movie: {item_counts.mean():.1f}")
    print(f"  Median interactions per movie: {item_counts.median():.1f}")

def main():
    """Main function"""
    print("🚀 BƯỚC 1: TIỀN XỬ LÝ DỮ LIỆU MOVIELENS-1M")
    print("=" * 60)
    
    try:
        # Bước 1: Load dữ liệu thô
        df = load_raw_data()
        
        # Bước 2: Chuyển đổi sang implicit feedback
        df_implicit = convert_to_implicit_feedback(df)
        
        # Bước 3: Tạo ID mappings
        df_mapped, user2id, item2id, num_users, num_items = create_id_mappings(df_implicit)
        
        # Bước 4: Chia train/test
        train_df, test_df = create_train_test_split(df_mapped)
        
        # Bước 5: Lưu dữ liệu
        save_processed_data(train_df, test_df, user2id, item2id)
        
        # Bước 6: In thống kê
        print_summary_stats(train_df, test_df, num_users, num_items)
        
        print("\n🎉 TIỀN XỬ LÝ DỮ LIỆU HOÀN THÀNH!")
        print("=" * 60)
        print("📁 Files created:")
        print("   - movielens_train.csv (training data)")
        print("   - movielens_test.csv (testing data)")
        print("   - user2id.json (user ID mapping)")
        print("   - item2id.json (movie ID mapping)")
        print("\n➡️ Next step: Chạy 02_build_adjacency_matrix.py")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Hãy kiểm tra lại dữ liệu và thử lại.")

if __name__ == "__main__":
    main()
