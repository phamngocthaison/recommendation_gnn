#!/usr/bin/env python3
"""
PRINT OUTPUTS SUMMARY
=====================

Script để hiển thị tổng quan về tất cả outputs của các steps
trong cấu trúc mới được tổ chức.
"""

from utils import print_step_outputs_summary

if __name__ == "__main__":
    print("📁 LIGHTGCN PIPELINE - ORGANIZED OUTPUTS STRUCTURE")
    print("=" * 60)
    print("Cấu trúc mới: Mỗi step có thư mục output riêng")
    print()
    
    print_step_outputs_summary()
    
    print("\n" + "=" * 60)
    print("📋 OUTPUT FOLDER STRUCTURE:")
    print("steps/outputs/")
    print("├── 01_preprocessing/     # Data preprocessing outputs")
    print("│   ├── movielens_train.csv")
    print("│   ├── movielens_test.csv")
    print("│   ├── user2id.json")
    print("│   └── item2id.json")
    print("├── 02_adjacency/         # Adjacency matrix outputs")
    print("│   └── norm_adj_matrix.npz")
    print("├── 03_training/          # Training outputs")
    print("│   ├── lightgcn_movielens.pt")
    print("│   ├── training_curves.png")
    print("│   └── training_log.json")
    print("├── 04_evaluation/        # Evaluation outputs")
    print("│   ├── evaluation_metrics.json")
    print("│   ├── sample_recommendations.json")
    print("│   ├── evaluation_summary.txt")
    print("│   └── evaluation_metrics.png")
    print("└── 05_visualization/     # Visualization outputs")
    print("    ├── demo_results.json")
    print("    ├── demo_summary.txt")
    print("    ├── embedding_visualization.png")
    print("    ├── movie_network.png")
    print("    └── interaction_heatmap.png")
    print()
    print("🎯 Benefits:")
    print("✅ Mỗi step có output riêng biệt")
    print("✅ Dễ dàng theo dõi và debug")
    print("✅ Không bị conflict giữa các steps")
    print("✅ Cấu trúc rõ ràng, dễ hiểu")
    print("✅ Dễ dàng clean up từng step")
