#!/usr/bin/env python3
"""
MAIN PIPELINE: LIGHTGCN RECOMMENDATION SYSTEM
==============================================

Pipeline hoàn chỉnh để xây dựng và evaluate LightGCN recommendation system:

1. Data Preprocessing (01_data_preprocessing.py)
2. Build Adjacency Matrix (02_build_adjacency_matrix.py)  
3. Train LightGCN Model (03_train_lightgcn.py)
4. Evaluate Model (04_evaluate_model.py)
5. Demo & Visualization (05_demo_visualization.py)

Usage:
    python main.py                    # Chạy toàn bộ pipeline
    python main.py --step 1           # Chạy từng bước cụ thể
    python main.py --skip-training    # Bỏ qua training (nếu đã có model)
    python main.py --help             # Hiển thị help

Author: AI Assistant
Date: 2024
"""

import os
import sys
import time
import argparse
import subprocess
from datetime import datetime

def print_banner():
    """In banner đẹp"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║  🚀 LIGHTGCN RECOMMENDATION SYSTEM PIPELINE                ║
    ║                                                              ║
    ║  A complete workflow for building and evaluating            ║
    ║  LightGCN-based recommendation systems                      ║
    ║                                                              ║
    ║  Paper: LightGCN: Simplifying and Powering Graph           ║
    ║         Convolution Network for Recommendation               ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_dependencies():
    """Kiểm tra dependencies"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'torch', 'numpy', 'pandas', 'matplotlib', 'seaborn', 
        'scikit-learn', 'scipy', 'networkx', 'tqdm'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ All dependencies are available!")
    return True

def check_data_files():
    """Kiểm tra data files"""
    print("\n📁 Checking data files...")
    
    required_files = [
        'ml-1m/ratings.dat',
        'ml-1m/movies.dat', 
        'ml-1m/users.dat'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            print(f"   ✅ {file_path} ({file_size:.1f} MB)")
        else:
            print(f"   ❌ {file_path} - MISSING")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️ Missing data files: {', '.join(missing_files)}")
        print("Please download MovieLens-1M dataset to ml-1m/ directory")
        return False
    
    print("✅ All data files are available!")
    return True

def run_step(step_number, step_name, script_path, description):
    """Chạy một bước cụ thể"""
    print(f"\n{'='*80}")
    print(f"🚀 BƯỚC {step_number}: {step_name}")
    print(f"{'='*80}")
    print(f"📝 {description}")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    try:
        # Chạy script
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=False, text=True)
        
        if result.returncode == 0:
            elapsed_time = time.time() - start_time
            print(f"\n✅ BƯỚC {step_number} HOÀN THÀNH!")
            print(f"⏱️ Thời gian: {elapsed_time:.1f} giây")
            return True
        else:
            print(f"\n❌ BƯỚC {step_number} THẤT BẠI!")
            print(f"Exit code: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"\n❌ Error running step {step_number}: {e}")
        return False

def run_full_pipeline(skip_training=False):
    """Chạy toàn bộ pipeline"""
    print("🚀 Starting full pipeline...")
    
    pipeline_steps = [
        {
            'number': 1,
            'name': 'DATA PREPROCESSING',
            'script': 'steps/01_data_preprocessing.py',
            'description': 'Load raw data, convert to implicit feedback, create ID mappings, split train/test'
        },
        {
            'number': 2,
            'name': 'BUILD ADJACENCY MATRIX',
            'script': 'steps/02_build_adjacency_matrix.py',
            'description': 'Build user-item graph, normalize adjacency matrix, create data loaders'
        }
    ]
    
    if not skip_training:
        pipeline_steps.extend([
            {
                'number': 3,
                'name': 'TRAIN LIGHTGCN MODEL',
                'script': 'steps/03_train_lightgcn.py',
                'description': 'Initialize LightGCN, train with BPR loss, early stopping, save model'
            }
        ])
    
    pipeline_steps.extend([
        {
            'number': 4 if skip_training else 4,
            'name': 'EVALUATE MODEL',
            'script': 'steps/04_evaluate_model.py',
            'description': 'Evaluate on test set, calculate metrics (Recall@K, NDCG@K, Precision@K)'
        },
        {
            'number': 5 if skip_training else 5,
            'name': 'DEMO & VISUALIZATION',
            'script': 'steps/05_demo_visualization.py',
            'description': 'Generate recommendations, create visualizations, demo results'
        }
    ])
    
    # Chạy từng bước
    for step in pipeline_steps:
        success = run_step(
            step['number'], 
            step['name'], 
            step['script'], 
            step['description']
        )
        
        if not success:
            print(f"\n❌ Pipeline failed at step {step['number']}")
            print("Please check the error and try again.")
            return False
        
        print(f"\n✅ Step {step['number']} completed successfully!")
        
        # Pause between steps (optional)
        if step['number'] < len(pipeline_steps):
            print("\n⏸️ Press Enter to continue to next step...")
            input()
    
    return True

def run_specific_step(step_number):
    """Chạy một bước cụ thể"""
    step_map = {
        1: ('steps/01_data_preprocessing.py', 'DATA PREPROCESSING'),
        2: ('steps/02_build_adjacency_matrix.py', 'BUILD ADJACENCY MATRIX'),
        3: ('steps/03_train_lightgcn.py', 'TRAIN LIGHTGCN MODEL'),
        4: ('steps/04_evaluate_model.py', 'EVALUATE MODEL'),
        5: ('steps/05_demo_visualization.py', 'DEMO & VISUALIZATION')
    }
    
    if step_number not in step_map:
        print(f"❌ Invalid step number: {step_number}")
        print("Valid steps: 1, 2, 3, 4, 5")
        return False
    
    script_path, step_name = step_map[step_number]
    
    if not os.path.exists(script_path):
        print(f"❌ Script not found: {script_path}")
        return False
    
    # Check dependencies for this step
    if step_number == 1:
        if not check_data_files():
            return False
    
    # Run the step
    descriptions = {
        1: 'Load raw data, convert to implicit feedback, create ID mappings, split train/test',
        2: 'Build user-item graph, normalize adjacency matrix, create data loaders',
        3: 'Initialize LightGCN, train with BPR loss, early stopping, save model',
        4: 'Evaluate on test set, calculate metrics (Recall@K, NDCG@K, Precision@K)',
        5: 'Generate recommendations, create visualizations, demo results'
    }
    
    return run_step(step_number, step_name, script_path, descriptions[step_number])

def print_pipeline_info():
    """In thông tin về pipeline"""
    print("\n📋 PIPELINE OVERVIEW")
    print("=" * 60)
    
    steps_info = [
        ("1", "Data Preprocessing", "Convert raw data to training format"),
        ("2", "Build Adjacency Matrix", "Create user-item graph structure"),
        ("3", "Train LightGCN Model", "Train neural network with BPR loss"),
        ("4", "Evaluate Model", "Calculate recommendation metrics"),
        ("5", "Demo & Visualization", "Generate recommendations and plots")
    ]
    
    for step_num, step_name, description in steps_info:
        print(f"{step_num}. {step_name:<25} - {description}")
    
    print("\n📁 Expected Output Files:")
    print("   - movielens_train.csv, movielens_test.csv")
    print("   - user2id.json, item2id.json")
    print("   - norm_adj_matrix.npz")
    print("   - lightgcn_movielens.pt")
    print("   - evaluation_results/")
    print("   - Various visualization plots")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='LightGCN Recommendation System Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run full pipeline
  python main.py --step 1           # Run only step 1
  python main.py --skip-training    # Skip training step
  python main.py --info             # Show pipeline info
        """
    )
    
    parser.add_argument('--step', type=int, choices=[1,2,3,4,5],
                       help='Run specific step (1-5)')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training step (step 3)')
    parser.add_argument('--info', action='store_true',
                       help='Show pipeline information')
    parser.add_argument('--check-deps', action='store_true',
                       help='Check dependencies only')
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Dependencies check failed. Please install missing packages.")
        return 1
    
    # Show info if requested
    if args.info:
        print_pipeline_info()
        return 0
    
    # Check dependencies only if requested
    if args.check_deps:
        print("✅ Dependencies check completed.")
        return 0
    
    # Check data files
    if not check_data_files():
        print("❌ Data files check failed. Please download MovieLens-1M dataset.")
        return 1
    
    # Run specific step if requested
    if args.step:
        print(f"🎯 Running specific step: {args.step}")
        success = run_specific_step(args.step)
        return 0 if success else 1
    
    # Run full pipeline
    print("🎯 Running full pipeline...")
    success = run_full_pipeline(skip_training=args.skip_training)
    
    if success:
        print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("📁 All output files have been generated.")
        print("🎬 You can now use the trained model for recommendations!")
        print("\n📊 Generated files:")
        print("   - Training data and model weights")
        print("   - Evaluation results and metrics")
        print("   - Visualizations and demo results")
        return 0
    else:
        print("\n❌ PIPELINE FAILED!")
        print("Please check the error messages and try again.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
