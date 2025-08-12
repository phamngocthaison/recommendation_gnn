#!/usr/bin/env python3
"""
MAIN PIPELINE SCRIPT
====================

Script chính để chạy toàn bộ pipeline LightGCN từ đầu đến cuối.
Mỗi step sẽ được chạy tuần tự và kiểm tra kết quả trước khi chuyển sang step tiếp theo.

Pipeline:
1. Data Preprocessing
2. Build Adjacency Matrix
3. Train LightGCN Model
4. Evaluate Model
5. Demo & Visualization

Usage:
    python run_pipeline.py [--step STEP] [--force] [--clean]
    
Options:
    --step STEP: Chạy từ step cụ thể (1-5)
    --force: Bỏ qua kiểm tra dependencies
    --clean: Xóa tất cả outputs trước khi chạy
"""

import os
import sys
import argparse
import subprocess
import time
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import (
    get_project_root, 
    get_step_output_dir, 
    list_step_outputs, 
    clear_step_outputs,
    print_step_outputs_summary
)

def print_header(title):
    """In header đẹp mắt"""
    print("\n" + "="*80)
    print(f"🚀 {title}")
    print("="*80)

def print_step_header(step_num, step_name, description):
    """In header cho từng step"""
    print(f"\n📋 STEP {step_num}: {step_name}")
    print("-" * 60)
    print(f"📝 {description}")
    print("-" * 60)

def check_step_dependencies(step_num):
    """Kiểm tra dependencies của step"""
    print(f"🔍 Checking dependencies for Step {step_num}...")
    
    if step_num == 1:
        # Step 1 không cần dependencies
        return True
    
    elif step_num == 2:
        # Step 2 cần outputs từ Step 1
        step1_outputs = list_step_outputs("01_preprocessing")
        required_files = ["movielens_train.csv", "movielens_test.csv", "user2id.json", "item2id.json"]
        
        missing_files = [f for f in required_files if f not in step1_outputs]
        if missing_files:
            print(f"❌ Missing required files from Step 1: {missing_files}")
            return False
        
        print("✅ All dependencies satisfied for Step 2")
        return True
    
    elif step_num == 3:
        # Step 3 cần outputs từ Step 1 và 2
        step1_outputs = list_step_outputs("01_preprocessing")
        step2_outputs = list_step_outputs("02_adjacency")
        
        required_step1 = ["movielens_train.csv", "user2id.json", "item2id.json"]
        required_step2 = ["norm_adj_matrix.npz"]
        
        missing_step1 = [f for f in required_step1 if f not in step1_outputs]
        missing_step2 = [f for f in required_step2 if f not in step2_outputs]
        
        if missing_step1:
            print(f"❌ Missing required files from Step 1: {missing_step1}")
            return False
        
        if missing_step2:
            print(f"❌ Missing required files from Step 2: {missing_step2}")
            return False
        
        print("✅ All dependencies satisfied for Step 3")
        return True
    
    elif step_num == 4:
        # Step 4 cần outputs từ Step 1, 2, và 3
        step1_outputs = list_step_outputs("01_preprocessing")
        step2_outputs = list_step_outputs("02_adjacency")
        step3_outputs = list_step_outputs("03_training")
        
        required_step1 = ["movielens_test.csv", "user2id.json", "item2id.json"]
        required_step2 = ["norm_adj_matrix.npz"]
        required_step3 = ["lightgcn_movielens.pt"]
        
        missing_step1 = [f for f in required_step1 if f not in step1_outputs]
        missing_step2 = [f for f in required_step2 if f not in step2_outputs]
        missing_step3 = [f for f in required_step3 if f not in step3_outputs]
        
        if missing_step1:
            print(f"❌ Missing required files from Step 1: {missing_step1}")
            return False
        
        if missing_step2:
            print(f"❌ Missing required files from Step 2: {missing_step2}")
            return False
        
        if missing_step3:
            print(f"❌ Missing required files from Step 3: {missing_step3}")
            return False
        
        print("✅ All dependencies satisfied for Step 4")
        return True
    
    elif step_num == 5:
        # Step 5 cần outputs từ tất cả steps trước
        step1_outputs = list_step_outputs("01_preprocessing")
        step2_outputs = list_step_outputs("02_adjacency")
        step3_outputs = list_step_outputs("03_training")
        
        required_step1 = ["movielens_train.csv", "user2id.json", "item2id.json"]
        required_step2 = ["norm_adj_matrix.npz"]
        required_step3 = ["lightgcn_movielens.pt"]
        
        missing_step1 = [f for f in required_step1 if f not in step1_outputs]
        missing_step2 = [f for f in required_step2 if f not in step2_outputs]
        missing_step3 = [f for f in required_step3 if f not in step3_outputs]
        
        if missing_step1:
            print(f"❌ Missing required files from Step 1: {missing_step1}")
            return False
        
        if missing_step2:
            print(f"❌ Missing required files from Step 2: {missing_step2}")
            return False
        
        if missing_step3:
            print(f"❌ Missing required files from Step 3: {missing_step3}")
            return False
        
        print("✅ All dependencies satisfied for Step 5")
        return True
    
    return False

def run_step(step_num, force=False):
    """Chạy một step cụ thể"""
    step_configs = {
        1: {
            "name": "DATA PREPROCESSING",
            "description": "Load và xử lý dữ liệu MovieLens-1M",
            "script": "01_data_preprocessing.py"
        },
        2: {
            "name": "BUILD ADJACENCY MATRIX", 
            "description": "Xây dựng adjacency matrix cho user-item graph",
            "script": "02_build_adjacency_matrix.py"
        },
        3: {
            "name": "TRAIN LIGHTGCN MODEL",
            "description": "Training LightGCN model với BPR loss",
            "script": "03_train_lightgcn.py"
        },
        4: {
            "name": "EVALUATE MODEL",
            "description": "Evaluate model với multiple metrics",
            "script": "04_evaluate_model.py"
        },
        5: {
            "name": "DEMO & VISUALIZATION",
            "description": "Tạo visualizations và demo recommendations",
            "script": "05_demo_visualization.py"
        }
    }
    
    if step_num not in step_configs:
        print(f"❌ Invalid step number: {step_num}")
        return False
    
    config = step_configs[step_num]
    print_step_header(step_num, config["name"], config["description"])
    
    # Kiểm tra dependencies (trừ khi force=True)
    if not force and not check_step_dependencies(step_num):
        print(f"❌ Dependencies not satisfied for Step {step_num}")
        return False
    
    # Chạy step script
    script_path = os.path.join(os.path.dirname(__file__), config["script"])
    print(f"🔄 Running {config['script']}...")
    
    try:
        start_time = time.time()
        result = subprocess.run([sys.executable, script_path], 
                              cwd=os.path.dirname(__file__),
                              check=True,
                              capture_output=False)
        end_time = time.time()
        
        if result.returncode == 0:
            print(f"✅ Step {step_num} completed successfully in {end_time - start_time:.1f}s")
            
            # Hiển thị outputs của step
            step_name = f"0{step_num}_{config['name'].lower().replace(' ', '_')}"
            outputs = list_step_outputs(step_name)
            if outputs:
                print(f"📁 Outputs created:")
                for output in outputs:
                    print(f"   - {output}")
            
            return True
        else:
            print(f"❌ Step {step_num} failed with return code {result.returncode}")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Step {step_num}: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error in Step {step_num}: {e}")
        return False

def run_full_pipeline(start_step=1, force=False, clean=False):
    """Chạy toàn bộ pipeline từ start_step"""
    print_header("LIGHTGCN RECOMMENDATION PIPELINE")
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if clean:
        print("\n🧹 Cleaning all outputs...")
        for step_num in range(1, 6):
            step_names = [
                "01_preprocessing", "02_adjacency", "03_training", 
                "04_evaluation", "05_visualization"
            ]
            clear_step_outputs(step_names[step_num - 1])
    
    # Chạy từng step
    for step_num in range(start_step, 6):
        success = run_step(step_num, force=force)
        
        if not success:
            print(f"\n❌ Pipeline failed at Step {step_num}")
            print("💡 Tips:")
            print("   - Check error messages above")
            print("   - Use --force to skip dependency checks")
            print("   - Use --step to start from a specific step")
            return False
        
        # Nghỉ giữa các steps
        if step_num < 5:
            print(f"\n⏳ Waiting 2 seconds before next step...")
            time.sleep(2)
    
    print_header("PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"🕐 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Hiển thị tổng quan outputs
    print_step_outputs_summary()
    
    return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="LightGCN Recommendation Pipeline")
    parser.add_argument("--step", type=int, choices=[1, 2, 3, 4, 5], default=1,
                       help="Start from specific step (default: 1)")
    parser.add_argument("--force", action="store_true",
                       help="Skip dependency checks")
    parser.add_argument("--clean", action="store_true",
                       help="Clean all outputs before running")
    
    args = parser.parse_args()
    
    # Chạy pipeline
    if args.step == 1:
        # Chạy toàn bộ pipeline
        success = run_full_pipeline(start_step=1, force=args.force, clean=args.clean)
    else:
        # Chạy từ step cụ thể
        success = run_step(args.step, force=args.force)
    
    if success:
        print("\n🎉 All done! Check the outputs in steps/outputs/ directory.")
        sys.exit(0)
    else:
        print("\n💥 Pipeline failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
