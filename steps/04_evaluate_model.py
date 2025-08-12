#!/usr/bin/env python3
"""
BƯỚC 4: EVALUATE LIGHTGCN MODEL
==================================

Mục đích:
- Chạy evaluation với multiple metrics
- Tạo recommendations cho sample users
- Lưu evaluation results

Output:
- evaluation_results.json: Kết quả evaluation
- sample_recommendations.json: Sample recommendations
- evaluation_summary.txt: Tóm tắt kết quả

Metrics: Recall@K, NDCG@K, Precision@K (K = 5, 10, 20)
"""

import subprocess
import sys
import os
import json
import time
from utils import get_step_output_dir, get_step_output_path

def run_evaluation():
    print("🚀 BƯỚC 4: EVALUATE LIGHTGCN MODEL")
    print("=" * 60)

    try:
        # Chạy file evaluate.py
        result = subprocess.run([sys.executable, "evaluate.py"], 
                              capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(__file__)))
        
        if result.returncode == 0:
            print("✅ Evaluation completed successfully!")
            print("\n" + "="*60)
            print("📊 EVALUATION RESULTS:")
            print("="*60)
            print(result.stdout)
            
            # Parse kết quả từ output
            results = parse_evaluation_output(result.stdout)
            
            # Lưu kết quả
            save_evaluation_results(results)
            
            return results
        else:
            print(f"❌ Evaluation failed with error:")
            print(result.stderr)
            return None
            
    except Exception as e:
        print(f"❌ Error running evaluation: {e}")
        return None

def parse_evaluation_output(output):
    """Parse kết quả từ output của evaluate.py"""
    results = {}
    
    # Tìm metrics từ output
    lines = output.split('\n')
    for line in lines:
        if ':' in line and any(metric in line for metric in ['recall@', 'ndcg@', 'precision@']):
            parts = line.split(':')
            if len(parts) == 2:
                metric = parts[0].strip()
                try:
                    value = float(parts[1].strip())
                    results[metric] = value
                except ValueError:
                    continue
    
    return results

def save_evaluation_results(results):
    """Lưu evaluation results"""
    print(f"\n💾 Saving evaluation results...")
    
    # Lưu vào thư mục output của step 04
    save_dir = get_step_output_dir("04_evaluation")
    
    # Lưu metrics
    with open(f"{save_dir}/evaluation_metrics.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Lưu summary
    with open(f"{save_dir}/evaluation_summary.txt", 'w') as f:
        f.write("LIGHTGCN EVALUATION SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("MODEL PERFORMANCE:\n")
        f.write("-" * 20 + "\n")
        for metric, value in results.items():
            f.write(f"{metric}: {value:.4f}\n")
        
        f.write(f"\nTIMESTAMP: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\nNOTE: Evaluation was performed using the existing evaluate.py script\n")
    
    print(f"✅ Saved results to {save_dir}/")
    print(f"   - evaluation_metrics.json")
    print(f"   - evaluation_summary.txt")

def print_evaluation_summary(results):
    """In tóm tắt evaluation results"""
    if not results:
        print("❌ No evaluation results to display")
        return
        
    print("\n📈 EVALUATION RESULTS SUMMARY")
    print("=" * 60)
    
    # Group metrics by K value
    k_values = set()
    for metric in results.keys():
        if '@' in metric:
            k = int(metric.split('@')[1])
            k_values.add(k)
    
    k_values = sorted(list(k_values))
    
    for k in k_values:
        print(f"\nK = {k}:")
        if f'recall@{k}' in results:
            print(f"  Recall@{k}:    {results[f'recall@{k}']:.4f}")
        if f'ndcg@{k}' in results:
            print(f"  NDCG@{k}:      {results[f'ndcg@{k}']:.4f}")
        if f'precision@{k}' in results:
            print(f"  Precision@{k}: {results[f'precision@{k}']:.4f}")

def main():
    """Main function"""
    try:
        # Chạy evaluation
        results = run_evaluation()
        
        if results:
            # In summary
            print_evaluation_summary(results)
            
            print("\n🎉 EVALUATION HOÀN THÀNH!")
            print("=" * 60)
            print("📁 Files created:")
            print("   - evaluation_results/evaluation_metrics.json")
            print("   - evaluation_summary.txt")
            print("\n➡️ Next step: Chạy 05_demo_visualization.py")
        else:
            print("❌ Evaluation failed. Please check the error messages above.")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print("Hãy kiểm tra lại dữ liệu và thử lại.")

if __name__ == "__main__":
    main()
