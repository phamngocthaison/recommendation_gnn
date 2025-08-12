#!/usr/bin/env python3
"""
BƯỚC 5: DEMO & VISUALIZATION
==================================

Mục đích:
- Sử dụng file evaluate.py có sẵn để tạo recommendations
- Tạo basic visualizations cho kết quả
- Lưu demo results và visualizations

Output:
- demo_results.json: Demo recommendations
- demo_summary.txt: Tóm tắt demo
- basic_visualizations.png: Basic charts
"""

import subprocess
import sys
import os
import json
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils import get_step_output_dir, get_step_output_path

def run_evaluation_for_demo():
    """Chạy evaluation để lấy recommendations cho demo"""
    print("🚀 BƯỚC 5: DEMO & VISUALIZATION")
    print("=" * 60)
    print("📁 Sử dụng file evaluate.py để tạo recommendations...")
    
    try:
        # Chạy file evaluate.py
        result = subprocess.run([sys.executable, "evaluate.py"], 
                              capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(__file__)))
        
        if result.returncode == 0:
            print("✅ Evaluation completed successfully!")
            
            # Parse kết quả từ output
            results = parse_evaluation_output(result.stdout)
            
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

def create_basic_visualizations(results, save_path="basic_visualizations.png"):
    """Tạo basic visualizations cho kết quả"""
    print(f"\n📊 Creating basic visualizations...")
    
    # Lưu vào thư mục output của step 05
    save_path = get_step_output_path("05_visualization", "basic_visualizations.png")
    
    if not results:
        print("❌ No results to visualize")
        return
    
    # Tạo figure với subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Group metrics by K value
    k_values = set()
    for metric in results.keys():
        if '@' in metric:
            k = int(metric.split('@')[1])
            k_values.add(k)
    
    k_values = sorted(list(k_values))
    
    # Plot Recall@K
    recall_values = [results.get(f'recall@{k}', 0) for k in k_values]
    axes[0].bar(k_values, recall_values, color='#2E86AB', alpha=0.8)
    axes[0].set_xlabel('K', fontsize=12)
    axes[0].set_ylabel('Recall@K', fontsize=12)
    axes[0].set_title('Recall@K Performance', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Plot NDCG@K
    ndcg_values = [results.get(f'ndcg@{k}', 0) for k in k_values]
    axes[1].bar(k_values, ndcg_values, color='#A23B72', alpha=0.8)
    axes[1].set_xlabel('K', fontsize=12)
    axes[1].set_ylabel('NDCG@K', fontsize=12)
    axes[1].set_title('NDCG@K Performance', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # Plot Precision@K
    precision_values = [results.get(f'precision@{k}', 0) for k in k_values]
    axes[2].bar(k_values, precision_values, color='#F18F01', alpha=0.8)
    axes[2].set_xlabel('K', fontsize=12)
    axes[2].set_ylabel('Precision@K', fontsize=12)
    axes[2].set_title('Precision@K Performance', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for ax, values in zip(axes, [recall_values, ndcg_values, precision_values]):
        for i, (bar, value) in enumerate(zip(ax.patches, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved visualizations to {save_path}")

def create_demo_results(results):
    """Tạo demo results từ evaluation results"""
    print(f"\n🎯 Creating demo results...")
    
    demo_results = {
        "model_performance": results,
        "summary": {
            "best_recall_k": None,
            "best_ndcg_k": None,
            "best_precision_k": None,
            "overall_assessment": ""
        },
        "recommendations_note": "Recommendations were generated using the existing evaluate.py script",
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Tìm best performing K cho mỗi metric
    k_values = set()
    for metric in results.keys():
        if '@' in metric:
            k = int(metric.split('@')[1])
            k_values.add(k)
    
    k_values = sorted(list(k_values))
    
    if k_values:
        best_recall_k = max(k_values, key=lambda k: results.get(f'recall@{k}', 0))
        best_ndcg_k = max(k_values, key=lambda k: results.get(f'ndcg@{k}', 0))
        best_precision_k = max(k_values, key=lambda k: results.get(f'precision@{k}', 0))
        
        demo_results["summary"]["best_recall_k"] = best_recall_k
        demo_results["summary"]["best_ndcg_k"] = best_ndcg_k
        demo_results["summary"]["best_precision_k"] = best_precision_k
        
        # Overall assessment
        avg_recall = np.mean([results.get(f'recall@{k}', 0) for k in k_values])
        avg_ndcg = np.mean([results.get(f'ndcg@{k}', 0) for k in k_values])
        avg_precision = np.mean([results.get(f'precision@{k}', 0) for k in k_values])
        
        if avg_recall > 0.1 and avg_ndcg > 0.1:
            demo_results["summary"]["overall_assessment"] = "Excellent performance"
        elif avg_recall > 0.05 and avg_ndcg > 0.05:
            demo_results["summary"]["overall_assessment"] = "Good performance"
        elif avg_recall > 0.02 and avg_ndcg > 0.02:
            demo_results["summary"]["overall_assessment"] = "Fair performance"
        else:
            demo_results["summary"]["overall_assessment"] = "Poor performance - needs improvement"
    
    return demo_results

def save_demo_results(demo_results, save_dir="demo_results"):
    """Lưu demo results"""
    print(f"\n💾 Saving demo results...")
    
    # Lưu vào thư mục output của step 05
    save_dir = get_step_output_dir("05_visualization")
    
    # Lưu demo results
    with open(f"{save_dir}/demo_results.json", 'w') as f:
        json.dump(demo_results, f, indent=2)
    
    # Lưu summary
    with open(f"{save_dir}/demo_summary.txt", 'w') as f:
        f.write("LIGHTGCN DEMO & VISUALIZATION SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("MODEL PERFORMANCE:\n")
        f.write("-" * 20 + "\n")
        for metric, value in demo_results["model_performance"].items():
            f.write(f"{metric}: {value:.4f}\n")
        
        f.write(f"\nPERFORMANCE SUMMARY:\n")
        f.write("-" * 20 + "\n")
        summary = demo_results["summary"]
        f.write(f"Best Recall@K: K={summary['best_recall_k']}\n")
        f.write(f"Best NDCG@K: K={summary['best_ndcg_k']}\n")
        f.write(f"Best Precision@K: K={summary['best_precision_k']}\n")
        f.write(f"Overall Assessment: {summary['overall_assessment']}\n")
        
        f.write(f"\nTIMESTAMP: {demo_results['timestamp']}\n")
        f.write(f"\nNOTE: {demo_results['recommendations_note']}\n")
    
    print(f"✅ Saved results to {save_dir}/")
    print(f"   - demo_results.json")
    print(f"   - demo_summary.txt")

def print_demo_summary(demo_results):
    """In tóm tắt demo results"""
    if not demo_results:
        print("❌ No demo results to display")
        return
        
    print("\n🎯 DEMO RESULTS SUMMARY")
    print("=" * 60)
    
    # Model performance
    print("\n📊 MODEL PERFORMANCE:")
    for metric, value in demo_results["model_performance"].items():
        print(f"  {metric}: {value:.4f}")
    
    # Performance summary
    summary = demo_results["summary"]
    print(f"\n🏆 PERFORMANCE SUMMARY:")
    print(f"  Best Recall@K: K={summary['best_recall_k']}")
    print(f"  Best NDCG@K: K={summary['best_ndcg_k']}")
    print(f"  Best Precision@K: K={summary['best_precision_k']}")
    print(f"  Overall Assessment: {summary['overall_assessment']}")
    
    print(f"\n⏰ Generated at: {demo_results['timestamp']}")

def main():
    """Main function"""
    try:
        # Chạy evaluation để lấy results
        results = run_evaluation_for_demo()
        
        if results:
            # Tạo demo results
            demo_results = create_demo_results(results)
            
            # Tạo visualizations
            create_basic_visualizations(results)
            
            # Lưu results
            save_demo_results(demo_results)
            
            # In summary
            print_demo_summary(demo_results)
            
            print("\n🎉 DEMO & VISUALIZATION HOÀN THÀNH!")
            print("=" * 60)
            print("📁 Files created:")
            print("   - demo_results/demo_results.json")
            print("   - demo_summary.txt")
            print("   - basic_visualizations.png")
            print("\n🎯 Demo sử dụng file evaluate.py có sẵn")
            print("📊 Basic visualizations đã được tạo")
        else:
            print("❌ Demo failed. Please check the error messages above.")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print("Hãy kiểm tra lại dữ liệu và thử lại.")

if __name__ == "__main__":
    main()
