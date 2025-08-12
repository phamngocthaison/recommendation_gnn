#!/usr/bin/env python3
"""
LIGHTGCN PIPELINE WRAPPER
==========================

Script wrapper để chạy pipeline từ thư mục gốc của project.
Script này sẽ chuyển vào thư mục steps/ và chạy pipeline chính.
"""

import os
import sys
import subprocess

def main():
    """Main function"""
    # Kiểm tra thư mục steps/ tồn tại
    steps_dir = "steps"
    if not os.path.exists(steps_dir):
        print("❌ Thư mục steps/ không tồn tại!")
        print("💡 Hãy đảm bảo bạn đang ở thư mục gốc của project")
        sys.exit(1)
    
    # Chuyển vào thư mục steps/
    os.chdir(steps_dir)
    print(f"📁 Changed to directory: {os.getcwd()}")
    
    # Chạy pipeline với tất cả arguments
    pipeline_script = "run_pipeline.py"
    if not os.path.exists(pipeline_script):
        print(f"❌ {pipeline_script} không tồn tại trong thư mục steps/")
        sys.exit(1)
    
    # Chuyển tất cả arguments cho pipeline script
    args = [sys.executable, pipeline_script] + sys.argv[1:]
    
    print(f"🚀 Running: {' '.join(args)}")
    print("=" * 60)
    
    try:
        # Chạy pipeline
        result = subprocess.run(args, check=True)
        sys.exit(result.returncode)
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Pipeline failed with return code {e.returncode}")
        sys.exit(e.returncode)
        
    except KeyboardInterrupt:
        print("\n⏹️ Pipeline interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
