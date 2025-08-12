#!/usr/bin/env python3
"""
TEST PIPELINE SETUP
===================

Script để test và verify setup của pipeline LightGCN.
Kiểm tra:
- File structure
- Dependencies
- Utility functions
- Step scripts
"""

import os
import sys
import importlib
import subprocess

def print_header(title):
    """In header đẹp mắt"""
    print("\n" + "="*60)
    print(f"🧪 {title}")
    print("="*60)

def test_file_structure():
    """Test cấu trúc file và thư mục"""
    print_header("TESTING FILE STRUCTURE")
    
    # Kiểm tra thư mục steps/ (từ bên trong steps/)
    current_dir = os.getcwd()
    if os.path.basename(current_dir) == "steps":
        print("✅ Đang ở trong thư mục steps/")
    else:
        print("❌ Không ở trong thư mục steps/")
        return False
    
    # Kiểm tra các step scripts
    step_scripts = [
        "01_data_preprocessing.py",
        "02_build_adjacency_matrix.py", 
        "03_train_lightgcn.py",
        "04_evaluate_model.py",
        "05_demo_visualization.py"
    ]
    
    missing_scripts = []
    for script in step_scripts:
        if os.path.exists(script):
            print(f"✅ {script} tồn tại")
        else:
            print(f"❌ {script} không tồn tại")
            missing_scripts.append(script)
    
    # Kiểm tra thư mục outputs
    outputs_dir = "outputs"
    if os.path.exists(outputs_dir):
        print("✅ Thư mục outputs/ tồn tại")
    else:
        print("❌ Thư mục outputs/ không tồn tại")
        os.makedirs(outputs_dir, exist_ok=True)
        print("✅ Đã tạo thư mục outputs/")
    
    # Kiểm tra thư mục ml-1m (từ thư mục gốc)
    ml1m_dir = "../ml-1m"
    if os.path.exists(ml1m_dir):
        print("✅ Thư mục ml-1m/ tồn tại")
        
        # Kiểm tra các file dữ liệu
        data_files = ["ratings.dat", "movies.dat", "users.dat"]
        for file in data_files:
            file_path = os.path.join(ml1m_dir, file)
            if os.path.exists(file_path):
                print(f"✅ {file} tồn tại")
            else:
                print(f"❌ {file} không tồn tại")
    else:
        print("❌ Thư mục ml-1m/ không tồn tại")
        print("💡 Hãy download MovieLens-1M dataset")
    
    return len(missing_scripts) == 0

def test_utility_functions():
    """Test utility functions"""
    print_header("TESTING UTILITY FUNCTIONS")
    
    try:
        # Import utils
        sys.path.append("steps")
        from utils import (
            get_project_root,
            get_step_output_dir,
            list_step_outputs,
            get_ml1m_path
        )
        print("✅ Import utils thành công")
        
        # Test các functions
        project_root = get_project_root()
        print(f"✅ get_project_root(): {project_root}")
        
        output_dir = get_step_output_dir("01_preprocessing")
        print(f"✅ get_step_output_dir(): {output_dir}")
        
        outputs = list_step_outputs("01_preprocessing")
        print(f"✅ list_step_outputs(): {outputs}")
        
        # Test với file không tồn tại
        try:
            get_ml1m_path("nonexistent.dat")
            print("❌ get_ml1m_path() không raise error với file không tồn tại")
        except FileNotFoundError:
            print("✅ get_ml1m_path() raise error đúng với file không tồn tại")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import utils thất bại: {e}")
        return False
    except Exception as e:
        print(f"❌ Test utils thất bại: {e}")
        return False

def test_step_scripts():
    """Test các step scripts có thể import được"""
    print_header("TESTING STEP SCRIPTS")
    
    step_scripts = [
        "01_data_preprocessing",
        "02_build_adjacency_matrix",
        "03_train_lightgcn", 
        "04_evaluate_model",
        "05_demo_visualization"
    ]
    
    failed_imports = []
    
    for script in step_scripts:
        try:
            # Thêm steps/ vào path
            sys.path.insert(0, "steps")
            
            # Import module
            module = importlib.import_module(script)
            print(f"✅ {script} import thành công")
            
            # Kiểm tra main function
            if hasattr(module, 'main'):
                print(f"✅ {script} có main function")
            else:
                print(f"⚠️ {script} không có main function")
                
        except ImportError as e:
            print(f"❌ {script} import thất bại: {e}")
            failed_imports.append(script)
        except Exception as e:
            print(f"❌ {script} test thất bại: {e}")
            failed_imports.append(script)
        finally:
            # Xóa steps/ khỏi path
            if "steps" in sys.path:
                sys.path.remove("steps")
    
    return len(failed_imports) == 0

def test_dependencies():
    """Test các dependencies cần thiết"""
    print_header("TESTING DEPENDENCIES")
    
    required_packages = [
        "torch", "numpy", "pandas", "scipy", "matplotlib", 
        "seaborn", "sklearn", "tqdm", "networkx"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package} đã cài đặt")
        except ImportError:
            print(f"❌ {package} chưa cài đặt")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n💡 Cài đặt các packages còn thiếu:")
        print(f"pip install {' '.join(missing_packages)}")
    
    return len(missing_packages) == 0

def test_pipeline_runner():
    """Test pipeline runner script"""
    print_header("TESTING PIPELINE RUNNER")
    
    runner_script = "run_pipeline.py"
    if not os.path.exists(runner_script):
        print("❌ run_pipeline.py không tồn tại")
        return False
    
    print("✅ run_pipeline.py tồn tại")
    
    # Test help
    try:
        result = subprocess.run([sys.executable, runner_script, "--help"], 
                              capture_output=True,
                              text=True,
                              timeout=10)
        
        if result.returncode == 0:
            print("✅ run_pipeline.py --help chạy thành công")
            return True
        else:
            print(f"❌ run_pipeline.py --help thất bại: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ run_pipeline.py --help timeout")
        return False
    except Exception as e:
        print(f"❌ Test run_pipeline.py thất bại: {e}")
        return False

def main():
    """Main function"""
    print_header("LIGHTGCN PIPELINE SETUP TEST")
    
    tests = [
        ("File Structure", test_file_structure),
        ("Utility Functions", test_utility_functions),
        ("Step Scripts", test_step_scripts),
        ("Dependencies", test_dependencies),
        ("Pipeline Runner", test_pipeline_runner)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Tổng kết
    print_header("TEST RESULTS SUMMARY")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Pipeline is ready to use.")
        print("\n💡 Next steps:")
        print("   1. cd steps/")
        print("   2. python run_pipeline.py --clean")
        print("   3. Wait for pipeline to complete")
    else:
        print(f"\n⚠️ {total - passed} tests failed. Please fix the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
