#!/usr/bin/env python3
"""
🧪 TEST PIPELINE SETUP
=======================

Script đơn giản để kiểm tra xem pipeline có được setup đúng không.
Chạy script này trước khi chạy pipeline chính.
"""

import os
import sys

def check_directory_structure():
    """Kiểm tra cấu trúc thư mục"""
    print("🔍 Kiểm tra cấu trúc thư mục...")
    
    current_dir = os.getcwd()
    print(f"📁 Thư mục hiện tại: {current_dir}")
    
    # Kiểm tra các thư mục và files quan trọng
    important_items = [
        ('ml-1m/', 'MovieLens dataset directory'),
        ('steps/', 'Pipeline steps directory'),
        ('README.md', 'Project documentation'),
        ('requirements.txt', 'Dependencies file')
    ]
    
    for item, description in important_items:
        if os.path.exists(item):
            if os.path.isdir(item):
                print(f"✅ {description}: {item}")
            else:
                file_size = os.path.getsize(item) / 1024  # KB
                print(f"✅ {description}: {item} ({file_size:.1f} KB)")
        else:
            print(f"❌ {description}: {item} - KHÔNG TỒN TẠI")
    
    return True

def check_ml1m_dataset():
    """Kiểm tra MovieLens dataset"""
    print("\n🎬 Kiểm tra MovieLens dataset...")
    
    ml1m_dir = '../ml-1m'
    if not os.path.exists(ml1m_dir):
        print(f"❌ Thư mục {ml1m_dir} không tồn tại!")
        return False
    
    required_files = [
        ('ratings.dat', 'User ratings data'),
        ('movies.dat', 'Movie metadata'),
        ('users.dat', 'User metadata')
    ]
    
    for filename, description in required_files:
        filepath = os.path.join(ml1m_dir, filename)
        if os.path.exists(filepath):
            file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
            print(f"✅ {description}: {filename} ({file_size:.1f} MB)")
        else:
            print(f"❌ {description}: {filename} - KHÔNG TỒN TẠI")
            return False
    
    return True

def check_steps_directory():
    """Kiểm tra thư mục steps"""
    print("\n🔧 Kiểm tra pipeline steps...")
    
    steps_dir = '../steps'
    if not os.path.exists(steps_dir):
        print(f"❌ Thư mục {steps_dir} không tồn tại!")
        return False
    
    required_scripts = [
        ('01_data_preprocessing.py', 'Data preprocessing'),
        ('02_build_adjacency_matrix.py', 'Build adjacency matrix'),
        ('03_train_lightgcn.py', 'Train LightGCN model'),
        ('04_evaluate_model.py', 'Evaluate model'),
        ('05_demo_visualization.py', 'Demo & visualization')
    ]
    
    for script, description in required_scripts:
        script_path = os.path.join(steps_dir, script)
        if os.path.exists(script_path):
            file_size = os.path.getsize(script_path) / 1024  # KB
            print(f"✅ {description}: {script} ({file_size:.1f} KB)")
        else:
            print(f"❌ {description}: {script} - KHÔNG TỒN TẠI")
            return False
    
    return True

def check_python_environment():
    """Kiểm tra Python environment"""
    print("\n🐍 Kiểm tra Python environment...")
    
    print(f"✅ Python version: {sys.version}")
    
    # Kiểm tra các packages quan trọng
    required_packages = [
        'torch', 'numpy', 'pandas', 'matplotlib', 
        'scikit-learn', 'scipy', 'tqdm'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Thiếu packages: {', '.join(missing_packages)}")
        print("Hãy cài đặt: pip install " + " ".join(missing_packages))
        return False
    
    return True

def test_file_access():
    """Test khả năng truy cập file"""
    print("\n📁 Test khả năng truy cập file...")
    
    try:
        # Test đọc ratings.dat
        ratings_path = '../ml-1m/ratings.dat'
        if os.path.exists(ratings_path):
            try:
                with open(ratings_path, 'r', encoding='latin-1') as f:
                    first_line = f.readline().strip()
                    print(f"✅ Đọc được ratings.dat: {first_line[:50]}...")
            except UnicodeDecodeError:
                # Thử với encoding khác
                with open(ratings_path, 'r', encoding='cp1252') as f:
                    first_line = f.readline().strip()
                    print(f"✅ Đọc được ratings.dat: {first_line[:50]}...")
        else:
            print("❌ Không thể đọc ratings.dat")
            return False
        
        # Test đọc movies.dat
        movies_path = '../ml-1m/movies.dat'
        if os.path.exists(movies_path):
            try:
                with open(movies_path, 'r', encoding='latin-1') as f:
                    first_line = f.readline().strip()
                    print(f"✅ Đọc được movies.dat: {first_line[:50]}...")
            except UnicodeDecodeError:
                # Thử với encoding khác
                with open(movies_path, 'r', encoding='cp1252') as f:
                    first_line = f.readline().strip()
                    print(f"✅ Đọc được movies.dat: {first_line[:50]}...")
        else:
            print("❌ Không thể đọc movies.dat")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Lỗi khi test file access: {e}")
        return False

def main():
    """Main function"""
    print("🧪 TEST PIPELINE SETUP")
    print("=" * 50)
    
    all_tests_passed = True
    
    # Chạy các tests
    if not check_directory_structure():
        all_tests_passed = False
    
    if not check_ml1m_dataset():
        all_tests_passed = False
    
    if not check_steps_directory():
        all_tests_passed = False
    
    if not check_python_environment():
        all_tests_passed = False
    
    if not test_file_access():
        all_tests_passed = False
    
    # Kết quả
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("🎉 TẤT CẢ TESTS ĐỀU PASSED!")
        print("✅ Pipeline đã sẵn sàng để chạy")
        print("\n🚀 Bạn có thể chạy:")
        print("   python main.py                    # Pipeline tự động")
        print("   python run_pipeline.py            # Pipeline tương tác")
        print("   python steps/01_data_preprocessing.py  # Từng bước")
    else:
        print("❌ MỘT SỐ TESTS FAILED!")
        print("⚠️ Hãy sửa các vấn đề trước khi chạy pipeline")
        print("\n🔧 Các bước cần làm:")
        print("   1. Đảm bảo đang ở thư mục gốc của project")
        print("   2. Download MovieLens-1M dataset")
        print("   3. Cài đặt dependencies: pip install -r requirements.txt")
    
    print("=" * 50)

if __name__ == "__main__":
    main()
