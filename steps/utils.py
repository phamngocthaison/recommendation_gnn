#!/usr/bin/env python3
"""
UTILITY FUNCTIONS FOR LIGHTGCN PIPELINE
=======================================

Các utility functions để xử lý đường dẫn file một cách nhất quán
trong tất cả các step scripts.
"""

import os

def get_project_root():
    """Lấy đường dẫn đến thư mục gốc của project"""
    current_dir = os.getcwd()
    
    # Nếu đang ở thư mục steps/, đi lên một cấp
    if os.path.basename(current_dir) == 'steps':
        return os.path.dirname(current_dir)
    
    # Nếu đang ở thư mục gốc
    if os.path.exists('ml-1m') and os.path.exists('steps'):
        return current_dir
    
    # Thử tìm thư mục gốc
    for parent in [os.path.dirname(current_dir), os.path.dirname(os.path.dirname(current_dir))]:
        if os.path.exists(os.path.join(parent, 'ml-1m')) and os.path.exists(os.path.join(parent, 'steps')):
            return parent
    
    return current_dir

def resolve_path(file_path):
    """
    Giải quyết đường dẫn file một cách thông minh
    
    Args:
        file_path (str): Đường dẫn file (có thể là relative hoặc absolute)
    
    Returns:
        str: Đường dẫn tuyệt đối đến file
    """
    project_root = get_project_root()
    
    # Nếu file_path đã là absolute path
    if os.path.isabs(file_path):
        return file_path
    
    # Thử đường dẫn tương đối từ thư mục hiện tại
    if os.path.exists(file_path):
        return os.path.abspath(file_path)
    
    # Thử đường dẫn từ project root
    full_path = os.path.join(project_root, file_path)
    if os.path.exists(full_path):
        return full_path
    
    # Thử đường dẫn tương đối từ thư mục steps/
    current_dir = os.getcwd()
    if os.path.basename(current_dir) == 'steps':
        steps_path = os.path.join(current_dir, file_path)
        if os.path.exists(steps_path):
            return os.path.abspath(steps_path)
    
    # Nếu không tìm thấy, trả về đường dẫn gốc
    return file_path

def check_file_exists(file_path, description=""):
    """
    Kiểm tra file tồn tại và trả về đường dẫn đúng
    
    Args:
        file_path (str): Đường dẫn file cần kiểm tra
        description (str): Mô tả file để hiển thị trong error message
    
    Returns:
        str: Đường dẫn đúng đến file
        
    Raises:
        FileNotFoundError: Nếu file không tồn tại
    """
    resolved_path = resolve_path(file_path)
    
    if not os.path.exists(resolved_path):
        if description:
            raise FileNotFoundError(f"File {description} không tồn tại: {file_path}")
        else:
            raise FileNotFoundError(f"File không tồn tại: {file_path}")
    
    return resolved_path

def get_ml1m_path(filename):
    """
    Lấy đường dẫn đến file trong thư mục ml-1m
    
    Args:
        filename (str): Tên file (ví dụ: 'ratings.dat', 'movies.dat')
    
    Returns:
        str: Đường dẫn đúng đến file
    """
    return check_file_exists(f'ml-1m/{filename}', f'ml-1m/{filename}')

def get_output_path(filename):
    """
    Lấy đường dẫn đến output file
    
    Args:
        filename (str): Tên file output
    
    Returns:
        str: Đường dẫn đúng đến file
    """
    return resolve_path(filename)

def get_step_output_dir(step_name):
    """
    Lấy đường dẫn đến thư mục output của step cụ thể
    
    Args:
        step_name (str): Tên step (ví dụ: '01_preprocessing', '02_adjacency')
    
    Returns:
        str: Đường dẫn đến thư mục output
    """
    project_root = get_project_root()
    output_dir = os.path.join(project_root, 'steps', 'outputs', step_name)
    
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir

def get_step_output_path(step_name, filename):
    """
    Lấy đường dẫn đầy đủ đến file output trong thư mục của step
    
    Args:
        step_name (str): Tên step (ví dụ: '01_preprocessing', '02_adjacency')
        filename (str): Tên file output
    
    Returns:
        str: Đường dẫn đầy đủ đến file output
    """
    output_dir = get_step_output_dir(step_name)
    return os.path.join(output_dir, filename)

def get_input_from_previous_step(step_name, filename):
    """
    Lấy đường dẫn đến file input từ step trước đó
    
    Args:
        step_name (str): Tên step trước đó (ví dụ: '01_preprocessing')
        filename (str): Tên file cần lấy
    
    Returns:
        str: Đường dẫn đến file input
    """
    return get_step_output_path(step_name, filename)

def list_step_outputs(step_name):
    """
    Liệt kê tất cả files trong thư mục output của step
    
    Args:
        step_name (str): Tên step
    
    Returns:
        list: Danh sách tên files
    """
    output_dir = get_step_output_dir(step_name)
    if os.path.exists(output_dir):
        return [f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))]
    return []

def clear_step_outputs(step_name):
    """
    Xóa tất cả files trong thư mục output của step
    
    Args:
        step_name (str): Tên step
    """
    output_dir = get_step_output_dir(step_name)
    if os.path.exists(output_dir):
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print(f"🧹 Cleared all outputs from {step_name}")

def print_step_outputs_summary():
    """
    In tổng quan về tất cả outputs của các steps
    """
    project_root = get_project_root()
    outputs_dir = os.path.join(project_root, 'steps', 'outputs')
    
    if not os.path.exists(outputs_dir):
        print("❌ No outputs directory found")
        return
    
    print("\n📁 STEP OUTPUTS SUMMARY")
    print("=" * 50)
    
    for step_dir in sorted(os.listdir(outputs_dir)):
        step_path = os.path.join(outputs_dir, step_dir)
        if os.path.isdir(step_path):
            files = [f for f in os.listdir(step_path) if os.path.isfile(os.path.join(step_path, f))]
            print(f"\n{step_dir.upper()}:")
            if files:
                for file in sorted(files):
                    file_path = os.path.join(step_path, file)
                    file_size = os.path.getsize(file_path)
                    print(f"  📄 {file} ({file_size:,} bytes)")
            else:
                print("  (no files)")
