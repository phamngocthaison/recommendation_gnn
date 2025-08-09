import subprocess
import sys

def fix_cuda_pytorch():
    """Fix PyTorch to use CUDA"""
    print("=== Fixing PyTorch CUDA Installation ===")
    
    # Check current PyTorch
    try:
        import torch
        print(f"Current PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
    except:
        print("PyTorch not found")
    
    print("\nUninstalling current PyTorch...")
    subprocess.run([sys.executable, '-m', 'pip', 'uninstall', 'torch', 'torchvision', 'torchaudio', '-y'])
    
    print("\nInstalling PyTorch with CUDA support...")
    cmd = [
        sys.executable, '-m', 'pip', 'install',
        'torch', 'torchvision', 'torchaudio',
        '--index-url', 'https://download.pytorch.org/whl/cu118'
    ]
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("\n✓ PyTorch with CUDA installed!")
        
        # Test the installation
        import torch
        print(f"New PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print("✓ CUDA is now working!")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("✗ CUDA still not available")
    else:
        print("✗ Installation failed")

if __name__ == "__main__":
    fix_cuda_pytorch() 