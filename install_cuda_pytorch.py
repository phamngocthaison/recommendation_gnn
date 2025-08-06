import subprocess
import sys
import os

def check_and_install_cuda_pytorch():
    """Check CUDA availability and install PyTorch with CUDA if needed"""
    print("=== CUDA PyTorch Installation Check ===")
    
    # Check if nvidia-smi is available
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ NVIDIA GPU detected")
            print("GPU Info:")
            print(result.stdout)
        else:
            print("✗ nvidia-smi not found - no NVIDIA GPU or drivers")
            return False
    except FileNotFoundError:
        print("✗ nvidia-smi not found - no NVIDIA GPU or drivers")
        return False
    
    # Check current PyTorch installation
    try:
        import torch
        print(f"\nCurrent PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print("✓ PyTorch with CUDA support is already installed")
            return True
        else:
            print("✗ PyTorch installed but CUDA not available")
            print("This might be a CPU-only PyTorch installation")
            
    except ImportError:
        print("✗ PyTorch not installed")
    
    # Offer to install PyTorch with CUDA
    print("\nWould you like to install PyTorch with CUDA support?")
    print("This will install PyTorch 2.7.1 with CUDA 11.8 support")
    
    # Auto-install for convenience
    print("\nInstalling PyTorch with CUDA support...")
    try:
        # Uninstall current PyTorch if exists
        subprocess.run([sys.executable, '-m', 'pip', 'uninstall', 'torch', 'torchvision', 'torchaudio', '-y'], 
                      capture_output=True)
        
        # Install PyTorch with CUDA
        install_cmd = [
            sys.executable, '-m', 'pip', 'install', 
            'torch', 'torchvision', 'torchaudio', 
            '--index-url', 'https://download.pytorch.org/whl/cu118'
        ]
        
        print("Running:", ' '.join(install_cmd))
        result = subprocess.run(install_cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ PyTorch with CUDA installed successfully")
            
            # Verify installation
            import torch
            print(f"New PyTorch version: {torch.__version__}")
            print(f"CUDA available: {torch.cuda.is_available()}")
            
            if torch.cuda.is_available():
                print("✓ CUDA is now available!")
                return True
            else:
                print("✗ CUDA still not available after installation")
                return False
        else:
            print("✗ Installation failed")
            print("Error:", result.stderr)
            return False
            
    except Exception as e:
        print(f"✗ Installation error: {e}")
        return False

if __name__ == "__main__":
    check_and_install_cuda_pytorch() 