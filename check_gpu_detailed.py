import torch
import sys
import os

def check_gpu_detailed():
    """Detailed GPU and CUDA check"""
    print("=== Detailed GPU Check ===")
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Check PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    
    # Check CUDA availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
            print(f"  Compute capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
        
        # Set current device
        torch.cuda.set_device(0)
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        
        # Test CUDA tensor creation
        try:
            test_tensor = torch.randn(10, 10).cuda()
            print(f"✓ CUDA tensor creation successful: {test_tensor.device}")
        except Exception as e:
            print(f"✗ CUDA tensor creation failed: {e}")
    else:
        print("No CUDA devices available")
        
        # Check if CUDA is installed but not detected
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if result.returncode == 0:
                print("nvidia-smi found but PyTorch CUDA not available")
                print("This might indicate a PyTorch installation issue")
            else:
                print("nvidia-smi not found - no NVIDIA drivers installed")
        except:
            print("Could not check nvidia-smi")
    
    # Check environment variables
    print(f"\nEnvironment variables:")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    print(f"CUDA_HOME: {os.environ.get('CUDA_HOME', 'Not set')}")
    
    # Test device selection
    print(f"\nDevice selection test:")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Selected device: {device}")
    
    # Test tensor on device
    test_tensor = torch.randn(5, 5).to(device)
    print(f"Test tensor device: {test_tensor.device}")
    
    return torch.cuda.is_available()

if __name__ == "__main__":
    check_gpu_detailed() 