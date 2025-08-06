import subprocess
import sys
import os

def run_webapp():
    """Run the Streamlit web app"""
    print("🎬 Starting LightGCN Movie Recommendation Web App...")
    print("=" * 60)
    
    # Check if model exists
    model_files = ["lightgcn_movielens_gpu.pt", "lightgcn_movielens.pt"]
    model_exists = False
    
    for model_file in model_files:
        if os.path.exists(model_file):
            print(f"✅ Found model: {model_file}")
            model_exists = True
            break
    
    if not model_exists:
        print("❌ No model file found!")
        print("Please run training first:")
        print("  python train_gpu.py  # For GPU training")
        print("  python train.py      # For CPU training")
        return
    
    # Check if required files exist
    required_files = ["user2id.json", "item2id.json", "movielens_train.csv"]
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        print("Please run preprocessing first:")
        print("  python '1. pre-processing.py'")
        return
    
    print("✅ All required files found!")
    print("\n🚀 Starting web app...")
    print("📱 The app will open in your browser at: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the app")
    print("=" * 60)
    
    try:
        # Run streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "web_app.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 Web app stopped!")
    except Exception as e:
        print(f"❌ Error running web app: {e}")
        print("\n💡 Make sure you have installed the required packages:")
        print("  pip install -r requirements_web.txt")

if __name__ == "__main__":
    run_webapp() 