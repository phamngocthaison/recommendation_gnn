# 🎬 LightGCN Movie Recommendation System

A Graph Neural Network (GNN) based movie recommendation system using LightGCN architecture, trained on the MovieLens-1M dataset with GPU acceleration support.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Web App Demo](#web-app-demo)
- [Model Architecture](#model-architecture)
- [Performance](#performance)
- [Customization](#customization)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

This project implements a **LightGCN (Light Graph Convolutional Network)** for movie recommendations using the MovieLens-1M dataset. The system provides both command-line tools and a beautiful web interface for exploring recommendations.

### Key Features:
- **LightGCN Architecture**: Efficient graph neural network for recommendations
- **GPU Acceleration**: Full CUDA support for faster training
- **Interactive Web App**: Beautiful Streamlit interface with real-time recommendations
- **Comprehensive Evaluation**: Multiple metrics (Recall@K, NDCG@K, Precision@K)
- **Early Stopping**: Prevents overfitting during training
- **Movie Information**: Real movie titles, years, and genres from MovieLens dataset

## ✨ Features

### 🚀 Core Features
- **Graph Neural Network**: LightGCN implementation with configurable layers
- **Implicit Feedback**: Treats ratings ≥ 4 as positive interactions
- **BPR Loss**: Bayesian Personalized Ranking for training
- **GPU Training**: Optimized for NVIDIA GPUs with CUDA support
- **Model Persistence**: Save and load trained models

### 🎨 Web App Features
- **Interactive Dashboard**: Real-time statistics and visualizations
- **User Recommendations**: Get personalized movie recommendations
- **Movie Details**: Display real movie titles, years, and genres
- **Model Analysis**: User embeddings visualization and similarity search
- **Random Demos**: Quick exploration of random user recommendations
- **Beautiful UI**: Modern gradient design with responsive layout

### 📊 Evaluation Features
- **Multiple Metrics**: Recall@K, NDCG@K, Precision@K (K=5,10,20)
- **Training Curves**: Visualize training progress
- **Performance Analysis**: Comprehensive model evaluation
- **Early Stopping**: Automatic training termination to prevent overfitting

## 📁 Project Structure

```
recommendation_gnn/
├── 📄 README.md                    # This file
├── 📄 requirements.txt             # Core dependencies
├── 📄 requirements_web.txt         # Web app dependencies
├── 🎬 web_app.py                   # Streamlit web application
├── 🚀 run_webapp.py                # Web app launcher
├── 🧠 lightgcn.py                  # LightGCN model implementation
├── 📊 data_loader.py               # Data loading and preprocessing
├── 🎯 evaluate.py                  # Model evaluation functions
├── 🏋️ train.py                     # CPU training script
├── ⚡ train_gpu.py                 # GPU training script (optimized)
├── 🎮 demo.py                      # Command-line demo
├── 🔧 steps/                       # Step-by-step pipeline
│   ├── 01_data_preprocessing.py    # Step 1: Data preprocessing
│   ├── 02_build_adjacency_matrix.py # Step 2: Build adjacency matrix
│   ├── 03_train_lightgcn.py       # Step 3: Train LightGCN model
│   ├── 04_evaluate_model.py        # Step 4: Evaluate model
│   └── 05_demo_visualization.py    # Step 5: Demo & visualization
├── 🚀 run_pipeline.py              # Interactive pipeline runner
├── 📊 movielens_train.csv          # Training data
├── 📊 movielens_test.csv           # Test data
├── 🗂️ user2id.json                 # User ID mappings
├── 🗂️ item2id.json                 # Item ID mappings
├── 📁 ml-1m/                       # MovieLens dataset
│   ├── movies.dat                  # Movie information
│   ├── ratings.dat                 # User ratings
│   └── users.dat                   # User information
└── 🧪 Debug & Test Files
    ├── debug_movie_mapping.py      # Movie ID mapping debug
    ├── test_movie_mapping.py       # Movie mapping test
    ├── check_gpu.py                # GPU availability check
    ├── check_gpu_detailed.py       # Detailed GPU diagnostics
    ├── fix_cuda_pytorch.py         # PyTorch CUDA fix
    └── test_fix.py                 # Comprehensive test script
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- NVIDIA GPU (optional, for GPU acceleration)
- CUDA 11.8+ (if using GPU)

### 1. Clone the Repository
```bash
git clone <repository-url>
cd recommendation_gnn
```

### 2. Install Dependencies

#### Core Dependencies
```bash
pip install -r requirements.txt
```

#### Web App Dependencies
```bash
pip install -r requirements_web.txt
```

#### GPU Support (Optional)
If you have an NVIDIA GPU and want GPU acceleration:
```bash
# Uninstall CPU-only PyTorch
pip uninstall torch torchvision torchaudio

# Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 🚀 Quick Start

> 📖 **Xem [QUICK_START.md](QUICK_START.md) để hướng dẫn chi tiết và troubleshooting!**

### 🚀 Option 1: Complete Pipeline (Recommended)
```bash
python main.py
```
This will run the entire pipeline automatically from start to finish!

### 🔧 Option 2: Interactive Step-by-Step Pipeline
```bash
python run_pipeline.py
```
Chạy từng bước với xác nhận từng bước, kiểm tra output files.

### 🔧 Option 3: Manual Step-by-Step Pipeline
```bash
# Step 1: Preprocess Data
python steps/01_data_preprocessing.py

# Step 2: Build Adjacency Matrix
python steps/02_build_adjacency_matrix.py

# Step 3: Train LightGCN Model
python steps/03_train_lightgcn.py

# Step 4: Evaluate Model
python steps/04_evaluate_model.py

# Step 5: Demo & Visualization
python steps/05_demo_visualization.py
```

### 🎯 Option 3: Individual Scripts
```bash
# Legacy data preprocessing
python "1. pre-processing.py"

# Direct training
python train_gpu.py  # GPU (faster)
python train.py      # CPU

# Evaluation and demo
python evaluate.py
python demo.py
```

### 🌐 Web App Demo
```bash
python run_webapp.py
```
Then open: http://localhost:8501

## 📖 Usage

### 🚀 Complete Pipeline (Recommended)
```bash
python main.py
```
Runs the complete pipeline from data preprocessing to evaluation:
1. **Data Preprocessing**: Load and process MovieLens-1M dataset
2. **Build Adjacency Matrix**: Create normalized graph structure
3. **Train LightGCN Model**: Train with GPU acceleration
4. **Evaluate Model**: Calculate performance metrics
5. **Demo & Visualization**: Interactive exploration

### 🔧 Step-by-Step Pipeline

#### Interactive Runner (Recommended)
```bash
python run_pipeline.py          # Chạy từng bước với xác nhận
python run_pipeline.py --auto   # Chạy tự động tất cả các bước
```

#### Manual Execution
```bash
# Step 1: Data Preprocessing
python steps/01_data_preprocessing.py

# Step 2: Build Adjacency Matrix  
python steps/02_build_adjacency_matrix.py

# Step 3: Train LightGCN Model
python steps/03_train_lightgcn.py

# Step 4: Evaluate Model
python steps/04_evaluate_model.py

# Step 5: Demo & Visualization
python steps/05_demo_visualization.py
```

### 🎯 Individual Scripts

#### Data Preprocessing
```bash
python "1. pre-processing.py"  # Legacy script
```
- Loads MovieLens-1M dataset
- Creates user/item ID mappings
- Splits data into train/test sets
- Generates CSV files for training

#### Model Training

#### GPU Training (Faster)
```bash
python train_gpu.py
```
Features:
- Automatic GPU detection
- Memory optimization
- Early stopping
- Training curves visualization
- Model checkpointing

#### CPU Training
```bash
python train.py
```
- Suitable for systems without GPU
- Same functionality as GPU version

#### Model Evaluation
```bash
python evaluate.py
```
Evaluates the trained model using:
- Recall@K (K=5,10,20)
- NDCG@K (K=5,10,20)
- Precision@K (K=5,10,20)

#### Interactive Demo
```bash
python demo.py
```
- Shows recommendations for sample users
- Interactive mode for custom user IDs
- Displays movie titles and scores

## 🌐 Web App Demo

The web app provides a beautiful, interactive interface for exploring the recommendation system.

### Features:

#### 🏠 Dashboard
- **Model Overview**: Statistics about users, movies, and interactions
- **User Activity Distribution**: Histogram of user interaction patterns
- **Popular Genres**: Top movie genres visualization

#### 🔍 User Recommendations
- **User Selection**: Choose any user ID (0-6037)
- **Personalized Recommendations**: Get top-K recommendations
- **User Profile**: View user's training interactions
- **Movie Details**: Real movie titles, years, and genres
- **Score Visualization**: Bar chart of recommendation scores

#### 📊 Model Analysis
- **Model Architecture**: Embedding dimensions, layers, parameters
- **User Embeddings**: 2D PCA visualization of user representations
- **Similar Users**: Find users with similar preferences
- **Cosine Similarity**: Interactive similarity search

#### 🎲 Random Recommendations
- **Random User Demo**: Explore recommendations for random users
- **Quick Exploration**: Fast way to see system capabilities

### Running the Web App:
```bash
python run_webapp.py
```

The app will:
- ✅ Check for required files
- ✅ Load the trained model
- ✅ Start Streamlit server
- ✅ Open browser automatically

Access at: http://localhost:8501

## 🧠 Model Architecture

### LightGCN
- **Architecture**: Light Graph Convolutional Network
- **Embedding Dimension**: 64 (configurable)
- **GCN Layers**: 3 (configurable)
- **Activation**: None (linear propagation)
- **Loss Function**: BPR (Bayesian Personalized Ranking)

### Key Components:
1. **User/Item Embeddings**: Learnable representations
2. **Graph Convolution**: Message passing on user-item graph
3. **Layer Aggregation**: Combine embeddings from all layers
4. **Prediction**: Inner product for recommendation scores

### Training Process:
- **Optimizer**: Adam
- **Learning Rate**: 0.001 (with ReduceLROnPlateau scheduler)
- **Batch Size**: 2048
- **Epochs**: 100 (with early stopping)
- **Regularization**: L2 weight decay

## 📈 Performance

### Model Performance (GPU Training)
```
=== Final Test Results (GPU) ===
recall@5: 0.0207
ndcg@5: 0.0128
precision@5: 0.0041
recall@10: 0.0411
ndcg@10: 0.0192
precision@10: 0.0041
recall@20: 0.0730
ndcg@20: 0.0272
precision@20: 0.0037
```

### Training Efficiency:
- **GPU Training**: ~18 epochs (with early stopping)
- **Training Time**: ~2-3 minutes on RTX GPU
- **Memory Usage**: ~2GB GPU memory
- **Model Size**: ~1.2MB

## ⚙️ Customization

### Hyperparameters
Edit `train_gpu.py` or `train.py`:
```python
# Model parameters
embed_dim = 64          # Embedding dimension
n_layers = 3           # Number of GCN layers

# Training parameters
learning_rate = 0.001  # Learning rate
batch_size = 2048      # Batch size
epochs = 100           # Maximum epochs
patience = 10          # Early stopping patience
```

### Dataset
To use a different dataset:
1. Replace `ml-1m/` folder with your dataset
2. Update data loading in `data_loader.py`
3. Modify preprocessing in `1. pre-processing.py`

### Model Architecture
Modify `lightgcn.py`:
- Change embedding dimensions
- Add/remove GCN layers
- Implement different aggregation methods
- Add regularization techniques

## 🔧 Troubleshooting

### Common Issues

#### 1. GPU Not Detected
```bash
# Check GPU availability
python check_gpu.py

# Detailed diagnostics
python check_gpu_detailed.py

# Fix PyTorch CUDA installation
python fix_cuda_pytorch.py
```

#### 2. Movie Names Not Showing
```bash
# Debug movie mapping
python debug_movie_mapping.py

# Test movie mapping
python test_movie_mapping.py
```

#### 3. Memory Issues
- Reduce batch size in training scripts
- Use CPU training if GPU memory is insufficient
- Close other GPU applications

#### 4. Web App Issues
```bash
# Install web dependencies
pip install -r requirements_web.txt

# Check Streamlit installation
streamlit --version
```

### Error Solutions

#### PyTorch CUDA Issues
```bash
# Uninstall CPU PyTorch
pip uninstall torch torchvision torchaudio

# Install CUDA PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Missing Dependencies
```bash
# Install all dependencies
pip install -r requirements.txt
pip install -r requirements_web.txt
```

#### File Not Found Errors
```bash
# Run preprocessing first
python "1. pre-processing.py"

# Check file existence
ls -la *.json *.csv
```

## 📚 Dependencies

### Core Dependencies
- `torch>=1.13.0`: PyTorch deep learning framework
- `pandas>=1.5.0`: Data manipulation
- `numpy>=1.24.0`: Numerical computing
- `scipy>=1.10.0`: Scientific computing
- `scikit-learn>=1.3.0`: Machine learning utilities

### Web App Dependencies
- `streamlit>=1.28.0`: Web application framework
- `plotly>=5.15.0`: Interactive visualizations
- `matplotlib>=3.7.0`: Static plotting
- `seaborn>=0.12.0`: Statistical visualizations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **LightGCN Paper**: He et al. "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation"
- **MovieLens Dataset**: GroupLens Research Group
- **PyTorch**: Facebook AI Research
- **Streamlit**: Streamlit Inc.

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review existing issues
3. Create a new issue with detailed information

---

**🎬 Enjoy exploring movie recommendations with LightGCN!** 