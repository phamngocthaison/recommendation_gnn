# LightGCN Recommendation Pipeline

Pipeline hoàn chỉnh để xây dựng và training LightGCN model cho recommendation system sử dụng MovieLens-1M dataset.

## 🏗️ Cấu trúc Pipeline

Pipeline được chia thành 5 steps tuần tự:

```
steps/
├── 01_data_preprocessing.py      # Tiền xử lý dữ liệu MovieLens-1M
├── 02_build_adjacency_matrix.py  # Xây dựng adjacency matrix
├── 03_train_lightgcn.py         # Training LightGCN model
├── 04_evaluate_model.py          # Evaluate model performance
├── 05_demo_visualization.py      # Demo và visualization
├── utils.py                      # Utility functions
├── run_pipeline.py               # Script chính để chạy pipeline
└── test_pipeline.py              # Script test setup
```

## 📁 Output Structure

Mỗi step sẽ tạo outputs trong thư mục riêng:

```
steps/outputs/
├── 01_preprocessing/             # Training data, test data, ID mappings
├── 02_adjacency/                 # Normalized adjacency matrix
├── 03_training/                  # Model weights, training curves, logs
├── 04_evaluation/                # Evaluation results, metrics
└── 05_visualization/             # Visualizations, demo results
```

## 🚀 Quick Start

### 1. Kiểm tra setup

```bash
cd steps/
python test_pipeline.py
```

### 2. Chạy toàn bộ pipeline

```bash
cd steps/
python run_pipeline.py --clean
```

### 3. Chạy từ step cụ thể

```bash
cd steps/
python run_pipeline.py --step 3  # Bắt đầu từ step 3
```

## 📋 Step Details

### Step 1: Data Preprocessing
- **Input**: `ml-1m/ratings.dat`, `ml-1m/movies.dat`, `ml-1m/users.dat`
- **Output**: 
  - `movielens_train.csv`: Training interactions
  - `movielens_test.csv`: Testing interactions  
  - `user2id.json`: User ID mapping
  - `item2id.json`: Movie ID mapping
- **Process**: Convert to implicit feedback (rating ≥ 4), create ID mappings, train/test split

### Step 2: Build Adjacency Matrix
- **Input**: Outputs từ Step 1
- **Output**: `norm_adj_matrix.npz` (normalized adjacency matrix)
- **Process**: Build user-item bipartite graph, normalize adjacency matrix

### Step 3: Train LightGCN Model
- **Input**: Outputs từ Step 1 và 2
- **Output**: 
  - `lightgcn_movielens.pt`: Trained model weights
  - `training_curves.png`: Training curves
  - `training_log.json`: Training logs
- **Process**: Initialize LightGCN, train with BPR loss, early stopping

### Step 4: Evaluate Model
- **Input**: Outputs từ Step 1, 2, và 3
- **Output**: 
  - `evaluation_results.json`: Evaluation metrics
  - `evaluation_metrics.png`: Metrics comparison
- **Process**: Calculate Recall@K, NDCG@K, Precision@K (K = 5, 10, 20)

### Step 5: Demo & Visualization
- **Input**: Outputs từ tất cả steps trước
- **Output**: 
  - `embedding_visualization.png`: T-SNE embedding plot
  - `movie_network.png`: Movie similarity network
  - `interaction_heatmap.png`: User-item interaction heatmap
  - `demo_results.json`: Sample recommendations

## 🛠️ Usage Options

### Pipeline Runner Options

```bash
python run_pipeline.py [OPTIONS]

Options:
  --step STEP     Start from specific step (1-5, default: 1)
  --force         Skip dependency checks
  --clean         Clean all outputs before running
  --help          Show help message
```

### Examples

```bash
# Chạy toàn bộ pipeline
python run_pipeline.py --clean

# Chạy từ step 3 (bỏ qua step 1-2)
python run_pipeline.py --step 3

# Bỏ qua kiểm tra dependencies
python run_pipeline.py --step 4 --force

# Xóa outputs cũ và chạy lại
python run_pipeline.py --clean
```

## 🔧 Dependencies

### Required Packages

```bash
pip install torch numpy pandas scipy matplotlib seaborn scikit-learn tqdm networkx
```

### Dataset

Download MovieLens-1M dataset và đặt trong thư mục `ml-1m/`:

```
ml-1m/
├── ratings.dat
├── movies.dat
└── users.dat
```

## 📊 Expected Results

### Model Performance
- **Dataset**: MovieLens-1M (6,040 users, 3,900 movies, ~1M ratings)
- **Model**: LightGCN (3 layers, 64 dimensions)
- **Expected Metrics**:
  - Recall@10: ~0.15-0.25
  - NDCG@10: ~0.12-0.20
  - Precision@10: ~0.08-0.15

### Training Time
- **Hardware**: CPU: ~30-60 minutes, GPU: ~5-15 minutes
- **Memory**: ~2-4 GB RAM

## 🐛 Troubleshooting

### Common Issues

1. **Import Error**: Đảm bảo đang chạy từ thư mục `steps/`
2. **File Not Found**: Kiểm tra MovieLens-1M dataset đã được download
3. **Memory Error**: Giảm batch size trong training script
4. **CUDA Error**: Kiểm tra PyTorch CUDA installation

### Debug Commands

```bash
# Test setup
python test_pipeline.py

# Check outputs
python print_outputs_summary.py

# Run single step
python 01_data_preprocessing.py
```

## 📚 References

- **Paper**: [LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation](https://arxiv.org/abs/2002.02126)
- **Dataset**: [MovieLens 1M Dataset](https://grouplens.org/datasets/movielens/1m/)
- **Implementation**: Based on LightGCN paper by He et al. (SIGIR 2020)

## 🤝 Contributing

Để cải thiện pipeline:

1. Fork repository
2. Tạo feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## 📄 License

MIT License - xem file LICENSE để biết thêm chi tiết.
