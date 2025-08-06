# LightGCN Movie Recommendation System

Hệ thống recommendation sử dụng Graph Neural Network (LightGCN) trên bộ dữ liệu MovieLens-1M.

## Tổng quan

LightGCN là một mô hình Graph Neural Network đơn giản và hiệu quả cho bài toán recommendation. Mô hình này:
- Loại bỏ các thành phần phức tạp như feature transformation và nonlinear activation
- Sử dụng graph convolution để học embeddings của users và items
- Đạt hiệu suất cao với ít tham số hơn các mô hình GNN khác

## Cấu trúc dự án

```
recommendation_gnn/
├── 1. pre-processing.py      # Tiền xử lý dữ liệu MovieLens-1M
├── data_loader.py            # Data loader và utility functions
├── lightgcn.py              # Implementation của LightGCN model
├── train.py                 # Script huấn luyện mô hình
├── evaluate.py              # Script đánh giá mô hình
├── demo.py                  # Demo và interactive recommendations
├── requirements.txt         # Dependencies
├── README.md               # Hướng dẫn sử dụng
├── ml-1m/                  # Dữ liệu MovieLens-1M
├── movielens_train.csv     # Dữ liệu train
├── movielens_test.csv      # Dữ liệu test
├── user2id.json           # Mapping user ID
└── item2id.json           # Mapping item ID
```

## Cài đặt

1. Clone repository và cài đặt dependencies:
```bash
pip install -r requirements.txt
```

2. Đảm bảo có bộ dữ liệu MovieLens-1M trong thư mục `ml-1m/`

## Sử dụng

### 1. Tiền xử lý dữ liệu

Chạy script tiền xử lý để chuẩn bị dữ liệu:
```bash
python "1. pre-processing.py"
```

Script này sẽ:
- Load dữ liệu từ `ml-1m/ratings.dat`
- Chuyển đổi thành implicit feedback (rating >= 4)
- Gán lại ID cho users và items
- Chia train/test (hold-out 1 interaction cuối cùng của mỗi user)
- Lưu mapping ID vào `user2id.json` và `item2id.json`

### 2. Huấn luyện mô hình

```bash
python train.py
```

Script training sẽ:
- Load dữ liệu đã tiền xử lý
- Xây dựng adjacency matrix
- Khởi tạo mô hình LightGCN
- Huấn luyện với BPR loss
- Sử dụng early stopping
- Lưu mô hình tốt nhất vào `lightgcn_movielens.pt`
- Vẽ biểu đồ training curves
- Đánh giá trên test set

### 3. Đánh giá mô hình

```bash
python evaluate.py
```

Script này sẽ:
- Load mô hình đã train
- Đánh giá với các metrics: Recall@K, NDCG@K, Precision@K
- Hiển thị kết quả và ví dụ recommendations

### 4. Demo và Interactive Mode

```bash
python demo.py
```

Script demo sẽ:
- Hiển thị recommendations cho một số users mẫu
- Cho phép nhập user ID để xem recommendations
- Hiển thị tên phim thay vì chỉ ID

## Hyperparameters

Các hyperparameters chính trong `train.py`:

- `embed_dim`: 64 (kích thước embedding)
- `n_layers`: 3 (số layer GCN)
- `dropout`: 0.1 (dropout rate)
- `lr`: 0.001 (learning rate)
- `batch_size`: 1024
- `num_epochs`: 100
- `early_stopping_patience`: 10

## Metrics

Hệ thống đánh giá với các metrics sau:
- **Recall@K**: Tỷ lệ items thực sự được recommend trong top-K
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **Precision@K**: Độ chính xác của top-K recommendations

## Kiến trúc LightGCN

LightGCN sử dụng:
1. **Graph Construction**: Xây dựng bipartite graph user-item
2. **Graph Convolution**: Propagate thông tin qua các layer
3. **Layer Combination**: Kết hợp embeddings từ tất cả layers
4. **Prediction**: Inner product của user và item embeddings

## Tùy chỉnh

### Thay đổi hyperparameters
Chỉnh sửa các giá trị trong `train.py`:
```python
embed_dim = 64      # Kích thước embedding
n_layers = 3        # Số layer
dropout = 0.1       # Dropout rate
lr = 0.001         # Learning rate
```

### Sử dụng dataset khác
1. Thay đổi format dữ liệu trong `data_loader.py`
2. Cập nhật preprocessing script
3. Điều chỉnh hyperparameters phù hợp

## Troubleshooting

### Lỗi CUDA
Nếu gặp lỗi CUDA, mô hình sẽ tự động chuyển sang CPU:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### Out of Memory
Giảm `batch_size` trong `train.py` nếu gặp lỗi memory.

### Model không converge
- Tăng `num_epochs`
- Giảm `lr`
- Tăng `early_stopping_patience`

## Kết quả mẫu

Sau khi train, bạn có thể mong đợi kết quả tương tự:
```
=== Final Test Results ===
recall@5: 0.1234
recall@10: 0.1890
recall@20: 0.2567
ndcg@5: 0.1456
ndcg@10: 0.1789
ndcg@20: 0.2012
precision@5: 0.0247
precision@10: 0.0189
precision@20: 0.0128
```

## Tài liệu tham khảo

- [LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation](https://arxiv.org/abs/2002.02126)
- [MovieLens 1M Dataset](https://grouplens.org/datasets/movielens/1m/)

## License

MIT License 