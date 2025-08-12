# 🚀 QUICK START GUIDE

## ⚡ Cách nhanh nhất để bắt đầu

### 1. Chạy toàn bộ pipeline tự động
```bash
python main.py
```
**Đây là cách đơn giản nhất!** Script sẽ tự động chạy tất cả các bước từ đầu đến cuối.

### 2. Chạy pipeline từng bước với kiểm soát
```bash
python run_pipeline.py --auto
```
Chạy tự động tất cả các bước, nhưng bạn có thể theo dõi tiến trình.

### 3. Chạy từng bước với xác nhận
```bash
python run_pipeline.py
```
Chạy từng bước một, yêu cầu xác nhận trước khi chuyển sang bước tiếp theo.

## 📋 Prerequisites

Trước khi chạy, hãy đảm bảo:

1. **Python 3.8+** đã được cài đặt
2. **MovieLens-1M dataset** đã được download vào thư mục `ml-1m/`
3. **Dependencies** đã được cài đặt:
   ```bash
   pip install -r requirements.txt
   ```

## 🔍 Kiểm tra setup

```bash
# Kiểm tra Python version
python --version

# Kiểm tra MovieLens dataset
ls ml-1m/

# Kiểm tra dependencies
python -c "import torch, numpy, pandas; print('✅ All good!')"
```

## 🎯 Các bước trong pipeline

| Bước | Script | Mô tả | Output |
|------|--------|-------|---------|
| 1 | `01_data_preprocessing.py` | Xử lý dữ liệu MovieLens-1M | `movielens_train.csv`, `movielens_test.csv` |
| 2 | `02_build_adjacency_matrix.py` | Tạo adjacency matrix | `norm_adj_matrix.npz` |
| 3 | `03_train_lightgcn.py` | Train LightGCN model | `lightgcn_movielens.pt` |
| 4 | `04_evaluate_model.py` | Đánh giá model | Metrics và kết quả |
| 5 | `05_demo_visualization.py` | Demo và visualization | Charts và recommendations |

## 🚨 Troubleshooting

### Lỗi thường gặp:

1. **"File ml-1m/ratings.dat không tồn tại"**
   - Download MovieLens-1M dataset
   - Đặt vào thư mục `ml-1m/`

2. **"ModuleNotFoundError"**
   - Cài đặt dependencies: `pip install -r requirements.txt`

3. **"CUDA out of memory"**
   - Giảm batch size trong training script
   - Hoặc sử dụng CPU training

4. **"Permission denied"**
   - Kiểm tra quyền ghi file trong thư mục hiện tại

## 🎉 Sau khi hoàn thành

Khi pipeline hoàn thành, bạn có thể:

1. **Chạy web app**:
   ```bash
   python run_webapp.py
   ```

2. **Command-line demo**:
   ```bash
   python demo.py
   ```

3. **Xem kết quả**:
   - Training curves: `training_curves.png`
   - Model weights: `lightgcn_movielens.pt`
   - Evaluation results: console output

## 📚 Tìm hiểu thêm

- **README.md**: Hướng dẫn chi tiết
- **main.py**: Pipeline chính với nhiều tùy chọn
- **steps/**: Các script từng bước chi tiết
- **web_app.py**: Giao diện web đẹp mắt

## 🆘 Cần giúp đỡ?

Nếu gặp vấn đề:

1. Kiểm tra error messages
2. Đọc troubleshooting section
3. Kiểm tra file logs
4. Đảm bảo tất cả prerequisites đã được đáp ứng

**Happy coding! 🎬🚀**
