# Hướng dẫn cài đặt TAMER trên Kaggle

Tài liệu này hướng dẫn chi tiết cách cài đặt và chạy mô hình TAMER trên Kaggle.

## Bước 1: Tạo một Notebook mới trên Kaggle

1. Đăng nhập vào [Kaggle](https://www.kaggle.com/)
2. Chọn "Code" > "New Notebook"
3. Đặt tên cho Notebook và lưu

## Bước 2: Cài đặt môi trường

Thêm đoạn code sau vào cell đầu tiên và chạy:

```python
# Kiểm tra phiên bản Python
!python --version

# Kiểm tra GPU
!nvidia-smi

# Cài đặt các thư viện cần thiết cho PyTorch Lightning
!pip install torch==2.2.1 torchvision==0.17.1
!pip install lightning==2.2.1 pytorch-lightning==2.2.1 lightning-utilities==0.10.0

# Cài đặt các thư viện khác
!pip install einops==0.7.0 editdistance==0.6.2 torchmetrics==1.2.1 jsonargparse[signatures]==4.27.1 typer==0.9.0 beautifulsoup4==4.12.3 lxml>=4.9.4 pandas==2.2.1
```

## Bước 3: Clone repository và cài đặt

Thêm vào cell tiếp theo:

```python
# Clone repository
!git clone https://github.com/your-github-username/TAMER.git
%cd TAMER
!pip install -e .
```

## Bước 4: Tải và chuẩn bị dữ liệu

Tạo cấu trúc thư mục và tải dữ liệu:

```python
# Tạo thư mục data
!mkdir -p data

# Với CROHME dataset, bạn cần:
# 1. Tải file từ link https://disk.pku.edu.cn/link/AAF10CCC4D539543F68847A9010C607139
# 2. Upload lên Kaggle
# 3. Giải nén vào thư mục data

# Hoặc bạn có thể tải trực tiếp nếu có URL công khai:
# !wget -O data/crohme.zip "URL_TO_CROHME_DATA"
# !unzip data/crohme.zip -d data/

# Kiểm tra cấu trúc thư mục
!ls -la data/
```

## Bước 5: Điều chỉnh cấu hình để chạy trên Kaggle

Tạo một file cấu hình mới cho Kaggle:

```python
%%writefile config/kaggle.yaml
seed_everything: 7
trainer:
  enable_checkpointing: true
  callbacks:
    - class_path: lightning.pytorch.callbacks.LearningRateMonitor
      init_args:
        logging_interval: epoch
    - class_path: lightning.pytorch.callbacks.ModelCheckpoint
      init_args:
        save_top_k: 1
        monitor: val_ExpRate
        mode: max
        filename: '{epoch}-{step}-{val_ExpRate:.4f}'
  devices: 1
  accelerator: gpu
  check_val_every_n_epoch: 2
  max_epochs: 100  # Giảm xuống để phù hợp với thời gian chạy trên Kaggle
  deterministic: true
model:
  d_model: 256
  # encoder
  growth_rate: 24
  num_layers: 16
  # decoder
  nhead: 8
  num_decoder_layers: 3
  dim_feedforward: 1024
  dc: 32
  dropout: 0.3
  vocab_size: 113  # 110 + 3
  cross_coverage: true
  self_coverage: true
  # beam search
  beam_size: 10
  max_len: 200
  alpha: 1.0
  early_stopping: false
  temperature: 1.0
  # training
  learning_rate: 1.0
  patience: 20
  milestones:
    - 50
    - 80
data:
  folder: data/crohme
  test_folder: 2014
  max_size: 320000
  scale_to_limit: true
  train_batch_size: 4  # Giảm batch size nếu gặp vấn đề OOM
  eval_batch_size: 2
  num_workers: 2
  scale_aug: false
```

## Bước 6: Huấn luyện mô hình

```python
# Huấn luyện
!python -u train.py --config config/kaggle.yaml
```

## Bước 7: Đánh giá mô hình

```python
# Đánh giá trên tập CROHME 2014
!python eval/test.py data/crohme 0 2014 320000 True

# Xem kết quả
!cat lightning_logs/version_0/2014.txt
```

## Lưu ý quan trọng cho Kaggle

1. **Thời gian chạy giới hạn**: Kaggle có giới hạn thời gian chạy. Điều chỉnh `max_epochs` nhỏ hơn.
2. **GPU Memory**: Điều chỉnh `batch_size` nhỏ hơn nếu gặp lỗi Out of Memory (OOM).
3. **Lưu trữ dữ liệu**: Sử dụng Kaggle Datasets để lưu trữ dữ liệu và checkpoints.
4. **Lưu checkpoint**: Sử dụng `InterruptCallback` để lưu kết quả định kỳ:

```python
from lightning.pytorch.callbacks import ModelCheckpoint

checkpoint_callback = ModelCheckpoint(
    dirpath='./checkpoints',
    filename='{epoch}-{val_ExpRate:.4f}',
    save_top_k=1,
    monitor='val_ExpRate',
    mode='max',
    save_last=True
)
```

5. **Tiếp tục huấn luyện**: Nếu bạn cần tiếp tục huấn luyện từ checkpoint đã lưu:

```python
# Tiếp tục huấn luyện từ checkpoint
!python -u train.py --config config/kaggle.yaml --ckpt_path lightning_logs/version_0/checkpoints/last.ckpt
``` 