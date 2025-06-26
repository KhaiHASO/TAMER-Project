# TAMER: Transformer-based Approach for Mathematical Expression Recognition

## Giới thiệu

TAMER là một dự án nghiên cứu nhằm cải thiện khả năng nhận dạng biểu thức toán học viết tay (Handwritten Mathematical Expression Recognition - HMER) thông qua nhiều kiến trúc mạng nơ-ron tiên tiến. Dự án phát triển dựa trên mô hình Transformer, áp dụng các kỹ thuật hiện đại để nâng cao chất lượng nhận dạng.

## Cấu trúc dự án

Dự án được tổ chức thành các phiên bản phát triển tiến tiến:

- **0-baseline**: Mô hình cơ sở sử dụng DenseNet và Transformer
- **1-tamer-vit**: Cải tiến encoder bằng Vision Transformer để trích xuất đặc trưng hình ảnh tốt hơn
- **2-tamer-gat**: Thay thế Tree-Aware Module (TAM) bằng Graph Attention Networks (GAT)
- **3-tamer-transformer**: Kết hợp ViT và GAT để tận dụng ưu điểm của cả hai phương pháp

## Yêu cầu cài đặt

```bash
# Cài đặt dependencies
pip install -r requirements.txt

# Hoặc cài đặt từng thư viện riêng biệt
pip install torch==1.10.0 torchvision==0.11.0
pip install pytorch-lightning==1.5.0
pip install einops editdistance
```

## Dữ liệu

Dự án sử dụng bộ dữ liệu CROHME (Competition on Recognition of Online Handwritten Mathematical Expressions):

```bash
# Tải và giải nén dữ liệu CROHME
mkdir -p data/CROHME
# Đặt dữ liệu CROHME.zip vào thư mục data
unzip data/CROHME.zip -d data/CROHME/
```

## Cách sử dụng

### Huấn luyện

Mô hình cơ sở có thể được huấn luyện như sau:

```bash
# Huấn luyện mô hình baseline
cd 0-baseline
python train.py --config config/crohme.yaml
```

### Đánh giá

```bash
# Đánh giá mô hình trên tập test
cd 0-baseline
python eval/test.py --ckpt <path-to-checkpoint> --config config/crohme.yaml
```

### Notebook tích hợp cho Kaggle và Colab

Dự án cung cấp các notebook Jupyter được tối ưu hóa để chạy trên các nền tảng phổ biến:

- **TAMER_Kaggle_Setup.ipynb**: Notebook tích hợp để thiết lập và chạy mô hình TAMER trên Kaggle
  - Tự động cài đặt các dependencies
  - Tải và tiền xử lý dữ liệu CROHME
  - Huấn luyện và đánh giá mô hình

- **tamer37_baseline.ipynb**: Notebook tương thích với Google Colab
  - Hướng dẫn chi tiết cho người mới bắt đầu
  - Tích hợp các đoạn code để kết nối với Google Drive
  - Các công cụ trực quan để phân tích kết quả
  
Các notebook này đã được tối ưu hóa để tận dụng tài nguyên GPU của Kaggle và Colab, cho phép người dùng huấn luyện và thử nghiệm mô hình mà không cần cấu hình phần cứng mạnh mẽ.

## Kiến trúc mô hình

### 0-baseline
Mô hình cơ sở sử dụng kiến trúc Encoder-Decoder:
- **Encoder**: DenseNet để trích xuất đặc trưng từ hình ảnh
- **Decoder**: Transformer để giải mã sang chuỗi LaTeX

## Giấy phép

Dự án này được phân phối dưới giấy phép [loại giấy phép]. Xem file `LICENSE` để biết thêm chi tiết.
