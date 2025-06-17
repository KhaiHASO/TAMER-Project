%%writefile setup_and_run.sh
#!/bin/bash

# --- BƯỚC 1: KHỞI TẠO MÔI TRƯỜNG ---
echo ">>> Creating Conda environment: TAMER with Python 3.7"
mamba create -y -n TAMER python=3.7

# --- BƯỚC 2: KÍCH HOẠT MÔI TRƯỜNG VÀ CÀI ĐẶT THƯ VIỆN ---
# Khởi tạo Conda cho phiên làm việc của script này
source /opt/conda/etc/profile.d/conda.sh
conda activate TAMER

echo ">>> Environment activated. Installing dependencies..."

# Cài PyTorch và các gói liên quan
mamba install -y pytorch=1.8.1 torchvision=0.2.2 cudatoolkit=11.1 pillow=8.4.0 -c pytorch -c nvidia

# Cài các gói khác từ conda-forge
mamba install -y pytorch-lightning=1.4.9 torchmetrics=0.6.0 pandoc -c conda-forge

# Cài đặt mã nguồn dự án bằng pip
# Lưu ý: Hãy chắc chắn rằng bạn đã tải/clone dự án TAMER vào đúng vị trí.
# Lệnh này giả sử thư mục TAMER nằm trong /kaggle/working/
echo ">>> Installing project source..."
cd /kaggle/working/TAMER  # <-- THAY ĐỔI ĐƯỜNG DẪN NÀY NẾU CẦN
pip install -e .

# --- BƯỚC 3: HUẤN LUYỆN MÔ HÌNH ---
echo ">>> Starting training..."
# Đừng quên chỉnh sửa file .yaml để dùng 1 GPU nếu cần
python -u train.py --config config/crohme.yaml

echo ">>> Script finished!"