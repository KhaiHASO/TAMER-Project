from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning import seed_everything
from tamer.datamodule import HMEDatamodule
from tamer.lit_tamer import LitTAMER

# Cấu hình seed cho tái sản xuất kết quả
seed_everything(7)

# Khởi tạo các đối tượng dữ liệu và mô hình với các tham số bổ sung từ config
datamodule = HMEDatamodule(
    folder="data/crohme",  # Thư mục dữ liệu
    test_folder="2014",  # Thư mục test
    max_size=320000,  # Giới hạn kích thước tối đa
    scale_to_limit=True,  # Nếu có scale
    train_batch_size=8,  # Batch size cho huấn luyện
    eval_batch_size=2,  # Batch size cho kiểm tra
    num_workers=5,  # Số lượng worker cho DataLoader
    scale_aug=False  # Điều chỉnh augmentation nếu cần
)

model = LitTAMER(
    d_model=256,
    growth_rate=24,
    num_layers=16,
    nhead=8,
    num_decoder_layers=3,
    dim_feedforward=1024,
    dc=32,
    dropout=0.3,
    vocab_size=113,  # 110 + 3
    cross_coverage=True,
    self_coverage=True,
    beam_size=10,
    max_len=200,
    alpha=1.0,
    early_stopping=False,
    temperature=1.0,
    learning_rate=1.0,
    patience=20,
    milestones=[300, 350]
)

# Các Callbacks
lr_monitor = LearningRateMonitor(logging_interval="epoch")
checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    monitor="val_ExpRate",  # Theo dõi giá trị val_ExpRate
    mode="max",  # Lưu mô hình khi giá trị `val_ExpRate` cao nhất
    filename="{epoch}-{step}-{val_ExpRate:.4f}",  # Tên file lưu checkpoint
)

# Cấu hình Trainer
trainer = Trainer(
    callbacks=[lr_monitor, checkpoint_callback],  # Cập nhật theo đúng cách sử dụng callbacks
    devices=1,  # Sử dụng 1 device (CPU)
    accelerator="cpu",  # Chỉ định sử dụng CPU
    check_val_every_n_epoch=2,  # Kiểm tra validation mỗi 2 epoch
    max_epochs=50,  # Số lượng epoch tối đa
    deterministic=True,  # Đảm bảo tính tái sản xuất kết quả
)

# Huấn luyện mô hình
trainer.fit(model, datamodule=datamodule)
#lấy bản này 