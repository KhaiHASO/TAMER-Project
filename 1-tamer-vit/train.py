from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning import seed_everything
from tamer.datamodule import HMEDatamodule
from tamer.lit_tamer import LitTAMER

# Cấu hình seed cho tái sản xuất kết quả
seed_everything(7)

# Khởi tạo các đối tượng dữ liệu và mô hình
datamodule = HMEDatamodule()
model = LitTAMER(
    d_model=256,
    growth_rate=24,
    num_layers=16,
    nhead=8,
    num_decoder_layers=3,
    dim_feedforward=1024,
    dc=32,
    dropout=0.3,
    vocab_size=113,
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
    monitor="val_ExpRate",
    mode="max",
    filename="{epoch}-{step}-{val_ExpRate:.4f}",
)

# Cấu hình Trainer
trainer = Trainer(
    callbacks=[lr_monitor, checkpoint_callback],  # Cập nhật theo đúng cách sử dụng callbacks
    devices=1,  # Thay gpus thành devices
    accelerator="auto",  # Tự động phát hiện và sử dụng GPU hoặc TPU
    check_val_every_n_epoch=2,
    max_epochs=400,
    deterministic=True,  # Đảm bảo tính tái sản xuất kết quả
)

# Huấn luyện mô hình
trainer.fit(model, datamodule=datamodule)
