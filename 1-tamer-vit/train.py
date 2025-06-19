from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import seed_everything
from tamer.datamodule import HMEDatamodule
from tamer.lit_tamer import LitTAMER
import shutil
import os

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
early_stopping = EarlyStopping(
    monitor="val_ExpRate",
    patience=20,
    mode="max",
    verbose=True
)

# Cấu hình Trainer
trainer = Trainer(
    callbacks=[lr_monitor, checkpoint_callback, early_stopping],
    devices=1,
    accelerator="auto",
    check_val_every_n_epoch=2,
    max_epochs=80,
    deterministic=True,
)

# Huấn luyện mô hình
trainer.fit(model, datamodule=datamodule)

# Đường dẫn file checkpoint tốt nhất
src = checkpoint_callback.best_model_path
if src:
    # Lấy tên file gốc
    filename = os.path.basename(src)
    dst = f"/kaggle/working/{filename}"
    shutil.copy(src, dst)
    print(f"Checkpoint đã được sao chép tới: {dst}")
else:
    print("Không tìm thấy checkpoint tốt nhất để sao chép!")
