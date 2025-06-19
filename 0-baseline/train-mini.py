from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning import seed_everything
from tamer.datamodule import HMEDatamodule
from tamer.lit_tamer import LitTAMER

seed_everything(7)

# 1. Datamodule giữ nguyên, vì Trainer sẽ chỉ lấy 1 batch từ nó
datamodule = HMEDatamodule(
    train_batch_size=2,
    eval_batch_size=2,
    num_workers=0,
    folder="data/crohme",
    test_folder="2014"
)

# 2. Model được điều chỉnh để dễ Overfit
model = LitTAMER(
    d_model=256,
    growth_rate=24,
    num_layers=16,
    nhead=8,
    num_decoder_layers=3,
    dim_feedforward=1024,
    dc=32,
    dropout=0.0,            # TẮT DROPOUT
    vocab_size=113,
    cross_coverage=True,
    self_coverage=True,
    beam_size=1,            # Giảm beam size để nhanh hơn
    max_len=200,
    alpha=1.0,
    early_stopping=False,
    temperature=1.0,
    learning_rate=1e-3,     # DÙNG LEARNING RATE HỢP LÝ
    patience=20,
    milestones=[300, 500]
)

lr_monitor = LearningRateMonitor(logging_interval="step")

# 3. Cấu hình Trainer đặc biệt cho Overfit Test
trainer = Trainer(
    callbacks=[lr_monitor], # Bỏ checkpoint và logger tự chế cho gọn
    devices=1,
    accelerator="gpu",
    precision=32,           # Dùng 32-bit cho ổn định khi debug
    
    # Cờ quan trọng nhất
    overfit_batches=1,
    
    # Chạy nhiều epoch trên cùng 1 batch đó để xem loss giảm
    max_epochs=200,
    
    deterministic=True,
)

# 4. Chạy và quan sát
print("--- Bắt đầu Overfit Test trên 1 batch ---")
trainer.fit(model, datamodule=datamodule)