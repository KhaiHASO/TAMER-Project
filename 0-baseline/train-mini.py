from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor
from tamer.datamodule import HMEDatamodule
from tamer.lit_tamer import LitTAMER

# Đặt seed để tái lập kết quả
seed_everything(7, workers=True)

# DataModule: dùng batch rất nhỏ để test overfit
datamodule = HMEDatamodule(
    folder="data/crohme",
    test_folder="2014",
    max_size=320000,
    scale_to_limit=True,
    scale_aug=False,
    train_batch_size=2,
    eval_batch_size=2,
    num_workers=0
)

# Mô hình đơn giản hóa để dễ overfit
model = LitTAMER(
    d_model=256,
    growth_rate=24,
    num_layers=8,
    nhead=4,
    num_decoder_layers=2,
    dim_feedforward=512,
    dc=16,
    dropout=0.0,
    vocab_size=113,
    cross_coverage=False,
    self_coverage=False,
    beam_size=5,
    max_len=40,
    alpha=1.0,
    early_stopping=False,
    temperature=1.0,
    learning_rate=3e-3,
    patience=10,
    milestones=[50, 100]
)

# Callback để quan sát learning rate
lr_monitor = LearningRateMonitor(logging_interval="step")

# Trainer cấu hình để overfit 1 batch duy nhất
trainer = Trainer(
    accelerator="cpu",
    # devices=1,
    precision=32,
    callbacks=[lr_monitor],
    max_epochs=50,
    overfit_batches=1,
    deterministic=True,
    log_every_n_steps=1
)

print("\n--- BẮT ĐẦU OVERFIT TEST TRÊN 1 BATCH ---\n")
trainer.fit(model, datamodule=datamodule)
