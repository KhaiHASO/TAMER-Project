from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning import seed_everything
from tamer.datamodule import HMEDatamodule
from tamer.lit_tamer import LitTAMER
import shutil
import os
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import Callback

# Set seed for reproducibility (from YAML)
seed_everything(7)

# Initialize data module with parameters from YAML if possible
# (Assuming HMEDatamodule can take batch size, num_workers, etc. If not, leave as default)
datamodule = HMEDatamodule(
    train_batch_size=8,
    eval_batch_size=2,
    num_workers=5,
    folder="data/crohme",
    test_folder="debug",
    max_size=320000,
    scale_to_limit=True,
    scale_aug=False
)

# Initialize model with parameters from YAML
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

# Callbacks as in YAML
lr_monitor = LearningRateMonitor(logging_interval="epoch")
checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    monitor="val_ExpRate",
    mode="max",
    filename="{epoch}-{step}-{val_ExpRate:.4f}",
)

class SimpleMetricsLogger(Callback):
    def __init__(self):
        self.epochs = []
        self.train_loss = []
        self.val_loss = []
        self.val_acc = []

    def on_train_epoch_end(self, trainer, pl_module):
        self.epochs.append(trainer.current_epoch)
        train_loss = trainer.callback_metrics.get("train_loss")
        self.train_loss.append(train_loss.cpu().item() if train_loss is not None else None)

    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get("val_loss")
        val_acc = trainer.callback_metrics.get("val_ExpRate")  # hoặc 'val_accuracy' nếu đúng tên
        self.val_loss.append(val_loss.cpu().item() if val_loss is not None else None)
        self.val_acc.append(val_acc.cpu().item() if val_acc is not None else None)

metrics_logger = SimpleMetricsLogger()

# Trainer configuration as in YAML
trainer = Trainer(
    callbacks=[lr_monitor, checkpoint_callback, metrics_logger],
    devices=1,
    accelerator="gpu",
    precision=16,
    #strategy="ddp",
    check_val_every_n_epoch=2,
    max_epochs=50,
    deterministic=True,
)

# Train the model
trainer.fit(model, datamodule=datamodule)


