import os
import lightning.pytorch as pl
from lightning.pytorch import Trainer, seed_everything
from tamer.datamodule import HMEDatamodule
from tamer.lit_tamer import LitTAMER
import yaml

# Đặt seed cho tính nhất quán
seed_everything(7)

# Đọc cấu hình từ file
with open('config/crohme.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Cấu hình trainer với overfit_batches
trainer_config = config['trainer']
trainer_config['overfit_batches'] = 1  # Overfit trên 1 batch
trainer_config['max_epochs'] = 30      # Chạy 30 epochs là đủ để debug

# Cấu hình mô hình
model_config = config['model']
model = LitTAMER(**model_config)

# Cấu hình datamodule
data_config = config['data']
dm = HMEDatamodule(**data_config)

# Tạo trainer
callbacks = []
for callback_conf in trainer_config.get('callbacks', []):
    cls_path = callback_conf['class_path']
    init_args = callback_conf.get('init_args', {})
    
    # Tách tên lớp từ đường dẫn
    module_path, class_name = cls_path.rsplit('.', 1)
    module = __import__(module_path, fromlist=[class_name])
    cls = getattr(module, class_name)
    
    callbacks.append(cls(**init_args))

trainer = Trainer(
    callbacks=callbacks,
    logger=pl.loggers.CSVLogger("lightning_logs", name="overfit_debug"),
    overfit_batches=1,
    max_epochs=30,
    accelerator=trainer_config.get('accelerator', 'cpu'),
    devices=trainer_config.get('devices', 1),
    check_val_every_n_epoch=1,
    deterministic=True
)

# Train mô hình
print("Bắt đầu training với overfit_batches=1...")
trainer.fit(model, dm)

# Kiểm tra mô hình trên validation
print("\nĐánh giá mô hình trên validation data...")
trainer.validate(model, dm)

print("Hoàn thành!") 