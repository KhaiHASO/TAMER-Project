from lightning.pytorch.cli import LightningCLI
from lightning_fabric.utilities.seed import seed_everything
from pytorch_lightning.plugins import DDPStrategy

from tamer.datamodule import HMEDatamodule
from tamer.lit_tamer import LitTAMER

# Đặt seed trước khi khởi tạo CLI
seed_everything(7, workers=True)

cli = LightningCLI(
    LitTAMER,
    HMEDatamodule,
    seed_everything_default=None,  # vì đã đặt thủ công ở trên
    save_config_overwrite=True,
    trainer_defaults={
        "strategy": DDPStrategy(find_unused_parameters=False),
    }
)
