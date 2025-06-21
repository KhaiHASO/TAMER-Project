from pytorch_lightning.utilities.cli import LightningCLI

from tamer.datamodule import HMEDatamodule
from tamer.lit_tamer import LitTAMER

cli = LightningCLI(
    LitTAMER,
    HMEDatamodule,
    save_config_overwrite=True,
    trainer_defaults={"strategy": "ddp", "find_unused_parameters": False},
)
