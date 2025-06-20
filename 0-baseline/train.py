from lightning.pytorch.cli import LightningCLI
from tamer.datamodule import HMEDatamodule
from tamer.lit_tamer import LitTAMER

cli = LightningCLI(
    LitTAMER,
    HMEDatamodule,
    save_config_kwargs={"overwrite": True}
)
