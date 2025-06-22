import zipfile
from typing import List
import editdistance
import json
import lightning.pytorch as pl
import torch
import torch.optim as optim
from torch import FloatTensor, LongTensor

from tamer.datamodule import vocab
from tamer.datamodule.datamodule import Batch
from tamer.model.tamer import TAMER
from tamer.utils.utils import Hypothesis, ExpRateRecorder, ce_loss, to_bi_tgt_out, to_struct_output


class LitTAMER(pl.LightningModule):
    def __init__(
        self,
        d_model: int,
        # encoder
        growth_rate: int,
        num_layers: int,
        # decoder
        nhead: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        dc: int,
        cross_coverage: bool,
        self_coverage: bool,
        # beam search
        beam_size: int,
        max_len: int,
        alpha: float,
        early_stopping: bool,
        temperature: float,
        # training
        learning_rate: float,
        patience: int,
        milestones: List[int] = [40, 55],
        vocab_size: int = 113,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.tamer_model = TAMER(
            d_model=d_model,
            growth_rate=growth_rate,
            num_layers=num_layers,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            dc=dc,
            cross_coverage=cross_coverage,
            self_coverage=self_coverage,
            vocab_size=vocab_size,
        )

        self.exprate_recorder = ExpRateRecorder()

    def forward(
        self, img: FloatTensor, img_mask: LongTensor, tgt: LongTensor
    ) -> FloatTensor:
        """run img and bi-tgt

        Parameters
        ----------
        img : FloatTensor
            [b, 1, h, w]
        img_mask: LongTensor
            [b, h, w]
        tgt : LongTensor
            [2b, l]

        Returns
        -------
        FloatTensor
            [2b, l, vocab_size]
        """
        return self.tamer_model(img, img_mask, tgt)

    def training_step(self, batch: Batch, _):
        # Ensure all batch items are on the same device as the model
        device = self.device
        batch.imgs = batch.imgs.to(device)
        batch.mask = batch.mask.to(device)
        
        tgt, out = to_bi_tgt_out(batch.indices, device)
        struct_out, _ = to_struct_output(batch.indices, device)
        out_hat, sim = self(batch.imgs, batch.mask, tgt)

        loss = ce_loss(out_hat, out)
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        struct_loss = ce_loss(sim, struct_out, ignore_idx=-1)
        self.log(
            "train/struct_loss",
            struct_loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        return loss + struct_loss


    def validation_step(self, batch: Batch, _):
        # Ensure all batch items are on the same device as the model
        device = self.device
        batch.imgs = batch.imgs.to(device)
        batch.mask = batch.mask.to(device)
        
        tgt, out = to_bi_tgt_out(batch.indices, device)
        struct_out, _ = to_struct_output(batch.indices, device)
        out_hat, sim = self(batch.imgs, batch.mask, tgt)

        loss = ce_loss(out_hat, out)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        struct_loss = ce_loss(sim, struct_out, ignore_idx=-1)
        self.log(
            "val/struct_loss",
            struct_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        # Run beam search if we're past the initial training epochs
        if self.current_epoch >= self.hparams.milestones[0]:
            hyps = self.approximate_joint_search(batch.imgs, batch.mask)
            self.exprate_recorder([h.seq for h in hyps], batch.indices)
            self.log(
                "val_ExpRate",
                self.exprate_recorder,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )
        else:
            # Log a default value when not calculating to avoid ModelCheckpoint error
            # Use a small value (0) so it won't be mistakenly chosen as the best model
            self.log(
                "val_ExpRate",
                0.0,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )

    def test_step(self, batch: Batch, _):
        # Ensure batch is on the correct device
        device = self.device
        batch.imgs = batch.imgs.to(device)
        batch.mask = batch.mask.to(device)
        
        hyps = self.approximate_joint_search(batch.imgs, batch.mask)
        self.exprate_recorder([h.seq for h in hyps], batch.indices)
        gts = [vocab.indices2words(ind) for ind in batch.indices]
        preds = [vocab.indices2words(h.seq) for h in hyps]

        return batch.img_bases, preds, gts

    def test_epoch_end(self, test_outputs) -> None:
        exprate = self.exprate_recorder.compute()
        print(f"Validation ExpRate: {exprate}")
        errors_dict = {}
        predictions_dict = {}
        with zipfile.ZipFile("result.zip", "w") as zip_f:
            for img_bases, preds, gts in test_outputs:
                for img_base, pred, gt in zip(img_bases, preds, gts):
                    content = f"%{img_base}\n${pred}$".encode()
                    with zip_f.open(f"{img_base}.txt", "w") as f:
                        f.write(content)
                    distance = editdistance.eval(pred, gt)
                    if distance > 0:
                        errors_dict[img_base] = {
                            "pred": " ".join(pred),
                            "gt": " ".join(gt),
                            "dist": distance,
                        }

                    predictions_dict[img_base] = {
                        "pred": " ".join(pred),
                        "gt": " ".join(gt),
                        "dist": distance,
                    }
        with open("errors.json", "w") as f:
            json.dump(errors_dict, f)
        with open("predictions.json", "w") as f:
            json.dump(predictions_dict, f)

    def approximate_joint_search(
        self, img: FloatTensor, mask: LongTensor
    ) -> List[Hypothesis]:
        """Run beam search on the model with error handling

        Parameters
        ----------
        img : FloatTensor
            Input image tensor [b, 1, h, w]
        mask : LongTensor
            Input mask tensor [b, h, w]

        Returns
        -------
        List[Hypothesis]
            List of beam search hypotheses
        """
        try:
            # Ensure inputs are on the correct device
            device = self.device
            img = img.to(device)
            mask = mask.to(device)
            
            # Extract beam search parameters from hparams
            beam_params = {
                'beam_size': self.hparams.beam_size,
                'max_len': self.hparams.max_len,
                'alpha': self.hparams.alpha,
                'early_stopping': self.hparams.early_stopping,
                'temperature': self.hparams.temperature
            }
            
            # Run beam search with error handling
            return self.tamer_model.beam_search(img, mask, **beam_params)
            
        except Exception as e:
            print(f"Error in beam search: {e}")
            # Return empty hypotheses in case of error
            batch_size = img.shape[0]
            return [Hypothesis([], 0.0, "l2r") for _ in range(batch_size)]

    def configure_optimizers(self):
        optimizer = optim.Adadelta(
            self.parameters(),
            lr=self.hparams.learning_rate,
            eps=1e-6,
            weight_decay=1e-4,
        )

        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.hparams.milestones, gamma=0.1
        )

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
