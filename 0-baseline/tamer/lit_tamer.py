import zipfile
from typing import List
import editdistance
import json
import lightning.pytorch as pl
import torch
import torch.optim as optim
from torch import FloatTensor, LongTensor

from tamer.datamodule import Batch, vocab
from tamer.model.tamer import TAMER
from tamer.utils.utils import (
    ExpRateRecorder, Hypothesis, ce_loss, to_bi_tgt_out, to_struct_output)


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
        tgt, out = to_bi_tgt_out(batch.indices, self.device)
        struct_out, _ = to_struct_output(batch.indices, self.device)
        out_hat, sim = self(batch.imgs, batch.mask, tgt)

        if self.global_step % 10 == 0:
            print("DEBUG: Training step output check")
            print(f"DEBUG: out_hat shape: {out_hat.shape}, out shape: {out.shape}")
            has_nan = torch.isnan(out_hat).any().item()
            has_inf = torch.isinf(out_hat).any().item()
            print(f"DEBUG: Has NaN: {has_nan}, Has Inf: {has_inf}")
            
            softmax_out = torch.softmax(out_hat, dim=-1)
            top_probs, top_indices = torch.topk(softmax_out[0, 0], 5)
            print(f"DEBUG: Top 5 probs for first token: {top_probs}")
            print(f"DEBUG: Top 5 indices for first token: {top_indices}")
            top_words = []
            for idx in top_indices:
                try:
                    top_words.append(vocab.indices2words([idx.item()])[0])
                except:
                    top_words.append("<error>")
            print(f"DEBUG: Top 5 words for first token: {top_words}")

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
        tgt, out = to_bi_tgt_out(batch.indices, self.device)
        struct_out, _ = to_struct_output(batch.indices, self.device)
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

        hyps = self.approximate_joint_search(batch.imgs, batch.mask)

        print("DEBUG: Beam search predictions:")
        pred_seqs = [h.seq for h in hyps]
        for i, (pred, gt) in enumerate(zip(pred_seqs, batch.indices)):
            pred_words = vocab.indices2words(pred)
            gt_words = vocab.indices2words(gt)
            print(f"Sample {i}")
            print(f"  Pred indices: {pred}")
            print(f"  GT indices: {gt}")
            print(f"  Pred words: {' '.join(pred_words)}")
            print(f"  GT words: {' '.join(gt_words)}")
            print(f"  Match: {pred == gt}")
        
        self.exprate_recorder([h.seq for h in hyps], batch.indices)
        exp_rate = self.exprate_recorder.compute()
        print(f"DEBUG: Current ExpRate: {exp_rate}")
        self.log(
            "val_ExpRate",
            self.exprate_recorder,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

    def test_step(self, batch: Batch, _):
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
        return self.tamer_model.beam_search(img, mask, **self.hparams)

    def configure_optimizers(self):
        optimizer = optim.Adadelta(
            self.parameters(),
            lr=self.hparams.learning_rate,
            eps=1e-6,
            weight_decay=1e-4,
        )
        # optimizer = optim.AdamW(
        #     self.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-4
        # )

        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.hparams.milestones, gamma=0.1
        )
        # reduce_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer,
        #     mode="max",
        #     factor=0.25,
        #     patience=self.hparams.patience // self.trainer.check_val_every_n_epoch,
        # )

        # scheduler = {
        #     "scheduler": reduce_scheduler,
        #     "monitor": "val_ExpRate",
        #     "interval": "epoch",
        #     "frequency": self.trainer.check_val_every_n_epoch,
        #     "strict": True,
        # }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
