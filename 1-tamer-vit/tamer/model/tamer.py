from typing import List

import pytorch_lightning as pl
import torch
from torch import FloatTensor, LongTensor

from tamer.utils.utils import Hypothesis

from .decoder import Decoder
from .encoder import Encoder
from .vit_encoder import ViTEncoder


class TAMER(pl.LightningModule):
    def __init__(
        self,
        d_model: int,
        # encoder
        encoder_type: str = "densenet",  # "densenet" or "vit"
        growth_rate: int = 24,
        num_layers: int = 16,
        # decoder
        nhead: int = 8,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 1024,
        dropout: float = 0.3,
        dc: int = 32,
        cross_coverage: bool = True,
        self_coverage: bool = True,
        vocab_size: int = 114,
    ):
        super().__init__()

        # Chọn encoder dựa trên encoder_type
        if encoder_type == "vit":
            self.encoder = ViTEncoder(d_model=d_model)
        else:  # mặc định là densenet
            self.encoder = Encoder(
                d_model=d_model, growth_rate=growth_rate, num_layers=num_layers
            )

        self.decoder = Decoder(
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            dc=dc,
            cross_coverage=cross_coverage,
            self_coverage=self_coverage,
            vocab_size=vocab_size,
        )

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
        feature, mask = self.encoder(img, img_mask)  # [b, t, d]
        feature = torch.cat((feature, feature), dim=0)  # [2b, t, d]
        mask = torch.cat((mask, mask), dim=0)

        return self.decoder(feature, mask, tgt)

    def beam_search(
        self,
        img: FloatTensor,
        img_mask: LongTensor,
        beam_size: int,
        max_len: int,
        alpha: float,
        early_stopping: bool,
        temperature: float,
        **kwargs,
    ) -> List[Hypothesis]:
        """run bi-direction beam search for given img

        Parameters
        ----------
        img : FloatTensor
            [b, 1, h', w']
        img_mask: LongTensor
            [b, h', w']
        beam_size : int
        max_len : int

        Returns
        -------
        List[Hypothesis]
        """
        feature, mask = self.encoder(img, img_mask)  # [b, t, d]
        return self.decoder.beam_search(
            [feature], [mask], beam_size, max_len, alpha, early_stopping, temperature
        )
