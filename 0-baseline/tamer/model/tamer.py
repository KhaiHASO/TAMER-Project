from typing import List

import lightning.pytorch as pl
import torch
from torch import FloatTensor, LongTensor

from tamer.utils.utils import Hypothesis

from .decoder import Decoder
from .encoder import Encoder


class TAMER(pl.LightningModule):
    def __init__(
        self,
        d_model: int,
        growth_rate: int,
        num_layers: int,
        nhead: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        dc: int,
        cross_coverage: bool,
        self_coverage: bool,
        vocab_size: int = 114,
    ):
        super().__init__()

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
        # Ensure all inputs are on the same device
        device = self.device
        img = img.to(device)
        img_mask = img_mask.to(device)
        tgt = tgt.to(device)
        
        # Process through encoder
        feature, mask = self.encoder(img, img_mask)  # [b, t, d]
        
        # Duplicate for bi-directional processing
        feature = torch.cat((feature, feature), dim=0)  # [2b, t, d]
        mask = torch.cat((mask, mask), dim=0)
        
        # Ensure all tensors are on the correct device
        feature = feature.to(device)
        mask = mask.to(device)
        
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
        # Ensure img and img_mask are on the same device as the model
        device = self.device
        img = img.to(device)
        img_mask = img_mask.to(device)
        
        # Process the image through the encoder
        feature, mask = self.encoder(img, img_mask)  # [b, t, d]
        
        # Ensure feature and mask are on the correct device
        feature = feature.to(device)
        mask = mask.to(device)
        
        # For bi-directional beam search, we need to duplicate the batch
        # This is because we'll run beam search for both left-to-right and right-to-left
        batch_size = feature.shape[0]
        feature_duplicated = torch.cat([feature, feature], dim=0)  # [2b, t, d]
        mask_duplicated = torch.cat([mask, mask], dim=0)  # [2b, t]
        
        # Now run beam search with the duplicated batch
        return self.decoder.beam_search(
            [feature_duplicated], [mask_duplicated], beam_size, max_len, alpha, early_stopping, temperature
        )
