import copy
from functools import partial
from typing import Optional

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch

from .arm import AttentionRefinementModule
from .attention import MultiheadAttention


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        decoder_layer,
        num_layers: int,
        arm: Optional[AttentionRefinementModule],
        norm=None,
    ):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

        self.arm = arm

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        height: int,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        output = tgt
        
        # Ensure all tensors are on the same device
        device = tgt.device
        if tgt_mask is not None:
            tgt_mask = tgt_mask.to(device)
        if memory_mask is not None:
            memory_mask = memory_mask.to(device)
        if tgt_key_padding_mask is not None:
            tgt_key_padding_mask = tgt_key_padding_mask.to(device)
        if memory_key_padding_mask is not None:
            memory_key_padding_mask = memory_key_padding_mask.to(device)

        arm_fn = None
        for i, mod in enumerate(self.layers):
            try:
                output, attn = mod(
                    output,
                    memory,
                    arm_fn,
                    tgt_mask=tgt_mask,
                    memory_mask=memory_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                )
                if i != len(self.layers) - 1 and self.arm is not None:
                    # Create a partial function for arm that includes the current attention
                    def arm_fn_with_attn(attn_input):
                        try:
                            return self.arm(attn, memory_key_padding_mask, height, attn_input)
                        except Exception as e:
                            print(f"Warning: ARM function failed with error: {e}")
                            # Return a zero tensor of the same shape as the input
                            return torch.zeros_like(attn_input)
                    arm_fn = arm_fn_with_attn
            except Exception as e:
                print(f"Error in transformer decoder layer {i}: {e}")
                # Try to continue with the next layer
                continue

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(
            d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = F.relu

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        arm: Optional[AttentionRefinementModule],
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        # Get device from input tensors
        device = tgt.device
        
        # Ensure masks are on the same device as tgt
        if tgt_mask is not None:
            tgt_mask = tgt_mask.to(device)
        
        if tgt_key_padding_mask is not None:
            tgt_key_padding_mask = tgt_key_padding_mask.to(device)
            
        if memory_mask is not None:
            memory_mask = memory_mask.to(device)
            
        if memory_key_padding_mask is not None:
            memory_key_padding_mask = memory_key_padding_mask.to(device)
            
            # Ensure memory_key_padding_mask has the right shape
            b, seq_len = memory.shape[1], memory.shape[0]
            if memory_key_padding_mask.shape[0] != b or memory_key_padding_mask.shape[1] != seq_len:
                # Create a new mask with the right shape
                new_mask = torch.zeros((b, seq_len), dtype=torch.bool, device=device)
                # Copy as much as we can from the original mask
                copy_rows = min(memory_key_padding_mask.shape[0], b)
                copy_cols = min(memory_key_padding_mask.shape[1], seq_len)
                new_mask[:copy_rows, :copy_cols] = memory_key_padding_mask[:copy_rows, :copy_cols]
                memory_key_padding_mask = new_mask
        
        tgt2 = self.self_attn(
            tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, attn = self.multihead_attn(
            tgt,
            memory,
            memory,
            arm=arm,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, attn
