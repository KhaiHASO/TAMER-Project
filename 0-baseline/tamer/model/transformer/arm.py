import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch import Tensor
from torch.nn.modules.batchnorm import BatchNorm1d


class MaskBatchNorm2d(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.bn = BatchNorm1d(num_features)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            [b, d, h, w]
        mask : Tensor
            [b, 1, h, w]

        Returns
        -------
        Tensor
            [b, d, h, w]
        """
        x = rearrange(x, "b d h w -> b h w d")
        mask = mask.squeeze(1)

        not_mask = ~mask

        flat_x = x[not_mask, :]
        flat_x = self.bn(flat_x)
        x[not_mask, :] = flat_x

        x = rearrange(x, "b h w d -> b d h w")

        return x


class AttentionRefinementModule(nn.Module):
    def __init__(self, nhead: int, dc: int, cross_coverage: bool, self_coverage: bool):
        super().__init__()
        assert cross_coverage or self_coverage
        self.nhead = nhead
        self.cross_coverage = cross_coverage
        self.self_coverage = self_coverage

        if cross_coverage and self_coverage:
            in_chs = 2 * nhead
        else:
            in_chs = nhead

        self.conv = nn.Conv2d(in_chs, dc, kernel_size=5, padding=2)
        self.act = nn.ReLU(inplace=True)

        self.proj = nn.Conv2d(dc, nhead, kernel_size=1, bias=False)
        self.post_norm = MaskBatchNorm2d(nhead)

    def forward(
        self, prev_attn: Tensor, key_padding_mask: Tensor, h: int, curr_attn: Tensor = None
    ) -> Tensor:
        """
        Parameters
        ----------
        prev_attn : Tensor
            [(b * nhead), t, l]
        key_padding_mask : Tensor
            [b, l]
        h : int
        curr_attn : Tensor, optional
            Current attention weights, same shape as prev_attn

        Returns
        -------
        Tensor
            [(b * nhead), t, l]
        """
        # If curr_attn is not provided, use prev_attn
        if curr_attn is None:
            curr_attn = prev_attn
            
        # Ensure both are on the same device
        device = prev_attn.device
        key_padding_mask = key_padding_mask.to(device)
        
        t = curr_attn.shape[1]
        b = key_padding_mask.shape[0]
        l = key_padding_mask.shape[1]
        
        # Calculate expected dimensions
        expected_h = int((l + h - 1) // h)  # Ceiling division
        
        # Make sure key_padding_mask has the right shape for the repeat operation
        if l != h * expected_h:
            # Pad or truncate key_padding_mask to match the expected size
            new_mask = torch.zeros((b, h * expected_h), dtype=key_padding_mask.dtype, device=device)
            copy_len = min(l, h * expected_h)
            new_mask[:, :copy_len] = key_padding_mask[:, :copy_len]
            key_padding_mask = new_mask
        
        try:
            # Try to create the mask with the expected dimensions
            mask = repeat(key_padding_mask, "b (h w) -> (b t) () h w", h=h, t=t, w=expected_h)
        except RuntimeError as e:
            # If that fails, create a mask with compatible dimensions
            print(f"Warning: Mask creation failed with error: {e}")
            print(f"key_padding_mask shape: {key_padding_mask.shape}, h: {h}, t: {t}")
            
            # Create a mask that matches the output shape of attns after rearrangement
            curr_attn_b = curr_attn.shape[0] // self.nhead
            mask = torch.zeros((curr_attn_b * t, 1, h, expected_h), dtype=torch.bool, device=device)

        curr_attn = rearrange(curr_attn, "(b n) t l -> b n t l", n=self.nhead)
        prev_attn = rearrange(prev_attn, "(b n) t l -> b n t l", n=self.nhead)

        attns = []
        if self.cross_coverage:
            attns.append(prev_attn)
        if self.self_coverage:
            attns.append(curr_attn)
        attns = torch.cat(attns, dim=1)

        attns = attns.cumsum(dim=2) - attns
        
        # Get the actual dimensions after rearrangement
        try:
            attns = rearrange(attns, "b n t (h w) -> (b t) n h w", h=h)
        except RuntimeError as e:
            # If rearrangement fails, adjust the dimensions
            print(f"Warning: Attention rearrangement failed with error: {e}")
            print(f"attns shape before rearrange: {attns.shape}, h: {h}")
            
            # Adjust attns to have a compatible shape
            b, n, t, l = attns.shape
            w = l // h
            if l % h != 0:
                # Pad attns to make l divisible by h
                pad_size = h - (l % h)
                padding = torch.zeros((b, n, t, pad_size), device=device)
                attns = torch.cat([attns, padding], dim=3)
                l = attns.shape[3]
                w = l // h
            
            attns = rearrange(attns, "b n t (h w) -> (b t) n h w", h=h, w=w)
            
            # Update mask to match the new dimensions
            if mask.shape[2:] != attns.shape[2:]:
                mask = torch.zeros((attns.shape[0], 1, attns.shape[2], attns.shape[3]), 
                                  dtype=torch.bool, device=device)

        # Ensure mask has the right shape for the operation
        if mask.shape[2:] != attns.shape[2:]:
            print(f"Warning: Mask shape {mask.shape} doesn't match attns shape {attns.shape}")
            mask = torch.zeros((attns.shape[0], 1, attns.shape[2], attns.shape[3]), 
                              dtype=torch.bool, device=device)

        cov = self.conv(attns)
        cov = self.act(cov)

        cov = cov.masked_fill(mask, 0.0)
        cov = self.proj(cov)

        cov = self.post_norm(cov, mask)

        cov = rearrange(cov, "(b t) n h w -> (b n) t (h w)", t=t)
        return cov
