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
        # Check for empty tensors
        if x.numel() == 0 or x.shape[0] == 0 or x.shape[2] == 0 or x.shape[3] == 0:
            print("Warning: Empty tensor in MaskBatchNorm2d, skipping normalization")
            return x
            
        # Ensure both tensors are on the same device
        device = x.device
        mask = mask.to(device)
        
        # Ensure mask has the right shape
        if mask.shape[0] != x.shape[0] or mask.shape[2] != x.shape[2] or mask.shape[3] != x.shape[3]:
            print(f"Warning: mask shape {mask.shape} doesn't match x shape {x.shape} in MaskBatchNorm2d")
            # Create a new mask with the right dimensions
            new_mask = torch.zeros((x.shape[0], 1, x.shape[2], x.shape[3]), dtype=mask.dtype, device=device)
            # Copy what we can
            copy_b = min(mask.shape[0], x.shape[0])
            copy_h = min(mask.shape[2], x.shape[2])
            copy_w = min(mask.shape[3], x.shape[3])
            if mask.shape[1] > 0:
                new_mask[:copy_b, 0, :copy_h, :copy_w] = mask[:copy_b, 0, :copy_h, :copy_w]
            mask = new_mask
        
        try:
            x = rearrange(x, "b d h w -> b h w d")
            mask = mask.squeeze(1)

            not_mask = ~mask

            # Check if there are any non-masked elements
            if not torch.any(not_mask):
                # If all elements are masked, return the input unchanged
                return x.permute(0, 3, 1, 2).contiguous()  # Reshape back to original format

            flat_x = x[not_mask, :]
            
            # Apply batch norm only if we have enough elements
            if flat_x.size(0) > 1:  # BatchNorm1d needs at least 2 elements
                flat_x = self.bn(flat_x)
                x[not_mask, :] = flat_x
            
            x = rearrange(x, "b h w d -> b d h w")
            return x
            
        except Exception as e:
            print(f"Error in MaskBatchNorm2d: {e}")
            # In case of error, return the input unchanged
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
            [(b * nhead), t, l] - Same shape as prev_attn
        """
        # Check for empty tensor - return early if empty
        if prev_attn.numel() == 0 or prev_attn.shape[0] == 0 or prev_attn.shape[1] == 0 or prev_attn.shape[2] == 0:
            print("Warning: Empty tensor in ARM module, skipping processing")
            return prev_attn

        # Store original shape and device
        original_shape = prev_attn.shape
        device = prev_attn.device
        
        # If curr_attn is not provided, use prev_attn
        if curr_attn is None:
            curr_attn = prev_attn
        
        # Make sure curr_attn is on the same device
        curr_attn = curr_attn.to(device)
        
        # Handle key_padding_mask
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.to(device)
        else:
            # Create a dummy mask if none provided
            b_times_nhead, _, l = prev_attn.shape
            b = b_times_nhead // self.nhead
            key_padding_mask = torch.zeros((b, l), dtype=torch.bool, device=device)
        
        # Get dimensions
        b_times_nhead, t, l = prev_attn.shape
        
        # Safety check for dimensions
        if b_times_nhead == 0 or t == 0 or l == 0:
            print("Warning: Zero dimension in ARM input tensor, skipping processing")
            return prev_attn
            
        nhead = self.nhead
        
        # Safety check for divisibility
        if b_times_nhead % nhead != 0:
            print(f"Warning: b_times_nhead ({b_times_nhead}) not divisible by nhead ({nhead})")
            # Adjust to make it divisible
            b = max(1, b_times_nhead // nhead)
            new_size = b * nhead
            
            # Create new tensors with adjusted dimensions
            new_prev_attn = torch.zeros((new_size, t, l), device=device)
            new_curr_attn = torch.zeros((new_size, t, l), device=device)
            
            # Copy data that fits
            copy_size = min(b_times_nhead, new_size)
            new_prev_attn[:copy_size] = prev_attn[:copy_size]
            new_curr_attn[:copy_size] = curr_attn[:copy_size]
            
            prev_attn = new_prev_attn
            curr_attn = new_curr_attn
            b_times_nhead = new_size
        
        b = b_times_nhead // nhead
        
        # Now reshape
        prev_attn_reshaped = prev_attn.view(b, nhead, t, l)
        curr_attn_reshaped = curr_attn.view(b, nhead, t, l)
        
        # Create attns tensor based on coverage types
        attns = []
        if self.cross_coverage:
            attns.append(prev_attn_reshaped)
        if self.self_coverage:
            attns.append(curr_attn_reshaped)
        
        # Safety check - if no attentions to process, return original
        if not attns:
            print("Warning: No attention types selected in ARM module")
            return prev_attn
            
        attns = torch.cat(attns, dim=1)
        
        # Cumulative sum along sequence dimension
        attns = attns.cumsum(dim=2) - attns
        
        # Reshape for 2D convolution
        n_channels = attns.shape[1]  # This should be nhead or 2*nhead
        
        # Check if l is divisible by h, if not, pad
        if l % h != 0:
            pad_size = h - (l % h)
            padding = torch.zeros((b, n_channels, t, pad_size), device=device)
            attns = torch.cat([attns, padding], dim=3)
            l = attns.shape[3]
        
        # Safety check for h value
        if h <= 0:
            print(f"Warning: Invalid h value ({h}) in ARM module, using default")
            h = 1
            
        w = max(1, l // h)  # Ensure w is at least 1
        
        # Reshape with safety checks
        try:
            attns = attns.reshape(b * t, n_channels, h, w)
        except RuntimeError as e:
            print(f"Error reshaping in ARM: {e}")
            print(f"Shapes: b={b}, t={t}, n_channels={n_channels}, h={h}, w={w}, l={l}")
            # Return original tensor if reshaping fails
            return prev_attn
        
        # Ensure the key_padding_mask has the right shape
        if key_padding_mask.shape[0] != b or key_padding_mask.shape[1] != l:
            # Create a new mask with the correct dimensions
            new_mask = torch.zeros((b, l), dtype=torch.bool, device=device)
            # Copy as much as we can from the original mask
            copy_rows = min(key_padding_mask.shape[0], b)
            copy_cols = min(key_padding_mask.shape[1], l)
            new_mask[:copy_rows, :copy_cols] = key_padding_mask[:copy_rows, :copy_cols]
            key_padding_mask = new_mask
        
        # Create 2D mask
        try:
            mask = key_padding_mask.reshape(b, 1, 1, l)
            mask = mask.expand(b, 1, t, l)
            mask = mask.reshape(b * t, 1, h, w)
        except RuntimeError as e:
            print(f"Error reshaping mask in ARM: {e}")
            # Create a default mask if reshaping fails
            mask = torch.zeros((b * t, 1, h, w), dtype=torch.bool, device=device)
        
        # Apply convolution
        cov = self.conv(attns)
        cov = self.act(cov)
        
        # Apply mask
        cov = cov.masked_fill(mask, 0.0)
        cov = self.proj(cov)
        
        # Apply batch norm
        cov = self.post_norm(cov, mask)
        
        # Reshape back to original format
        try:
            cov = cov.reshape(b, t, nhead, h * w)
            cov = cov.permute(0, 2, 1, 3).contiguous()
        except RuntimeError as e:
            print(f"Error in final reshape of ARM: {e}")
            # Return original if reshaping fails
            return prev_attn
        
        # Trim to the original length if needed
        if cov.shape[3] > original_shape[2]:
            cov = cov[:, :, :, :original_shape[2]]
        
        # Reshape to match the original input format
        cov = cov.view(b * nhead, t, -1)
        
        # Ensure the output size matches the input
        if cov.shape[2] < original_shape[2]:
            # If too short, pad
            padding = torch.zeros((cov.shape[0], cov.shape[1], original_shape[2] - cov.shape[2]), 
                               device=device)
            cov = torch.cat([cov, padding], dim=2)
        elif cov.shape[2] > original_shape[2]:
            # If too long, truncate
            cov = cov[:, :, :original_shape[2]]
        
        return cov
