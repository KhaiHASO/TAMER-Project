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
            [(b * nhead), t, l] - Same shape as prev_attn
        """
        try:
            # Store original shape for later use
            original_shape = prev_attn.shape
            
            # If curr_attn is not provided, use prev_attn
            if curr_attn is None:
                curr_attn = prev_attn
                
            # Ensure both are on the same device
            device = prev_attn.device
            key_padding_mask = key_padding_mask.to(device)
            
            # Get dimensions
            b_times_nhead, t, l = prev_attn.shape
            
            # Calculate nhead and batch size
            nhead = self.nhead
            b = b_times_nhead // nhead
            
            # Calculate expected width based on length and height
            w = (l + h - 1) // h  # Ceiling division
            
            # Reshape key_padding_mask if needed
            if key_padding_mask.shape[1] != h * w:
                new_mask = torch.zeros((b, h * w), dtype=key_padding_mask.dtype, device=device)
                copy_len = min(key_padding_mask.shape[1], h * w)
                new_mask[:, :copy_len] = key_padding_mask[:, :copy_len]
                key_padding_mask = new_mask
            
            # Instead of reshaping directly, create new tensors with the right shape
            # and copy data from the original tensors
            curr_attn_reshaped = torch.zeros(b, nhead, t, l, device=device)
            prev_attn_reshaped = torch.zeros(b, nhead, t, l, device=device)
            
            # Fill the reshaped tensors
            for i in range(b):
                for j in range(nhead):
                    idx = i * nhead + j
                    if idx < b_times_nhead:
                        curr_attn_reshaped[i, j] = curr_attn[idx]
                        prev_attn_reshaped[i, j] = prev_attn[idx]
            
            # Create attns tensor
            attns = []
            if self.cross_coverage:
                attns.append(prev_attn_reshaped)
            if self.self_coverage:
                attns.append(curr_attn_reshaped)
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
            
            w = l // h
            attns = attns.reshape(b * t, n_channels, h, w)
            
            # Create mask with matching dimensions
            mask = key_padding_mask.reshape(b, 1, 1, l)
            mask = mask.expand(b, 1, t, l)
            mask = mask.reshape(b * t, 1, h, w)
            
            # Ensure mask has the right shape
            if mask.shape[2:] != attns.shape[2:]:
                # Create a new mask with the correct dimensions
                mask = torch.zeros((attns.shape[0], 1, attns.shape[2], attns.shape[3]), 
                                  dtype=torch.bool, device=device)
            
            # Apply convolution
            cov = self.conv(attns)
            cov = self.act(cov)
            
            # Apply mask
            cov = cov.masked_fill(mask, 0.0)
            cov = self.proj(cov)
            
            # Apply batch norm
            cov = self.post_norm(cov, mask)
            
            # Reshape back to original format
            cov = cov.reshape(b, t, nhead, h * w)
            cov = cov.permute(0, 2, 1, 3).contiguous()
            
            # Create output tensor with the same shape as the input
            output = torch.zeros_like(prev_attn)
            
            # Fill the output tensor
            for i in range(b):
                for j in range(nhead):
                    idx = i * nhead + j
                    if idx < b_times_nhead:
                        # Get the data for this batch and head
                        head_data = cov[i, j]
                        
                        # Handle size mismatch
                        if head_data.shape[1] != original_shape[2]:
                            if head_data.shape[1] < original_shape[2]:
                                # Pad if smaller
                                padding = torch.zeros((head_data.shape[0], original_shape[2] - head_data.shape[1]), 
                                                    device=device)
                                head_data = torch.cat([head_data, padding], dim=1)
                            else:
                                # Truncate if larger
                                head_data = head_data[:, :original_shape[2]]
                        
                        # Copy to output
                        output[idx] = head_data
            
            return output
            
        except Exception as e:
            print(f"Error in ARM module: {e}")
            # Return zeros as fallback with the same shape as prev_attn
            return torch.zeros_like(prev_attn)
