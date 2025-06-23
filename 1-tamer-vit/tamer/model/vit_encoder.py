import math
from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import FloatTensor, LongTensor

from .pos_enc import ImgPosEnc

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding """
    def __init__(self, img_size=224, patch_size=16, in_chans=1, embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        return self.proj(x)  # (B, embed_dim, H', W')

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

def interpolate_pos_embed(pos_embed, target_len):
    """Interpolate position embeddings to a new size."""
    # Interpolate position embedding for different sequence lengths
    # Consider the cls token separately
    cls_pos_embed = pos_embed[:, 0:1, :]
    patch_pos_embed = pos_embed[:, 1:, :]
    
    # Original length of the position embedding
    src_len = patch_pos_embed.shape[1]
    
    # Required length for the sequence
    tgt_len = target_len - 1  # -1 for cls token
    
    # Calculate ratio for interpolation
    ratio = tgt_len / src_len
    
    # Interpolate the patch position embeddings
    interploated_patch_pos_embed = F.interpolate(
        patch_pos_embed.permute(0, 2, 1),  # [B, D, L]
        size=tgt_len,
        mode='linear'
    ).permute(0, 2, 1)  # [B, L, D]
    
    # Concatenate with the cls token position embedding
    return torch.cat([cls_pos_embed, interploated_patch_pos_embed], dim=1)

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=1, embed_dim=256, depth=12,
                 num_heads=8, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
            
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim))
        
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate)
            for i in range(depth)])
            
        self.norm = nn.LayerNorm(embed_dim)

        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward_features(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)
        
        # Reshape to sequence form
        x = x.flatten(2).transpose(1, 2)  # (B, n_patches, embed_dim)
        
        # Add cls token and position embedding
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        # Add interpolated positional embedding if sequence length doesn't match
        seq_len = x.size(1)
        if seq_len != self.pos_embed.size(1):
            pos_embed = interpolate_pos_embed(self.pos_embed, seq_len)
            x = x + pos_embed
        else:
            x = x + self.pos_embed
        
        # Apply transformer blocks
        for blk in self.blocks:
            x = blk(x)
            
        x = self.norm(x)
        return x

    def forward(self, x):
        return self.forward_features(x)

class ViTEncoder(pl.LightningModule):
    def __init__(self, d_model: int, img_size: int = 224, patch_size: int = 16, 
                 depth: int = 12, num_heads: int = 8, drop_rate: float = 0.1):
        super().__init__()
        
        self.vit = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=1,
            embed_dim=d_model,
            depth=depth,
            num_heads=num_heads,
            drop_rate=drop_rate,
        )
        
        self.feature_proj = nn.Linear(d_model, d_model)
        self.pos_enc_2d = ImgPosEnc(d_model, normalize=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, img: FloatTensor, img_mask: LongTensor) -> Tuple[FloatTensor, LongTensor]:
        """encode image to feature

        Parameters
        ----------
        img : FloatTensor
            [b, 1, h', w']
        img_mask: LongTensor
            [b, h', w']

        Returns
        -------
        Tuple[FloatTensor, LongTensor]
            [b, h, w, d], [b, h, w]
        """
        # Tính toán kích thước và đảm bảo chia hết cho patch_size
        b, c, h, w = img.shape
        patch_size = self.vit.patch_size
        
        h_padded = math.ceil(h / patch_size) * patch_size
        w_padded = math.ceil(w / patch_size) * patch_size
        
        if h != h_padded or w != w_padded:
            pad_h = h_padded - h
            pad_w = w_padded - w
            img = F.pad(img, (0, pad_w, 0, pad_h))
            img_mask = F.pad(img_mask, (0, pad_w, 0, pad_h), value=1)
        
        # Extract features from ViT
        features = self.vit(img)  # [b, n_patches+1, d]
        
        # Skip class token
        patch_features = features[:, 1:]  # [b, n_patches, d]
        
        # Reshape to 2D grid
        h_patches = h_padded // patch_size
        w_patches = w_padded // patch_size
        feature_grid = patch_features.reshape(b, h_patches, w_patches, -1)  # [b, h', w', d]
        
        # Project features
        feature_grid = self.feature_proj(feature_grid)
        
        # Create mask for features
        mask_downsampled = img_mask[:, ::patch_size, ::patch_size]
        mask_downsampled = mask_downsampled[:, :h_patches, :w_patches]
        
        # Add positional encoding
        feature_grid = self.pos_enc_2d(feature_grid, mask_downsampled)
        feature_grid = self.norm(feature_grid)
        
        return feature_grid, mask_downsampled 