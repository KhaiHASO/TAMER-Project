import math
from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange
from torch import FloatTensor, LongTensor
import timm
import cv2
import torchvision.transforms as T

from .pos_enc import ImgPosEnc


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 1,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        
        # Load ViT model với pretrained=False để tránh lỗi mismatch
        self.vit = timm.create_model(
            'vit_base_patch16_224',
            pretrained=False,  # Để False vì dùng custom cấu hình
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
        )
        
        # Remove classification head
        self.vit.head = nn.Identity()
        
        self.out_channels = embed_dim

    def forward(self, x, x_mask):
        # ViT expects input in format [B, C, H, W]
        # Get features from ViT
        features = self.vit.forward_features(x)
        
        # Reshape features to match expected format
        # ViT outputs [B, N, C] where N = (H/P) * (W/P)
        # We need to reshape to [B, H/P, W/P, C]
        B, N, C = features.shape
        
        # Tính toán đúng kích thước H, W dựa trên số patch thực tế
        # N = (H/P) * (W/P) = (224/16) * (224/16) = 14 * 14 = 196
        # Nhưng có thể có thêm 1 patch cho cls_token
        if N == 197:  # 196 + 1 (cls_token)
            H = W = 14
            # Bỏ qua cls_token, chỉ lấy patch tokens
            features = features[:, 1:, :]  # Remove cls_token
        elif N == 196:  # Chỉ có patch tokens
            H = W = 14
        else:
            # Fallback: tính toán dựa trên sqrt
        H = W = int(math.sqrt(N))
        
        features = features.reshape(B, H, W, C)
        
        # Update mask to match new dimensions
        out_mask = x_mask[:, ::16, ::16]  # Assuming patch_size=16
        
        return features, out_mask


class Encoder(pl.LightningModule):
    def __init__(self, d_model: int, growth_rate: int, num_layers: int):
        super().__init__()

        # Initialize ViT
        self.model = VisionTransformer(
            img_size=224,  # Adjust based on your input size
            patch_size=16,
            in_chans=1,
            embed_dim=d_model,  # Use d_model as embed_dim
            depth=num_layers,  # Use num_layers as depth
            num_heads=8,  # Use 8 heads as in original config
            mlp_ratio=4.0,
            drop_rate=0.3,  # Use dropout from config
        )

        self.pos_enc_2d = ImgPosEnc(d_model, normalize=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, img: FloatTensor, img_mask: LongTensor
    ) -> Tuple[FloatTensor, LongTensor]:
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
        # extract feature
        feature, mask = self.model(img, img_mask)

        # positional encoding
        feature = self.pos_enc_2d(feature, mask)
        feature = self.norm(feature)

        return feature, mask
