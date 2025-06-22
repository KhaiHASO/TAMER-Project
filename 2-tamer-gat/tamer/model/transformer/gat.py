import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class GraphAttentionLayer(nn.Module):
    """
    Graph attention layer based on the GAT paper:
    "Graph Attention Networks" - Veličković et al. (2018)
    """
    def __init__(self, in_features, out_features, dropout=0.3, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha

        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, h, adj_mat=None):
        """
        h: node features [B, N, F]
        adj_mat: adjacency matrix [B, N, N] or None (fully connected)
        """
        batch_size, N, _ = h.shape
        Wh = self.W(h)  # [B, N, F']
        
        # Create all pairs of nodes for attention computation
        a_input = torch.cat([
            Wh.repeat(1, 1, N).view(batch_size, N * N, -1),
            Wh.repeat(1, N, 1)
        ], dim=2).view(batch_size, N, N, 2 * self.out_features)
        
        # Attention coefficients
        e = self.leakyrelu(self.a(a_input).squeeze(-1))  # [B, N, N]
        
        # Mask out attention to non-connected nodes
        if adj_mat is not None:
            e = e.masked_fill(adj_mat == 0, -9e15)
        
        # Attention weights through softmax
        attention = F.softmax(e, dim=2)  # [B, N, N]
        attention = self.dropout_layer(attention)
        
        # Apply attention to node features
        h_prime = torch.bmm(attention, Wh)  # [B, N, F']
        
        return h_prime


class MultiHeadGAT(nn.Module):
    """Multi-head Graph Attention Network"""
    def __init__(self, in_features, out_features, nhead=8, dropout=0.3, alpha=0.2, concat=True):
        super(MultiHeadGAT, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.nhead = nhead
        self.concat = concat
        
        if concat:
            assert out_features % nhead == 0, "out_features must be divisible by nhead if concat=True"
            self.out_features_per_head = out_features // nhead
        else:
            self.out_features_per_head = out_features
            
        self.attentions = nn.ModuleList([
            GraphAttentionLayer(
                in_features, 
                self.out_features_per_head,
                dropout=dropout, 
                alpha=alpha
            ) for _ in range(nhead)
        ])
        
        self.out_proj = nn.Linear(
            self.out_features_per_head * nhead if concat else out_features,
            out_features
        )
        
    def forward(self, x, adj_mat=None):
        """
        x: node features [B, N, F]
        adj_mat: adjacency matrix [B, N, N] or None
        """
        if self.concat:
            return self.out_proj(
                torch.cat([att(x, adj_mat) for att in self.attentions], dim=2)
            )
        else:
            return self.out_proj(
                torch.mean(
                    torch.stack([att(x, adj_mat) for att in self.attentions]), 
                    dim=0
                )
            )


class GATModule(nn.Module):
    """
    GAT module to replace the Tree-Aware module in TAMER
    """
    def __init__(self, d_model, nhead=8, num_layers=2, dropout=0.3):
        super(GATModule, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        
        self.gat_layers = nn.ModuleList([
            MultiHeadGAT(
                d_model if i == 0 else d_model,
                d_model,
                nhead=nhead,
                dropout=dropout
            ) for i in range(num_layers)
        ])
        
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])
        
    def forward(self, x, padding_mask=None):
        """
        x: input tensor [T, B, D] where T is sequence length, B is batch size, D is dimension
        padding_mask: [B, T] where 1 indicates padding
        """
        # Rearrange to [B, T, D]
        x = x.permute(1, 0, 2)
        
        # Create attention mask from padding mask
        if padding_mask is not None:
            B, T = padding_mask.size()
            # Convert padding mask to attention mask
            adj_mat = (~padding_mask.unsqueeze(1).repeat(1, T, 1)).float()
        else:
            adj_mat = None
            
        # Apply GAT layers with residual connections and layer normalization
        for i in range(self.num_layers):
            residual = x
            x = self.gat_layers[i](x, adj_mat)
            x = self.norm_layers[i](x + residual)
            
        # Return to [T, B, D] format
        return x.permute(1, 0, 2) 