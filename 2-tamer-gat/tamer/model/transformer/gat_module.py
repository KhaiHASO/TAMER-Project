import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class GraphAttentionModule(nn.Module):
    def __init__(self, in_channels, out_channels, heads=1, concat=True, dropout=0.0):
        super().__init__()
        self.gat = GATConv(in_channels, out_channels, heads=heads, concat=concat, dropout=dropout)

    def forward(self, x, edge_index):
        # x: [N, in_channels], edge_index: [2, E]
        return self.gat(x, edge_index) 