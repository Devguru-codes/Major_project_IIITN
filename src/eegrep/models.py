"""The fixed model (the control variable, not the contribution).

DenseGCN (Kipf & Welling 2017) on 19-node graphs with dense normalised adjacency:
    H1 = Dropout(BN(ReLU(Â X W0)))    W0: F×64
    H2 = ReLU(Â H1 W1)                W1: 64×32
    g  = mean over nodes -> Linear(32→3)
≈2.8k parameters for F=7. DenseGAT (single head) is the A6 sensitivity check.
Identity adjacency turns either model into a per-node MLP (ablation A1).
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DenseGCNLayer(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out)

    def forward(self, x, a):            # x: (B, C, F), a: (B, C, C) or (C, C)
        return a @ self.lin(x)


class DenseGATLayer(nn.Module):
    """Single-head GAT restricted to the edges of `a` (self-loops included by Â)."""

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out, bias=False)
        self.att_src = nn.Parameter(torch.empty(d_out))
        self.att_dst = nn.Parameter(torch.empty(d_out))
        self.bias = nn.Parameter(torch.zeros(d_out))
        nn.init.normal_(self.att_src, std=0.1)
        nn.init.normal_(self.att_dst, std=0.1)

    def forward(self, x, a):
        h = self.lin(x)                                        # (B, C, D)
        e = F.leaky_relu((h @ self.att_dst)[..., :, None] + (h @ self.att_src)[..., None, :], 0.2)
        e = e.masked_fill(a.expand_as(e) <= 0, float("-inf"))
        return torch.softmax(e, dim=-1) @ h + self.bias


class GraphClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden=(64, 32), n_classes: int = 3, dropout: float = 0.5,
                 batchnorm: bool = True, arch: str = "gcn"):
        super().__init__()
        layer = {"gcn": DenseGCNLayer, "gat": DenseGATLayer}[arch]
        dims = [in_dim, *hidden]
        self.layers = nn.ModuleList(layer(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.bn = nn.BatchNorm1d(hidden[0]) if batchnorm else None
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden[-1], n_classes)

    def embed(self, x, a):
        h = x
        for i, layer in enumerate(self.layers):
            h = F.relu(layer(h, a))
            if i == 0:
                if self.bn is not None:
                    h = self.bn(h.transpose(1, 2)).transpose(1, 2)
                h = self.dropout(h)
        return h.mean(dim=1)                                    # mean-pool over nodes

    def forward(self, x, a):
        return self.head(self.embed(x, a))


def build_model(in_dim: int, model_cfg: dict, n_classes: int = 3) -> GraphClassifier:
    return GraphClassifier(in_dim, tuple(model_cfg["hidden"]), n_classes, model_cfg["dropout"],
                           model_cfg["batchnorm"], model_cfg["arch"])


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
