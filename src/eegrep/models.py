"""The fixed model (the control variable, not the contribution).

DenseGCN (Kipf & Welling 2017) on 19-node graphs with dense normalised adjacency:
    H1 = Dropout(BN(ReLU(Â X W0)))    W0: F×64
    H2 = ReLU(Â H1 W1)                W1: 64×32
    g  = mean over nodes -> Linear(32→3)
≈2.8k parameters for F=7. DenseGAT (single head) is the A6 sensitivity check.
Identity adjacency turns either model into a per-node MLP (ablation A1). EEGNet and a 1D-CNN on the raw
window are the deep-learning baselines (NB06; arch "eegnet" | "cnn1d").
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


class EEGNet(nn.Module):
    """Deep-learning baseline (NB06): EEGNet-8,2 (Lawhern et al. 2018) on the raw window (B, C, T); the graph is
    ignored. Temporal conv (64 taps = 0.5 s at 128 Hz) -> depthwise spatial conv (D=2) -> separable conv ->
    linear. Max-norm weight constraints of the original are omitted (AdamW weight decay instead)."""

    def __init__(self, n_channels: int, n_times: int, n_classes: int = 3, dropout: float = 0.5,
                 f1: int = 8, d: int = 2, f2: int = 16, kern: int = 64):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, f1, (1, kern), padding="same", bias=False), nn.BatchNorm2d(f1),
            nn.Conv2d(f1, f1 * d, (n_channels, 1), groups=f1, bias=False), nn.BatchNorm2d(f1 * d), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(dropout))
        self.block2 = nn.Sequential(
            nn.Conv2d(f1 * d, f1 * d, (1, 16), padding="same", groups=f1 * d, bias=False),
            nn.Conv2d(f1 * d, f2, 1, bias=False), nn.BatchNorm2d(f2), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(dropout))
        self.head = nn.Linear(f2 * (n_times // 32), n_classes)

    def forward(self, x, a=None):
        h = self.block2(self.block1(x.unsqueeze(1)))
        return self.head(h.flatten(1))


class CNN1D(nn.Module):
    """Deep-learning baseline (NB06): three temporal conv blocks over channels-as-features, global average
    pooling, linear head; the graph is ignored."""

    def __init__(self, n_channels: int, n_times: int, n_classes: int = 3, dropout: float = 0.5,
                 widths=(32, 64, 64), kern: int = 7):
        super().__init__()
        layers, c = [], n_channels
        for i, w in enumerate(widths):
            layers += [nn.Conv1d(c, w, kern, padding=kern // 2), nn.BatchNorm1d(w), nn.ReLU()]
            if i < len(widths) - 1:
                layers.append(nn.MaxPool1d(4))
            c = w
        self.features = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(c, n_classes)

    def forward(self, x, a=None):
        return self.head(self.dropout(self.features(x).mean(dim=-1)))


def build_model(in_dim: int, model_cfg: dict, n_classes: int = 3, n_nodes: int = 19) -> nn.Module:
    """in_dim = node-feature size F (for the raw-window baselines: the number of samples per window)."""
    if model_cfg["arch"] == "eegnet":
        return EEGNet(n_nodes, in_dim, n_classes, model_cfg["dropout"])
    if model_cfg["arch"] == "cnn1d":
        return CNN1D(n_nodes, in_dim, n_classes, model_cfg["dropout"])
    return GraphClassifier(in_dim, tuple(model_cfg["hidden"]), n_classes, model_cfg["dropout"],
                           model_cfg["batchnorm"], model_cfg["arch"])


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
