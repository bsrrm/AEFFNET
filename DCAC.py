"""AC extension for AEFFNet: contrast-guided directional adaptive convolution.

Public CFBlock/C3k2_CFBlock names stay compatible with the supplied detector.
The default is the proposed CDAC architecture. Set legacy=True for the supplied
fixed-strip baseline. New parameters require training; old checkpoints do not
acquire the proposed architecture by importing this file.

Standalone dependency: PyTorch only. See README.md for integration and limits.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["CFBlock", "CDACAttention", "DetailGatedFFN", "MLP",
           "ConvolutionalAttention", "collect_cdac_ignored_layers"]


class DropPath(nn.Module):
    """Per-sample stochastic depth; disabled in evaluation or at probability 0."""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        if not 0 <= drop_prob < 1:
            raise ValueError("drop_path_rate must be in [0, 1)")
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep


class _LegacyConv(nn.Module):
    """Equivalent 1x1 Conv-BN-SiLU adapter used only for legacy unequal widths."""
    def __init__(self, c1, c2):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class MLP(nn.Module):
    """Unchanged two-dense-3x3 baseline feed-forward architecture."""
    def __init__(self, in_channels, hidden_channels=None, out_channels=None, drop_rate=0.):
        super().__init__()
        hidden_channels = hidden_channels or in_channels
        out_channels = out_channels or in_channels
        self.norm = nn.SyncBatchNorm(in_channels, eps=1e-06)
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 3, 1, 1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, 3, 1, 1)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.norm(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x)
        return self.drop(x)


class ConvolutionalAttention(nn.Module):
    """Supplied fixed-strip, shared-weight baseline (kept for ablations)."""
    def __init__(self, in_channels, out_channels, inter_channels, num_heads=8):
        super().__init__()
        if out_channels % num_heads or inter_channels % num_heads:
            raise ValueError("legacy out_channels and inter_channels must divide by num_heads")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inter_channels = inter_channels
        self.num_heads = num_heads
        self.norm = nn.SyncBatchNorm(in_channels)
        self.kv = nn.Parameter(torch.zeros(inter_channels, in_channels, 7, 1))
        self.kv3 = nn.Parameter(torch.zeros(inter_channels, in_channels, 1, 7))
        nn.init.trunc_normal_(self.kv, std=0.001)
        nn.init.trunc_normal_(self.kv3, std=0.001)

    def _act_dn(self, x):
        n, _, h, w = x.shape
        x = x.reshape(n, self.num_heads, self.inter_channels // self.num_heads, -1)
        x = F.softmax(x, dim=3)
        x = x / (torch.sum(x, dim=2, keepdim=True) + 1e-06)
        return x.reshape(n, self.inter_channels, h, w)

    def forward(self, x):
        x = self.norm(x)
        x1 = self._act_dn(F.conv2d(x, self.kv, padding=(3, 0)))
        x1 = F.conv2d(x1, self.kv.transpose(1, 0), padding=(3, 0))
        x3 = self._act_dn(F.conv2d(x, self.kv3, padding=(0, 3)))
        x3 = F.conv2d(x3, self.kv3.transpose(1, 0), padding=(0, 3))
        return x1 + x3


def _local_mean(x):
    return F.avg_pool2d(x, 3, stride=1, padding=1, count_include_pad=False)


class CDACAttention(nn.Module):
    """Local/global routing over directional scales and a high-pass branch.

    One routing weight per branch and pixel is broadcast over latent channels.
    All branches are evaluated; this is feature adaptation, not conditional
    execution. Spatial normalization competes across branches, not positions.
    """
    def __init__(self, channels, inter_channels=None, kernels=(3, 7),
                 use_detail=True, use_contrast=True, use_routing=True,
                 use_spatial_routing=True, adaptive_temperature=True):
        super().__init__()
        kernels = tuple(int(k) for k in kernels)
        if not kernels or len(set(kernels)) != len(kernels) or any(k < 1 or k % 2 == 0 for k in kernels):
            raise ValueError("kernels must contain distinct positive odd integers")
        latent = min(64, max(8, channels // 2)) if inter_channels is None else int(inter_channels)
        if channels < 2 or latent < 2:
            raise ValueError("CDAC requires channels >= 2 and inter_channels >= 2")
        self.use_detail = bool(use_detail)
        self.use_contrast = bool(use_contrast)
        self.use_routing = bool(use_routing)
        self.use_spatial_routing = bool(use_spatial_routing)
        self.adaptive_temperature = bool(adaptive_temperature)
        self.norm = nn.GroupNorm(1, channels, eps=1e-6)
        self.project_in = nn.Conv2d(channels, latent, 1, bias=False)
        self.branches = nn.ModuleList()
        for k in kernels:
            self.branches.extend([
                nn.Conv2d(latent, latent, (1, k), padding=(0, k // 2), groups=latent, bias=False),
                nn.Conv2d(latent, latent, (k, 1), padding=(k // 2, 0), groups=latent, bias=False),
            ])
        self.detail = nn.Conv2d(latent, latent, 3, padding=1, groups=latent, bias=False) if use_detail else None
        self.branch_count = len(self.branches) + int(use_detail)
        router_width = max(8, min(32, latent // 4))
        self.spatial_router = nn.Sequential(
            nn.Conv2d(4, router_width, 1), nn.SiLU(),
            nn.Conv2d(router_width, self.branch_count + 1, 1))
        self.global_router = nn.Sequential(
            nn.Conv2d(latent, router_width, 1), nn.SiLU(),
            nn.Conv2d(router_width, self.branch_count + 1, 1))
        self.project_out = nn.Conv2d(latent, channels, 1, bias=False)
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8.0
        self.register_buffer("sobel_x", sobel_x[None, None], persistent=True)
        self.register_buffer("sobel_y", sobel_x.t()[None, None].contiguous(), persistent=True)

    def routing_cues(self, z):
        # Float32 reductions limit overflow/underflow under mixed precision.
        # These are learned-feature cues, not calibrated physical illumination.
        with torch.autocast(device_type=z.device.type, enabled=False):
            a = z.float().abs().mean(dim=1, keepdim=True)
            b = _local_mean(a)
            contrast = (a - b).abs() / (b + 1e-6)
            if not self.use_contrast:
                contrast = torch.zeros_like(contrast)
            padded = F.pad(a, (1, 1, 1, 1), mode="replicate")
            gx = F.conv2d(padded, self.sobel_x.float()).abs() / (b + 1e-6)
            gy = F.conv2d(padded, self.sobel_y.float()).abs() / (b + 1e-6)
            amplitude = a / (a.mean(dim=(2, 3), keepdim=True) + 1e-6)
            return torch.tanh(torch.cat((amplitude, contrast, gx, gy), dim=1)).to(z.dtype)

    def branch_weights(self, z):
        """Return BxMxHxW weights and Bx1xHxW bounded temperature."""
        if not self.use_routing:
            shape = (z.shape[0], self.branch_count, z.shape[2], z.shape[3])
            return z.new_full(shape, 1.0 / self.branch_count), z.new_ones((z.shape[0], 1, z.shape[2], z.shape[3]))
        logits = self.global_router(z.mean(dim=(2, 3), keepdim=True))
        if self.use_spatial_routing:
            logits = logits + self.spatial_router(self.routing_cues(z))
        # Expand gives global-only ablations the same spatial API.
        logits = logits.expand(-1, -1, z.shape[2], z.shape[3]).float()
        scores = logits[:, :self.branch_count]
        temperature = 0.5 + 1.5 * torch.sigmoid(logits[:, self.branch_count:]) if self.adaptive_temperature else torch.ones_like(logits[:, :1])
        weights = torch.softmax(scores / temperature, dim=1)
        return weights.to(z.dtype), temperature.to(z.dtype)

    def forward(self, x):
        z = self.project_in(self.norm(x))
        weights, _ = self.branch_weights(z)
        mixed = torch.zeros_like(z)
        for i, branch in enumerate(self.branches):
            mixed = mixed + weights[:, i:i + 1] * branch(z)
        if self.detail is not None:
            mixed = mixed + weights[:, -1:] * self.detail(z - _local_mean(z))
        return self.project_out(mixed)


class DetailGatedFFN(nn.Module):
    """Depthwise value transform gated by context and a local high-pass term."""
    def __init__(self, channels, expansion=2.0, use_detail=True, drop_rate=0.):
        super().__init__()
        hidden = max(2, int(channels * expansion))
        self.norm = nn.GroupNorm(1, channels, eps=1e-6)
        self.value = nn.Conv2d(channels, hidden, 1, bias=False)
        self.value_dw = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False)
        self.gate = nn.Conv2d(channels, hidden, 1, bias=False)
        self.detail_gate = nn.Conv2d(channels, hidden, 1, bias=False) if use_detail else None
        self.project_out = nn.Conv2d(hidden, channels, 1, bias=False)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        n = self.norm(x)
        v = self.value_dw(self.value(n))
        g = self.gate(n)
        if self.detail_gate is not None:
            g = g + self.detail_gate(n - _local_mean(n))
        y = self.drop(F.gelu(v) * torch.sigmoid(g))
        return self.drop(self.project_out(y))


class CFBlock(nn.Module):
    """Drop-in class with a new CDAC default and an explicit legacy baseline.

    The first five positional arguments match the supplied CFBlock. num_heads
    is used by legacy=True only; CDAC uses scalar branch maps and has no fixed
    head/channel divisibility requirement. New options are keyword-only.
    """
    def __init__(self, in_channels, out_channels, num_heads=8, drop_rate=0.,
                 drop_path_rate=0., *, legacy=False, inter_channels=None,
                 kernels=(3, 7), use_detail=True, use_contrast=True,
                 use_routing=True, use_spatial_routing=True,
                 adaptive_temperature=True, use_gated_ffn=True, expansion=2.0):
        super().__init__()
        if min(in_channels, out_channels) < 2:
            raise ValueError("CFBlock requires at least two input/output channels")
        if not 0 <= drop_rate < 1 or not 0 <= drop_path_rate < 1 or expansion <= 0:
            raise ValueError("drop_rate and drop_path_rate must be in [0,1); expansion must be positive")
        self.legacy_mode = bool(legacy)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        if legacy:
            self.conv1x1 = _LegacyConv(in_channels, out_channels) if in_channels != out_channels else nn.Identity()
            self.attn_l = ConvolutionalAttention(out_channels, out_channels, 64, num_heads)
            self.mlp_l = MLP(out_channels, drop_rate=drop_rate)
        else:
            self.conv1x1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.GroupNorm(1, out_channels, eps=1e-6), nn.SiLU()) if in_channels != out_channels else nn.Identity()
            self.attn_l = CDACAttention(out_channels, inter_channels, kernels, use_detail,
                                       use_contrast, use_routing, use_spatial_routing, adaptive_temperature)
            self.mlp_l = DetailGatedFFN(out_channels, expansion, use_detail, drop_rate) if use_gated_ffn else MLP(out_channels, drop_rate=drop_rate)

    def forward(self, x):
        # Existing full-model pickle objects have the same three child names.
        # They continue executing their original modules; no architecture is
        # silently substituted during checkpoint deserialization.
        x = self.conv1x1(x)
        x = x + self.drop_path(self.attn_l(x))
        return x + self.drop_path(self.mlp_l(x))


def collect_cdac_ignored_layers(model):
    """Fixed routing-head outputs represent branches plus temperature.

    Add these Conv2d objects to the compressor's ignored_layers. This protects
    their output widths; it does not certify the full detector's pruning path.
    """
    ignored = []
    for module in model.modules():
        if isinstance(module, CDACAttention):
            ignored.extend((module.spatial_router[-1], module.global_router[-1]))
    return ignored
