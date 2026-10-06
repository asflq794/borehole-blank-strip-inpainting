from __future__ import annotations

# NAF-style blocks adapted from megvii-research/NAFNet (MIT).
# Modified for this restoration method; see LICENSE_NAFNet.txt.
import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 32
STAGE_STEM_WIDTH = 16
GUIDANCE_STEM_WIDTH = 16
ENC_BLKS = [2, 2, 4, 4]
MIDDLE_BLKS = 4
DEC_BLKS = [2, 2, 2, 2]
RGB_WEIGHT = 1.0
LOW_WEIGHT = 0.2
MID_WEIGHT = 0.3
HIGH_WEIGHT = 0.1

class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, channels: int, dw_expand: int = 2, ffn_expand: int = 2) -> None:
        super().__init__()
        dw_channels = channels * dw_expand
        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, dw_channels, 1)
        self.conv2 = nn.Conv2d(dw_channels, dw_channels, 3, padding=1, groups=dw_channels)
        self.sg = SimpleGate()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channels // 2, dw_channels // 2, 1),
        )
        self.conv3 = nn.Conv2d(dw_channels // 2, channels, 1)
        self.norm2 = LayerNorm2d(channels)
        ffn_channels = channels * ffn_expand
        self.conv4 = nn.Conv2d(channels, ffn_channels, 1)
        self.conv5 = nn.Conv2d(ffn_channels // 2, channels, 1)
        self.beta = nn.Parameter(torch.zeros((1, channels, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, channels, 1, 1)), requires_grad=True)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        y = inp + x * self.beta
        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        return y + x * self.gamma


class ErrorGuidedJointRefiner(nn.Module):
    """Frozen PreviousJoint architecture used by the formal base-A builder."""

    def __init__(
        self,
        width: int = WIDTH,
        enc_blk_nums: list[int] | None = None,
        middle_blk_num: int = MIDDLE_BLKS,
        dec_blk_nums: list[int] | None = None,
    ) -> None:
        super().__init__()
        enc_blk_nums = enc_blk_nums or ENC_BLKS
        dec_blk_nums = dec_blk_nums or DEC_BLKS
        self.stage1_stem = nn.Conv2d(3, STAGE_STEM_WIDTH, 3, padding=1)
        self.guidance_stem = nn.Conv2d(4, GUIDANCE_STEM_WIDTH, 3, padding=1)
        self.fusion = nn.Conv2d(STAGE_STEM_WIDTH + GUIDANCE_STEM_WIDTH, width, 3, padding=1)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        channels = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(channels) for _ in range(num)]))
            self.downs.append(nn.Conv2d(channels, channels * 2, 2, stride=2))
            channels *= 2
        self.middle = nn.Sequential(*[NAFBlock(channels) for _ in range(middle_blk_num)])
        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(nn.Conv2d(channels, channels * 2, 1, bias=False), nn.PixelShuffle(2)))
            channels //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(channels) for _ in range(num)]))
        self.ending = nn.Conv2d(width, 3, 3, padding=1)
        self.padder_size = 2 ** len(self.encoders)
        nn.init.zeros_(self.ending.weight)
        nn.init.zeros_(self.ending.bias)

    def forward(self, stage1: torch.Tensor, e_known: torch.Tensor, g_mask: torch.Tensor) -> torch.Tensor:
        h, w = stage1.shape[-2:]
        stage1, e_known, g_mask = self.pad_inputs(stage1, e_known, g_mask)
        s_feat = self.stage1_stem(stage1)
        eg_feat = self.guidance_stem(torch.cat([e_known, g_mask], dim=1))
        x = self.fusion(torch.cat([s_feat, eg_feat], dim=1))
        skips = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            skips.append(x)
            x = down(x)
        x = self.middle(x)
        for decoder, up, skip in zip(self.decoders, self.ups, skips[::-1]):
            x = up(x) + skip
            x = decoder(x)
        residual = self.ending(x)
        return residual[:, :, :h, :w]

    def pad_inputs(self, *items: torch.Tensor) -> tuple[torch.Tensor, ...]:
        h, w = items[0].shape[-2:]
        pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        if pad_h == 0 and pad_w == 0:
            return items
        return tuple(F.pad(item, (0, pad_w, 0, pad_h), mode="reflect") for item in items)


def y_channel(x: torch.Tensor) -> torch.Tensor:
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def gaussian_kernel(device: torch.device, channels: int = 1, size: int = 15, sigma: float = 4.0) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, size, size).repeat(channels, 1, 1, 1)


def torch_blur(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    pad = kernel.shape[-1] // 2
    return F.conv2d(F.pad(x, (pad, pad, pad, pad), mode="reflect"), kernel, groups=x.shape[1])


def frequency_decompose_y(x: torch.Tensor, low_kernel: torch.Tensor, mid_kernel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    y = y_channel(x)
    low = torch_blur(y, low_kernel)
    g15 = torch_blur(y, mid_kernel)
    mid = g15 - low
    high = y - g15
    return low, mid, high


def masked_mean_rgb(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask3 = mask.expand(-1, 3, -1, -1)
    denom = mask3.sum().clamp_min(1.0)
    return (value * mask3).sum() / denom


def masked_mean_band(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum().clamp_min(1.0)
    return (value * mask).sum() / denom


def build_gt_safe(original: torch.Tensor, stage1: torch.Tensor, h0_mask: torch.Tensor) -> torch.Tensor:
    h0_rgb = h0_mask.expand(-1, 3, -1, -1)
    return original * (1.0 - h0_rgb) + stage1 * h0_rgb


def build_prediction(
    model: nn.Module,
    original: torch.Tensor,
    stage1: torch.Tensor,
    h0_mask: torch.Tensor,
    t_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    u_mask = torch.clamp(h0_mask + t_mask, 0.0, 1.0)
    g_mask = 1.0 - u_mask
    e_known = (original - stage1) * g_mask
    c_base = original * g_mask + stage1 * u_mask
    residual = model(stage1, e_known, g_mask)
    pred = c_base + u_mask * residual
    return pred, {
        "u_mask": u_mask,
        "g_mask": g_mask,
        "e_known": e_known,
        "c_base": c_base,
        "residual": residual,
    }


def compute_clean_loss(
    pred: torch.Tensor,
    original: torch.Tensor,
    stage1: torch.Tensor,
    h0_mask: torch.Tensor,
    t_mask: torch.Tensor,
    low_kernel: torch.Tensor,
    mid_kernel: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    gt_safe = build_gt_safe(original, stage1, h0_mask)
    pred_low, pred_mid, pred_high = frequency_decompose_y(pred, low_kernel, mid_kernel)
    gt_low, gt_mid, gt_high = frequency_decompose_y(gt_safe, low_kernel, mid_kernel)
    rgb_loss = masked_mean_rgb(torch.abs(pred - original), t_mask)
    low_loss = masked_mean_band(torch.abs(pred_low - gt_low), t_mask)
    mid_loss = masked_mean_band(torch.abs(pred_mid - gt_mid), t_mask)
    high_loss = masked_mean_band(torch.abs(pred_high - gt_high), t_mask)
    total = RGB_WEIGHT * rgb_loss + LOW_WEIGHT * low_loss + MID_WEIGHT * mid_loss + HIGH_WEIGHT * high_loss
    return total, {
        "total_loss": float(total.detach().cpu()),
        "rgb_loss": float(rgb_loss.detach().cpu()),
        "low_loss": float(low_loss.detach().cpu()),
        "mid_loss": float(mid_loss.detach().cpu()),
        "high_loss": float(high_loss.detach().cpu()),
    }
