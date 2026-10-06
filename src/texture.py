from __future__ import annotations

from dataclasses import dataclass
from numpy.lib.stride_tricks import sliding_window_view
import math
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGE_SIZE = 512
CHANNELS = [16, 32, 64, 64, 64, 64]
TEXTURE_FEATURE_CHANNELS = 5
SOURCE_SAFE_RADIUS = 2
ROUTING_COLLAPSE_THRESHOLD = 0.50
GAUSSIAN_LOWPASS_SIGMA = 9.0
GAUSSIAN_LOWPASS_PADDING = 'reflect'

@dataclass
class RoutingMap:
    scale_name: str
    height: int
    width: int
    block: int
    source_x: np.ndarray
    source_y: np.ndarray
    dx: np.ndarray
    dy: np.ndarray
    hole_mask: np.ndarray
    safe_known: np.ndarray
    block_records: list[dict[str, float | int]]
    diagnostics: dict[str, float | int | bool | str]


def gaussian_lowpass(
    x: torch.Tensor,
    sigma: float = GAUSSIAN_LOWPASS_SIGMA,
) -> torch.Tensor:
    """Differentiable depthwise separable Gaussian low-pass."""
    if x.ndim != 4:
        raise ValueError(f"gaussian_lowpass expects [B,C,H,W], got {tuple(x.shape)}")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    radius = math.ceil(3.0 * float(sigma))
    coords = torch.arange(-radius, radius + 1, device=x.device, dtype=torch.float32)
    kernel = torch.exp(-(coords * coords) / (2.0 * float(sigma) * float(sigma)))
    kernel = (kernel / kernel.sum()).to(dtype=x.dtype)
    channels = int(x.shape[1])
    horizontal_kernel = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    vertical_kernel = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    horizontal = F.pad(x, (radius, radius, 0, 0), mode=GAUSSIAN_LOWPASS_PADDING)
    horizontal = F.conv2d(horizontal, horizontal_kernel, groups=channels)
    vertical = F.pad(horizontal, (0, 0, radius, radius), mode=GAUSSIAN_LOWPASS_PADDING)
    return F.conv2d(vertical, vertical_kernel, groups=channels)


def apply_low_frequency_projection(
    y_raw: torch.Tensor,
    a_full: torch.Tensor,
) -> torch.Tensor:
    """Replace only the Gaussian low-frequency component with A_full."""
    low_anchor = gaussian_lowpass(a_full, sigma=GAUSSIAN_LOWPASS_SIGMA).detach()
    low_raw = gaussian_lowpass(y_raw, sigma=GAUSSIAN_LOWPASS_SIGMA)
    return y_raw + (low_anchor - low_raw)


def resize_rgb_float(arr_u8: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(arr_u8.astype(np.float32) / 255.0, (size, size), interpolation=cv2.INTER_AREA)


def resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    resized = cv2.resize(mask.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
    return resized > 0


def valid_source_cells_all_k(k_mask: np.ndarray, size: int) -> np.ndarray:
    if k_mask.shape != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f"Expected native K mask {(IMAGE_SIZE, IMAGE_SIZE)}, got {k_mask.shape}")
    if IMAGE_SIZE % size != 0:
        raise ValueError(f"Feature size {size} is not an exact divisor of {IMAGE_SIZE}")
    factor = IMAGE_SIZE // size
    cropped = k_mask[: size * factor, : size * factor]
    return cropped.reshape(size, factor, size, factor).all(axis=(1, 3))


def luminance(rgb: np.ndarray) -> np.ndarray:
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def gaussian_rgb(rgb: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    return cv2.GaussianBlur(rgb, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)


def gaussian_rgb_source_texture(rgb: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(rgb, (3, 3), sigmaX=0.8, sigmaY=0.8, borderType=cv2.BORDER_REFLECT)


def sobel_xy(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gx = cv2.Sobel(y.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(y.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    return gx, gy


def build_texture_deficit_feature(
    original_u8: np.ndarray,
    a_full_u8: np.ndarray,
    size: int,
    safe_known: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    original = resize_rgb_float(original_u8, size)
    a_full = resize_rgb_float(a_full_u8, size)
    deficit = original - a_full
    detail = deficit - gaussian_rgb_source_texture(deficit)
    y_detail = luminance(detail)
    gx, gy = sobel_xy(y_detail)
    raw = np.concatenate([detail.transpose(2, 0, 1), gx[None], gy[None]], axis=0).astype(np.float32)
    scales: list[float] = []
    normalized = np.empty_like(raw)
    for channel in range(raw.shape[0]):
        values = np.abs(raw[channel][safe_known]) if np.any(safe_known) else np.empty(0, dtype=np.float32)
        p95 = float(np.percentile(values, 95)) if values.size else 1e-6
        scale = max(p95, 1e-6)
        scales.append(scale)
        normalized[channel] = np.clip(raw[channel] / scale, -1.0, 1.0)
    normalized[:, ~safe_known] = 0.0
    return normalized, detail.astype(np.float32), scales


def build_structure_guide(a_full_u8: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rgb = resize_rgb_float(a_full_u8, size)
    guide_rgb = gaussian_rgb(rgb, sigma=1.0).astype(np.float32)
    guide_y = luminance(guide_rgb).astype(np.float32)
    guide_gx, guide_gy = sobel_xy(guide_y)
    return guide_rgb, guide_y, guide_gx.astype(np.float32), guide_gy.astype(np.float32)


def build_safe_known_mask(known_s: np.ndarray, radius: int = SOURCE_SAFE_RADIUS) -> np.ndarray:
    kernel = np.ones((radius * 2 + 1, radius * 2 + 1), np.uint8)
    eroded = cv2.erode(known_s.astype(np.uint8), kernel, iterations=1)
    return eroded > 0


def zscore_patch_descriptor(guide_y: np.ndarray, guide_gx: np.ndarray, guide_gy: np.ndarray, guide_rgb: np.ndarray) -> np.ndarray:
    padded_y = np.pad(guide_y, 2, mode="reflect")
    padded_gx = np.pad(guide_gx, 2, mode="reflect")
    padded_gy = np.pad(guide_gy, 2, mode="reflect")
    y_patch = sliding_window_view(padded_y, (5, 5)).reshape(guide_y.shape[0], guide_y.shape[1], 25).astype(np.float32)
    gx_patch = sliding_window_view(padded_gx, (5, 5)).reshape(guide_y.shape[0], guide_y.shape[1], 25).astype(np.float32)
    gy_patch = sliding_window_view(padded_gy, (5, 5)).reshape(guide_y.shape[0], guide_y.shape[1], 25).astype(np.float32)
    mean = y_patch.mean(axis=2, keepdims=True)
    std = y_patch.std(axis=2, keepdims=True)
    y_z = (y_patch - mean) / (std + 1e-6)
    rgb_mean = cv2.blur(guide_rgb, (5, 5), borderType=cv2.BORDER_REFLECT).astype(np.float32)
    return np.concatenate([y_z, gx_patch, gy_patch, rgb_mean], axis=2).astype(np.float32)


def descriptor_score(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    y_score = np.mean(np.abs(candidates[:, 0:25] - query[0:25]), axis=1)
    gx_score = np.mean(np.abs(candidates[:, 25:50] - query[25:50]), axis=1)
    gy_score = np.mean(np.abs(candidates[:, 50:75] - query[50:75]), axis=1)
    rgb_score = np.mean(np.abs(candidates[:, 75:78] - query[75:78]), axis=1)
    return y_score + 0.30 * gx_score + 0.30 * gy_score + 0.20 * rgb_score


def choose_candidates(candidate_yx: np.ndarray, qy: int, ry: int) -> np.ndarray:
    dy = np.abs(candidate_yx[:, 0] - qy)
    idx = np.where(dy <= ry)[0]
    if len(idx) > 0:
        return idx
    idx = np.where(dy <= ry * 2)[0]
    if len(idx) > 0:
        return idx
    min_dy = int(dy.min())
    return np.where(dy == min_dy)[0]


def block_offset_jump_stats(block_records: list[dict[str, float | int]], block: int) -> tuple[dict[str, float], np.ndarray]:
    if not block_records:
        return {
            "route_offset_jump_mean": math.nan,
            "route_offset_jump_median": math.nan,
            "route_offset_jump_p90": math.nan,
            "route_offset_jump_max": math.nan,
        }, np.zeros((1, 1), dtype=np.float32)
    grid: dict[tuple[int, int], tuple[int, int]] = {}
    max_by = 0
    max_bx = 0
    for rec in block_records:
        by = int(rec["target_y"]) // block
        bx = int(rec["target_x"]) // block
        grid[(by, bx)] = (int(rec["source_dx"]), int(rec["source_dy"]))
        max_by = max(max_by, by)
        max_bx = max(max_bx, bx)
    heat = np.zeros((max_by + 1, max_bx + 1), dtype=np.float32)
    jumps: list[float] = []
    for (by, bx), (dx, dy) in grid.items():
        local: list[float] = []
        for nb in ((by + 1, bx), (by, bx + 1)):
            if nb not in grid:
                continue
            ndx, ndy = grid[nb]
            jump = float(abs(dx - ndx) + abs(dy - ndy))
            jumps.append(jump)
            local.append(jump)
        if local:
            heat[by, bx] = max(local)
    arr = np.array(jumps, dtype=np.float32)
    return {
        "route_offset_jump_mean": float(arr.mean()) if arr.size else math.nan,
        "route_offset_jump_median": float(np.median(arr)) if arr.size else math.nan,
        "route_offset_jump_p90": float(np.percentile(arr, 90)) if arr.size else math.nan,
        "route_offset_jump_max": float(arr.max()) if arr.size else math.nan,
    }, heat


def build_block_routing_map(
    original_u8: np.ndarray,
    a_full_u8: np.ndarray,
    guide_u8: np.ndarray,
    effective_mask: np.ndarray,
    scale_name: str,
    size: int,
    block: int,
    ry: int,
) -> tuple[RoutingMap | None, np.ndarray, list[float]]:
    guide_rgb, guide_y, guide_gx, guide_gy = build_structure_guide(guide_u8, size)
    mask_s = resize_mask(effective_mask, size)
    known_s = ~mask_s
    valid_cells = valid_source_cells_all_k(~effective_mask.astype(bool), size)
    safe_known = build_safe_known_mask(valid_cells, radius=SOURCE_SAFE_RADIUS)
    texture, _detail_rgb, p95_scales = build_texture_deficit_feature(original_u8, a_full_u8, size, safe_known)
    candidate_yx = np.argwhere(safe_known)
    if candidate_yx.size == 0:
        return None, texture, p95_scales
    desc = zscore_patch_descriptor(guide_y, guide_gx, guide_gy, guide_rgb)
    cand_desc = desc[candidate_yx[:, 0], candidate_yx[:, 1]]

    source_y = np.full((size, size), -1, dtype=np.int16)
    source_x = np.full((size, size), -1, dtype=np.int16)
    dx_map = np.zeros((size, size), dtype=np.int16)
    dy_map = np.zeros((size, size), dtype=np.int16)
    block_records: list[dict[str, float | int]] = []
    source_counter: dict[tuple[int, int], int] = {}
    source_block_counter: dict[tuple[int, int], int] = {}

    for y0 in range(0, size, block):
        for x0 in range(0, size, block):
            y1 = min(size, y0 + block)
            x1 = min(size, x0 + block)
            if not np.any(mask_s[y0:y1, x0:x1]):
                continue
            qy = min(size - 1, y0 + (y1 - y0) // 2)
            qx = min(size - 1, x0 + (x1 - x0) // 2)
            cand_idx = choose_candidates(candidate_yx, qy, ry)
            q_desc = desc[qy, qx]
            scores = descriptor_score(q_desc, cand_desc[cand_idx])
            best_local = int(np.argmin(scores))
            best_global = int(cand_idx[best_local])
            cy, cx = map(int, candidate_yx[best_global])
            score = float(scores[best_local])
            dy = cy - qy
            dx = cx - qx
            used = 0
            for yy in range(y0, y1):
                for xx in range(x0, x1):
                    if not mask_s[yy, xx]:
                        continue
                    sy = int(np.clip(cy + (yy - qy), 0, size - 1))
                    sx = int(np.clip(cx + (xx - qx), 0, size - 1))
                    if not safe_known[sy, sx]:
                        sy, sx = cy, cx
                    if not safe_known[sy, sx]:
                        raise RuntimeError(f"{scale_name}: routed source entered unsafe region")
                    source_y[yy, xx] = sy
                    source_x[yy, xx] = sx
                    dy_map[yy, xx] = sy - yy
                    dx_map[yy, xx] = sx - xx
                    source_counter[(sy, sx)] = source_counter.get((sy, sx), 0) + 1
                    used += 1
            block_records.append(
                {
                    "target_y": qy,
                    "target_x": qx,
                    "source_y": cy,
                    "source_x": cx,
                    "source_dy": dy,
                    "source_dx": dx,
                    "match_score": score,
                    "pixels_routed": used,
                }
            )
            source_block_counter[(cy // block, cx // block)] = source_block_counter.get((cy // block, cx // block), 0) + 1

    if np.any(mask_s & (source_y < 0)):
        raise RuntimeError(f"{scale_name}: not every hole pixel received a source")
    if np.any(mask_s & ~safe_known[source_y.clip(0), source_x.clip(0)]):
        raise RuntimeError(f"{scale_name}: source safety assertion failed")

    scores = np.array([r["match_score"] for r in block_records], dtype=np.float32)
    dxs = dx_map[mask_s].astype(np.float32)
    dys = dy_map[mask_s].astype(np.float32)
    total_routed = int(mask_s.sum())
    unique_source_pixels = len(source_counter)
    most_used = max(source_counter.values()) if source_counter else 0
    unique_source_blocks = len(source_block_counter)
    most_used_source_block = max(source_block_counter.values()) if source_block_counter else 0
    jump_stats, _ = block_offset_jump_stats(block_records, block)
    diagnostics = {
        "scale": scale_name,
        "feature_height": size,
        "feature_width": size,
        "routing_block": block,
        "vertical_search_radius": ry,
        "safe_known_erosion_radius": SOURCE_SAFE_RADIUS,
        "source_support_rule": "low-resolution factor x factor cell must be ALL K before 5x5 erosion / radius=2",
        "source_support_all_K": True,
        "texture_source": "static signed deficit D0 = Original - A_full; K only",
        "texture_deficit_gaussian_kernel": "3x3",
        "texture_deficit_gaussian_sigma": 0.8,
        "texture_deficit_feature_channels": "D_detail_RGB + Gx + Gy",
        "texture_deficit_p95_scales": p95_scales,
        "num_hole_blocks": len(block_records),
        "num_safe_known_candidates": int(candidate_yx.shape[0]),
        "mean_match_score": float(scores.mean()) if scores.size else math.nan,
        "median_match_score": float(np.median(scores)) if scores.size else math.nan,
        "p90_match_score": float(np.percentile(scores, 90)) if scores.size else math.nan,
        "mean_abs_dx": float(np.mean(np.abs(dxs))) if dxs.size else math.nan,
        "mean_abs_dy": float(np.mean(np.abs(dys))) if dys.size else math.nan,
        "max_abs_dx": float(np.max(np.abs(dxs))) if dxs.size else 0.0,
        "max_abs_dy": float(np.max(np.abs(dys))) if dys.size else 0.0,
        "unique_source_pixel_count": unique_source_pixels,
        "unique_source_pixel_ratio": float(unique_source_pixels / max(total_routed, 1)),
        "most_used_source_fraction": float(most_used / max(total_routed, 1)),
        "ROUTING_COLLAPSE_WARNING": bool(most_used / max(total_routed, 1) >= ROUTING_COLLAPSE_THRESHOLD),
        "unique_source_block_count": unique_source_blocks,
        "unique_source_block_ratio": float(unique_source_blocks / max(len(block_records), 1)),
        "most_used_source_block_fraction": float(most_used_source_block / max(len(block_records), 1)),
        "ROUTING_BLOCK_COLLAPSE_WARNING": bool(most_used_source_block / max(len(block_records), 1) >= ROUTING_COLLAPSE_THRESHOLD),
        **jump_stats,
    }
    routing = RoutingMap(scale_name, size, size, block, source_x, source_y, dx_map, dy_map, mask_s, safe_known, block_records, diagnostics)
    return routing, texture, p95_scales


def apply_block_routing(texture: np.ndarray, route: RoutingMap) -> np.ndarray:
    routed = texture.copy()
    ys, xs = np.where(route.hole_mask)
    routed[:, ys, xs] = texture[:, route.source_y[ys, xs], route.source_x[ys, xs]]
    if not np.all(route.safe_known[route.source_y[ys, xs], route.source_x[ys, xs]]):
        raise RuntimeError(f"{route.scale_name}: source coordinates are not all safe-known")
    return routed.astype(np.float32)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_ch, out_ch, 3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_ch, out_ch, 3),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ProjectedStage2DirectNet(nn.Module):
    def __init__(self, in_ch: int, variant: str) -> None:
        super().__init__()
        if variant != "routed_skip":
            raise ValueError(variant)
        self.variant = variant
        c = CHANNELS
        self.enc0 = ConvBlock(in_ch, c[0])
        self.down1 = nn.Conv2d(c[0], c[1], 3, stride=2, padding=1)
        self.enc1 = ConvBlock(c[1], c[1])
        self.down2 = nn.Conv2d(c[1], c[2], 3, stride=2, padding=1)
        self.enc2 = ConvBlock(c[2], c[2])
        self.down3 = nn.Conv2d(c[2], c[3], 3, stride=2, padding=1)
        self.enc3 = ConvBlock(c[3], c[3])
        self.down4 = nn.Conv2d(c[3], c[4], 3, stride=2, padding=1)
        self.enc4 = ConvBlock(c[4], c[4])
        self.down5 = nn.Conv2d(c[4], c[5], 3, stride=2, padding=1)
        self.mid = ConvBlock(c[5], c[5])

        self.dec4 = ConvBlock(c[5] + c[4], c[4])
        self.dec3_direct = ConvBlock(c[4] + c[3], c[3])
        self.dec3_routed = ConvBlock(c[4] + TEXTURE_FEATURE_CHANNELS, c[3])
        self.dec2_direct = ConvBlock(c[3] + c[2], c[2])
        self.dec2_routed = ConvBlock(c[3] + TEXTURE_FEATURE_CHANNELS, c[2])
        self.dec1_direct = ConvBlock(c[2] + c[1], c[1])
        self.dec1_routed = ConvBlock(c[2] + TEXTURE_FEATURE_CHANNELS, c[1])
        self.dec_full = ConvBlock(c[1], c[0])
        self.head = nn.Conv2d(c[0], 3, 1)
        self.sigmoid = nn.Sigmoid()
        self.last_diagnostic_stats: dict[str, float] = {}

    def forward(
        self,
        x: torch.Tensor,
        route_eighth: torch.Tensor | None = None,
        route_quarter: torch.Tensor | None = None,
        route_half: torch.Tensor | None = None,
        diagnostic_stats: bool = False,
    ) -> torch.Tensor:
        self.last_diagnostic_stats = {}
        e0 = self.enc0(x)
        e1 = self.enc1(self.down1(e0))
        e2 = self.enc2(self.down2(e1))
        e3 = self.enc3(self.down3(e2))
        e4 = self.enc4(self.down4(e3))
        m = self.mid(self.down5(e4))

        d = F.interpolate(m, size=e4.shape[-2:], mode="bilinear", align_corners=False)
        d = self.dec4(torch.cat([d, e4], dim=1))
        d = F.interpolate(d, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        if self.variant == "routed_skip" and route_eighth is not None:
            if diagnostic_stats:
                dec_abs = float(torch.mean(torch.abs(d)).detach().cpu())
                route_abs = float(torch.mean(torch.abs(route_eighth)).detach().cpu())
                self.last_diagnostic_stats["decoder_pre_route_abs_mean_1_8"] = dec_abs
                self.last_diagnostic_stats["route_abs_mean_1_8"] = route_abs
                self.last_diagnostic_stats["route_to_decoder_ratio_1_8"] = route_abs / max(dec_abs, 1e-12)
            d = self.dec3_routed(torch.cat([d, route_eighth], dim=1))
        else:
            d = self.dec3_direct(torch.cat([d, e3], dim=1))
        d = F.interpolate(d, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        if self.variant == "routed_skip":
            if route_quarter is not None:
                if diagnostic_stats:
                    dec_abs = float(torch.mean(torch.abs(d)).detach().cpu())
                    route_abs = float(torch.mean(torch.abs(route_quarter)).detach().cpu())
                    self.last_diagnostic_stats["decoder_pre_route_abs_mean_1_4"] = dec_abs
                    self.last_diagnostic_stats["route_abs_mean_1_4"] = route_abs
                    self.last_diagnostic_stats["route_to_decoder_ratio_1_4"] = route_abs / max(dec_abs, 1e-12)
                d = self.dec2_routed(torch.cat([d, route_quarter], dim=1))
            else:
                d = self.dec2_direct(torch.cat([d, e2], dim=1))
        else:
            d = self.dec2_direct(torch.cat([d, e2], dim=1))

        d = F.interpolate(d, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        if self.variant == "routed_skip" and route_half is not None:
            if diagnostic_stats:
                dec_abs = float(torch.mean(torch.abs(d)).detach().cpu())
                route_abs = float(torch.mean(torch.abs(route_half)).detach().cpu())
                self.last_diagnostic_stats["decoder_pre_route_abs_mean_1_2"] = dec_abs
                self.last_diagnostic_stats["route_abs_mean_1_2"] = route_abs
                self.last_diagnostic_stats["route_to_decoder_ratio_1_2"] = route_abs / max(dec_abs, 1e-12)
            d = self.dec1_routed(torch.cat([d, route_half], dim=1))
        else:
            d = self.dec1_direct(torch.cat([d, e1], dim=1))

        d = F.interpolate(d, size=e0.shape[-2:], mode="bilinear", align_corners=False)
        d = self.dec_full(d)
        return self.sigmoid(self.head(d))


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.expand(-1, x.shape[1], -1, -1)
    return torch.sum(x * m) / torch.clamp(torch.sum(m), min=1.0)
