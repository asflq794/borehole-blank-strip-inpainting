"""Minimal single-image entry points for the three-stage restoration method."""
from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F

from .refiner import ErrorGuidedJointRefiner
from .structure import (
    StructureNet,
    TeleaHorizontalASEStage1Net,
    build_telea_scaffold,
    get_structured_noise,
)
from .texture import (
    ProjectedStage2DirectNet,
    apply_block_routing,
    apply_low_frequency_projection,
    build_block_routing_map,
    masked_mean,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(image.copy()).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32) / 255.0


def to_image(tensor: torch.Tensor) -> np.ndarray:
    return np.rint(tensor.detach().clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def restore_stage1(image: np.ndarray, hole: np.ndarray, device: torch.device, iterations: int = 650) -> np.ndarray:
    if image.shape != (512, 512, 3) or hole.shape != (512, 512):
        raise ValueError("Stage1 requires a 512x512 RGB image and mask")
    set_seed(42)
    scaffold = build_telea_scaffold(image, hole, radius=3.0)
    noise = get_structured_noise(16, (256, 256)).to(device)
    scaffold_256 = F.interpolate(to_tensor(scaffold, device), size=(256, 256), mode="bilinear", align_corners=False)
    net_input = torch.cat([noise[:, :13].clone(), scaffold_256], dim=1).detach()
    template = StructureNet(16, 3, depth=6, pad="reflection").to(device)
    state = {key: value.detach().clone() for key, value in template.state_dict().items()}
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    del template
    model = TeleaHorizontalASEStage1Net(16, 3, depth=6, pad="reflection").to(device)
    for part in ("core", "ase", "head"):
        getattr(model, part).load_state_dict({key[len(part) + 1:]: value for key, value in state.items() if key.startswith(part + ".")}, strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    torch.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)
    known = torch.from_numpy((~hole).copy()).to(device=device, dtype=torch.float32)[None, None]
    original = to_tensor(image, device)
    perturb = torch.zeros_like(net_input)
    perturb[:, :13] = 1
    buffer = torch.empty_like(net_input)
    model.train()
    for _ in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        output = model(net_input + buffer.normal_() * 0.01 * perturb)
        output = F.interpolate(output, size=(512, 512), mode="bilinear", align_corners=False)
        loss = F.l1_loss(output * known, original * known)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        output = F.interpolate(model(net_input), size=(512, 512), mode="bilinear", align_corners=False)
    return to_image(output)


def refine_structure(image: np.ndarray, stage1: np.ndarray, hole: np.ndarray, model: ErrorGuidedJointRefiner, device: torch.device) -> np.ndarray:
    original_t = to_tensor(image, device)
    stage1_t = to_tensor(stage1, device)
    known = torch.from_numpy((~hole).copy()).to(device=device, dtype=torch.float32)[None, None]
    model.eval()
    with torch.no_grad():
        e_known = (original_t - stage1_t) * known
        a_full = stage1_t + model(stage1_t, e_known, known)
    return to_image(a_full)


def routed_features(image: np.ndarray, a_full: np.ndarray, hole: np.ndarray, device: torch.device) -> tuple[torch.Tensor | None, ...]:
    configs = (("eighth", 64, 1, 3), ("quarter", 128, 2, 6), ("half", 256, 4, 12))
    features = []
    for label, size, block, ry in configs:
        route, texture, _ = build_block_routing_map(image, a_full, a_full, hole, label, size, block, ry)
        if route is None:
            features.append(None)
        else:
            routed = apply_block_routing(texture, route)
            features.append(torch.from_numpy(routed.copy()).unsqueeze(0).to(device=device, dtype=torch.float32))
    return tuple(features)


def restore_stage2(image: np.ndarray, a_full: np.ndarray, hole: np.ndarray, device: torch.device, iterations: int = 500) -> np.ndarray:
    routes = routed_features(image, a_full, hole, device)
    original_t = to_tensor(image, device)
    a_full_t = to_tensor(a_full, device)
    known = torch.from_numpy((~hole).copy()).to(device=device, dtype=torch.float32)[None, None]
    set_seed(1042)
    noise = get_structured_noise(16, (512, 512)).to(device)
    net_input = torch.cat([noise, a_full_t.detach()], dim=1).detach()
    buffer = torch.empty_like(net_input)
    set_seed(2042)
    model = ProjectedStage2DirectNet(19, variant="routed_skip").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for _ in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        raw = model(net_input + buffer.normal_() * 0.01, *routes)
        projected = apply_low_frequency_projection(raw, a_full_t)
        loss = masked_mean(torch.abs(projected - original_t), known)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    model.eval()
    with torch.no_grad():
        raw = model(net_input, *routes)
        projected = apply_low_frequency_projection(raw, a_full_t)
    result = image.copy()
    result[hole] = to_image(projected)[hole]
    if not np.array_equal(result[~hole], image[~hole]):
        raise RuntimeError("Known-region pasteback failed")
    return result
