"""A one-forward synthetic smoke test of the core modules."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.pipeline import set_seed
from src.refiner import ErrorGuidedJointRefiner
from src.structure import TeleaHorizontalASEStage1Net, get_structured_noise
from src.texture import ProjectedStage2DirectNet, apply_low_frequency_projection


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal synthetic execution test")
    parser.add_argument("--output", type=Path, default=Path("demo_output.png"), help="Output PNG path")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    set_seed(42)
    with torch.no_grad():
        noise = get_structured_noise(16, (256, 256)).to(device)
        scaffold = torch.full((1, 3, 256, 256), 0.5, device=device)
        stage1 = TeleaHorizontalASEStage1Net(16).to(device).eval()
        s1 = stage1(torch.cat([noise[:, :13], scaffold], dim=1))
        s1 = F.interpolate(s1, size=(64, 64), mode="bilinear", align_corners=False)
        known = torch.ones((1, 1, 64, 64), device=device)
        known[:, :, :, 25:37] = 0
        refiner = ErrorGuidedJointRefiner().to(device).eval()
        a_full = s1 + refiner(s1, torch.zeros_like(s1), known)
        stage2 = ProjectedStage2DirectNet(19, "routed_skip").to(device).eval()
        raw = stage2(torch.cat([get_structured_noise(16, (64, 64)).to(device), a_full], dim=1))
        projected = apply_low_frequency_projection(raw, a_full)
        final = torch.where(known.bool().expand_as(projected), s1, projected)
        image = np.rint(final.squeeze(0).permute(1, 2, 0).cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise RuntimeError("Cannot save demo output")
    print(f"Demo smoke test passed: {args.output}")


if __name__ == "__main__":
    main()
