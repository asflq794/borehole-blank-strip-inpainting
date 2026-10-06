"""Run the complete restoration method on one RGB image."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from src.pipeline import refine_structure, restore_stage1, restore_stage2
from src.refiner import ErrorGuidedJointRefiner


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.shape != (512, 512, 3):
        raise ValueError(f"Expected a readable 512x512 RGB image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != (512, 512):
        raise ValueError(f"Expected a readable 512x512 mask: {path}")
    if not np.isin(mask, (0, 255)).all() or not np.any(mask == 255) or not np.any(mask == 0):
        raise ValueError("Mask must be binary, nonempty, and use white (255) for missing pixels")
    return mask == 255


def load_refiner(checkpoint: Path, device: torch.device) -> ErrorGuidedJointRefiner:
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    if isinstance(payload, dict):
        state = next((payload[key] for key in ("model", "model_state_dict", "state_dict") if isinstance(payload.get(key), dict)), payload)
    else:
        raise ValueError("Refiner checkpoint has no state dictionary")
    if all(key.startswith("module.") for key in state):
        state = {key[len("module."):]: value for key, value in state.items()}
    model = ErrorGuidedJointRefiner().to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Restore one borehole image")
    parser.add_argument("--image", type=Path, required=True, help="512x512 RGB input image")
    parser.add_argument("--mask", type=Path, required=True, help="Binary mask: white denotes missing pixels")
    parser.add_argument("--refiner-checkpoint", type=Path, required=True, help="Trained residual-refiner checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="Restored PNG path")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    image = read_rgb(args.image)
    hole = read_mask(args.mask)
    model = load_refiner(args.refiner_checkpoint, device)
    stage1 = restore_stage1(image, hole, device)
    a_full = refine_structure(image, stage1, hole, model, device)
    restored = restore_stage2(image, a_full, hole, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), cv2.cvtColor(restored, cv2.COLOR_RGB2BGR))
    print(f"Restored image: {args.output}")


if __name__ == "__main__":
    main()
