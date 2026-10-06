"""Self-supervised training of the observable-residual refiner."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch

from src.pipeline import set_seed, to_tensor
from src.refiner import ErrorGuidedJointRefiner, build_prediction, compute_clean_loss, gaussian_kernel


def read_pair(case: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    def rgb(name: str) -> np.ndarray:
        image = cv2.imread(str(case / name), cv2.IMREAD_COLOR)
        if image is None or image.shape != (512, 512, 3):
            raise ValueError(f"Invalid image: {case / name}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def mask(name: str) -> np.ndarray:
        image = cv2.imread(str(case / name), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != (512, 512) or not np.isin(image, (0, 255)).all():
            raise ValueError(f"Invalid binary mask: {case / name}")
        return image == 255

    original = rgb("original.png")
    stage1 = rgb("stage1_raw.png")
    h0 = mask("validated_real_gap_mask.png")
    t = mask("train_synthetic_mask.png")
    if np.any(h0 & t) or not np.any(t):
        raise ValueError(f"Training mask must satisfy nonempty T and H0 intersection T = empty: {case}")
    return original, stage1, h0, t


def load_warmstart(model: ErrorGuidedJointRefiner, path: Path, device: torch.device) -> None:
    payload = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Warm-start checkpoint has no state dictionary")
    state = next((payload[key] for key in ("model", "model_state_dict", "state_dict") if isinstance(payload.get(key), dict)), payload)
    if all(key.startswith("module.") for key in state):
        state = {key[len("module."):]: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the residual refiner on mask-first Stage1 pairs")
    parser.add_argument("--pairs", type=Path, required=True, help="Root containing per-case Stage1 training pairs")
    parser.add_argument("--warmstart", type=Path, required=True, help="Compatible residual-refiner initialization")
    parser.add_argument("--output", type=Path, required=True, help="Output checkpoint path")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--iterations", type=int, default=25000, help="Optimization iterations")
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    cases = sorted({path.parent for path in args.pairs.rglob("train_synthetic_mask.png")})
    if not cases:
        parser.error("No Stage1 training pairs found")
    set_seed(42)
    rng = random.Random(42)
    model = ErrorGuidedJointRefiner().to(device)
    load_warmstart(model, args.warmstart, device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    low_kernel = gaussian_kernel(device, sigma=4.0)
    mid_kernel = gaussian_kernel(device, sigma=1.5)
    recent_losses: list[float] = []
    best_window_mean = float("inf")
    bad_windows = 0
    stopped_at = args.iterations
    for iteration in range(1, args.iterations + 1):
        original, stage1, h0, t = read_pair(rng.choice(cases))
        y_t, x_t = np.argwhere(t)[rng.randrange(int(t.sum()))]
        y0 = int(np.clip(int(y_t) - rng.randint(0, 255), 0, 256))
        x0 = int(np.clip(int(x_t) - rng.randint(0, 255), 0, 256))
        sl = np.s_[y0:y0 + 256, x0:x0 + 256]
        original_t = to_tensor(original[sl], device)
        stage1_t = to_tensor(stage1[sl], device)
        h0_t = torch.from_numpy(h0[sl].copy()).to(device=device, dtype=torch.float32)[None, None]
        t_t = torch.from_numpy(t[sl].copy()).to(device=device, dtype=torch.float32)[None, None]
        optimizer.zero_grad(set_to_none=True)
        pred, _ = build_prediction(model, original_t, stage1_t, h0_t, t_t)
        loss, _ = compute_clean_loss(pred, original_t, stage1_t, h0_t, t_t, low_kernel, mid_kernel)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at iteration {iteration}")
        loss.backward()
        optimizer.step()
        recent_losses.append(float(loss.detach().cpu()))
        if iteration == 1 or iteration % 100 == 0 or iteration == args.iterations:
            print(f"iteration={iteration} loss={float(loss.detach().cpu()):.6f}", flush=True)
        if iteration % 1000 == 0:
            window_mean = float(np.mean(recent_losses[-1000:]))
            if iteration == 10000:
                best_window_mean = window_mean
            elif iteration > 10000:
                relative_improvement = (best_window_mean - window_mean) / max(abs(best_window_mean), 1e-12)
                if relative_improvement >= 0.01:
                    best_window_mean = window_mean
                    bad_windows = 0
                else:
                    bad_windows += 1
                if bad_windows >= 3:
                    stopped_at = iteration
                    print(f"Early stopping at iteration {iteration}", flush=True)
                    break
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "iteration": stopped_at}, args.output)
    print(f"Checkpoint: {args.output}")


if __name__ == "__main__":
    main()
