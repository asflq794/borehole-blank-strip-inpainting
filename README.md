# Borehole Image Gap Restoration

Code for structural prior reconstruction, observable-residual refinement, and safe measured-texture routing with low-frequency consistency projection.

## Requirements

Python 3.10+, PyTorch, NumPy, and OpenCV. Install with `pip install -r requirements.txt`.

## Repository structure

`src/` contains the core method. `run.py` restores one image, `train_refiner.py` trains the residual refiner, and `demo.py` runs a small synthetic smoke test. `examples/` contains representative final restorations.

## Usage

### Train the residual refiner

Train from random initialization (seed 42) using mask-first Stage1 pairs containing `original.png`, `stage1_raw.png`, `validated_real_gap_mask.png`, and `train_synthetic_mask.png`:

```bash
python train_refiner.py --pairs PATH_TO_PAIRS --output refiner.pth
```

To continue training from compatible refiner weights, add `--warmstart PATH_TO_INITIAL_WEIGHTS`. The Stage1 training pairs and trained refiner weights are not included in this repository; supply your own pairs for training. A new random-initialization run is not expected to reproduce the historical trained weights or manuscript metrics without the same training data and protocol.

### Run restoration

```bash
python run.py --image input.png --mask missing.png --refiner-checkpoint refiner.pth --output restored.png
```

### Minimal demo

```bash
python demo.py --output demo_output.png
```

## Input and output

- Input: a 512 × 512 RGB borehole image and a binary mask; white (255) denotes missing pixels.
- Output: a restored RGB image.

## Examples

Representative final restoration results are provided in `examples/`.

## License

Original code in this repository is licensed under the Apache License, Version 2.0 (see `LICENSE`). Third-party components retain their original license notices in `LICENSE_DIP.txt` and `LICENSE_NAFNet.txt`.
