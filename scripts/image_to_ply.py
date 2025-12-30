"""
image_to_ply.py
----------------
Converts a single RGB image into a 3D point cloud (.ply)
using the TRELLIS Image-to-3D pipeline.

This script is designed for:
- reproducible inference
- accurate performance profiling
- research and production-ready usage
"""

# ============================================================
# Ensure project root is on PYTHONPATH
# ============================================================

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# Standard imports
# ============================================================

import time
import torch
from PIL import Image

from trellis.pipelines import TrellisImageTo3DPipeline


# ============================================================
# Configuration
# ============================================================

MODEL_DIR = Path(r"C:/temp_model/TRELLIS-image-large")

INPUT_IMAGE_PATH = PROJECT_ROOT / "ply_folder" / "phone_holder.jpg"
OUTPUT_PLY_PATH  = PROJECT_ROOT / "ply_folder" / "phone_holder.ply"

SEED = 1

GAUSSIAN_DOWNSAMPLE_RATIO = 0.1   # 10% resolution for faster inference
SPARSE_SAMPLER_STEPS = 8
SLAT_SAMPLER_STEPS = 8


# ============================================================
# Utility
# ============================================================

def log_elapsed(start_time: float, message: str) -> None:
    """Logs elapsed wall-clock time with a consistent format."""
    elapsed = time.time() - start_time
    print(f"[TIME] {message}: {elapsed:.2f} seconds")


# ============================================================
# Main Pipeline
# ============================================================

def main() -> None:
    # --------------------------------------------------------
    # 1. Load TRELLIS pipeline (model + weights)
    # --------------------------------------------------------
    t0 = time.time()

    pipeline = TrellisImageTo3DPipeline.from_pretrained(MODEL_DIR)
    pipeline.gaussian_downsample_ratio = GAUSSIAN_DOWNSAMPLE_RATIO
    pipeline.cuda()

    log_elapsed(t0, "Model loaded and moved to CUDA")

    # --------------------------------------------------------
    # 2. Load input image (disk I/O only)
    # --------------------------------------------------------
    t0 = time.time()

    if not INPUT_IMAGE_PATH.exists():
        raise FileNotFoundError(f"Input image not found: {INPUT_IMAGE_PATH}")

    image = Image.open(INPUT_IMAGE_PATH)

    log_elapsed(t0, "Image loaded from disk")

    # --------------------------------------------------------
    # 3. Run inference (GPU-timing safe)
    # --------------------------------------------------------
    torch.cuda.synchronize()   # ensure clean start
    t0 = time.time()

    outputs = pipeline.run(
        image=image,
        seed=SEED,
        preprocess_image=True,
        formats=["gaussian"],
        sparse_structure_sampler_params={
            "steps": SPARSE_SAMPLER_STEPS
        },
        slat_sampler_params={
            "steps": SLAT_SAMPLER_STEPS
        }
    )

    torch.cuda.synchronize()   # ensure GPU work is finished
    log_elapsed(t0, "Inference completed")

    # --------------------------------------------------------
    # 4. Export result to PLY
    # --------------------------------------------------------
    t0 = time.time()

    gaussian_result = outputs["gaussian"][0]
    gaussian_result.save_ply(OUTPUT_PLY_PATH)

    log_elapsed(t0, "PLY export completed")

    print(f"[OK] PLY file saved to: {OUTPUT_PLY_PATH.resolve()}")


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()
