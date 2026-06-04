from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.multimodal_captcha.generator import generate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic multimodal CAPTCHA data.")
    parser.add_argument("--output-dir", default="data/synthetic_captcha")
    parser.add_argument("--num-samples", type=int, default=600)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--image-size", type=int, default=192)
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="medium")
    parser.add_argument("--debug-labels", action="store_true", help="Draw color/object labels inside cells for visual debugging.")
    args = parser.parse_args()

    manifest = generate_dataset(args.output_dir, args.num_samples, args.seed, args.image_size, args.debug_labels, args.difficulty)
    print(f"Generated {args.num_samples} samples at {manifest}")


if __name__ == "__main__":
    main()
