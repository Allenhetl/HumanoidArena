#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path


DEFAULT_DATASET_ROOT = Path(
    "/home/dreams/Users/taowen/HumanoidArena/lerobot/outputs/HumanoidArena_datasets_v2"
)
DEFAULT_OUTPUT_ROOT = Path("/home/dreams/Users/taowen/debug_dataset_v2")


def find_dataset_roots(root: Path) -> list[Path]:
    return sorted(info_path.parent.parent for info_path in root.rglob("meta/info.json"))


def load_total_episodes(dataset_root: Path) -> int:
    info_path = dataset_root / "meta" / "info.json"
    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)
    total_episodes = int(info.get("total_episodes", 0))
    if total_episodes <= 0:
        raise ValueError(f"{dataset_root} has invalid total_episodes={total_episodes}")
    return total_episodes


def sample_episodes(total_episodes: int, count: int, rng: random.Random) -> list[int]:
    count = min(int(count), int(total_episodes))
    return sorted(rng.sample(range(total_episodes), count))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Randomly render local-delta v2 LeRobot datasets with the three-panel MuJoCo visualizer. "
            "The output directory mirrors the input dataset-root structure."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--num-samples", type=int, default=10, help="Episodes to sample per dataset.")
    parser.add_argument("--seed", type=int, default=20260509)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--mujoco-gl", type=str, default="egl")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--visualizer", type=Path, default=Path(__file__).with_name("visualize_localdelta_v2_mujoco.py"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    visualizer = args.visualizer.expanduser().resolve()
    python_bin = args.python.expanduser().resolve()

    dataset_roots = find_dataset_roots(dataset_root)
    if not dataset_roots:
        raise FileNotFoundError(f"No LeRobot datasets found under {dataset_root}")
    if not visualizer.is_file():
        raise FileNotFoundError(f"Visualizer script not found: {visualizer}")

    rng = random.Random(args.seed)
    failures: list[tuple[Path, int, int]] = []
    scheduled = 0

    print(f"dataset_root: {dataset_root}")
    print(f"output_root: {output_root}")
    print(f"datasets: {len(dataset_roots)}")
    print(f"num_samples_per_dataset: {args.num_samples}")

    for current_dataset_root in dataset_roots:
        rel_dataset = current_dataset_root.relative_to(dataset_root)
        total_episodes = load_total_episodes(current_dataset_root)
        episodes = sample_episodes(total_episodes, args.num_samples, rng)
        current_output_dir = output_root / rel_dataset
        if not args.dry_run:
            current_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"== {rel_dataset} total_episodes={total_episodes} samples={episodes} ==")
        for sample_idx, episode_index in enumerate(episodes):
            output_path = current_output_dir / f"sample_{sample_idx:02d}_episode_{episode_index:06d}.mp4"
            if output_path.exists() and not args.overwrite:
                print(f"SKIP exists: {output_path}")
                continue

            cmd = [
                str(python_bin),
                str(visualizer),
                "--dataset-root",
                str(current_dataset_root),
                "--episode",
                str(episode_index),
                "--output",
                str(output_path),
                "--max-frames",
                str(args.max_frames),
                "--stride",
                str(args.stride),
                "--mujoco-gl",
                str(args.mujoco_gl),
            ]
            print("+ " + " ".join(cmd))
            scheduled += 1
            if args.dry_run:
                continue

            result = subprocess.run(cmd, check=False)
            if result.returncode != 0:
                failures.append((current_dataset_root, episode_index, result.returncode))
                print(
                    f"ERROR dataset={current_dataset_root} episode={episode_index} "
                    f"returncode={result.returncode}",
                    file=sys.stderr,
                )
                if args.fail_fast:
                    return result.returncode

    if failures:
        print(f"Completed with {len(failures)} failures out of {scheduled} scheduled renders.", file=sys.stderr)
        for dataset, episode_index, returncode in failures:
            print(f"FAILED {dataset} episode={episode_index} returncode={returncode}", file=sys.stderr)
        return 1

    print(f"Done. scheduled={scheduled} output_root={output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
