#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_seed(spec):
    label, path = spec.split("=", 1)
    data = np.load(path)
    return label, Path(path), data


def require_same_grid(seed_data):
    ref = seed_data[0][2]
    x_ref = ref["x_grid"]
    y_ref = ref["y_grid"]
    grid_ref = float(ref["grid"])
    for label, path, data in seed_data[1:]:
        if data["x_grid"].shape != x_ref.shape or data["y_grid"].shape != y_ref.shape:
            raise RuntimeError(f"Grid shape mismatch for {label}: {path}")
        if not np.allclose(data["x_grid"], x_ref) or not np.allclose(data["y_grid"], y_ref):
            raise RuntimeError(f"Grid coordinates mismatch for {label}: {path}")
        if abs(float(data["grid"]) - grid_ref) > 1e-9:
            raise RuntimeError(f"Grid resolution mismatch for {label}: {path}")
    return x_ref, y_ref, grid_ref


def plot_union(union_before, union_after, vote_before, vote_after, x_grid, y_grid, starts, out_path):
    extent = [float(x_grid.min()), float(x_grid.max()), float(y_grid.min()), float(y_grid.max())]
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), dpi=150)
    panels = [
        (union_before.astype(float), "Union reachable before overhead/table blocking", "Greens", 0, 1),
        (union_after.astype(float), "Union reachable after overhead/table blocking", "Greens", 0, 1),
        (vote_before, "Vote count before overhead/table blocking", "viridis", 0, max(1, int(vote_before.max()))),
        (vote_after, "Vote count after overhead/table blocking", "viridis", 0, max(1, int(vote_after.max()))),
    ]
    for ax, (arr, title, cmap, vmin, vmax) in zip(axes.ravel(), panels):
        im = ax.imshow(arr, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        for label, xy in starts.items():
            ax.scatter([xy[0]], [xy[1]], s=45, marker="x", label=label)
        ax.set_aspect("equal")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=7)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_seed_overlay(seed_data, mask_key, x_grid, y_grid, out_path):
    extent = [float(x_grid.min()), float(x_grid.max()), float(y_grid.min()), float(y_grid.max())]
    fig, ax = plt.subplots(figsize=(11, 10), dpi=170)
    colors = ["#22c55e", "#2563eb", "#f97316", "#a855f7", "#ef4444", "#14b8a6", "#eab308"]
    for idx, (label, _path, data) in enumerate(seed_data):
        mask = data[mask_key].astype(bool)
        rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
        rgb = tuple(int(colors[idx % len(colors)].lstrip("#")[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
        rgba[mask] = (*rgb, 0.35)
        ax.imshow(rgba, origin="lower", extent=extent, interpolation="nearest")
        start = data["start_xy"]
        ax.scatter([start[0]], [start[1]], c=[colors[idx % len(colors)]], s=60, marker="x", label=label)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"Per-seed overlay: {mask_key}")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_blocking(union_before, union_after, blocked_any, x_grid, y_grid, starts, out_path):
    extent = [float(x_grid.min()), float(x_grid.max()), float(y_grid.min()), float(y_grid.max())]
    img = np.zeros((*union_before.shape, 4), dtype=np.float32)
    img[union_before] = (0.25, 0.55, 1.0, 0.45)
    img[union_after] = (0.05, 0.85, 0.25, 0.85)
    img[union_before & ~union_after] = (0.95, 0.10, 0.10, 0.85)
    img[blocked_any & ~union_before] = (0.75, 0.10, 0.95, 0.35)
    fig, ax = plt.subplots(figsize=(11, 10), dpi=170)
    ax.imshow(img, origin="lower", extent=extent, interpolation="nearest")
    for label, xy in starts.items():
        ax.scatter([xy[0]], [xy[1]], s=55, marker="x", label=label)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Multi-seed stitch: blue=before, green=after, red=removed by table/overhead projection")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", action="append", required=True, help="label=/path/to/floor_masks.npz")
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    seed_data = [load_seed(s) for s in args.seed]
    x_grid, y_grid, grid = require_same_grid(seed_data)

    before_stack = np.stack([d["reachable_before_overhead"].astype(bool) for _, _, d in seed_data], axis=0)
    after_stack = np.stack([d["reachable"].astype(bool) for _, _, d in seed_data], axis=0)
    connected_stack = np.stack([d["connected"].astype(bool) for _, _, d in seed_data], axis=0)
    blocked_stack = np.stack([d["overhead_blocked"].astype(bool) for _, _, d in seed_data], axis=0)
    valid_stack = np.stack([d["valid_floor"].astype(bool) for _, _, d in seed_data], axis=0)

    vote_before = before_stack.sum(axis=0).astype(np.int16)
    vote_after = after_stack.sum(axis=0).astype(np.int16)
    vote_connected = connected_stack.sum(axis=0).astype(np.int16)
    vote_valid = valid_stack.sum(axis=0).astype(np.int16)
    blocked_any = blocked_stack.any(axis=0)
    union_before = vote_before > 0
    union_after = vote_after > 0
    union_connected = vote_connected > 0
    union_valid = vote_valid > 0

    starts = {label: d["start_xy"].tolist() for label, _, d in seed_data}
    plot_union(union_before, union_after, vote_before, vote_after, x_grid, y_grid, starts, args.out_dir / "multi_seed_union_vote.png")
    plot_seed_overlay(seed_data, "reachable_before_overhead", x_grid, y_grid, args.out_dir / "per_seed_reachable_before_overhead_overlay.png")
    plot_seed_overlay(seed_data, "reachable", x_grid, y_grid, args.out_dir / "per_seed_reachable_after_overhead_overlay.png")
    plot_blocking(union_before, union_after, blocked_any, x_grid, y_grid, starts, args.out_dir / "multi_seed_overhead_blocking_effect.png")

    meta = {
        "seeds": [
            {
                "label": label,
                "path": str(path),
                "start_xy": d["start_xy"].tolist(),
                "plane_coef": d["plane_coef"].tolist(),
                "valid_floor_area_m2": float(d["valid_floor"].sum() * grid * grid),
                "connected_area_m2": float(d["connected"].sum() * grid * grid),
                "reachable_before_overhead_area_m2": float(d["reachable_before_overhead"].sum() * grid * grid),
                "reachable_area_m2": float(d["reachable"].sum() * grid * grid),
            }
            for label, path, d in seed_data
        ],
        "grid_m": grid,
        "union_valid_area_m2": float(union_valid.sum() * grid * grid),
        "union_connected_area_m2": float(union_connected.sum() * grid * grid),
        "union_reachable_before_overhead_area_m2": float(union_before.sum() * grid * grid),
        "union_reachable_after_overhead_area_m2": float(union_after.sum() * grid * grid),
        "blocked_from_union_area_m2": float((union_before & ~union_after).sum() * grid * grid),
        "vote_before_max": int(vote_before.max()),
        "vote_after_max": int(vote_after.max()),
        "outputs": [
            "multi_seed_union_vote.png",
            "per_seed_reachable_before_overhead_overlay.png",
            "per_seed_reachable_after_overhead_overlay.png",
            "multi_seed_overhead_blocking_effect.png",
            "multi_seed_stitched_masks.npz",
            "multi_seed_stitch_report.json",
            "REPORT.md",
        ],
    }

    np.savez_compressed(
        args.out_dir / "multi_seed_stitched_masks.npz",
        x_grid=x_grid,
        y_grid=y_grid,
        grid=np.array(grid, dtype=np.float64),
        union_valid=union_valid,
        union_connected=union_connected,
        union_reachable_before_overhead=union_before,
        union_reachable=union_after,
        vote_valid=vote_valid,
        vote_connected=vote_connected,
        vote_reachable_before_overhead=vote_before,
        vote_reachable=vote_after,
        blocked_any=blocked_any,
    )
    (args.out_dir / "multi_seed_stitch_report.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    md = ["# Multi-Seed Real Scene Reachability Stitch", "", "## Seeds", ""]
    for seed in meta["seeds"]:
        md.append(
            f"- `{seed['label']}` start={seed['start_xy']} before={seed['reachable_before_overhead_area_m2']:.2f}m^2 after={seed['reachable_area_m2']:.2f}m^2 plane={seed['plane_coef']}"
        )
    md.extend(
        [
            "",
            "## Union",
            "",
            f"- Union reachable before overhead/table blocking: {meta['union_reachable_before_overhead_area_m2']:.2f} m^2",
            f"- Union reachable after overhead/table blocking: {meta['union_reachable_after_overhead_area_m2']:.2f} m^2",
            f"- Area removed by any overhead/table projection from the union: {meta['blocked_from_union_area_m2']:.2f} m^2",
            "",
            "## Figures",
            "",
            "- `multi_seed_union_vote.png`",
            "- `per_seed_reachable_before_overhead_overlay.png`",
            "- `per_seed_reachable_after_overhead_overlay.png`",
            "- `multi_seed_overhead_blocking_effect.png`",
        ]
    )
    (args.out_dir / "REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
