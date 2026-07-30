#!/usr/bin/env python3
"""Export raw checkpoint Gaussians without dataset or camera normalization."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from threedgrut.export.formats.ply import PLYExporter
from threedgrut.export.usd.nurec.exporter import NuRecExporter
from threedgrut.model.model import MixtureOfGaussians


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--usdz", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, weights_only=False)
    conf = checkpoint["config"]
    conf.export_usd.apply_normalizing_transform = False
    model = MixtureOfGaussians(conf)
    model.init_from_checkpoint(checkpoint, setup_optimizer=False)

    args.ply.parent.mkdir(parents=True, exist_ok=True)
    args.usdz.parent.mkdir(parents=True, exist_ok=True)
    PLYExporter().export(model, args.ply, dataset=None, conf=conf)
    NuRecExporter(export_cameras=False, export_post_processing=False).export(
        model,
        args.usdz,
        dataset=None,
        conf=conf,
    )
    print(f"Exported raw PLY: {args.ply}")
    print(f"Exported raw NuRec: {args.usdz}")


if __name__ == "__main__":
    main()
