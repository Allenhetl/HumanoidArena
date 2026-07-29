#!/usr/bin/env python3
"""Plan mask-constrained overhead tube lights from a real-scene mesh."""

import argparse
from pathlib import Path
import json
import math

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=Path, required=True, help="Input binary big-endian PLY mesh")
    parser.add_argument("--output", type=Path, required=True, help="Directory for the layout and plots")
    parser.add_argument("--robot-xy", type=float, nargs=2, default=(-3.239, -4.425), metavar=("X", "Y"))
    parser.add_argument("--ball-xy", type=float, nargs=2, default=(-3.239, -3.225), metavar=("X", "Y"))
    return parser.parse_args()


def _dilate(a: np.ndarray, it: int = 1) -> np.ndarray:
    out = a.copy()
    for _ in range(it):
        p = np.pad(out, 1, constant_values=False)
        out = (
            p[1:-1, 1:-1]
            | p[:-2, 1:-1]
            | p[2:, 1:-1]
            | p[1:-1, :-2]
            | p[1:-1, 2:]
            | p[:-2, :-2]
            | p[:-2, 2:]
            | p[2:, :-2]
            | p[2:, 2:]
        )
    return out


def _erode(a: np.ndarray, it: int = 1) -> np.ndarray:
    out = a.copy()
    for _ in range(it):
        p = np.pad(out, 1, constant_values=False)
        out = (
            p[1:-1, 1:-1]
            & p[:-2, 1:-1]
            & p[2:, 1:-1]
            & p[1:-1, :-2]
            & p[1:-1, 2:]
            & p[:-2, :-2]
            & p[:-2, 2:]
            & p[2:, :-2]
            & p[2:, 2:]
        )
    return out


def _connected_component(mask: np.ndarray, start_rc: tuple[int, int]) -> np.ndarray:
    visited = np.zeros_like(mask, dtype=bool)
    q = [start_rc]
    visited[start_rc] = True
    head = 0
    while head < len(q):
        r, c = q[head]
        head += 1
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and mask[rr, cc] and not visited[rr, cc]:
                visited[rr, cc] = True
                q.append((rr, cc))
    return visited


def _world_to_rc(xy: np.ndarray, min_xy: np.ndarray, cell: float) -> np.ndarray:
    ij = np.floor((xy - min_xy) / cell).astype(int)
    return np.stack([ij[..., 1], ij[..., 0]], axis=-1)


def _inside_mask(points_xy: np.ndarray, mask: np.ndarray, min_xy: np.ndarray, cell: float) -> np.ndarray:
    rc = _world_to_rc(points_xy, min_xy, cell)
    r = rc[..., 0]
    c = rc[..., 1]
    valid = (r >= 0) & (r < mask.shape[0]) & (c >= 0) & (c < mask.shape[1])
    out = np.zeros(points_xy.shape[:-1], dtype=bool)
    out[valid] = mask[r[valid], c[valid]]
    return out


def _estimate_oriented_frame(points_xy: np.ndarray) -> dict:
    """Robust minimum-area oriented rectangle by brute-force angle search.

    Uses percentile extents rather than raw min/max to reduce sensitivity to isolated
    floor-like outliers. Returns long-axis basis u and short-axis basis v.
    """
    pts = points_xy.astype(np.float64)
    center = np.median(pts, axis=0)
    pts0 = pts - center
    # Search 0..180 deg. 0.5 deg resolution is enough for layout planning.
    best = None
    for deg in np.linspace(0.0, 179.5, 360):
        th = math.radians(float(deg))
        u = np.array([math.cos(th), math.sin(th)])
        v = np.array([-math.sin(th), math.cos(th)])
        pu = pts0 @ u
        pv = pts0 @ v
        u0, u1 = np.percentile(pu, [2.0, 98.0])
        v0, v1 = np.percentile(pv, [2.0, 98.0])
        area = max(1e-6, (u1 - u0) * (v1 - v0))
        if best is None or area < best[0]:
            best = (area, deg, u, v, u0, u1, v0, v1)
    _area, deg, u, v, u0, u1, v0, v1 = best
    # Make u the long axis.
    if (u1 - u0) < (v1 - v0):
        u, v = v, -u
        u0, u1, v0, v1 = v0, v1, -u1, -u0
        deg = (deg + 90.0) % 180.0
    return {
        'center': center,
        'u': u,
        'v': v,
        'u_min': float(u0),
        'u_max': float(u1),
        'v_min': float(v0),
        'v_max': float(v1),
        'yaw_deg': float(deg),
    }


def main():
    args = parse_args()
    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ply_path = args.mesh.resolve()
    robot_xy = np.asarray(args.robot_xy, dtype=np.float32)
    ball_xy = np.asarray(args.ball_xy, dtype=np.float32)

    with ply_path.open('rb') as f:
        n_vertex = None
        fmt = None
        while True:
            line = f.readline().decode('utf-8', 'replace').strip()
            if line.startswith('format '):
                fmt = line.split()[1]
            if line.startswith('element vertex '):
                n_vertex = int(line.split()[-1])
            if line == 'end_header':
                break
        if fmt != 'binary_big_endian':
            raise SystemExit(f'unsupported PLY format: {fmt}')
        raw = f.read(n_vertex * 6 * 4)
    verts = np.frombuffer(raw, dtype='>f4').reshape(n_vertex, 6).astype(np.float32)
    xyz = verts[:, :3]
    nrm = verts[:, 3:6]
    z = xyz[:, 2]
    nz = nrm[:, 2]
    finite = np.isfinite(xyz).all(axis=1) & np.isfinite(nrm).all(axis=1)

    # Floor-like points.
    lo, hi = np.percentile(z[finite], [0.5, 99.5])
    horiz = finite & (nz > 0.45) & (z > lo) & (z < hi)
    z_h = z[horiz]
    low_cut = np.percentile(z_h, 35) if len(z_h) else np.percentile(z[finite], 35)
    z_floor_candidates = z_h[z_h <= low_cut]
    if len(z_floor_candidates) < 1000:
        z_floor_candidates = z_h
    bins = np.arange(float(z_floor_candidates.min()), float(z_floor_candidates.max()) + 0.01, 0.01)
    hist, edges = np.histogram(z_floor_candidates, bins=bins)
    floor_z = float((edges[int(hist.argmax())] + edges[int(hist.argmax()) + 1]) * 0.5)
    near = horiz & (np.abs(z - floor_z) < 0.04)
    if near.sum() > 1000:
        floor_z = float(np.median(z[near]))
    ground = finite & (nz > 0.35) & (np.abs(z - floor_z) < 0.08)
    gxy = xyz[ground, :2]

    # Occupancy mask and connected floor around robot start.
    cell = 0.10
    margin = 0.5
    min_xy = np.floor((gxy.min(axis=0) - margin) / cell) * cell
    max_xy = np.ceil((gxy.max(axis=0) + margin) / cell) * cell
    shape = np.ceil((max_xy - min_xy) / cell).astype(int) + 1
    if shape[0] * shape[1] > 2_000_000:
        cell = 0.15
        shape = np.ceil((max_xy - min_xy) / cell).astype(int) + 1
    occ = np.zeros((shape[1], shape[0]), dtype=bool)
    ij = np.floor((gxy - min_xy) / cell).astype(int)
    ij = ij[(ij[:, 0] >= 0) & (ij[:, 0] < shape[0]) & (ij[:, 1] >= 0) & (ij[:, 1] < shape[1])]
    occ[ij[:, 1], ij[:, 0]] = True
    occ_clean = _erode(_dilate(occ, 2), 1)
    occ_for_lights = _erode(occ_clean, 3)  # keep lights away from detected floor boundary.

    start_rc = tuple(int(v) for v in _world_to_rc(robot_xy[None, :], min_xy, cell)[0])
    occ_idx = np.argwhere(occ_clean)
    if not (0 <= start_rc[0] < occ_clean.shape[0] and 0 <= start_rc[1] < occ_clean.shape[1] and occ_clean[start_rc]):
        xy_occ = np.stack([min_xy[0] + (occ_idx[:, 1] + 0.5) * cell, min_xy[1] + (occ_idx[:, 0] + 0.5) * cell], axis=1)
        k = int(np.argmin(np.sum((xy_occ - robot_xy[None, :]) ** 2, axis=1)))
        start_rc = tuple(int(v) for v in occ_idx[k])
    comp = _connected_component(occ_clean, start_rc)
    comp_for_lights = comp & occ_for_lights
    comp_idx = np.argwhere(comp)
    comp_xy = np.stack([min_xy[0] + (comp_idx[:, 1] + 0.5) * cell, min_xy[1] + (comp_idx[:, 0] + 0.5) * cell], axis=1)

    frame = _estimate_oriented_frame(comp_xy)
    center = frame['center']
    u = frame['u']
    v = frame['v']
    u_min, u_max = frame['u_min'], frame['u_max']
    v_min, v_max = frame['v_min'], frame['v_max']

    # Ceiling/light height over connected footprint extents.
    rel = xyz[:, :2].astype(np.float64) - center[None, :]
    pu = rel @ u
    pv = rel @ v
    in_oriented_rect = finite & (pu >= u_min) & (pu <= u_max) & (pv >= v_min) & (pv <= v_max)
    z_region = z[in_oriented_rect]
    ceil_z = float(np.percentile(z_region, 95)) if len(z_region) else floor_z + 2.6
    light_z = float(np.clip(ceil_z - 0.25, floor_z + 1.8, floor_z + 3.0))

    # Generate candidate tubes in oriented room frame, then reject tubes outside floor mask.
    edge_margin = 0.55
    row_spacing = 1.2
    tube_max_len = 3.0
    coverage_threshold = 0.82
    u0, u1 = u_min + edge_margin, u_max - edge_margin
    v0, v1 = v_min + edge_margin, v_max - edge_margin
    long_len = max(0.1, u1 - u0)
    short_len = max(0.1, v1 - v0)
    n_rows = min(8, max(1, int(math.ceil(short_len / row_spacing))))
    n_seg = min(6, max(1, int(math.ceil(long_len / tube_max_len))))
    row_coords = np.linspace(v0 + short_len / (2 * n_rows), v1 - short_len / (2 * n_rows), n_rows) if n_rows > 1 else np.array([(v0 + v1) / 2])
    seg_centers = np.linspace(u0 + long_len / (2 * n_seg), u1 - long_len / (2 * n_seg), n_seg) if n_seg > 1 else np.array([(u0 + u1) / 2])
    seg_len = min(tube_max_len, long_len / n_seg * 0.85 if n_seg else long_len)

    accepted = []
    rejected = []
    sample_t = np.linspace(-0.5, 0.5, 31)
    for ri, vv in enumerate(row_coords):
        for si, uu in enumerate(seg_centers):
            center_xy = center + uu * u + vv * v
            pts = center_xy[None, :] + (sample_t[:, None] * seg_len) * u[None, :]
            inside = _inside_mask(pts, comp_for_lights, min_xy, cell)
            center_inside = bool(_inside_mask(center_xy[None, :], comp_for_lights, min_xy, cell)[0])
            coverage = float(inside.mean())
            rec = {
                'name': f'auto_tube_{ri}_{si}',
                'pos': [round(float(center_xy[0]), 3), round(float(center_xy[1]), 3), round(float(light_z), 3)],
                'axis_xy': [round(float(u[0]), 4), round(float(u[1]), 4)],
                'yaw_deg': round(float(frame['yaw_deg']), 2),
                'length': round(float(seg_len), 3),
                'radius': 0.05,
                'intensity': 3500.0,
                'mask_coverage': round(coverage, 3),
                'center_inside_mask': center_inside,
            }
            if center_inside and coverage >= coverage_threshold:
                accepted.append(rec)
            else:
                rejected.append(rec)

    # If overly sparse, keep best coverage near robot/ball area for visualization but mark threshold issue.
    layout = {
        'scene': 'ipark_t2_505_20260721',
        'source_ply': str(ply_path),
        'method': 'oriented room frame from robust min-area rectangle over connected floor component; tube candidates filtered by eroded connected floor mask coverage',
        'cell_size_m': cell,
        'robot_xy': [float(robot_xy[0]), float(robot_xy[1])],
        'ball_xy': [float(ball_xy[0]), float(ball_xy[1])],
        'floor_z_est': round(float(floor_z), 4),
        'ceiling_z_p95_est': round(float(ceil_z), 4),
        'light_z': round(float(light_z), 4),
        'room_frame': {
            'center_xy': [round(float(center[0]), 3), round(float(center[1]), 3)],
            'u_long_axis_xy': [round(float(u[0]), 4), round(float(u[1]), 4)],
            'v_short_axis_xy': [round(float(v[0]), 4), round(float(v[1]), 4)],
            'yaw_deg': round(float(frame['yaw_deg']), 2),
            'u_minmax': [round(float(u_min), 3), round(float(u_max), 3)],
            'v_minmax': [round(float(v_min), 3), round(float(v_max), 3)],
        },
        'candidate_count': len(accepted) + len(rejected),
        'accepted_count': len(accepted),
        'rejected_count': len(rejected),
        'coverage_threshold': coverage_threshold,
        'tube_lights': accepted,
        'rejected_tube_lights': rejected,
    }
    (out_dir / 'ipark_auto_tube_light_layout_v2.json').write_text(json.dumps(layout, indent=2), encoding='utf-8')

    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    rng = np.random.default_rng(0)
    plot_gxy = gxy[rng.choice(len(gxy), min(len(gxy), 160000), replace=False)] if len(gxy) > 160000 else gxy
    comp_plot = comp_xy
    if len(comp_plot) > 100000:
        comp_plot = comp_plot[rng.choice(len(comp_plot), 100000, replace=False)]

    corners_uv = np.array([[u_min, v_min], [u_max, v_min], [u_max, v_max], [u_min, v_max]])
    corners_xy = center[None, :] + corners_uv[:, 0:1] * u[None, :] + corners_uv[:, 1:2] * v[None, :]

    def draw_tube(ax, lt, color, lw=3.5, alpha=1.0, ls='-'):
        x, y, _ = lt['pos']
        L = lt['length']
        ax_u = np.array(lt['axis_xy'], dtype=float)
        p0 = np.array([x, y]) - 0.5 * L * ax_u
        p1 = np.array([x, y]) + 0.5 * L * ax_u
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, lw=lw, alpha=alpha, ls=ls, solid_capstyle='round')

    fig, ax = plt.subplots(figsize=(10, 10), dpi=180)
    ax.scatter(plot_gxy[:, 0], plot_gxy[:, 1], s=0.12, c='lightgray', alpha=0.55, label='floor-like mesh points')
    ax.scatter(comp_plot[:, 0], comp_plot[:, 1], s=1.0, c='#66aaff', alpha=0.35, label='connected floor component')
    ax.add_patch(Polygon(corners_xy, closed=True, fill=False, edgecolor='green', linewidth=2.0, label='oriented room rectangle'))
    for lt in rejected:
        draw_tube(ax, lt, color='red', lw=2.2, alpha=0.45, ls='--')
    for lt in accepted:
        draw_tube(ax, lt, color='orange', lw=4.0, alpha=1.0)
    ax.scatter([robot_xy[0]], [robot_xy[1]], c='red', s=80, marker='*', label='robot init')
    ax.scatter([ball_xy[0]], [ball_xy[1]], c='black', s=45, marker='o', label='football')
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f'Ipark oriented mask-constrained tube layout\nyaw={frame["yaw_deg"]:.1f}°, floor_z={floor_z:.3f}, light_z={light_z:.3f}, accepted={len(accepted)}/{len(accepted)+len(rejected)}')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.grid(True, alpha=0.25)
    ax.legend(loc='best', fontsize=8)
    pad = 1.0
    ax.set_xlim(corners_xy[:, 0].min() - pad, corners_xy[:, 0].max() + pad)
    ax.set_ylim(corners_xy[:, 1].min() - pad, corners_xy[:, 1].max() + pad)
    fig.tight_layout()
    fig.savefig(out_dir / 'ipark_auto_tube_light_layout_v2_topdown.png')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 10), dpi=180)
    ax.scatter(plot_gxy[:, 0], plot_gxy[:, 1], s=0.10, c='gray', alpha=0.45, label='floor-like points')
    ax.add_patch(Polygon(corners_xy, closed=True, fill=False, edgecolor='green', linewidth=1.8, label='oriented room rectangle'))
    for lt in rejected:
        draw_tube(ax, lt, color='red', lw=1.8, alpha=0.35, ls='--')
    for lt in accepted:
        draw_tube(ax, lt, color='orange', lw=3.0, alpha=0.95)
    ax.scatter([robot_xy[0]], [robot_xy[1]], c='red', s=80, marker='*', label='robot init')
    ax.scatter([ball_xy[0]], [ball_xy[1]], c='black', s=45, marker='o', label='football')
    ax.set_aspect('equal', adjustable='box')
    ax.set_title('Ipark overview: accepted orange, rejected red dashed')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.grid(True, alpha=0.25)
    ax.legend(loc='best', fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / 'ipark_auto_tube_light_layout_v2_overview.png')
    plt.close(fig)

    print(json.dumps(layout, indent=2))
    print('wrote', out_dir)


if __name__ == '__main__':
    main()
