# Real Scene Collision Repair Visual Report

## Purpose

This report records where the scene collision was modified and what the modification changes geometrically. It is intended to complement first-person standing videos with whole-scene, top-down diagnostics.

## Inputs

- Scene: `assets/objects/real_scene/small_warehouse_digital_twin_office_fixed_gauss_xneg90_ccm1.usda`
- Mesh path: `/World/mesh`
- Masks: `analysis_outputs/real_scene_floor_extract_start_1p5_4p0_ccm1_plane15_close35_v2/floor_masks.npz`
- Mask key: `reachable`
- Fitted plane: `z = -0.051646155*x + 0.028293407*y + -1.092631070`

## Current Split vs Face-Graph Prototype

| Metric | Current centroid split | Face-graph connected prototype |
| --- | ---: | ---: |
| Removed triangles | 17245 | 16122 |
| Removed area | 28.144 m^2 | 26.734 m^2 |
| Current-only triangles | 1123 | - |
| Graph-only triangles | - | 0 |

Interpretation:

- `current centroid split` is the already generated repaired-collision asset rule.
- `face-graph connected prototype` only removes candidate floor faces that are connected to the start-floor component in the original mesh face graph.
- If current-only regions appear around furniture, walls, or disconnected floor islands, graph-connected split is safer.
- If graph-connected removes too little, the original mesh may be disconnected by reconstruction cracks and needs grid-assisted bridging or small-gap stitching.

## Figures

1. `01_floor_masks_and_residuals.png`: floor masks, reachable area, residual heatmap, clearance.
2. `02_split_current_vs_graph.png`: current removed faces versus graph-connected prototype.
3. `03_repair_effect_overview.png`: before-repair residual, high/low confidence repair zones, removed face overview.

## Key Numbers

- Reachable cells: 11713
- Reachable area: 29.282 m^2
- Reachable residual p50/p90/p95/max: 61.5 / 120.6 / 131.0 / 150.0 mm
- High-confidence reachable cells (<=100mm residual): 7070
- Low-confidence reachable cells (>100mm residual): 1958

## Recommended Next Asset Trial

Generate a second repaired wrapper using the face-graph connected split if visual inspection shows current-only deletion outside the true start-connected floor. Then run the same static-ref A/B and add feet contact / foot height metrics.
