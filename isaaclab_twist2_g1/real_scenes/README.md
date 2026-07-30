# Real-Scene Release Registry

This directory is the Git-owned control plane for reproducible real-scene assets. Large runtime payloads remain outside Git under `assets/objects/real_scene/` and will move to a versioned HuggingFace asset repository.

## Ownership

HumanoidArena owns:

- Pipeline adapters and validation tools.
- Portable scene descriptors and acceptance policies.
- Locked artifact names, sizes, hashes, and external source commits.
- Small USD composition wrappers.
- Task configuration and experiment documentation.

External reconstruction workspaces own source images, COLMAP databases/models, checkpoints, raw PLY files, alignment work products, and full logs. The external 3DGRUT checkout remains a pinned dependency rather than a vendored repository.

## Release Layout

```text
real_scenes/
|-- acceptance_policy.json
|-- schema/
|   `-- scene-release.schema.json
`-- scenes/
    `-- <scene-id>/
        |-- scene.yaml
        |-- manifest.lock.json
        |-- acceptance.json
        `-- scene.usda
```

The runtime cache remains:

```text
assets/objects/real_scene/
```

## Pipeline Contract

Every scene advances through explicit, restartable stages:

```text
capture
-> reconstruct_colmap
-> diagnose_and_repair
-> train_3dgrut
-> export_nurec
-> align_to_sim
-> compose_collision
-> validate_release
-> deploy
-> smoke_test
-> formal_test
```

Each stage must consume named files, write immutable outputs or a new versioned directory, and emit a JSON report. A failed stage must not overwrite the last accepted release.

Required provenance includes:

- Source image set identity and frame count.
- COLMAP model hashes and registration metrics.
- External 3DGRUT repository URL, commit, config, and checkpoint hash.
- Alignment convention, transform, input hashes, and residual metrics.
- NuRec and collision payload hashes.
- Task config hash, spawn, smoke result, and formal-run status.

## Validation

Run metadata-only gates from the repository root:

```bash
python isaaclab_twist2_g1/tools/real_scene/validate_scene_release.py \
  --scene-dir isaaclab_twist2_g1/real_scenes/scenes/odin1_colmap_independent_repaired_3dgrut_30k
```

Also verify deployed binary assets:

```bash
python isaaclab_twist2_g1/tools/real_scene/validate_scene_release.py \
  --scene-dir isaaclab_twist2_g1/real_scenes/scenes/odin1_colmap_independent_repaired_3dgrut_30k \
  --asset-dir isaaclab_twist2_g1/assets/objects/real_scene
```

Required gate failures return a non-zero exit status. Warning gates report known limitations without blocking a release, such as camera residuals that are superseded by accepted LiDAR structural alignment.

## Artifact Distribution

The current Odin release is `server-local`: payloads remain on the lab host and are identified by SHA-256. When the HuggingFace asset repository is available, update only the distribution section of `manifest.lock.json` with the repository, revision, and paths. Do not replace hashes or silently mutate an existing release.
