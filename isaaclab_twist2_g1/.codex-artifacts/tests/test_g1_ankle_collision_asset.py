import argparse
import os
import sys

from isaaclab.app import AppLauncher


PROJECT_ROOT = "/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1"
ASSET_PATH = os.path.join(
    PROJECT_ROOT,
    "assets/robots/g1-29dof_wholebody_dex3/g1_29dof_with_dex3_rev_1_0.usd",
)
EXPECTED_PITCH_APPROXIMATIONS = {
    "left_ankle_pitch_link/collisions": "convexHull",
    "right_ankle_pitch_link/collisions": "convexHull",
}
EXPECTED_ROLL_MESHES = {
    "left_ankle_roll_link/collisions/sole_mesh": (0.18, 0.065, 0.014),
    "right_ankle_roll_link/collisions/sole_mesh": (0.18, 0.065, 0.014),
}
SIZE_TOLERANCE = 1e-3


def main() -> int:
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(["--headless", "--device", "cpu"])
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    try:
        from pxr import Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.Open(ASSET_PATH)
        if stage is None:
            raise RuntimeError(f"Failed to open USD asset: {ASSET_PATH}")

        for suffix, expected in EXPECTED_PITCH_APPROXIMATIONS.items():
            matches = [str(prim.GetPath()) for prim in stage.Traverse() if str(prim.GetPath()).endswith(suffix)]
            if not matches:
                raise RuntimeError(f"Missing prim ending with '{suffix}'")
            if len(matches) != 1:
                raise RuntimeError(f"Expected one prim ending with '{suffix}', found {matches}")

            prim_path = matches[0]
            collision_api = UsdPhysics.MeshCollisionAPI.Get(stage, prim_path)
            approximation = collision_api.GetApproximationAttr().Get() if collision_api else None
            print(f"{prim_path} -> {approximation}")
            if approximation != expected:
                raise RuntimeError(
                    f"{prim_path} approximation mismatch: expected {expected}, got {approximation}"
                )

        for suffix, expected_size in EXPECTED_ROLL_MESHES.items():
            matches = [str(prim.GetPath()) for prim in stage.Traverse() if str(prim.GetPath()).endswith(suffix)]
            if not matches:
                raise RuntimeError(f"Missing roll collider mesh ending with '{suffix}'")
            if len(matches) != 1:
                raise RuntimeError(f"Expected one roll collider mesh ending with '{suffix}', found {matches}")

            prim_path = matches[0]
            mesh_prim = stage.GetPrimAtPath(prim_path)
            mesh = UsdGeom.Mesh(mesh_prim)
            extent = mesh.GetExtentAttr().Get()
            if extent is None or len(extent) != 2:
                raise RuntimeError(f"{prim_path} has invalid extent: {extent}")
            size = tuple(max_v - min_v for min_v, max_v in zip(extent[0], extent[1]))

            collision_api = UsdPhysics.MeshCollisionAPI.Get(stage, prim_path)
            approximation = collision_api.GetApproximationAttr().Get() if collision_api else None
            print(f"{prim_path} -> size={size} approximation={approximation}")

            if approximation != "convexDecomposition":
                raise RuntimeError(
                    f"{prim_path} approximation mismatch: expected convexDecomposition, got {approximation}"
                )
            for axis_size, expected_axis_size in zip(size, expected_size):
                if abs(axis_size - expected_axis_size) > SIZE_TOLERANCE:
                    raise RuntimeError(
                        f"{prim_path} size mismatch: expected {expected_size}, got {size}"
                    )
        return 0
    finally:
        simulation_app.close()


if __name__ == "__main__":
    sys.exit(main())
