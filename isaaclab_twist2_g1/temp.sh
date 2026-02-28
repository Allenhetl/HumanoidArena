(unitree_sim_env) hcl4070-1@hcl4070-1:~/Desktop/taowen/projects/isaaclab_twist2_g1$ bash run.sh
PY: /home/hcl4070-1/.conda/envs/unitree_sim_env/bin/python
coverage: 7.13.0 /home/hcl4070-1/.conda/envs/unitree_sim_env/lib/python3.10/site-packages/coverage/__init__.py
has Tracer: True
sys.path head: ['/home/hcl4070-1/Desktop/taowen/projects/isaaclab_twist2_g1', '/home/hcl4070-1/.conda/envs/unitree_sim_env/lib/python310.zip', '/home/hcl4070-1/.conda/envs/unitree_sim_env/lib/python3.10', '/home/hcl4070-1/.conda/envs/unitree_sim_env/lib/python3.10/lib-dynload', '/home/hcl4070-1/.conda/envs/unitree_sim_env/lib/python3.10/site-packages', '/home/hcl4070-1/.conda/envs/unitree_sim_env/lib/python3.10/site-packages/cmeel.prefix/lib/python3.10/site-packages', '/home/hcl4070-1/.conda/envs/unitree_sim_env/lib/python3.10/site-packages/rerun_sdk']
[DDSManager] DDS system initialized
[DDSManager] DDSManager initialized
[INFO][AppLauncher]: Using device: cuda
[INFO][AppLauncher]: Loading experience file: /home/hcl4070-1/Desktop/taowen/projects/IsaacLab/apps/isaaclab.python.rendering.kit
[Warning] [simulation_app.simulation_app] Modules: ['omni.kit_app'] were loaded before SimulationApp was started and might not be loaded correctly.
[Warning] [simulation_app.simulation_app] Please check to make sure no extra omniverse or pxr modules are imported before the call to SimulationApp(...)
Loading user config located at: '/home/hcl4070-1/.conda/envs/unitree_sim_env/lib/python3.10/site-packages/omni/data/Kit/Isaac-Sim/4.5/user.config.json'
[Info] [carb] Logging to file: /home/hcl4070-1/.conda/envs/unitree_sim_env/lib/python3.10/site-packages/omni/logs/Kit/Isaac-Sim/4.5/kit_20260109_125710.log
2026-01-09 04:57:10 [0ms] [Warning] [omni.kit.app.plugin] No crash reporter present, dumps uploading isn't available.
2026-01-09 04:57:10 [4ms] [Warning] [omni.ext.plugin] [ext: rendering_modes] Extensions config 'extension.toml' doesn't exist '/home/hcl4070-1/Desktop/taowen/projects/IsaacLab/apps/rendering_modes' or '/home/hcl4070-1/Desktop/taowen/projects/IsaacLab/apps/rendering_modes/config'
2026-01-09 04:57:10 [288ms] [Warning] [omni.datastore] OmniHub is inaccessible
2026-01-09 04:57:12 [2,554ms] [Warning] [gpu.foundation.plugin] Skipping unsupported non-NVIDIA GPU: Intel(R) UHD Graphics (ADL-S GT1)
2026-01-09 04:57:12 [2,554ms] [Warning] [gpu.foundation.plugin] Skipping unsupported non-NVIDIA GPU: Intel(R) UHD Graphics (ADL-S GT1)

|---------------------------------------------------------------------------------------------|
| Driver Version: 580.95.05     | Graphics API: Vulkan
|=============================================================================================|
| GPU | Name                             | Active | LDA | GPU Memory | Vendor-ID | LUID       |
|     |                                  |        |     |            | Device-ID | UUID       |
|     |                                  |        |     |            | Bus-ID    |            |
|---------------------------------------------------------------------------------------------|
| 0   | NVIDIA GeForce RTX 4070 Laptop.. | Yes: 0 |     | 8188    MB | 10de      | 0          |
|     |                                  |        |     |            | 2820      | df1a73f1.. |
|     |                                  |        |     |            | 1         |            |
|---------------------------------------------------------------------------------------------|
| 1   | Intel(R) UHD Graphics (ADL-S G.. |        |     | 23846   MB | 8086      | 0          |
|     |                                  |        |     |            | 4688      | 86808846.. |
|     |                                  |        |     |            | 0         |            |
|=============================================================================================|
| OS: 22.04.5 LTS (Jammy Jellyfish) ubuntu, Version: 22.04.5, Kernel: 6.8.0-40-generic
| XServer Vendor: The X.Org Foundation, XServer Version: 12101004 (1.21.1.4)
| Processor: 13th Gen Intel(R) Core(TM) i7-13700HX
| Bare Metal Cores: 16 | Bare Metal Logical Cores: 32
| Available Cores:  24
|---------------------------------------------------------------------------------------------|
| Total Memory (MB): 31795 | Free Memory: 27378
| Total Page/Swap (MB): 2047 | Free Page/Swap: 2047
|---------------------------------------------------------------------------------------------|
2026-01-09 04:57:13 [3,012ms] [Warning] [gpu.foundation.plugin] IOMMU is enabled.
2026-01-09 04:57:13 [3,012ms] [Warning] [gpu.foundation.plugin] Detected IOMMU is enabled. Running CUDA peer-to-peer bandwidth and latency validation.
Unidirectional P2P=Enabled Bandwidth (P2P Writes) Matrix (GB/s)
   D\D     0
     0 197.51
P2P=Enabled Latency (P2P Writes) Matrix (us)
   GPU     0
     0   1.38

   CPU     0
     0   3.15
2026-01-09 04:57:14 [4,870ms] [Warning] [omni.log] Source: omni.hydra was already registered.
2026-01-09 04:57:15 [5,510ms] [Warning] [omni.isaac.dynamic_control] omni.isaac.dynamic_control is deprecated as of Isaac Sim 4.5. No action is needed from end-users.
2026-01-09 04:57:16 [6,708ms] [Warning] [omni.replicator.core.scripts.extension] No material configuration file, adding configuration to material settings directly.
2026-01-09 04:57:18 [8,248ms] [Warning] [omni.kit.menu.utils.app_menu] add_menu_items: menu [<MenuItemDescription name:'New'>, <MenuItemDescription name:'Open'>, <MenuItemDescription name:'Re-open with New Edit Layer'>, <MenuItemDescription name:'Save'>, <MenuItemDescription name:'Save With Options'>, <MenuItemDescription name:'Save As...'>, <MenuItemDescription name:'Save Flattened As...'>, <MenuItemDescription name:'Add Reference'>, <MenuItemDescription name:'Add Payload'>, <MenuItemDescription name:'Exit'>] cannot change delegate
/home/hcl4070-1/.conda/envs/unitree_sim_env/lib/python3.10/site-packages/wandb/sdk/internal/internal_api.py:13: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
  from pkg_resources import parse_version
2026-01-09 04:57:23 [12,980ms] [Warning] [omni.usd-abi.plugin] No setting was found for '/rtx-defaults/ambientOcclusion/enabled'
2026-01-09 04:57:23 [12,980ms] [Warning] [omni.usd-abi.plugin] No setting was found for '/rtx-defaults/directLighting/sampledLighting/enabled'
2026-01-09 04:57:23 [12,980ms] [Warning] [omni.usd-abi.plugin] No setting was found for '/rtx-defaults/indirectDiffuse/enabled'
2026-01-09 04:57:23 [12,980ms] [Warning] [omni.usd-abi.plugin] No setting was found for '/rtx-defaults/raytracing/cached/enabled'
2026-01-09 04:57:23 [12,980ms] [Warning] [omni.usd-abi.plugin] No setting was found for '/rtx-defaults/reflections/enabled'
2026-01-09 04:57:23 [12,980ms] [Warning] [omni.usd-abi.plugin] No setting was found for '/rtx-defaults/sceneDb/ambientLightIntensity'
2026-01-09 04:57:23 [12,981ms] [Warning] [omni.usd-abi.plugin] No setting was found for '/rtx-defaults/translucency/enabled'
2026-01-09 04:57:23 [12,981ms] [Warning] [omni.usd-abi.plugin] No setting was found for '/rtx-defaults/viewTile/limit'
2026-01-09 04:57:23 [13,174ms] [Warning] [rtx.scenedb.plugin] SceneDbContext : TLAS limit buffer size 7508933632
2026-01-09 04:57:23 [13,174ms] [Warning] [rtx.scenedb.plugin] SceneDbContext : TLAS limit : valid false, within: false
2026-01-09 04:57:23 [13,174ms] [Warning] [rtx.scenedb.plugin] SceneDbContext : TLAS limit : decrement: 167690, decrement size: 7433845248
2026-01-09 04:57:23 [13,174ms] [Warning] [rtx.scenedb.plugin] SceneDbContext : New limit 9574251 (slope: 447, intercept: 13179904)
2026-01-09 04:57:23 [13,174ms] [Warning] [rtx.scenedb.plugin] SceneDbContext : TLAS limit buffer size 4287216384
2026-01-09 04:57:23 [13,174ms] [Warning] [rtx.scenedb.plugin] SceneDbContext : TLAS limit : valid true, within: true
2026-01-09 04:57:23 [13,364ms] [Warning] [omni.usd-abi.plugin] No setting was found for '/rtx-defaults/directLighting/sampledLighting/samplesPerPixel'
2026-01-09 04:57:23 [13,366ms] [Warning] [omni.usd-abi.plugin] No setting was found for '/rtx-defaults/pathtracing/maxSamplesPerLaunch'
2026-01-09 04:57:23 [13,369ms] [Warning] [omni.usd-abi.plugin] No setting was found for '/rtx-defaults-transient/meshlights/forceDisable'
2026-01-09 04:57:23 [13,448ms] [Warning] [omni.usd-abi.plugin] No setting was found for '/rtx-defaults/post/dlss/execMode'
[MultiImageWriter] Shared memory initialized: isaac_multi_image_shm
============================================================
robot control system started
Task: Isaac-Move-Cylinder-G129-Dex1-Wholebody
Action source: dds
============================================================
[INFO]: Parsing configuration from: <class 'tasks.g1_tasks.move_cylinder_g1_29dof_dex1_wholebody.move_cylinder_g1_29dof_dex1_hw_env_cfg.MoveCylinderG129Dex1WholebodyEnvCfg'>

create environment...
2026-01-09 04:57:29 [19,389ms] [Warning] [isaaclab.envs.manager_based_env] Seed not set for the environment. The environment creation may not be deterministic.
[INFO]: Base environment:
	Environment device    : cuda:0
	Environment seed      : None
	Physics step-size     : 0.001
	Rendering step-size   : 0.01
	Environment step-size : 0.01
[INFO]: Time taken for scene creation : 0.555349 seconds
[INFO]: Scene manager:  <class InteractiveScene>
	Number of environments: 1
	Environment spacing   : 2.5
	Source prim name      : /World/envs/env_0
	Global prim paths     : []
	Replicate physics     : True
[INFO]: Starting the simulation. This may take a few seconds. Please wait...
2026-01-09 04:57:30 [20,499ms] [Warning] [omni.physx.plugin] PhysicsUSD: PhysxMaterialAPI at prim /World/envs/env_0/Room/Assets/dollies/dolly/wheelMaterial has attribute "improvePatchFriction" set to false. This setting is deprecated and support will end soon.
2026-01-09 04:57:30 [20,585ms] [Warning] [omni.physx.plugin] PhysX warning: PxMaterial::setFlag(): the friction behavior with the flag PxMaterialFlag::eIMPROVED_PATCH_FRICTION cleared is deprecated and support will end soon., FILE /builds/omniverse/physics/physx/source/physx/src/NpMaterial.cpp, LINE 186
2026-01-09 04:57:43 [33,324ms] [Warning] [rtx.postprocessing.plugin] DLSS increasing input dimensions: Render resolution of (371, 278) is below minimal input resolution of 300.
[INFO]: Time taken for simulation start : 13.563939 seconds
[INFO] Command Manager:  <CommandManager> contains 0 active terms.
+------------------------+
|  Active Command Terms  |
+--------+-------+-------+
| Index  | Name  |  Type |
+--------+-------+-------+
+--------+-------+-------+

[INFO] Event Manager:  <EventManager> contains 0 active terms.

[INFO] Recorder Manager:  <RecorderManager> contains 0 active terms.
+---------------------+
| Active Recorder Terms |
+-----------+---------+
|   Index   | Name    |
+-----------+---------+
+-----------+---------+

[INFO] Action Manager:  <ActionManager> contains 1 active terms.
+------------------------------------+
|  Active Action Terms (shape: 33)   |
+--------+-------------+-------------+
| Index  | Name        |   Dimension |
+--------+-------------+-------------+
|   0    | joint_pos   |          33 |
+--------+-------------+-------------+

dds_manager: <dds.dds_master.DDSManager object at 0x7ea0bcbcc2e0>
[DDSManager] object 'g129' not found, objects: dict_keys([])
[g1_state] G1 robot DDS communication instance obtained
g1_robot_dds is not initialized
[DDSManager] object 'dex1' not found, objects: dict_keys([])
[Observations] DDS communication instance obtained
[INFO] Observation Manager: <ObservationManager> contains 1 groups.
+------------------------------------------------+
|  Active Observation Terms in Group: 'policy'   |
+--------+----------------------+----------------+
| Index  | Name                 |     Shape      |
+--------+----------------------+----------------+
|   0    | robot_joint_state    |     (87,)      |
|   1    | robot_gipper_state   |      (2,)      |
|   2    | camera_image         | (480, 640, 3)  |
+--------+----------------------+----------------+

[INFO] Termination Manager:  <TerminationManager> contains 0 active terms.
+----------------------------+
|  Active Termination Terms  |
+--------+-------+-----------+
| Index  | Name  |  Time Out |
+--------+-------+-----------+
+--------+-------+-----------+

[INFO] Reward Manager:  <RewardManager> contains 1 active terms.
+-------------------------+
|   Active Reward Terms   |
+-------+--------+--------+
| Index | Name   | Weight |
+-------+--------+--------+
|   0   | reward |    1.0 |
+-------+--------+--------+

[INFO] Curriculum Manager:  <CurriculumManager> contains 0 active terms.
+----------------------+
| Active Curriculum Terms |
+-----------+----------+
|   Index   | Name     |
+-----------+----------+
+-----------+----------+

Creating window for environment.
[INFO]: Completed setting up the environment...

create environment success ...
robot cfg init pos: (-3.9, -2.81811, 0.8)
robot usd: /home/hcl4070-1/Desktop/taowen/projects/isaaclab_twist2_g1/assets/robots/g1-29dof_wholebody_dex1/g1_29dof_with_dex1_rev_1_0.usd

============================================================
 Getting robot stiffness and damping parameters from runtime environment
============================================================
 Getting joint kp/kd parameters from runtime environment...
 Available entities in scene: ['terrain', 'robot', 'world_camera', 'contact_forces', 'front_camera', 'robot_camera', 'room_walls', 'light']
✅ Found robot object: <class 'isaaclab.assets.articulation.articulation.Articulation'>
✅ Found robot.data object: <class 'isaaclab.assets.articulation.articulation_data.ArticulationData'>

 Available attributes in robot.data:
------------------------------------------------------------
 Total attributes: 116

 Joint-related attributes:
    body_incoming_joint_wrench_b: <class 'torch.Tensor'> - shape: torch.Size([1, 51, 6])
    default_joint_armature: <class 'torch.Tensor'> - shape: torch.Size([1, 33])
2026-01-09 04:57:43 [33,881ms] [Warning] [isaaclab.assets.articulation.articulation_data] The `default_joint_friction` property will be deprecated in a future release. Please use `default_joint_friction_coeff` instead.
    default_joint_friction: <class 'torch.Tensor'> - shape: torch.Size([1, 33])
    default_joint_friction_coeff: <class 'torch.Tensor'> - shape: torch.Size([1, 33])
2026-01-09 04:57:43 [33,881ms] [Warning] [isaaclab.assets.articulation.articulation_data] The `default_joint_limits` property will be deprecated in a future release. Please use `default_joint_pos_limits` instead.
    default_joint_limits: <class 'torch.Tensor'> - shape: torch.Size([1, 33, 2])
    default_joint_pos: <class 'torch.Tensor'> - shape: torch.Size([1, 33])
    default_joint_pos_limits: <class 'torch.Tensor'> - shape: torch.Size([1, 33, 2])
    default_joint_vel: <class 'torch.Tensor'> - shape: torch.Size([1, 33])
    joint_acc: <class 'torch.Tensor'> - shape: torch.Size([1, 33])
    joint_armature: <class 'torch.Tensor'> - shape: torch.Size([1, 33])
    joint_effort_limits: <class 'torch.Tensor'> - shape: torch.Size([1, 33])
    joint_effort_target: <class 'torch.Tensor'> - shape: torch.Size([1, 33])
2026-01-09 04:57:43 [33,882ms] [Warning] [isaaclab.assets.articulation.articulation_data] The `joint_friction` property will be deprecated in a future release. Please use `joint_friction_coeff` instead.
    joint_friction: <class 'torch.Tensor'> - shape: torch.Size([1, 33])
    joint_friction_coeff: <class 'torch.Tensor'> - shape: torch.Size([1, 33])
2026-01-09 04:57:43 [33,882ms] [Warning] [isaaclab.assets.articulation.articulation_data] The `joint_limits` property will be deprecated in a future release. Please use `joint_pos_limits` instead.
    joint_limits: <class 'torch.Tensor'> - shape: torch.Size([1, 33, 2])
    joint_names: <class 'list'> - shape: N/A
    joint_pos: <class 'torch.Tensor'> - shape: torch.Size([1, 33])
    joint_pos_limits: <class 'torch.Tensor'> - shape: torch.Size([1, 33, 2])
    joint_pos_target: <class 'torch.Tensor'> - shape: torch.Size([1, 33])
    joint_vel: <class 'torch.Tensor'> - shape: torch.Size([1, 33])
    joint_vel_limits: <class 'torch.Tensor'> - shape: torch.Size([1, 33])
    joint_vel_target: <class 'torch.Tensor'> - shape: torch.Size([1, 33])
2026-01-09 04:57:43 [33,882ms] [Warning] [isaaclab.assets.articulation.articulation_data] The `joint_velocity_limits` property will be deprecated in a future release. Please use `joint_vel_limits` instead.
    joint_velocity_limits: <class 'torch.Tensor'> - shape: torch.Size([1, 33])
    soft_joint_pos_limits: <class 'torch.Tensor'> - shape: torch.Size([1, 33, 2])
    soft_joint_vel_limits: <class 'torch.Tensor'> - shape: torch.Size([1, 33])

 Stiffness-related attributes:
    default_fixed_tendon_limit_stiffness: <class 'NoneType'> - shape: N/A
    default_fixed_tendon_stiffness: <class 'NoneType'> - shape: N/A
    default_joint_stiffness: <class 'torch.Tensor'> - shape: torch.Size([1, 33])
      Value: tensor([[100., 100., 150., 100., 100., 150., 100., 100., 150., 150., 150.,  40.,
          40.,  40.,  40.,  40.,  40.,  40.,  40.,  40.,  40.,  40.,  40.,   4.,
           4.,   4.,   4.,   4.,   4., 800., 800., 800., 800.]],
       device='cuda:0')
    fixed_tendon_limit_stiffness: <class 'NoneType'> - shape: N/A
    fixed_tendon_stiffness: <class 'NoneType'> - shape: N/A
    joint_stiffness: <class 'torch.Tensor'> - shape: torch.Size([1, 33])
      Value: tensor([[100., 100., 150., 100., 100., 150., 100., 100., 150., 150., 150.,  40.,
          40.,  40.,  40.,  40.,  40.,  40.,  40.,  40.,  40.,  40.,  40.,   4.,
           4.,   4.,   4.,   4.,   4., 800., 800., 800., 800.]],
       device='cuda:0')

️ Damping-related attributes:
    default_fixed_tendon_damping: <class 'NoneType'> - shape: N/A
    default_joint_damping: <class 'torch.Tensor'> - shape: torch.Size([1, 33])
      Value: tensor([[2.0000, 2.0000, 4.0000, 2.0000, 2.0000, 4.0000, 2.0000, 2.0000, 4.0000,
         4.0000, 4.0000, 5.0000, 5.0000, 2.0000, 2.0000, 5.0000, 5.0000, 2.0000,
         2.0000, 5.0000, 5.0000, 5.0000, 5.0000, 0.2000, 0.2000, 0.2000, 0.2000,
         0.2000, 0.2000, 3.0000, 3.0000, 3.0000, 3.0000]], device='cuda:0')
    fixed_tendon_damping: <class 'NoneType'> - shape: N/A
    joint_damping: <class 'torch.Tensor'> - shape: torch.Size([1, 33])
      Value: tensor([[2.0000, 2.0000, 4.0000, 2.0000, 2.0000, 4.0000, 2.0000, 2.0000, 4.0000,
         4.0000, 4.0000, 5.0000, 5.0000, 2.0000, 2.0000, 5.0000, 5.0000, 2.0000,
         2.0000, 5.0000, 5.0000, 5.0000, 5.0000, 0.2000, 0.2000, 0.2000, 0.2000,
         0.2000, 0.2000, 3.0000, 3.0000, 3.0000, 3.0000]], device='cuda:0')

⚙️ All potentially relevant attributes (first 20):
    body_ang_vel_w: Tensor - shape: torch.Size([1, 51, 3])
    body_com_ang_vel_w: Tensor - shape: torch.Size([1, 51, 3])
    body_com_lin_vel_w: Tensor - shape: torch.Size([1, 51, 3])
    body_com_pos_b: Tensor - shape: torch.Size([1, 51, 3])
    body_com_pos_w: Tensor - shape: torch.Size([1, 51, 3])
    body_com_pose_b: Tensor - shape: torch.Size([1, 51, 7])
    body_com_pose_w: Tensor - shape: torch.Size([1, 51, 7])
    body_com_vel_w: Tensor - shape: torch.Size([1, 51, 6])
    body_incoming_joint_wrench_b: Tensor - shape: torch.Size([1, 51, 6])
    body_lin_vel_w: Tensor - shape: torch.Size([1, 51, 3])
2026-01-09 04:57:44 [33,952ms] [Warning] [isaaclab.utils.math] The function 'quat_rotate' will be deprecated in favor of the faster method 'quat_apply'. Please use 'quat_apply' instead....
    body_link_ang_vel_w: Tensor - shape: torch.Size([1, 51, 3])
    body_link_lin_vel_w: Tensor - shape: torch.Size([1, 51, 3])
    body_link_pos_w: Tensor - shape: torch.Size([1, 51, 3])
    body_link_pose_w: Tensor - shape: torch.Size([1, 51, 7])
    body_link_vel_w: Tensor - shape: torch.Size([1, 51, 6])
    body_pos_w: Tensor - shape: torch.Size([1, 51, 3])
    body_pose_w: Tensor - shape: torch.Size([1, 51, 7])
    body_vel_w: Tensor - shape: torch.Size([1, 51, 6])
    com_pos_b: Tensor - shape: torch.Size([1, 51, 3])
    default_fixed_tendon_damping: NoneType - shape: N/A
   ... 63 more relevant attributes

 Trying common parameter names:
   ✅ Found stiffness: default_joint_stiffness
      Type: <class 'torch.Tensor'>
      Shape: torch.Size([1, 33])
      Value: tensor([[100., 100., 150., 100., 100., 150., 100., 100., 150., 150., 150.,  40.,
          40.,  40.,  40.,  40.,  40.,  40.,  40.,  40.,  40.,  40.,  40.,   4.,
           4.,   4.,   4.,   4.,   4., 800., 800., 800., 800.]],
       device='cuda:0')
   ✅ Found damping: default_joint_damping
      Type: <class 'torch.Tensor'>
      Shape: torch.Size([1, 33])
      Value: tensor([[2.0000, 2.0000, 4.0000, 2.0000, 2.0000, 4.0000, 2.0000, 2.0000, 4.0000,
         4.0000, 4.0000, 5.0000, 5.0000, 2.0000, 2.0000, 5.0000, 5.0000, 2.0000,
         2.0000, 5.0000, 5.0000, 5.0000, 5.0000, 0.2000, 0.2000, 0.2000, 0.2000,
         0.2000, 0.2000, 3.0000, 3.0000, 3.0000, 3.0000]], device='cuda:0')

✅ Parameter acquisition complete!
 Successfully found at least one parameter!
✅ Successfully got robot parameters!
============================================================


***  Please left-click on the Sim window to activate rendering. ***


2026-01-09 04:57:44 [34,309ms] [Warning] [omni.physx.plugin] PhysicsUSD: PhysxMaterialAPI at prim /World/envs/env_0/Room/Assets/dollies/dolly/wheelMaterial has attribute "improvePatchFriction" set to false. This setting is deprecated and support will end soon.
2026-01-09 04:57:44 [34,395ms] [Warning] [omni.physx.plugin] PhysX warning: PxMaterial::setFlag(): the friction behavior with the flag PxMaterialFlag::eIMPROVED_PATCH_FRICTION cleared is deprecated and support will end soon., FILE /builds/omniverse/physics/physx/source/physx/src/NpMaterial.cpp, LINE 186
dds_manager: <dds.dds_master.DDSManager object at 0x7ea0bcbcc2e0>
[DDSManager] object 'g129' not found, objects: dict_keys([])
[g1_state] G1 robot DDS communication instance obtained
g1_robot_dds is not initialized
[DDSManager] object 'dex1' not found, objects: dict_keys([])
[Observations] DDS communication instance obtained
✅ Viewport active camera set to: /World/envs/env_0/Robot/d435_link/front_cam
⚠️  Overriding step_hz 500 -> 100 to match TWIST2 policy rate
========= create image server =========
[Image Server] Initializing multi-image server from shared memory
[MultiImageReader] Shared memory opened: isaac_multi_image_shm
[Image Server] Multi-image server initialized (xrobot)
[Image Server] Starting send_process from shared memory...
[Image Server] Multi-image publishing thread started
========= create image server success =========
========= create dds =========
[g1_robot] Input shared memory: psm_4a30c818
[g1_robot] Output shared memory: psm_d06cc1cf
[g1_robot] G1 robot DDS node initialized
[DDSManager] register object 'g129' success (category: No category)
[gripper] Input shared memory: psm_08e5686f
[gripper] Output shared memory: psm_87dc9503
[gripper] Gripper DDS node initialized
[DDSManager] register object 'dex1' success (category: No category)
[run_command_dds] Input shared memory: psm_fec51ef8
[run_command_dds] Output shared memory: psm_4fdf2fd5
[run_command_dds] Run command DDS node initialized
[DDSManager] register object 'run_command' success (category: No category)
[reset_pose_dds] Output shared memory: psm_1ec4eea8
[reset_pose_dds] Reset pose DDS node initialized
[DDSManager] register object 'reset_pose' success (category: No category)
[sim_state_dds] Input shared memory: psm_0d3d1f3c
[sim_state_dds] Sim state DDS node initialized
[DDSManager] register object 'sim_state' success (category: No category)
[Image Server] XRobot connect failed: [Errno 113] No route to host
[Image Server] XRobot connect failed to 10.42.0.35:12345 (resolved: 10.42.0.35): [Errno 113] No route to host
[g1_robot] State publisher initialized (rt/lowstate)
[gripper] Gripper state publisher initialized
[sim_state_dds] Sim state publisher initialized
[DDSManager] publish loop thread started
[DDSManager] manager started, managing 5 objects
[g1_robot] Create ChannelSubscriber...
[gripper] Gripper command subscriber initialized
[run_command_dds] Run command subscriber initialized
[reset_pose_dds] Reset pose command subscriber initialized
========= create dds success =========

create action provider: dds_wholebody...
args_cli.task: Isaac-Move-Cylinder-G129-Dex1-Wholebody
ActionProvider init
enable_robot: g129
enable_gripper: True
enable_dex3: False
[DDSActionProvider] DDS communication initialized
[DDSActionProvider] ONNX policy loaded with providers: ['CPUExecutionProvider']
========= create controller =========
  - control frequency: 100Hz
[SimpleController] set the action provider: DDSActionProvider
========= create controller success =========
performance analysis enabled, report every 500 steps
Note: The DDS in Sim transmits messages on channel 1. Please ensure that other DDS instances use the same channel for message exchange by setting: ChannelFactoryInitialize(1).
========= start controller =========
[DDSActionProvider] ActionProvider started
[SimpleController] the controller is started
========= start controller success =========
reward: 0.0
dds_manager: <dds.dds_master.DDSManager object at 0x7ea0bcbcc2e0>
[g1_state] G1 robot DDS communication instance obtained
[Observations] DDS communication instance obtained
reward: 0.0
reward: 0.0
reward: 0.0
reward: 0.0
[Image Server] XRobot connect failed: [Errno 113] No route to host
[Image Server] XRobot connect failed to 10.42.0.35:12345 (resolved: 10.42.0.35): [Errno 113] No route to host
reward: 0.0
reward: 0.0
reward: 0.0
reward: 0.0
reward: 0.0
reward: 0.0
reward: 0.0
reward: 0.0
reward: 0.0
[Image Server] XRobot connect failed: [Errno 113] No route to host
[Image Server] XRobot connect failed to 10.42.0.35:12345 (resolved: 10.42.0.35): [Errno 113] No route to host
reward: 0.0
^C
received signal 2, stopping controller...
[DDSActionProvider] ActionProvider stop
[DDSActionProvider] ActionProvider stopped
[SimpleController] the controller is stopped
[MultiImageReader] Shared memory closed: isaac_multi_image_shm
[Image Server] Multi-image server closed
[Image Server] Publishing thread stopped
[MultiImageReader] Shared memory closed: isaac_multi_image_shm
[Image Server] Multi-image server closed
[DDSManager] publish loop thread stopped
run.sh: line 14: 1109716 Killed                  python sim_main.py --device cuda --enable_cameras --task Isaac-Move-Cylinder-G129-Dex1-Wholebody --robot_type g129 --enable_dex1_dds --image_transport xrobot --image_xrobot_host 10.42.0.35 --image_xrobot_port 12345 --image_xrobot_width 640 --image_xrobot_height 480 --image_xrobot_bitrate 4194304 --image_fps 30 --image_xrobot_ffmpeg /usr/bin/ffmpeg
(unitree_sim_env) hcl4070-1@hcl4070-1:~/Desktop/taowen/projects/isaaclab_twist2_g1$ /home/hcl4070-1/.conda/envs/unitree_sim_env/lib/python3.10/multiprocessing/resource_tracker.py:224: UserWarning: resource_tracker: There appear to be 9 leaked shared_memory objects to clean up at shutdown
  warnings.warn('resource_tracker: There appear to be %d '

