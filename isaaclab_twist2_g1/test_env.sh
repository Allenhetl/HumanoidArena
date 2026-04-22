  #!/usr/bin/env bash
  set -euo pipefail

  echo '=== 1. NVML / CUDA ==='
  nvidia-smi

  echo
  echo '=== 2. Vulkan tool ==='
  if command -v vulkaninfo >/dev/null 2>&1; then
    echo "vulkaninfo=$(command -v vulkaninfo)"
  else
    echo "vulkaninfo not found"
  fi

  echo
  echo '=== 3. Key Vulkan / NVIDIA graphics runtime files ==='
  ls -l /usr/share/vulkan/icd.d/nvidia_icd.json 2>/dev/null || echo 'missing: /usr/share/vulkan/icd.d/nvidia_icd.json'
  ls -l /usr/lib/x86_64-linux-gnu/libEGL_nvidia.so* 2>/dev/null || echo 'missing: libEGL_nvidia.so*'
  ls -l /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so* 2>/dev/null || echo 'missing: libGLX_nvidia.so*'
  ls -l /usr/lib/x86_64-linux-gnu/libnvidia-glvkspirv.so* 2>/dev/null || echo 'missing: libnvidia-glvkspirv.so*'

  echo
  echo '=== 4. Vulkan enumerate ==='
  if command -v vulkaninfo >/dev/null 2>&1 && [ -f /usr/share/vulkan/icd.d/nvidia_icd.json ]; then
    export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
    export XDG_RUNTIME_DIR=/tmp/runtime-root
    mkdir -p "$XDG_RUNTIME_DIR"
    chmod 700 "$XDG_RUNTIME_DIR"
    vulkaninfo | grep -E 'GPU id|deviceName|driverVersion' -n | sed -n '1,80p'
  else
    echo 'skip vulkan enumerate: vulkaninfo or nvidia_icd.json missing'
  fi

  echo
  echo '=== 5. Minimal Isaac smoke ==='
  cd /ai/Yichi/taowen/HumanoidArena/isaaclab_twist2_g1

  export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
  export XDG_RUNTIME_DIR=/tmp/runtime-root
  mkdir -p "$XDG_RUNTIME_DIR"
  chmod 700 "$XDG_RUNTIME_DIR"

  python script/eval_scripts/sonic/sim_eval_vla.py \
    --task Isaac-Move-Football-Single-G129-Dex3-Wholebody \
    --checkpoint /ai/Yichi/taowen/ckpts/0421/act_sonic_football_0416/pretrained_model \
    --server-url http://127.0.0.1:19999 \
    --device cuda:1 \
    --headless \
    --max-steps 5 \
    --seed 0 \
    --repeat-idx 0 \
    --episode-seed 1132158683 \
    --results-dir /tmp/isaac_smoke_results \
    --video-fps 30 \
    --post-termination-record-steps 0