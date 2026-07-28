#!/usr/bin/env bash
# Shared helpers for campaign-style batch evaluation scripts.

if [[ -n "${_EVAL_BATCH_UTILS_SH:-}" ]]; then return 0; fi
readonly _EVAL_BATCH_UTILS_SH=1

_EVAL_BATCH_UTILS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_EVAL_BATCH_BATCH_DIR="$(cd "${_EVAL_BATCH_UTILS_DIR}/.." && pwd)"
_EVAL_BATCH_ISAACLAB_ROOT="$(cd "${_EVAL_BATCH_BATCH_DIR}/.." && pwd)"
_EVAL_BATCH_REPO_ROOT="$(cd "${_EVAL_BATCH_ISAACLAB_ROOT}/.." && pwd)"
source "${_EVAL_BATCH_ISAACLAB_ROOT}/script/common/runtime_paths.sh"
export EVAL_BATCH_REPO_ROOT="${_EVAL_BATCH_REPO_ROOT}"

eval_batch_backend_script_dir() {
  case "$1" in
    twist2|sonic|sonic_low_latency|mimic_lite)
      printf '%s/script/eval_scripts/%s' "${_EVAL_BATCH_ISAACLAB_ROOT}" "$1" ;;
    *) echo "Error: unknown backend '$1'" >&2; return 1 ;;
  esac
}

eval_batch_task_launcher() {
  case "$1" in
    open_door) printf '%s' HSI_open_door_run_vla_eval_parallel.sh ;;
    football) printf '%s' HOI_football_run_vla_eval_parallel.sh ;;
    doubledesk) printf '%s' HOI_double_desk_run_vla_eval_parallel.sh ;;
    pp_box) printf '%s' HOI_pp_box_run_vla_eval_parallel.sh ;;
    boxing) printf '%s' HSI_boxing_run_vla_eval_parallel.sh ;;
    sit_sofa) printf '%s' HSI_sit_sofa_run_vla_eval_parallel.sh ;;
    vision_navi) printf '%s' HSI_vision_navi_run_vla_eval_parallel.sh ;;
    *) echo "Error: unknown task '$1'" >&2; return 1 ;;
  esac
}

eval_batch_launcher_path() {
  printf '%s/%s' "$(eval_batch_backend_script_dir "$1")" "$(eval_batch_task_launcher "$2")"
}

# Twist2 has distinct test timing. Every other supported executor uses Sonic timing.
# A task-specific override such as OPEN_DOOR_ENV_CONFIG_YAML always wins.
eval_batch_env_config_yaml() {
  local backend="$1" task="$2" override_name value filename
  override_name="${task^^}_ENV_CONFIG_YAML"
  value="${!override_name:-}"
  if [[ -z "${value}" ]]; then
    case "${backend}:${task}" in
      twist2:open_door) filename=open_door_twist2_test.yaml ;;
      twist2:football) filename=football_single_twist2_test.yaml ;;
      sonic:open_door|sonic_low_latency:open_door|mimic_lite:open_door) filename=open_door_sonic_test.yaml ;;
      sonic:football|sonic_low_latency:football|mimic_lite:football) filename=football_single_sonic_test.yaml ;;
      *) echo "Error: no env config mapping for backend=$backend task=$task" >&2; return 1 ;;
    esac
    value="${_EVAL_BATCH_ISAACLAB_ROOT}/tasks/common_test_config/base_test/${filename}"
  elif [[ "${value}" != /* ]]; then
    value="${_EVAL_BATCH_ISAACLAB_ROOT}/${value}"
  fi
  [[ -f "${value}" ]] || { echo "Error: env config does not exist: ${value}" >&2; return 1; }
  (cd "$(dirname "${value}")" && printf '%s/%s' "$PWD" "$(basename "${value}")")
}

eval_batch_execution_gmt() {
  case "$1" in
    sonic|sonic_low_latency) printf '%s' sonic ;;
    twist2|mimic_lite) printf '%s' "$1" ;;
    *) echo "Error: unknown backend '$1'" >&2; return 1 ;;
  esac
}

eval_batch_gmt_relation() {
  [[ "$1" == "$2" ]] && printf '%s' in_gmt || printf '%s' cross_gmt
}

eval_batch_setup_backend_env() {
  case "$1" in
    mimic_lite)
      export MIMIC_LITE_ROBOT_CFG=1
      export VLA_MIMICLITE_YAW_CALIB_DEG="${EVAL_BATCH_MIMIC_YAW_CALIB_DEG:-${VLA_MIMICLITE_YAW_CALIB_DEG:-0.0}}" ;;
    twist2|sonic|sonic_low_latency)
      unset MIMIC_LITE_ROBOT_CFG VLA_MIMICLITE_YAW_CALIB_DEG || true ;;
    *) echo "Error: unknown backend '$1'" >&2; return 1 ;;
  esac
}

eval_batch_find_latest_complete_model() {
  local root="$1" latest=-1 result= d step n
  for d in "${root}"/*; do
    [[ -d "$d" ]] || continue
    step="$(basename "$d")"; [[ "$step" =~ ^[0-9]+$ ]] || continue
    [[ -s "$d/pretrained_model/config.json" && -s "$d/pretrained_model/model.safetensors" ]] || continue
    n=$((10#$step)); (( n > latest )) && { latest=$n; result="$d/pretrained_model"; }
  done
  [[ -n "$result" ]] || { echo "Error: no complete numeric checkpoint found under $root" >&2; return 1; }
  printf '%s' "$result"
}

eval_batch_validate_model_complete() {
  local f
  for f in config.json model.safetensors; do
    [[ -s "$1/$f" ]] || { echo "Error: incomplete VLA model: $1/$f" >&2; return 1; }
  done
}

eval_batch_extract_policy_id() {
  local model="$1" parent upper
  parent="$(dirname "$model")"; upper="$(dirname "$parent")"
  if [[ "$(basename "$upper")" == checkpoints ]]; then basename "$(dirname "$upper")"; else basename "$parent"; fi
}

eval_batch_extract_checkpoint_step() {
  local parent upper step
  parent="$(dirname "$1")"; upper="$(dirname "$parent")"; step="$(basename "$parent")"
  if [[ "$(basename "$upper")" == checkpoints && "$step" =~ ^[0-9]+$ ]]; then printf '%s' "$step"; else printf '%s' final; fi
}

eval_batch_sanitize_component() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | tr -c '[:alnum:]' '_' | sed 's/^_//; s/_$//; s/__*/_/g'
}

eval_batch_compute_stage_dir() {
  printf '%s__%s__step_%s' "$(eval_batch_sanitize_component "$1")" "$(eval_batch_sanitize_component "$2")" "$3"
}

eval_batch_backend_results_root() {
  printf '%s/eval_results/%s' "$(eval_batch_backend_script_dir "$1")" "$2"
}

# Resolve one campaign root shared by every requested backend without creating it.
eval_batch_resolve_run_id() {
  local campaign="$1" backends="$2" backend root latest= candidate= run_id ts
  if [[ -n "${RUN_TIMESTAMP:-}" ]]; then
    printf '%s__%s' "$campaign" "$RUN_TIMESTAMP"
    return
  fi
  if [[ "${RESUME_LATEST:-0}" == 1 ]]; then
    for backend in $backends; do
      root="$(eval_batch_backend_script_dir "$backend")/eval_results"
      latest="$(python3 - "$root" "$campaign" <<'PY'
import pathlib, sys
root, prefix = pathlib.Path(sys.argv[1]), sys.argv[2] + "__"
items = sorted(p.name for p in root.glob(prefix + "*") if p.is_dir()) if root.is_dir() else []
print(items[-1] if items else "")
PY
)"
      [[ -n "$latest" ]] || { echo "Error: RESUME_LATEST=1 but no $campaign campaign exists for $backend" >&2; return 1; }
      if [[ -z "$candidate" ]]; then candidate="$latest"; elif [[ "$candidate" != "$latest" ]]; then
        echo "Error: inconsistent latest campaign across backends: $candidate versus $latest ($backend)" >&2; return 1
      fi
    done
    printf '%s' "$candidate"
    return
  fi
  ts="$(date +%Y%m%d_%H%M%S)"; run_id="${campaign}__${ts}"
  for backend in $backends; do
    [[ ! -e "$(eval_batch_backend_results_root "$backend" "$run_id")" ]] || {
      echo "Error: refusing to reuse existing campaign root with RESUME_LATEST=0: $run_id" >&2; return 1;
    }
  done
  printf '%s' "$run_id"
}

# Tab-separated schema. Tabs/newlines in values are rejected rather than producing bad JSON.
# backend, execution_gmt, training_gmt, relation, task, policy, step, model, results, yaml, status
eval_batch_manifest_stage_fields() {
  local v
  for v in "$@"; do
    [[ "$v" != *$'\t'* && "$v" != *$'\n'* ]] || { echo "Error: manifest value contains a tab/newline" >&2; return 1; }
  done
  [[ $# -eq 11 ]] || { echo "Error: manifest stage requires 11 fields" >&2; return 1; }
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' "$@"
}

eval_batch_write_manifest() {
  local output="$1"
  python3 -c '
import json, os, subprocess, sys
out = sys.argv[1]
stages = []
keys = ("execution_backend", "execution_gmt", "training_gmt", "gmt_relation", "task",
        "policy_id", "checkpoint_step", "model_path", "results_dir", "env_config_yaml", "status")
for number, line in enumerate(sys.stdin, 1):
    parts = line.rstrip("\n").split("\t")
    if len(parts) != len(keys):
        raise SystemExit(f"malformed manifest row {number}: expected {len(keys)} fields, got {len(parts)}")
    row = dict(zip(keys, parts))
    try: row["checkpoint_step"] = int(row["checkpoint_step"])
    except ValueError: pass
    stages.append(row)
if not stages:
    raise SystemExit("manifest has no stages")
root = os.environ["EVAL_BATCH_REPO_ROOT"]
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
backends = os.environ.get("EVAL_BACKENDS", "").split()
resources = {}
for backend in backends:
    if backend == "mimic_lite":
        resources[backend] = {
            "onnx_path": os.environ.get("MIMIC_LITE_ONNX_PATH", ""),
            "yaml_path": os.environ.get("MIMIC_LITE_YAML_PATH", ""),
            "yaw_calibration_deg": os.environ.get("EVAL_BATCH_MIMIC_YAW_CALIB_DEG", "0.0"),
        }
    elif backend in ("sonic", "sonic_low_latency"):
        prefix = "SONIC_LOW_LATENCY" if backend == "sonic_low_latency" else "SONIC_RELEASE"
        resources[backend] = {
            "encoder_path": os.environ.get(prefix + "_ENCODER_PATH", ""),
            "decoder_path": os.environ.get(prefix + "_DECODER_PATH", ""),
        }
manifest = {
    "schema_version": 2,
    "campaign_id": os.environ.get("EVAL_BATCH_CAMPAIGN_ID", ""),
    "run_id": os.environ.get("EVAL_BATCH_RUN_ID", ""),
    "run_timestamp": os.environ.get("RUN_TIMESTAMP", ""),
    "run_label": os.environ.get("RUN_LABEL", ""),
    "git_commit": commit,
    "execution_backends": backends,
    "num_workers": int(os.environ.get("NUM_WORKERS", "0")),
    "seeds": os.environ.get("EVAL_BATCH_ACTUAL_SEEDS", "").split(),
    "repeats_per_seed": os.environ.get("EVAL_BATCH_ACTUAL_REPEATS", ""),
    "max_steps": json.loads(os.environ.get("EVAL_BATCH_ACTUAL_MAX_STEPS_JSON", "{}")),
    "config_provenance": {
        "batch_script": os.environ.get("EVAL_BATCH_SCRIPT", ""),
        "model_source": os.environ.get("EVAL_BATCH_MODEL_SOURCE", ""),
        "env_config_selection": "task override variable, otherwise backend+task mapping",
    },
    "backend_resources": resources,
    "stages": stages,
}
text = json.dumps(manifest, indent=2) + "\n"
if out == "-":
    sys.stdout.write(text)
else:
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f: f.write(text)
    os.replace(tmp, out)
' "$output"
}

# Export resolved resource defaults so every manifest records what launchers use.
export MIMIC_LITE_ONNX_PATH="${MIMIC_LITE_ONNX_PATH:-${_EVAL_BATCH_ISAACLAB_ROOT}/assets/checkpoints/mimic_lite/policy-xua2csee-4000.onnx}"
export MIMIC_LITE_YAML_PATH="${MIMIC_LITE_YAML_PATH:-${_EVAL_BATCH_ISAACLAB_ROOT}/assets/checkpoints/mimic_lite/policy-xua2csee-4000.yaml}"
export SONIC_RELEASE_ENCODER_PATH="${SONIC_ENCODER_PATH:-${SONIC_POLICY_ROOT}/model_encoder.onnx}"
export SONIC_RELEASE_DECODER_PATH="${SONIC_DECODER_PATH:-${SONIC_POLICY_ROOT}/model_decoder.onnx}"
export SONIC_LOW_LATENCY_ENCODER_PATH="${SONIC_LOW_LATENCY_ENCODER_PATH:-${SONIC_ENCODER_PATH:-${SONIC_LOW_LATENCY_POLICY_ROOT}/model_encoder.onnx}}"
export SONIC_LOW_LATENCY_DECODER_PATH="${SONIC_LOW_LATENCY_DECODER_PATH:-${SONIC_DECODER_PATH:-${SONIC_LOW_LATENCY_POLICY_ROOT}/model_decoder.onnx}}"
