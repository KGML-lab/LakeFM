#!/usr/bin/env bash
set -euo pipefail

eval_dataset="LakeBeD"
node="${NPROC_PER_NODE:-1}"
denorm_eval=False
plot=False

var_names_subset=null
depth_name=""

# Masking defaults
mask_variable=null
mask_depth=null
mask_var_across_depths=false
mask_depth_across_variables=false

usage() {
  echo "Usage:"
  echo "  bash scripts/driver.sh <run_name> <lake_name> [depth_m] [--plot] [--depth depth_m] [--denorm] [--gpus N] [--vars JSON_LIST] [--mask-vars JSON_LIST] [--mask-depths JSON_LIST_OR_NUMBER] [--eval-dataset NAME]"
  echo ""
  echo "Notes:"
  echo "  - depth is OPTIONAL for non-plot evaluation."
  echo "  - If --plot is set, depth MUST be provided (either as 3rd positional arg or via --depth)."
  exit 2
}

positional=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --denorm)
      denorm_eval=True
      shift
      ;;
    --plot)
      plot=True
      shift
      ;;
    --depth)
      depth_name="${2:-}"
      [[ -z "$depth_name" ]] && usage
      shift 2
      ;;
    --gpus)
      node="${2:-}"
      [[ -z "$node" ]] && usage
      shift 2
      ;;
    --vars|--var-names-subset)
      var_names_subset="${2:-}"
      [[ -z "$var_names_subset" ]] && usage
      shift 2
      ;;
    --mask-vars|--mask-variables)
      mask_variable="${2:-}"
      [[ -z "$mask_variable" ]] && usage
      mask_var_across_depths=true
      mask_depth_across_variables=false
      mask_depth=null
      shift 2
      ;;
    --mask-depths|--mask-depth)
      mask_depth="${2:-}"
      [[ -z "$mask_depth" ]] && usage
      mask_depth_across_variables=true
      mask_var_across_depths=false
      mask_variable=null
      shift 2
      ;;
    --eval-dataset)
      eval_dataset="${2:-}"
      [[ -z "$eval_dataset" ]] && usage
      shift 2
      ;;
    -h|--help)
      usage
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "Unknown option: $1"
      usage
      ;;
    *)
      positional+=("$1")
      shift
      ;;
  esac
done

if [[ $# -gt 0 ]]; then
  positional+=("$@")
fi

if [[ ${#positional[@]} -lt 2 ]]; then
  usage
fi

run_name="${positional[0]}"
lake_name="${positional[1]}"

# Optional positional depth (3rd arg)
if [[ ${#positional[@]} -ge 3 && -z "$depth_name" ]]; then
  depth_name="${positional[2]}"
fi

if [[ "$plot" == "True" && -z "$depth_name" ]]; then
  echo "Error: --plot requires a depth. Provide [depth_m] or --depth <depth_m>."
  usage
fi

SERVER_PREFIX="../resources"
CKPT_FOLDER="lakefm/dev/pretrain_ckpts"
EVAL_OUT_FOLDER="lakefm/dev/evaluations"
ckpt_name="lakefm5m"
CKPT_PATH="${SERVER_PREFIX}/${CKPT_FOLDER}/${ckpt_name}.pth"
EVAL_OUTPUT_PATH="${SERVER_PREFIX}/${EVAL_OUT_FOLDER}/${ckpt_name}/${eval_dataset}"

cmd=(
  torchrun --nproc_per_node="$node" -m cli.main -cp conf/pretrain task_name="evaluate"
  project_name="lakefm_eval"
  run_name="$run_name"
  server_prefix="$SERVER_PREFIX"
  evaluator.eval_dataset="$eval_dataset"
  model.add_or_concat="concat"
  evaluator.ckpt_path="$CKPT_PATH"
  evaluator.ckpt_name="$ckpt_name"
  evaluator.num_trials=1
  evaluator.output_dir="$EVAL_OUTPUT_PATH"
  dataloader.batch_size=32
  dataloader.shuffle=False
  data.use_global_lake_filter=true
  data.lake_ids="[\"${lake_name}\"]"
  data.lake_ids_format=list
  model.revin=False
  data.ds_plot_id=0
  dataloader.sharding_mode=ddp
  plot_merged=True
  plot_interval=True
  num_plot_batches=-1
  forecast_plot_type="line"
  model.variate_wise_df=true
  model.shared_variate_embedding_for_df=true
  model.num_layers=12
  evaluator.denorm_eval="$denorm_eval"
  plotter.var_names_subset="$var_names_subset"
  mask_variable="$mask_variable"
  mask_depth="$mask_depth"
  mask_var_across_depths="$mask_var_across_depths"
  mask_depth_across_variables="$mask_depth_across_variables"
  evaluator.plot="$plot"
  dataloader.num_workers=12
)

if [[ -n "$depth_name" ]]; then
  cmd+=(plotter.depth_name="$depth_name")
fi

"${cmd[@]}"