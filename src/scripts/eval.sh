#!/usr/bin/env bash
set -euo pipefail

eval_dataset="LakeBeD"
node="${NPROC_PER_NODE:-1}"
denorm_eval=False
plot=False
# If not provided, we plot ALL variables (up to the plotter's max_features).
var_names_subset=null

# Masking defaults
mask_variable=null
mask_depth=null
mask_var_across_depths=false
mask_depth_across_variables=false

usage() {
  echo "Usage: bash scripts/driver.sh <run_name> <lake_name> <depth_m> [--denorm] [--plot] [--gpus N] [--vars JSON_LIST] [--eval-dataset NAME]"
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

if [[ ${#positional[@]} -lt 3 ]]; then
  usage
fi

run_name="${positional[0]}"
lake_name="${positional[1]}"
depth_name="${positional[2]}"

SERVER_PREFIX="resources"
CKPT_FOLDER="lakefm/dev/pretrain_ckpts"
EVAL_OUT_FOLDER="lakefm/dev/evaluations"
ckpt_name="ckpt"
CKPT_PATH="${SERVER_PREFIX}/${CKPT_FOLDER}/${ckpt_name}.pth"
EVAL_OUTPUT_PATH="${SERVER_PREFIX}/${EVAL_OUT_FOLDER}/${ckpt_name}/${eval_dataset}"

torchrun --nproc_per_node=$node -m cli.main -cp conf/pretrain task_name="evaluate" \
        project_name="lakefm_eval" \
        run_name=$run_name \
        server_prefix=$SERVER_PREFIX   \
        evaluator.eval_dataset=$eval_dataset \
        model.add_or_concat="concat" \
        evaluator.ckpt_path=$CKPT_PATH \
        evaluator.ckpt_name=$ckpt_name \
        evaluator.num_trials=1 \
        evaluator.output_dir=$EVAL_OUTPUT_PATH \
        dataloader.batch_size=32 \
        dataloader.shuffle=False \
        data.use_global_lake_filter=true \
        data.lake_ids="[\"${lake_name}\"]" \
        data.lake_ids_format=list \
        model.revin=False \
        data.ds_plot_id=0 \
        dataloader.sharding_mode=ddp \
        plot_merged=True \
        data.norm_override=True \
        plot_interval=True \
        num_plot_batches=-1 \
        forecast_plot_type="line" \
        model.variate_wise_df=true \
        model.shared_variate_embedding_for_df=true \
        model.num_layers=12 \
        evaluator.denorm_eval=$denorm_eval \
        plotter.depth_name=$depth_name \
        plotter.var_names_subset=$var_names_subset \
        mask_variable=$mask_variable \
        mask_depth=$mask_depth \
        mask_var_across_depths=$mask_var_across_depths \
        mask_depth_across_variables=$mask_depth_across_variables \
        evaluator.plot=$plot \
        dataloader.num_workers=12