eval_dataset="LakeBeD"
node=$1
ckpt_name=$2
run_name=$3
lake_id=$4
depth_index=$5
depth_name=$6
var_names_subset=$7
mask_value=$8
mask_kind=$9
denorm_eval=${10}
plot=${11}

SERVER_PREFIX="resources"
CKPT_FOLDER="lakefm/dev/pretrain_ckpts"
EVAL_OUT_FOLDER="lakefm/dev/evaluations"
CKPT_PATH="${SERVER_PREFIX}/${CKPT_FOLDER}/${ckpt_name}"
EVAL_OUTPUT_PATH="${SERVER_PREFIX}/${EVAL_OUT_FOLDER}/${ckpt_name}/${eval_dataset}"

# Select masking scenario
if [ "$mask_kind" = "none" ]; then
  mask_variable=null
  mask_depth=null
  mask_var_across_depths=false
  mask_depth_across_variables=false
elif [ "$mask_kind" = "variate" ]; then
  mask_variable=$mask_value
  mask_depth=null
  mask_var_across_depths=true
  mask_depth_across_variables=false
else
  mask_variable=null
  mask_depth=$mask_value
  mask_var_across_depths=false
  mask_depth_across_variables=true
fi

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
        data.lake_ids="[[${lake_id}]]" \
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
        plotter.depth_index=$depth_index \
        plotter.depth_name=$depth_name \
        plotter.var_names_subset="${var_names_subset}" \
        mask_variable="${mask_variable}" \
        mask_depth=$mask_depth \
        mask_var_across_depths=$mask_var_across_depths \
        mask_depth_across_variables=$mask_depth_across_variables \
        evaluator.plot=$plot \
        dataloader.num_workers=12