from hydra.utils import instantiate
from omegaconf import OmegaConf
from omegaconf import ListConfig
from utils.exp_utils import pretty_print
from types import SimpleNamespace
from copy import deepcopy

import copy
import os

def load_config_from_path(yaml_filename: str, yaml_dir: str):
    # Construct full path
    config_path = os.path.join(yaml_dir, yaml_filename)
    
    # Load into DictConfig
    cfg = OmegaConf.load(config_path)
    return cfg

class BaseLakeBuilder():
    def __init__(self, dataset_names,
                 config_dir="cli/conf/pretrain/data/",
                 **kwargs):
        self.config_dir = config_dir
        self.dataset_names = dataset_names
        self.cfg = kwargs
        self.pretrain_dataset = self.cfg['pretrain_dataset']
        self.global_lake_id_prefix = 0
        self.var2id_key = self.cfg['var_to_id']
        self.id2var_key = self.cfg['id_to_var']
        self.glbl_id_prefix_key = self.cfg['global_lake_id_prefix']
        self.ds_plot_id = self.cfg['ds_plot_id']
        self.norm_override = self.cfg['norm_override']

    def load_dataset(self, server_prefix='raid', rank=0, world_size=1, root_cfg=None):
        all_datasets = []
        if rank==0:
            pretty_print(f"Loading all builders using BaseBuilder")
        
        prefix = self.global_lake_id_prefix
        
        if root_cfg.task_name=="pretrain" and self.pretrain_dataset:
            dataset_ls = self.pretrain_dataset
        elif root_cfg.task_name=="evaluate" and root_cfg.evaluator.eval_dataset:
            dataset_ls = [root_cfg.evaluator.eval_dataset]
        elif root_cfg.task_name=="evaluate" and self.pretrain_dataset:
            dataset_ls = self.pretrain_dataset
        else:
            dataset_ls = self.dataset_names
        
        plot_ds = None

        if isinstance(dataset_ls, ListConfig):
            dataset_ls = list(dataset_ls)
        elif not isinstance(dataset_ls, list):
            dataset_ls = [dataset_ls]

        use_filter = self.cfg.get("use_global_lake_filter", False)
        global_ids    = self.cfg.get("lake_ids", None)
        global_formats    = self.cfg.get("lake_ids_format", None)

        if use_filter:
            if isinstance(global_ids, ListConfig): gl_ids = list(global_ids)
            if isinstance(global_formats, ListConfig): gl_fmt = list(global_formats)

        for idx, name in enumerate(dataset_ls):
            config_file = f"{name}.yaml"
            cfg = OmegaConf.load(os.path.join(self.config_dir, config_file))
            # Override values dynamically
            overrides = OmegaConf.create({
                "server_prefix": server_prefix,
                "task_name": root_cfg.task_name,
            })
            mv = getattr(root_cfg, "mask_variable", None)
            if isinstance(mv, ListConfig):
                mv = list(mv)
            overrides["mask_variable"] = mv
            overrides["mask_depth"] = getattr(root_cfg, "mask_depth", None)
            overrides["mask_var_across_depths"] = getattr(root_cfg, "mask_var_across_depths", False)
            overrides["mask_depth_for_all_vars"] = getattr(root_cfg, "mask_depth_across_variables", False)
            overrides["eval_time_grid"] = getattr(root_cfg, "eval_time_grid", "regular")
            
            if hasattr(root_cfg, "window"):
                overrides["dynamic_windows"] = root_cfg.window.dynamic_windows
                overrides["window_pairs"] = root_cfg.window.window_pairs
                overrides["window_sampling_strategy"] = root_cfg.window.window_sampling_strategy

            if use_filter and global_ids is not None:
                if isinstance(global_ids[0], (list, ListConfig)):
                    if global_ids[idx]:
                        overrides["lake_ids"]        = global_ids[idx]
                        overrides["lake_ids_format"] = global_formats[idx] if isinstance(global_formats, (list, ListConfig)) else global_formats
                else:
                    if global_ids:
                        overrides["lake_ids"]        = global_ids
                        overrides["lake_ids_format"] = global_formats

            cfg = OmegaConf.merge(cfg, overrides)
            cfg["context_len"] = root_cfg["seq_len"]
            cfg["prediction_len"]= root_cfg["pred_len"]
            
            builder = instantiate(cfg, base_builder=deepcopy(self))
            datasets = builder.load_dataset(prefix=prefix, rank=rank,
                                            world_size=world_size,
                                            sharding_mode=root_cfg.dataloader['sharding_mode'],
                                            root_cfg=root_cfg
                                            )
            all_datasets.extend(datasets)

            raw_ids = builder.split_ids(builder.lake_ids, builder.lake_ids_format)
            prefix += len(raw_ids)
        # store back the final prefix
        self.global_lake_id_prefix = prefix    
        if rank==0:
            plot_ds = all_datasets[self.ds_plot_id]
        
        return all_datasets, plot_ds