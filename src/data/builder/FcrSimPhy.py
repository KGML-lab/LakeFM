import os.path as osp
import pandas as pd
import numpy as np

from tqdm import tqdm
from data.dataset import LakeDataset
from data.eval_dataset import LakeEvalDataset
from torch.utils.data import ConcatDataset
from utils.exp_utils import pretty_print
from data.builder.base import BaseLakeBuilder
from torch.distributed import get_rank, get_world_size

'''
Lake specific dataset builder for FCRSimPhy
'''
class FCRSimPhyBuilder():
    def __init__(self, 
                root_dir, 
                lake_data_file, 
                driver_file, 
                params_file, 
                lake_ids, 
                lake_ids_format,
                base_builder: BaseLakeBuilder,
                **kwargs):
        
        self.root_dir = root_dir
        self.lake_data_file = lake_data_file
        self.driver_file = driver_file
        self.params_file = params_file
        self.lake_ids = lake_ids
        self.lake_ids_format = lake_ids_format
        self.cfg = kwargs 
        self.norm_path = self.cfg['norm_path']
        self.lake_id_col = self.cfg['lake_id_col']
        self.base = base_builder

        if self.base.norm_override:
            self.norm_path = None

        self.vars1d = self.cfg['vars_1d']
        self.vars2d = self.cfg['vars_2d']
        self.depth_col = self.cfg['depth_col']
        self.date_col = self.cfg['date_col']
        self.inverse_params = self.cfg['inverse_params']

        self.var2id = self.base.var2id_key
        self.id2var = self.base.id2var_key

    def get_variate_ids(self, lake_df, driver_df):
        """
        Get the variate ids for 2D and 1D variables from the lake and driver dataframes.
        """
        variate_ids_2D = []
        variate_ids_1D = []

        for col in self.vars2d:
            if col in self.var2id:
                variate_ids_2D.append(self.var2id[col])
        
        for col in self.vars1d:
            if col in self.var2id:
                variate_ids_1D.append(self.var2id[col])
        
        return variate_ids_2D, variate_ids_1D

    def load_dataset(self, prefix, rank=0, world_size=1, sharding_mode = "ddp", root_cfg=None):
        self.task_name = root_cfg.task_name
        self.server_prefix = root_cfg.server_prefix
        self.run_name = root_cfg.run_name
        self.ckpt_name = root_cfg.evaluator.ckpt_name
        self.regular_grid_forecasting = root_cfg.regular_grid_forecasting
        self.regular_grid_depths = root_cfg.regular_grid_depths
        self.regular_grid_max_depth = root_cfg.regular_grid_max_depth
        
        if rank==0:
            pretty_print(f"FCRSimPhyBuilder initialized")

        raw_ids = self.split_ids(self.lake_ids, self.lake_ids_format) # list of lake names
        global_ids = [prefix + id_ for id_ in raw_ids]

        # each rank loads only its shard
        rank = get_rank()
        world_size = get_world_size()
        # local_ids = lake_id_list[rank::world_size]

        # sharding strategy
        if sharding_mode == "dataset":    
            all_pairs   = list(zip(raw_ids, global_ids))
            local_pairs = all_pairs[rank::world_size]                   # legacy behaviour
        else:                                             # "ddp": no manual shard
            local_pairs  = list(zip(raw_ids, global_ids)) 

        if self.cfg['lake_vars']:
            lake_df = pd.read_csv(osp.join(self.root_dir, self.lake_data_file))
        else:
            lake_df = None
        
        if self.cfg['driver_vars']:
            driver_df = pd.read_csv(osp.join(self.root_dir, self.driver_file))
            cols = [self.date_col] + self.vars1d
            driver_df = driver_df[cols]
        else:
            driver_df = None

        if self.cfg['param_vars']:
            param_df = pd.read_csv(osp.join(self.root_dir, self.params_file))
        else:
            param_df = None

        datasets=[]
        
        if rank==0:
            pretty_print(f"Loading FCR Simulation datasets")
        
        variate_ids_2D, variate_ids_1D = self.get_variate_ids(lake_df, 
                                                            driver_df)

        dataset_class = None
        if self.task_name == "pretrain":
            dataset_class = LakeDataset
            is_test_fraction = False
        elif self.task_name == "evaluate" or self.task_name == "infer" or self.task_name == "plot_predictions":
            dataset_class = LakeEvalDataset
            is_test_fraction = True
        else:
            raise ValueError(f"FCRSimPhyBuilder:: Invalid task name: {self.task_name}")

        # for lake_id in tqdm(local_ids):
        for raw_id, global_lake_id in tqdm(local_pairs, total=len(raw_ids)):
            save_normalization_file = "FcrSimPhy_"+str(raw_id)
            if self.norm_path:
                normalization_stats_path = osp.join(self.norm_path, save_normalization_file)
            else: # ood sites
                normalization_stats_path = osp.join(f"{self.server_prefix}/lakefm/dev/norm_stats", "global_variable_stats.json")
                # normalization_stats_path = None
            if lake_df is not None:
                lake_subset = lake_df[lake_df[self.lake_id_col] == raw_id].copy()
                cols = [self.date_col, self.depth_col] + self.vars2d
                lake_subset = lake_subset[cols]
            else:
                lake_subset = None
            
            if param_df is not None:
                param_subset = param_df[param_df[self.lake_id_col] == raw_id].copy()
                cols = self.inverse_params
                param_subset = param_subset[cols]
            else:
                param_subset = None

            ds = dataset_class(
                    lake_df=lake_subset,
                    driver_df=driver_df,
                    param_df=param_subset,
                    lake_id=global_lake_id,
                    cfg=self.cfg,
                    variate_ids_2D=variate_ids_2D,
                    variate_ids_1D=variate_ids_1D,
                    id2var=self.id2var,
                    var_names_2D=self.vars2d,
                    var_names_1D=self.vars1d,
                    is_test_fraction=is_test_fraction,
                    normalization_stats_path=normalization_stats_path,
                    run_name=self.run_name,
                    ckpt_name=self.ckpt_name,
                    regular_grid_forecasting=self.regular_grid_forecasting,
                    regular_grid_depths=self.regular_grid_depths,
                    regular_grid_max_depth=self.regular_grid_max_depth)
                    
            datasets.append(ds)
        print(f"Rank {rank}: Loaded {len(datasets)} datasets from FCR Simulations")
        return datasets

    def split_ids(self, lake_ids, lake_ids_format):
        return list(range(*lake_ids)) if lake_ids_format == "range" else lake_ids
