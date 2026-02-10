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
Lake specific dataset builder for WQHansonSim
'''
class WQHansonSimBuilder():
    def __init__(self, 
                root_dir,  
                lake_ids, 
                lake_ids_format,
                base_builder: BaseLakeBuilder,
                **kwargs):
        
        self.root_dir = root_dir
        self.cfg = kwargs
        self.norm_path = self.cfg['norm_path']
        self.lake_ids = lake_ids
        self.lake_ids_format = lake_ids_format
        self.lake_id_col = self.cfg['lake_id_col']
        self.date_col = self.cfg['date_col']
        self.depth_col = self.cfg['depth_col']
        self.base = base_builder
        self.var2id = self.base.var2id_key
        self.id2var = self.base.id2var_key
        
        if self.base.norm_override:
            self.norm_path = None

        self.vars2d = self.cfg['vars_2d']
        self.vars1d = self.cfg['vars_1d']
        self.lake_suffix = self.cfg['lake_suffix']
        self.driver_suffix = self.cfg['driver_suffix']

    def get_variate_ids(self):
        """
        Get the variate ids for 2D and 1D variables from the lake and driver dataframes.
        """
        variate_ids_2D = []
        variate_ids_1D = []

        for col in self.vars2d:
            variate_ids_2D.append(self.var2id[col])
        
        for col in self.vars1d:
            variate_ids_1D.append(self.var2id[col])
        
        return variate_ids_2D, variate_ids_1D

    def load_dataset(self, prefix, rank=0, world_size=1, sharding_mode="ddp", root_cfg=None):
        self.task_name = root_cfg.task_name
        self.server_prefix = root_cfg.server_prefix
        self.run_name = root_cfg.run_name
        self.ckpt_name = root_cfg.evaluator.ckpt_name
        self.regular_grid_forecasting = root_cfg.regular_grid_forecasting
        self.regular_grid_depths = root_cfg.regular_grid_depths
        self.regular_grid_max_depth = root_cfg.regular_grid_max_depth

        if rank==0:
            pretty_print(f"WQHansonSimBuilder initialized")

        raw_ids = self.split_ids(self.lake_ids, self.lake_ids_format) # list of lake names

        if raw_ids and isinstance(raw_ids[0], int):
            lake_keys = [self.cfg["lake_names"][i-1] for i in raw_ids]
        else:
            lake_keys = raw_ids

        id_to_dataset_name = {idx+1: name for idx, name in enumerate(lake_keys)}
        id_list = list(id_to_dataset_name.keys())
        global_ids = [prefix + id_ for id_ in id_list]

        # each rank loads only its shard
        rank = get_rank()
        world_size = get_world_size()
        
        # sharding strategy
        all_pairs   = list(zip(id_list, global_ids))
        if sharding_mode == "dataset":                       # legacy behaviour
            local_pairs = all_pairs[rank::world_size]
        else:                                             # "ddp": no manual shard
            local_pairs = all_pairs                        # full set on every rank

        datasets=[]
        
        if rank==0:
            pretty_print(f"Loading WQ Hanson Simulation datasets")
        
        variate_ids_2D, variate_ids_1D = self.get_variate_ids()

        param_file = self.cfg['param']
        param_df = pd.read_csv(osp.join(self.root_dir, param_file))

        dataset_class = None
        if self.task_name == "pretrain":
            dataset_class = LakeDataset
            is_test_fraction = False
        elif self.task_name == "evaluate" or self.task_name == "infer" or self.task_name == "plot_predictions":
            dataset_class = LakeEvalDataset
            is_test_fraction = True
        else:
            raise ValueError(f"WQHansonSimBuilder:: Invalid task name: {self.task_name}")

        for i, (raw_id, global_id) in enumerate(tqdm(local_pairs)):
            save_normalization_file = "WQHansonSim_"+str(raw_ids[i])
            if self.norm_path:
                normalization_stats_path = osp.join(self.norm_path, save_normalization_file)
            else: # ood sites
                normalization_stats_path = osp.join(f"{self.server_prefix}/lakefm/dev/norm_stats", "global_variable_stats.json")
                
            dataset = id_to_dataset_name[raw_id]
            filename = self.cfg[dataset]
            
            # load driver
            driver_df = pd.read_csv(osp.join(self.root_dir, filename+self.driver_suffix))
            cols = [self.date_col] + self.vars1d
            driver_df = driver_df[cols]
            
            # load lake
            lake_df = pd.read_csv(osp.join(self.root_dir, filename+self.lake_suffix))
            cols = [self.date_col, self.depth_col] + self.vars2d
            lake_df = lake_df[cols]
                
            ds = dataset_class(
                    lake_df=lake_df,
                    driver_df=driver_df,
                    param_df=param_df,
                    lake_id=global_id,
                    cfg=self.cfg,
                    variate_ids_2D=variate_ids_2D,
                    variate_ids_1D=variate_ids_1D,
                    id2var=self.id2var,
                    var_names_2D=self.vars2d,
                    var_names_1D=self.vars1d,
                    is_test_fraction=is_test_fraction,
                    normalization_stats_path=normalization_stats_path,
                    lakename=dataset,
                    run_name=self.run_name,
                    ckpt_name=self.ckpt_name,
                    regular_grid_forecasting=self.regular_grid_forecasting,
                    regular_grid_depths=self.regular_grid_depths,
                    regular_grid_max_depth=self.regular_grid_max_depth)
                    
            datasets.append(ds)
        print(f"Rank {rank}: Loaded {len(datasets)} datasets from WQHanson Simulations")
        return datasets

    def split_ids(self, lake_ids, lake_ids_format):
        return list(range(*lake_ids)) if lake_ids_format == "range" else lake_ids