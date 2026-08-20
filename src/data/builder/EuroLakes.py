import os.path as osp
import pandas as pd

from tqdm import tqdm
from data.dataset import LakeDataset
from data.eval_dataset_regular import LakeEvalDataset
from utils.exp_utils import pretty_print
from data.builder.base import BaseLakeBuilder
from torch.distributed import get_rank, get_world_size


class EuroLakesBuilder:
    """
    Builder for per-lake single CSV files containing both 2D lake variables and
    optional 1D driver variables.
    """

    def __init__(
        self,
        root_dir,
        lake_ids,
        lake_ids_format,
        base_builder: BaseLakeBuilder,
        **kwargs,
    ):
        self.root_dir = root_dir
        self.cfg = kwargs
        self.norm_path = self.cfg["norm_path"]
        self.lake_ids = lake_ids
        self.lake_ids_format = lake_ids_format
        self.date_col = self.cfg["date_col"]
        self.depth_col = self.cfg["depth_col"]
        self.base = base_builder
        self.var2id = self.base.var2id_key
        self.id2var = self.base.id2var_key
        self.vars2d = self.cfg["vars_2d"]
        self.vars1d = self.cfg["vars_1d"]
        self.file_suffix = self.cfg["file_suffix"]

        # For OOD sites, normalization is independently done for each context window.
        if self.base.norm_override:
            self.norm_path = None

    def split_ids(self, lake_ids, lake_ids_format):
        return list(range(*lake_ids)) if lake_ids_format == "range" else lake_ids

    def get_variate_ids(self, vars2d, vars1d):
        variate_ids_2d = [self.var2id[c] for c in vars2d if c in self.var2id]
        variate_ids_1d = [self.var2id[c] for c in vars1d if c in self.var2id]
        return variate_ids_2d, variate_ids_1d

    def load_dataset(self, prefix, rank=0, world_size=1, sharding_mode="ddp", root_cfg=None):
        self.task_name = root_cfg.task_name
        self.server_prefix = root_cfg.server_prefix
        self.run_name = root_cfg.run_name
        self.ckpt_name = root_cfg.evaluator.ckpt_name
        self.regular_grid_forecasting = root_cfg.regular_grid_forecasting
        self.regular_grid_depths = root_cfg.regular_grid_depths
        self.regular_grid_max_depth = root_cfg.regular_grid_max_depth

        if rank == 0:
            pretty_print("EuroLakesBuilder initialized")

        raw_ids = self.split_ids(self.lake_ids, self.lake_ids_format)  # list of lake names or 1-based ints
        if raw_ids and isinstance(raw_ids[0], int):
            lake_keys = [self.cfg["lake_names"][i - 1] for i in raw_ids]
        else:
            lake_keys = raw_ids
        id_to_dataset_name = {idx + 1: name for idx, name in enumerate(lake_keys)}
        id_list = list(id_to_dataset_name.keys())
        global_ids = [prefix + id_ for id_ in id_list]

        rank = get_rank()
        world_size = get_world_size()

        all_pairs = list(zip(id_list, global_ids))
        if sharding_mode == "dataset":
            local_pairs = all_pairs[rank::world_size]
        else:
            local_pairs = all_pairs

        if self.task_name == "pretrain":
            dataset_class = LakeDataset
        elif self.task_name in {"evaluate", "infer", "plot_predictions"}:
            dataset_class = LakeEvalDataset
        else:
            raise ValueError(f"EuroLakesBuilder:: Invalid task name: {self.task_name}")

        datasets = []
        if rank == 0:
            pretty_print("Loading EuroLakes CSV datasets")

        for i, (raw_id, global_id) in enumerate(tqdm(local_pairs)):
            save_normalization_file = "EuroLakes_" + str(raw_ids[i])
            if self.norm_path:
                normalization_stats_path = osp.join(self.norm_path, save_normalization_file)
            else:
                normalization_stats_path = osp.join(
                    f"{self.server_prefix}/lakefm/dev/norm_stats", "global_variable_stats.json"
                )
                
            dataset = id_to_dataset_name[raw_id]
            filename = self.cfg[dataset]
            csv_path = osp.join(self.root_dir, filename + self.file_suffix)
            df = pd.read_csv(csv_path)

            df_cols = df.columns.tolist()
            var2d_cols = [c for c in self.vars2d if c in df_cols and c in self.var2id]
            var1d_cols = [c for c in self.vars1d if c in df_cols and c in self.var2id]

            lake_cols = [self.date_col, self.depth_col] + var2d_cols
            lake_df = df[lake_cols].copy()

            # Build 1D driver frame from same CSV by collapsing duplicate depth rows per date.
            driver_cols = [self.date_col] + var1d_cols
            if var1d_cols:
                driver_df = (
                    df[driver_cols]
                    .sort_values(self.date_col)
                    .groupby(self.date_col, as_index=False)
                    .first()
                )
            else:
                # Keep date signal even if no explicit 1D variables are provided.
                driver_df = (
                    df[[self.date_col]]
                    .sort_values(self.date_col)
                    .drop_duplicates(subset=[self.date_col])
                    .reset_index(drop=True)
                )

            variate_ids_2d, variate_ids_1d = self.get_variate_ids(
                vars2d=var2d_cols, vars1d=var1d_cols
            )

            ds = dataset_class(
                lake_df=lake_df,
                driver_df=driver_df,
                param_df=None,
                lake_id=global_id,
                cfg=self.cfg,
                variate_ids_2D=variate_ids_2d,
                variate_ids_1D=variate_ids_1d,
                id2var=self.id2var,
                var_names_2D=var2d_cols,
                var_names_1D=var1d_cols,
                lakename=dataset,
                normalization_stats_path=normalization_stats_path,
                run_name=self.run_name,
                ckpt_name=self.ckpt_name,
                regular_grid_forecasting=self.regular_grid_forecasting,
                regular_grid_depths=self.regular_grid_depths,
                regular_grid_max_depth=self.regular_grid_max_depth,
            )
            datasets.append(ds)

        print(f"Rank {rank}: Loaded {len(datasets)} datasets from EuroLakes CSV")
        return datasets
