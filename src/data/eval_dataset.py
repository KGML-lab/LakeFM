import pandas as pd
import torch
import numpy as np
import json
import os
from torch.utils.data import Dataset
from utils.util import Normalizer
from utils.exp_utils import pretty_print
from omegaconf import ListConfig

def var2id_helper(mask_variable, var2id):
    if mask_variable:
        if isinstance(mask_variable, ListConfig):
            mask_variable = list(mask_variable)
        if isinstance(mask_variable, (list, tuple, np.ndarray)):
            mask_variable_list = []
            for v in mask_variable:
                if isinstance(v, str) and v in var2id:
                    mask_variable_list.append(var2id[v])
                elif isinstance(v, (int, np.integer)):
                    mask_variable_list.append(int(v))
            mask_variable = mask_variable_list if mask_variable_list else None
        elif isinstance(mask_variable, str) and mask_variable in var2id:
            mask_variable = var2id[mask_variable]
        else:
            mask_variable = mask_variable
    else:
        mask_variable = None

    return mask_variable

class LakeEvalDataset(Dataset):
    
    def __init__(self, 
                 lake_df,
                 driver_df, 
                 param_df=None,
                 lake_id=None,
                 cfg=None,
                 variate_ids_2D=None,
                 variate_ids_1D=None,
                 id2var=None,
                 var_names_2D=None,
                 var_names_1D=None,
                 timeenc=0,
                 lakename=None,
                 coverage_threshold=None,
                 is_test_fraction=False,
                 normalization_stats_path=None,
                 create_global_stats=True,
                 regular_grid_forecasting=False,
                 regular_grid_depths=20,
                 regular_grid_max_depth=None,
                 run_name=None,
                 ckpt_name=None):

        self.lake_df = lake_df
        self.driver_df = driver_df
        self.param_df = param_df
        self.lake_id = lake_id
        self.variate_ids_2D = variate_ids_2D
        self.variate_ids_1D = variate_ids_1D
        self.id2var = id2var
        self.var2id = {v: k for k, v in self.id2var.items()} if self.id2var else {}
        self.var_names_2D = var_names_2D or []
        self.var_names_1D = var_names_1D or []
        self.id_str = f"lake_{lake_id}" if lake_id is not None else "lake"
        self.lake_name = lakename
        self.coverage_threshold = coverage_threshold
        self.is_test_fraction = is_test_fraction
        self.normalization_stats_path = normalization_stats_path
        self.create_global_stats = create_global_stats
        
        # Regular grid forecasting configuration
        self.regular_grid_forecasting = regular_grid_forecasting
        self.regular_grid_depths = regular_grid_depths
        self.regular_grid_max_depth = regular_grid_max_depth
        self.run_name = run_name
        self.timeenc = timeenc
        self.ckpt_name = ckpt_name
        self.cfg = cfg

        self.train_frac = self.cfg['train_fraction']
        self.val_frac = self.cfg['val_fraction']
        self.test_frac = self.cfg['test_fraction']

        self.context_len = self.cfg['context_len']
        self.prediction_len = self.cfg['prediction_len']

        self.context_window_range = tuple(self.cfg.get('context_window_range', (self.context_len, self.context_len)))
        self.pred_window_range = tuple(self.cfg.get('pred_window_range', (self.prediction_len, self.prediction_len)))
        # Current effective lengths (may change per-batch when dynamic_windows=True)
        self.current_context_len = self.context_len
        self.current_prediction_len = self.prediction_len

        self.freq = self.cfg['frequency']
        self.date_col = self.cfg['date_col']
        self.lake_id_col = self.cfg['lake_id_col']
        self.depth_col = self.cfg['depth_col']
        self.feats_to_drop = self.cfg['feature_cols_to_drop']
        
        self.type_map = {'train': 0, 'val': 1, 'test': 2}

        # Evaluation time grid selection:
        # - "regular": build a daily grid from min..max date in the split (current default behavior)
        # - "irregular": use observed timestamps directly (self.unique_datetimes), like training
        self.eval_time_grid = str(self.cfg.get("eval_time_grid", "regular")).lower()
        if self.eval_time_grid not in {"regular", "irregular"}:
            print(f"Warning: unknown eval_time_grid={self.eval_time_grid!r}; falling back to 'regular'")
            self.eval_time_grid = "regular"
        
        # accept single variable id or list of ids; None means no variable masking
        self.mask_variable = self.cfg.get('mask_variable', None)
        self.mask_depth = self.cfg.get('mask_depth', None)
        if isinstance(self.mask_variable, ListConfig):
            self.mask_variable = list(self.mask_variable)
        # Normalize mask_depth to allow either a single depth or a list of depths
        if isinstance(self.mask_depth, ListConfig):
            self.mask_depth = list(self.mask_depth)
        if isinstance(self.mask_depth, str) and self.mask_depth.strip().lower() in {"null", "none", ""}:
            self.mask_depth = None
        self.mask_variable = var2id_helper(self.mask_variable, self.var2id)
        if isinstance(self.mask_depth, (list, tuple, np.ndarray)):
            self.mask_depth = [float(d) for d in self.mask_depth]
        elif self.mask_depth is not None:
            self.mask_depth = float(self.mask_depth)

        if isinstance(self.mask_variable, (list, tuple, np.ndarray)):
            self.mask_variable_idx = [int(v) for v in self.mask_variable]
        elif self.mask_variable is not None:
            self.mask_variable_idx = [int(self.mask_variable)]
        else:
            self.mask_variable_idx = None

        # When True and mask_variable provided: mask that variable across all depths
        self.mask_var_across_depths = bool(self.cfg.get('mask_var_across_depths', False))
        # When True and mask_depth provided: mask that depth for all variables
        self.mask_depth_for_all_vars = bool(self.cfg.get('mask_depth_for_all_vars', False))

        self.mean = None
        self.std = None
        self.valid_idx = None

        self.__process_data__()
        self.__split__()
    
    def _apply_context_mask_tokens(self,
                                   driver_data_x,
                                   driver_var_ids_x,
                                   driver_depth_vals_x,
                                   driver_time_ids_x,
                                   driver_time_values_x,
                                   driver_datetime_strs_x,
                                   lake_data_x,
                                   lake_var_ids_x,
                                   lake_depth_vals_x,
                                   lake_time_ids_x,
                                   lake_time_values_x,
                                   lake_datetime_strs_x):
        if self.mask_variable_idx is None and self.mask_depth is None:
            return (driver_data_x,
                    driver_var_ids_x,
                    driver_depth_vals_x,
                    driver_time_ids_x,
                    driver_time_values_x,
                    driver_datetime_strs_x,
                    lake_data_x,
                    lake_var_ids_x,
                    lake_depth_vals_x,
                    lake_time_ids_x,
                    lake_time_values_x,
                    lake_datetime_strs_x)

        num_lake = lake_data_x.shape[0] if isinstance(lake_data_x, torch.Tensor) else 0
        if num_lake == 0:
            return (driver_data_x,
                    driver_var_ids_x,
                    driver_depth_vals_x,
                    driver_time_ids_x,
                    driver_time_values_x,
                    driver_datetime_strs_x,
                    lake_data_x,
                    lake_var_ids_x,
                    lake_depth_vals_x,
                    lake_time_ids_x,
                    lake_time_values_x,
                    lake_datetime_strs_x)

        mask = torch.zeros(num_lake, dtype=torch.bool)

        if self.mask_variable_idx is not None:
            var_list = torch.tensor(self.mask_variable_idx, dtype=lake_var_ids_x.dtype, device=lake_var_ids_x.device)
            is_var = torch.isin(lake_var_ids_x, var_list)
        else:
            is_var = torch.zeros_like(mask)

        is_depth = torch.zeros_like(mask)
        if self.mask_depth is not None:
            depth_vals_for_mask = lake_depth_vals_x
            if self.max_depth > self.min_depth:
                depth_vals_for_mask = lake_depth_vals_x * (self.max_depth - self.min_depth) + self.min_depth
            if isinstance(self.mask_depth, (list, tuple, np.ndarray)):
                depth_list = torch.tensor(self.mask_depth,
                                          dtype=depth_vals_for_mask.dtype,
                                          device=depth_vals_for_mask.device)
                is_depth = torch.isclose(depth_vals_for_mask.unsqueeze(1), depth_list.unsqueeze(0)).any(dim=1)
            else:
                is_depth = torch.isclose(depth_vals_for_mask, torch.tensor(float(self.mask_depth),
                                                                           dtype=depth_vals_for_mask.dtype,
                                                                           device=depth_vals_for_mask.device))
        # Compose scenarios
        # 1) Mask variable(s) across all depths
        if self.mask_variable_idx is not None and self.mask_var_across_depths:
            mask = mask | is_var
        # 2) Mask depth across all variables
        if self.mask_depth is not None and self.mask_depth_for_all_vars:
            mask = mask | is_depth
        # 3) Variable(s) at a given depth (cross-variable coupling at a depth)
        if self.mask_variable_idx is not None and self.mask_depth is not None and not self.mask_var_across_depths and not self.mask_depth_for_all_vars:
            mask = mask | (is_var & is_depth)
        # If only variable provided without flags, default to variable at all depths
        if self.mask_variable_idx is not None and self.mask_depth is None and not self.mask_var_across_depths:
            mask = mask | is_var
        # If only depth provided without flag, default to depth across all variables
        if self.mask_variable_idx is None and self.mask_depth is not None and not self.mask_depth_for_all_vars:
            mask = mask | is_depth
        # If nothing to mask, return as-is
        if not torch.any(mask):
            return (driver_data_x,
                    driver_var_ids_x,
                    driver_depth_vals_x,
                    driver_time_ids_x,
                    driver_time_values_x,
                    driver_datetime_strs_x,
                    lake_data_x,
                    lake_var_ids_x,
                    lake_depth_vals_x,
                    lake_time_ids_x,
                    lake_time_values_x,
                    lake_datetime_strs_x)

        lake_data_x = lake_data_x.clone()
        lake_data_x[mask] = float('nan')

        return (driver_data_x,
                driver_var_ids_x,
                driver_depth_vals_x,
                driver_time_ids_x,
                driver_time_values_x,
                driver_datetime_strs_x,
                lake_data_x,
                lake_var_ids_x,
                lake_depth_vals_x,
                lake_time_ids_x,
                lake_time_values_x,
                lake_datetime_strs_x)
    
    def _create_datetime_var_depth_map(self):
        """
        Create a mapping from datetime to available (variable, depth) combinations.
        Returns dict: {datetime: [(var1, depth1, row_idx), (var1, depth2, row_idx), ...]}
        """
        datetime_map = {}
        
        for idx, row in self.split_df_lake.iterrows():
            datetime = row[self.date_col]
            depth = row[self.depth_col]
            
            if datetime not in datetime_map:
                datetime_map[datetime] = []
            
            for i, var_idx in enumerate(self.variate_ids_2D):
                var_name = self.var_names_2D[i] if i < len(self.var_names_2D) else None
                if var_name and not pd.isna(row[var_name]):
                    datetime_map[datetime].append((var_idx, depth, idx))
        
        return datetime_map
    
    def _create_regular_date_grid(self):
        """
        Create a regular date grid from min to max datetime in the current split.
        This is used for evaluation/testing to ensure consistent temporal coverage.
        
        For training, we use irregular sampling (self.unique_datetimes).
        For eval/test, we use this regular grid (self.regular_date_grid).
        """
        if len(self.unique_datetimes) == 0:
            self.regular_date_grid = []
            print(f"No unique datetimes found ")
            return
        
        min_date = pd.to_datetime(self.unique_datetimes[0])
        max_date = pd.to_datetime(self.unique_datetimes[-1])
        
        # Create daily date range from min to max
        date_range = pd.date_range(start=min_date, end=max_date, freq='D')
        
        self.regular_date_grid = [d.strftime('%Y-%m-%d') for d in date_range]
        
        print(f"Created regular date grid for {self.lake_name or self.lake_id}: "
              f"{len(self.regular_date_grid)} days from {self.regular_date_grid[0]} to {self.regular_date_grid[-1]}")
    
    def _generate_regular_grid_queries(self, prediction_date_range, base_dt):
        if not self.regular_grid_forecasting:
            return None
            
        # Get available variables
        available_vars = list(self.variate_ids_2D) if self.variate_ids_2D else []
        if not available_vars:
            return None
            
        # datetime_var_depth_map[dt] is a list of tuples: (var_id, depth, row_idx)
        depths_set = set()
        for dt in prediction_date_range:
            for _, depth, _ in self.datetime_var_depth_map.get(dt, []):
                if not pd.isna(depth):
                    depths_set.add(round(float(depth), 3))

        if not depths_set:
            return None

        max_depth = self.regular_grid_max_depth if self.regular_grid_max_depth is not None else max(depths_set)

        regular_depths = np.array(sorted([d for d in depths_set if d <= max_depth]), dtype=np.float32)

        if self.regular_grid_depths is not None and len(regular_depths) > self.regular_grid_depths:
            idx = np.linspace(0, len(regular_depths) - 1, self.regular_grid_depths).round().astype(int)
            regular_depths = regular_depths[idx]
        
        T = len(prediction_date_range)
        D = len(regular_depths)
        V = len(available_vars)
        
        # Generate all combinations
        query_times = []
        query_depths = []
        query_vars = []
        query_lake_ids = []
        query_time_ids = []
        query_time_values = []
        query_datetime_strs = []
        
        for t_idx, dt_str in enumerate(prediction_date_range):
            dt_obj = pd.to_datetime(dt_str)
            time_id = (dt_obj - base_dt).days if base_dt else t_idx
            time_value = (dt_obj - pd.Timestamp('2020-01-01')).days / 365.25
            
            for d_idx, depth in enumerate(regular_depths):
                for v_idx, var_id in enumerate(available_vars):
                    query_times.append(t_idx)
                    query_depths.append(depth)
                    query_vars.append(var_id)
                    query_lake_ids.append(self.lake_id)
                    query_time_ids.append(time_id)
                    query_time_values.append(time_value)
                    query_datetime_strs.append(dt_str)
        
        query_dict = {
            'time_values': torch.tensor(query_times, dtype=torch.float32),
            'depth_values': torch.tensor(query_depths, dtype=torch.float32),
            'var_ids': torch.tensor(query_vars, dtype=torch.long),
            'lake_ids': torch.tensor(query_lake_ids, dtype=torch.long),
            'time_ids': torch.tensor(query_time_ids, dtype=torch.long),
            'time_values_norm': torch.tensor(query_time_values, dtype=torch.float32),
            'datetime_strs': query_datetime_strs,
            'datetime_vals': [pd.to_datetime(dt) for dt in prediction_date_range],
            'depth_vals': regular_depths,
            'shape': (T, D, V)
        }
        
        return query_dict
    
    def _build_valid_indices_for_eval(self, date_grid=None):
        """
        Build valid indices for evaluation where each index in the selected date grid
        has sufficient token-level coverage using observed (t, v, d) tokens:
        - Context window: >= 3 observed tokens
        - Prediction window: >= 1 observed token
        
        This ensures we only evaluate on windows with meaningful data.
        """
        grid = date_grid if date_grid is not None else getattr(self, 'regular_date_grid', [])
        if grid is None or len(grid) == 0:
            self.valid_idx = []
            print(f"No date grid found for evaluation")
            return
        
        valid_indices = []
        min_context_tokens = 3
        min_pred_tokens = 1
        
        context_len = self.current_context_len
        pred_len = self.current_prediction_len
        
        dt_map = getattr(self, 'datetime_var_depth_map', {})
        
        for idx in range(len(grid)):
            if idx + context_len + pred_len > len(grid):
                break
            
            context_dates = grid[idx:idx + context_len]
            pred_dates = grid[idx + context_len:idx + context_len + pred_len]
            
            context_token_count = sum(len(dt_map[d]) for d in context_dates if d in dt_map)
            pred_token_count = sum(len(dt_map[d]) for d in pred_dates if d in dt_map)
            
            if context_token_count >= min_context_tokens and pred_token_count >= min_pred_tokens:
                valid_indices.append(idx)
        
        self.valid_idx = valid_indices
        
        print(f"Built {len(self.valid_idx)} valid evaluation indices out of "
              f"{len(grid) - context_len - pred_len + 1} possible windows "
              f"(token thresholds: ctx>={min_context_tokens}, pred>={min_pred_tokens})")
    
    def _save_valid_indices(self, flag):
        """
        Save valid indices to a JSON file for reproducibility across baselines.
        
        Args:
            flag: 'train', 'val', or 'test' to indicate which split
        """
        if not hasattr(self, 'valid_idx') or self.valid_idx is None:
            return
        
        stats_path = os.path.dirname(self.normalization_stats_path)
        stats_path = os.path.join(stats_path, self.ckpt_name)
        stats_dir = os.path.join(stats_path, f"valid_indices/{self.lake_name or self.lake_id}")
                
        os.makedirs(stats_dir, exist_ok=True)
        
        valid_idx_path = os.path.join(stats_dir, f"valid_indices_{flag}.json")

        if getattr(self, "eval_time_grid", "regular") == "regular" and getattr(self, "regular_date_grid", None):
            grid = self.regular_date_grid
            grid_type = "regular"
        else:
            grid = getattr(self, "unique_datetimes", [])
            grid_type = "irregular"
        
        valid_idx_data = {
            'lake_id': self.lake_id,
            'lake_name': self.lake_name,
            'split': flag,
            'context_len': self.current_context_len,
            'prediction_len': self.current_prediction_len,
            'num_valid_indices': len(self.valid_idx),
            'valid_indices': self.valid_idx,
            'eval_time_grid': grid_type,
            'date_grid_start': grid[0] if grid else None,
            'date_grid_end': grid[-1] if grid else None,
            'num_grid_dates': len(grid),
            'regular_date_grid_start': self.regular_date_grid[0] if getattr(self, "regular_date_grid", None) else None,
            'regular_date_grid_end': self.regular_date_grid[-1] if getattr(self, "regular_date_grid", None) else None,
            'num_regular_grid_dates': len(self.regular_date_grid) if getattr(self, "regular_date_grid", None) else 0
        }
        
        with open(valid_idx_path, 'w') as f:
            json.dump(valid_idx_data, f, indent=2)
        
        print(f"Saved valid indices to {valid_idx_path}")
    
    def _extract_irregular_data(self, datetimes, base_dt=None):
        """
        Extract lake data for given datetimes with irregular depth structure.
        """
        all_values = []
        all_var_ids = []
        all_depth_vals = []
        all_time_ids = []
        all_time_values = []
        
        var_depth_combinations = set()
        for dt in datetimes:
            if dt in self.datetime_var_depth_map:
                for v_idx, depth, row_idx in self.datetime_var_depth_map[dt]:
                    var_depth_combinations.add((v_idx, depth))
        
        # Sort combinations for consistent ordering: v1_d1, v1_d2, v1_d3, v2_d1, v2_d2, ...
        var_depth_combinations = sorted(var_depth_combinations, key=lambda x: (x[0], x[1]))
        
        # Base datetime for relative time id computation
        if base_dt is None:
            base_dt = pd.to_datetime(datetimes[0]) if len(datetimes) > 0 else None
        else:
            base_dt = pd.to_datetime(base_dt)

        for v_idx, depth in var_depth_combinations:
            for time_idx, dt in enumerate(datetimes):
                if dt in self.datetime_var_depth_map:
                    for var_idx, d, row_idx in self.datetime_var_depth_map[dt]:
                        if var_idx == v_idx and d == depth:
                            if row_idx in self.split_df_lake.index:
                                var_idx_in_list = self.variate_ids_2D.index(v_idx) if v_idx in self.variate_ids_2D else -1
                                if var_idx_in_list >= 0 and var_idx_in_list < len(self.var_names_2D):
                                    var_name = self.var_names_2D[var_idx_in_list]
                                    value = self.split_df_lake.loc[row_idx, var_name]
                                else:
                                    value = np.nan
                                
                                if not np.isnan(value):
                                    all_values.append(value)
                                    all_var_ids.append(v_idx)
                                    all_depth_vals.append(depth)
                                    # time_ids are days since first datetime in the window
                                    if base_dt is not None:
                                        rel_days = (pd.to_datetime(dt) - base_dt).days
                                    else:
                                        rel_days = 0
                                    all_time_ids.append(rel_days)
                                    all_time_values.append(dt)
                            break
                        
        time_values_numeric = []
        for dt_str in all_time_values:
            dt = pd.to_datetime(dt_str)
            day_of_year = dt.dayofyear - 1
            normalized_day_of_year = day_of_year / 365.25
            time_values_numeric.append(normalized_day_of_year)
        
        return (torch.tensor(all_values, dtype=torch.float32),
                torch.tensor(all_var_ids, dtype=torch.long), 
                torch.tensor(all_depth_vals, dtype=torch.float32),
                torch.tensor(all_time_ids, dtype=torch.long),
                torch.tensor(time_values_numeric, dtype=torch.float32),
                all_time_values)
    
    def _extract_driver_as_surface_variables(self, driver_data, datetimes, base_dt=None):
        """
        Extract driver variables as surface (depth=0) variables
        """
        all_values = []
        all_var_ids = []
        all_depth_vals = []
        all_time_ids = []
        all_time_values = []
        # Base datetime for relative time id computation
        if base_dt is None:
            base_dt = pd.to_datetime(datetimes[0]) if len(datetimes) > 0 else None
        else:
            base_dt = pd.to_datetime(base_dt)

        for var_idx, var_id in enumerate(self.variate_ids_1D):
            for time_idx, dt in enumerate(datetimes):
                if time_idx < driver_data.shape[0] and var_idx < driver_data.shape[1]:
                    value = driver_data[time_idx, var_idx].item()
                    
                    if not np.isnan(value):
                        all_values.append(value)
                        all_var_ids.append(var_id)
                        all_depth_vals.append(0.0)
                        # time_ids are days since first datetime in the window
                        if base_dt is not None:
                            rel_days = (pd.to_datetime(dt) - base_dt).days
                        else:
                            rel_days = 0
                        all_time_ids.append(rel_days)
                        all_time_values.append(dt)  # Use actual datetime

        time_values_numeric = []
        for dt_str in all_time_values:
            dt = pd.to_datetime(dt_str)
            day_of_year = dt.dayofyear - 1
            normalized_day_of_year = day_of_year / 365.25
            time_values_numeric.append(normalized_day_of_year)
              
        return (torch.tensor(all_values, dtype=torch.float32),
                torch.tensor(all_var_ids, dtype=torch.long), 
                torch.tensor(all_depth_vals, dtype=torch.float32),
                torch.tensor(all_time_ids, dtype=torch.long),
                torch.tensor(time_values_numeric, dtype=torch.float32),
                all_time_values)

    def __split__(self, flag='test'):
        self.set_type = self.type_map[flag]

        '''
        Split data
        '''
        self.border1_DR = self.border1s_DR[self.set_type]
        self.border2_DR = self.border2s_DR[self.set_type]
        
        self.border1_DF = self.border1s_DF[self.set_type]
        self.border2_DF = self.border2s_DF[self.set_type]

        df_stamp = self.df_ts[self.border1_DR:self.border2_DR]
        
        '''
        Time features generation
        '''
        if self.timeenc == 1:
            df_stamp_dt = df_stamp.copy()
            df_stamp_dt[self.date_col] = pd.to_datetime(df_stamp_dt[self.date_col])
            df_stamp_dt['month'] = df_stamp_dt[self.date_col].apply(lambda row: row.month, 1)
            df_stamp_dt['day'] = df_stamp_dt[self.date_col].apply(lambda row: row.day, 1)
            df_stamp_dt['weekday'] = df_stamp_dt[self.date_col].apply(lambda row: row.weekday(), 1)
            df_stamp_dt['hour'] = df_stamp_dt[self.date_col].apply(lambda row: row.hour, 1)
            data_stamp = df_stamp_dt.drop([self.date_col], 1).values
        else:
            data_stamp = df_stamp.values

        driver_cols = [col for col in self.df_driver.columns if col != self.date_col]
        lake_cols = [col for col in self.df_lake.columns if col not in [self.date_col, self.depth_col]]
        
        self.data_DR = self.df_driver[driver_cols][self.border1_DR:self.border2_DR].values
        self.split_df_lake = self.df_lake.iloc[self.border1_DF:self.border2_DF].copy()
        self.data_DF = self.split_df_lake[lake_cols]
        
        self.data_stamp = data_stamp
        unique_dates = self.split_df_lake[self.date_col].unique()
        self.unique_datetimes = sorted(unique_dates)
        self.datetime_var_depth_map = self._create_datetime_var_depth_map()
        
        if flag in ['test']:
            if getattr(self, "eval_time_grid", "regular") == "irregular":
                self.regular_date_grid = []
                date_grid = self.unique_datetimes
            else:
                self._create_regular_date_grid()
                date_grid = self.regular_date_grid

            self._build_valid_indices_for_eval(date_grid=date_grid)
            self._save_valid_indices(flag)

    def _standardize_date_columns(self):
        if self.date_col in self.lake_df.columns:
            self.lake_df[self.date_col] = pd.to_datetime(self.lake_df[self.date_col]).dt.date.astype(str)
        
        if self.date_col in self.driver_df.columns:
            self.driver_df[self.date_col] = pd.to_datetime(self.driver_df[self.date_col]).dt.date.astype(str)
        
        if self.param_df is not None and self.date_col in self.param_df.columns:
            self.param_df[self.date_col] = pd.to_datetime(self.param_df[self.date_col]).dt.date.astype(str)

    def __process_data__(self):
        
        self._standardize_date_columns()
        
        self.norm = Normalizer(
            lake_id=self.lake_id, 
            lake_name=self.lake_name,
            id2var=self.id2var,
            variate_ids_2D=self.variate_ids_2D,
            variate_ids_1D=self.variate_ids_1D,
            run_name=self.run_name,
            ckpt_name=self.ckpt_name
        )

        if self.param_df is not None:
            self.param_df = self.features_processing(self.param_df)

        self.depth_values = sorted(self.lake_df[self.depth_col].unique())
        self.num_unique_depths = len(self.depth_values)
        self.max_depth = max(self.depth_values) if len(self.depth_values) > 0 else 1.0
        self.min_depth = min(self.depth_values) if len(self.depth_values) > 0 else 0.0

        if self.max_depth > self.min_depth:
            self.lake_df[self.depth_col] = (
                (self.lake_df[self.depth_col] - self.min_depth) / (self.max_depth - self.min_depth)
            )
        
        unique_dates = self.lake_df[self.date_col].unique()
        unique_datetimes = sorted(unique_dates)
        
        df_lake=self.features_processing(self.lake_df)
        
        self.df_lake_with_datetime = self.lake_df.copy()
        self.df_ts=self.driver_df[[self.date_col]].copy()
        
        df_driver=self.features_processing(self.driver_df)
        
        num_unique_datetimes = len(unique_datetimes)
        num_train_datetimes = int(num_unique_datetimes * self.train_frac)
        num_test_datetimes = int(num_unique_datetimes * self.test_frac)
        num_val_datetimes = num_unique_datetimes - num_train_datetimes - num_test_datetimes
        
        train_end_date = unique_datetimes[num_train_datetimes - 1]
        val_end_date = unique_datetimes[num_train_datetimes + num_val_datetimes - 1]
        
        num_train_DF = len(self.df_lake_with_datetime[self.df_lake_with_datetime[self.date_col] <= train_end_date])
        num_val_DF = len(self.df_lake_with_datetime[
            (self.df_lake_with_datetime[self.date_col] > train_end_date) & 
            (self.df_lake_with_datetime[self.date_col] <= val_end_date)
        ])
        
        num_train_DR = len(self.df_ts[self.df_ts[self.date_col] <= train_end_date])
        num_val_DR = len(self.df_ts[
            (self.df_ts[self.date_col] > train_end_date) & 
            (self.df_ts[self.date_col] <= val_end_date)
        ])
        num_test_DR = len(self.df_ts[self.df_ts[self.date_col] > val_end_date])
        num_vali_DR = num_val_DR  # Keep consistent naming
        
        self.border1s_DR = [0, num_train_DR, len(self.df_ts) - num_test_DR]
        self.border2s_DR = [num_train_DR, num_train_DR + num_vali_DR, len(self.df_ts)]
        
        self.border1s_DF = [0, num_train_DF, len(df_lake) - len(self.df_lake_with_datetime[self.df_lake_with_datetime[self.date_col] > val_end_date])]
        self.border2s_DF = [num_train_DF, num_train_DF + num_val_DF, len(df_lake)]
        
        print(f"\n\n normalization_stats_path: {self.normalization_stats_path}\n\n")
        if not self.normalization_stats_path:
            raise RuntimeError(
                "No normalization_stats_path provided. Evaluation requires either per-lake "
                "normalization stats or global_variable_stats.json."
            )
        is_global_stats = os.path.basename(self.normalization_stats_path) == "global_variable_stats.json"

        try:
            if is_global_stats:
                self.norm.apply_global_normalization(self.normalization_stats_path)
                print("Applied global per-variable normalization for evaluation")
            else:
                self.norm.apply_normalization_from_stats(self.normalization_stats_path)
                print("Applied per-lake normalization for evaluation")
        except FileNotFoundError as e:
            raise RuntimeError(
                f"normalization stats not found. "
            ) from e
        
        df_driver_scaled, df_lake_scaled = self.norm.transform_data(df_driver, df_lake)
        
        # Convert back to DataFrames and add back metadata columns
        self.df_driver = pd.DataFrame(df_driver_scaled, columns=df_driver.columns)
        self.df_driver[self.date_col] = self.driver_df[self.date_col].values
        
        self.df_lake = pd.DataFrame(df_lake_scaled, columns=df_lake.columns)
        self.df_lake[self.date_col] = self.lake_df[self.date_col].values
        self.df_lake[self.depth_col] = self.lake_df[self.depth_col].values

    def __getitem__(self, idx):
        if self.valid_idx is not None:
            idx = self.valid_idx[idx]
            
        # Choose the date grid based on eval_time_grid (default: regular grid for test)
        if getattr(self, "eval_time_grid", "regular") == "regular" and getattr(self, "regular_date_grid", None):
            date_grid = self.regular_date_grid
        else:
            date_grid = self.unique_datetimes

        # idx is position in date_grid
        s_begin_grid = idx
        s_end_grid = s_begin_grid + self.current_context_len
        r_begin_grid = s_end_grid
        r_end_grid = r_begin_grid + self.current_prediction_len
        
        # date ranges from selected grid
        context_date_range = date_grid[s_begin_grid:s_end_grid]
        prediction_date_range = date_grid[r_begin_grid:r_end_grid]
        
        context_datetimes = [d for d in context_date_range if d in self.datetime_var_depth_map]
        prediction_datetimes = [d for d in prediction_date_range if d in self.datetime_var_depth_map]
            
        split_driver_dates = self.df_ts[self.date_col].values[self.border1_DR:self.border2_DR]
        driver_date_to_idx = {date: i for i, date in enumerate(split_driver_dates)}
        
        context_driver_indices = [driver_date_to_idx.get(d) for d in context_date_range if d in driver_date_to_idx]
        pred_driver_indices = [driver_date_to_idx.get(d) for d in prediction_date_range if d in driver_date_to_idx]
        
        if len(context_driver_indices) > 0:
            driver1D_x = torch.tensor(self.data_DR[context_driver_indices])
        else:
            driver1D_x = torch.zeros((len(context_date_range), self.data_DR.shape[1]))
        
        if len(pred_driver_indices) > 0:
            driver1D_y = torch.tensor(self.data_DR[pred_driver_indices])
        else:
            driver1D_y = torch.zeros((len(prediction_date_range), self.data_DR.shape[1]))
        
        
        base_dt_ctx = pd.to_datetime(context_date_range[0]) if len(context_date_range) > 0 else None
        lake_data_x, lake_var_ids_x, lake_depth_vals_x, lake_time_ids_x, lake_time_values_x, lake_datetime_strs_x = self._extract_irregular_data(
            context_datetimes, base_dt=base_dt_ctx
        )
        lake_data_y, lake_var_ids_y, lake_depth_vals_y, lake_time_ids_y, lake_time_values_y, lake_datetime_strs_y = self._extract_irregular_data(
            prediction_datetimes, base_dt=base_dt_ctx
        )
        
        regular_grid_queries = None
        if self.regular_grid_forecasting:
            regular_grid_queries = self._generate_regular_grid_queries(
                prediction_date_range, base_dt=base_dt_ctx
            )   

        if len(context_driver_indices) > 0:
            date_values_x = self.data_stamp[context_driver_indices]
        else:
            date_values_x = np.array([])
        
        if len(pred_driver_indices) > 0:
            date_values_y = self.data_stamp[pred_driver_indices]
        else:
            date_values_y = np.array([])

        if self.param_df is not None:
            simulation_params = torch.tensor(self.param_df.values)
        else:
            simulation_params = None

        driver_data_x, driver_var_ids_x, driver_depth_vals_x, driver_time_ids_x, driver_time_values_x, driver_datetime_strs_x = self._extract_driver_as_surface_variables(
            driver1D_x, context_datetimes, base_dt=base_dt_ctx
        )
        (driver_data_x,
         driver_var_ids_x,
         driver_depth_vals_x,
         driver_time_ids_x,
         driver_time_values_x,
         driver_datetime_strs_x,
         lake_data_x,
         lake_var_ids_x,
         lake_depth_vals_x,
         lake_time_ids_x,
         lake_time_values_x,
         lake_datetime_strs_x) = self._apply_context_mask_tokens(
            driver_data_x,
            driver_var_ids_x,
            driver_depth_vals_x,
            driver_time_ids_x,
            driver_time_values_x,
            driver_datetime_strs_x,
            lake_data_x,
            lake_var_ids_x,
            lake_depth_vals_x,
            lake_time_ids_x,
            lake_time_values_x,
            lake_datetime_strs_x
        )
        flat_seq_x = torch.cat([driver_data_x, lake_data_x]) if driver_data_x.numel() > 0 else lake_data_x
        
        flat_seq_y = lake_data_y
        
        if driver_var_ids_x.numel() > 0:
            var_ids_x = torch.cat([driver_var_ids_x, lake_var_ids_x])
            depth_vals_x = torch.cat([driver_depth_vals_x, lake_depth_vals_x])
            time_values_x = torch.cat([driver_time_values_x, lake_time_values_x])
            time_ids_x = torch.cat([driver_time_ids_x, lake_time_ids_x])
        else:
            var_ids_x = lake_var_ids_x
            depth_vals_x = lake_depth_vals_x
            time_values_x = lake_time_values_x
            time_ids_x = lake_time_ids_x

        var_ids_y = lake_var_ids_y
        depth_vals_y = lake_depth_vals_y
        time_values_y = lake_time_values_y
        time_ids_y = torch.cat([lake_time_ids_y])

        sample_ids_x, sample_ids_y = self.get_sample_ids(time_ids_x, time_ids_y)

        num_nans_in_seq_x = torch.isnan(flat_seq_x).sum().item()
        flat_mask_x = (~torch.isnan(flat_seq_x)).float()
        flat_mask_y = (~torch.isnan(flat_seq_y)).float()
        
        flat_seq_x = torch.nan_to_num(flat_seq_x).float()
        flat_seq_y = torch.nan_to_num(flat_seq_y).float()

        sample = {"time_values_x": time_values_x,    # Time values (normalized days)
                  "time_values_y": time_values_y,    # Time values (normalized days)
                  "datetime_strs_x": np.array(lake_datetime_strs_x, dtype='datetime64[ns]'),
                  "datetime_strs_y": np.array(lake_datetime_strs_y, dtype='datetime64[ns]'),
                  "depth_values_x": depth_vals_x,          # Depth values  
                  "depth_values_y": depth_vals_y,          # Depth values  
                  "var_ids_x": var_ids_x,                  # Variable IDs
                  "var_ids_y": var_ids_y,                  # Variable IDs
                  "time_ids_x": time_ids_x,
                  "time_ids_y": time_ids_y,
                  "flat_seq_x": flat_seq_x,
                  "flat_seq_y": flat_seq_y,
                  "flat_mask_x": flat_mask_x,
                  "flat_mask_y": flat_mask_y,
                  "lake_id": self.lake_id,
                  "sample_ids_x": sample_ids_x,
                  "sample_ids_y": sample_ids_y,
                  "num2Dvars": len(self.variate_ids_2D),
                  "num1Dvars": len(self.variate_ids_1D), 
                  "num_depths": self.num_unique_depths,
                  "lake_name": self.lake_name,
                  "idx": idx
                  } 
        
        if regular_grid_queries is not None:
            sample["tgt_variate_ids"] = regular_grid_queries["var_ids"]
            sample["tgt_time_values"] = regular_grid_queries["time_values_norm"]
            sample["tgt_time_ids"] = regular_grid_queries["time_ids"]
            sample["tgt_depth_values"] = regular_grid_queries["depth_values"]
            sample["tgt_datetime_strs"] = np.array(regular_grid_queries["datetime_strs"], dtype="datetime64[ns]")
            sample["tgt_padding_mask"] = torch.ones_like(regular_grid_queries["var_ids"], dtype=torch.bool)
        
        return sample
    
    def __len__(self):
        max_ctx = max(self.context_window_range)
        max_pred = max(self.pred_window_range)
        
        if self.valid_idx is not None:
            return max(0, len(self.valid_idx))
        else:
            if hasattr(self, 'regular_date_grid') and len(self.regular_date_grid) > 0:
                computed_len = len(self.regular_date_grid) - max_ctx - max_pred + 1
            else:
                computed_len = len(self.unique_datetimes) - max_ctx - max_pred + 1
            return max(0, computed_len)

    def set_window_lengths(self, context_len: int, prediction_len: int):
        self.current_context_len = int(context_len)
        self.current_prediction_len = int(prediction_len)

    def sample_feasible_window(self, rng: np.random.Generator = None):
        """
        Sample a feasible (context, prediction) pair under split length.
        If not dynamic, returns fixed config lengths.
        """
        if not self.dynamic_windows:
            return self.context_len, self.prediction_len
        if rng is None:
            rng = np.random.default_rng()
        available = len(self.unique_datetimes)
        min_ctx, max_ctx = self.context_window_range
        min_pred, max_pred = self.pred_window_range
        max_ctx = max(min_ctx, min(max_ctx, available))
        max_pred = max(min_pred, min(max_pred, available))
        for _ in range(10):
            ctx = int(rng.integers(min_ctx, max_ctx + 1))
            pred = int(rng.integers(min_pred, max_pred + 1))
            if ctx + pred <= available:
                return ctx, pred
        total = min(available, max_ctx + max_pred)
        ctx = min(max_ctx, total - min_pred)
        pred = total - ctx
        ctx = max(ctx, min_ctx)
        pred = max(pred, min_pred)
        return ctx, pred
    
    def features_processing(self, df):
        
        cols=df.columns
        cols_to_drop=[self.date_col] + self.feats_to_drop
        cols=[col for col in cols if not col.startswith('Flag')]
        cols=[col for col in cols if col not in cols_to_drop]
        
        return df[cols]

    def get_metadata(self):
        return {
            "lake_id": self.lake_id,
            "id_str": self.id_str,
            "num_depths": self.num_unique_depths,
            "context_len": self.context_len,
            "prediction_len": self.prediction_len
        }
    
    def _flatten_data(self, data):
        
        if data.ndim == 3:
            # (S, V, D) → (V, D, S) → (V * D * S)
            return data.permute(1, 2, 0).reshape(-1)
        elif len(data.shape)==2: 
            return data.permute(1, 0).reshape(-1)
        else:
            return ValueError("Unsupported data shape for flattening")
    
    def get_sample_ids(self, time_ids_x, time_ids_y):
        """
        Generate sample IDs for irregular data structure using time_ids.
        
        Each time_id=0 indicates the start of a new sample. All tokens with
        time_id=0,1,2,... until the next time_id=0 belong to the same sample.
        """
        sample_ids_x = []
        sample_ids_y = []
        
        current_sample_id = 1
        for time_id in time_ids_x:
            if time_id == 0:
                current_sample_id += 1
            sample_ids_x.append(current_sample_id)
        
        for time_id in time_ids_y:
            if time_id == 0:
                current_sample_id += 1
            sample_ids_y.append(current_sample_id)
        
        
        return torch.tensor(sample_ids_x, dtype=torch.long), torch.tensor(sample_ids_y, dtype=torch.long)

    def __repr__(self):
        return f"LakeEvalDataset(id={self.id_str}, len={len(self)}, depths={self.num_unique_depths})"
