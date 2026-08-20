import pandas as pd
import torch
import numpy as np
import json
import os
from torch.utils.data import Dataset
from utils.util import Normalizer

class LakeDataset(Dataset):
    """
    Updated LakeDataset to handle irregular grid data structure.
    
    Key Changes:
    1. No longer assumes structured data on regular grid
    2. Handles variable depths per datetime
    3. Implements (time, variable, depth, value) token structure
    4. New flattening: v1_d1_sequence -> v1_d2_sequence -> v2_d1_sequence...
    5. Updated train-test split for datetime-based splitting with variable depths
    6. Stores corresponding variable and time IDs for each token
    7. Each token is now (time, depth, variable, value) for consistent embedding
    8. Tokens are passed as separate lists instead of stacked tensors
    """
    
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
                 mask_variable=None,
                 mask_depth=None,
                 create_global_stats=True,
                 run_name=None,
                 ckpt_name=None,
                 regular_grid_forecasting=False,
                 regular_grid_depths=12,
                 regular_grid_max_depth=None):

        self.lake_df = lake_df
        self.driver_df = driver_df
        self.param_df = param_df
        self.lake_id = lake_id
        self.variate_ids_2D = variate_ids_2D
        self.variate_ids_1D = variate_ids_1D
        self.id2var = id2var
        self.var_names_2D = var_names_2D or []
        self.var_names_1D = var_names_1D or []
        self.id_str = f"lake_{lake_id}" if lake_id is not None else "lake"
        self.lake_name = lakename
        self.coverage_threshold = coverage_threshold
        self.is_test_fraction = is_test_fraction
        self.normalization_stats_path = normalization_stats_path
        self.create_global_stats = create_global_stats
        self.run_name = run_name
        self.ckpt_name = ckpt_name
        self.timeenc = timeenc

        self.cfg = cfg

        self.train_frac = self.cfg['train_fraction']
        self.val_frac = self.cfg['val_fraction']
        self.test_frac = self.cfg['test_fraction']

        self.context_len = self.cfg['context_len']
        self.prediction_len = self.cfg['prediction_len']

        # Dynamic windowing config (optional)
        self.dynamic_windows = bool(self.cfg.get('dynamic_windows', False))
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
        
        # masking variables and depths
        self.mask_variable_idx = mask_variable
        self.mask_depth = mask_depth

        self.mean = None
        self.std = None
        self.valid_idx = None

        self.__process_data__()
        self.__split__()
        # build_valid_idx is not needed for irregular data
        # if self.coverage_threshold is not None:
        #     self.__build_valid_idx__()
    
    def __build_valid_idx__(self):
        """
        Build valid datetime-window starts for current (context, prediction) lengths.
        A start index is valid iff BOTH context and prediction windows contain at least
        one observed lake token (based on datetime_var_depth_map).
        """
        L_ctx, L_pred = self.current_context_len, self.current_prediction_len
        num_dt = len(self.unique_datetimes)
        num_candidates = num_dt - L_ctx - L_pred + 1

        if num_candidates <= 0:
            self.valid_idx = []
            return

        per_dt_obs = np.array(
            [len(self.datetime_var_depth_map.get(dt, [])) for dt in self.unique_datetimes],
            dtype=np.int64,
        )
        prefix = np.zeros(num_dt + 1, dtype=np.int64)
        prefix[1:] = np.cumsum(per_dt_obs)

        valid = []
        for start_idx in range(num_candidates):
            ctx_obs = prefix[start_idx + L_ctx] - prefix[start_idx]
            pred_start = start_idx + L_ctx
            pred_obs = prefix[pred_start + L_pred] - prefix[pred_start]
            if ctx_obs > 0 and pred_obs > 0:
                valid.append(start_idx)

        self.valid_idx = valid
    
    def _create_datetime_var_depth_map(self):
        """
        Create a mapping from datetime to available (variable, depth) combinations.
        Returns dict: {datetime: [(var1, depth1, row_idx), (var1, depth2, row_idx), ...]}
        """
        datetime_map = {}
        
        # Iterate over the split DataFrame that retains date/depth columns
        for idx, row in self.split_df_lake.iterrows():
            datetime = row[self.date_col]
            # Datetime is now standardized to YYYY-MM-DD string format
            depth = row[self.depth_col]
            
            if datetime not in datetime_map:
                datetime_map[datetime] = []
            
            # Store available variable-depth combinations for this datetime
            for i, var_idx in enumerate(self.variate_ids_2D):
                var_name = self.var_names_2D[i] if i < len(self.var_names_2D) else None
                if var_name and not pd.isna(row[var_name]):  # Only include non-NaN values
                    datetime_map[datetime].append((var_idx, depth, idx))
        
        return datetime_map
    
    def _extract_irregular_data(self, datetimes, base_dt=None):
        """
        Extract lake data for given datetimes with irregular depth structure.
        
        For irregular data, we simply extract all available (time, variable, depth, value) 
        combinations that exist in the data, without trying to force a regular grid structure.
        
        Args:
            datetimes: List of datetime values to extract data for
        
        Returns:
            values: Flattened tensor of values
            var_ids: Variable IDs for each token  
            depth_vals: Depth values for each token
            time_ids: Relative day indices per token (int)
            time_values: Normalized day-of-year per token (float)
            datetime_strs: Raw datetime strings per token (list[str])
        """
        all_values = []
        all_var_ids = []
        all_depth_vals = []
        all_time_ids = []
        all_time_values = []
        
        # Collect all unique (variable, depth) combinations that appear in the datetimes
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
                            # Use split-scoped DataFrame to ensure we stay within the active split
                            if row_idx in self.split_df_lake.index:
                                # Use actual variable name instead of standardized name
                                var_idx_in_list = self.variate_ids_2D.index(v_idx) if v_idx in self.variate_ids_2D else -1
                                if var_idx_in_list >= 0 and var_idx_in_list < len(self.var_names_2D):
                                    var_name = self.var_names_2D[var_idx_in_list]
                                    value = self.split_df_lake.loc[row_idx, var_name]
                                else:
                                    value = np.nan
                                
                                # Only add if we have a valid observed value
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
            # Use day of year [0-365] instead of days since epoch for better seasonality encoding
            day_of_year = dt.dayofyear - 1  # 0-indexed (0-364 or 0-365 for leap years)
            # Normalize to [0, 1] range for numerical stability
            normalized_day_of_year = day_of_year / 365.25  # 365.25 accounts for leap years
            time_values_numeric.append(normalized_day_of_year)
        
        # sanity check
        if len(all_values) == 0:
            print(f"Debug: _extract_irregular_data returning empty tensors for datetimes: {datetimes}")
            print(f"Debug: var_depth_combinations: {var_depth_combinations}")
            for dt in datetimes:
                if dt in self.datetime_var_depth_map:
                    print(f"Debug: {dt} has {len(self.datetime_var_depth_map[dt])} observations")
                else:
                    print(f"Debug: {dt} NOT in datetime_var_depth_map")
        
        return (torch.tensor(all_values, dtype=torch.float32),
                torch.tensor(all_var_ids, dtype=torch.long), 
                torch.tensor(all_depth_vals, dtype=torch.float32),
                torch.tensor(all_time_ids, dtype=torch.long),
                torch.tensor(time_values_numeric, dtype=torch.float32),
                all_time_values)
    
    def _extract_driver_as_surface_variables(self, driver_data, datetimes, base_dt=None):
        """
        Extract driver variables as surface (depth=0) variables.
        Creates temporal sequences for each driver variable, similar to lake variables.
        
        Args:
            driver_data: Driver data tensor (time, variables)
            datetimes: List of datetime values
            
        Returns:
            values: Flattened tensor of driver values
            var_ids: Variable IDs for each token
            depth_vals: Depth values (all 0.0 for surface)
            time_ids: Relative day indices per token (int)
            time_values: Normalized day-of-year per token (float)
            datetime_strs: Raw datetime strings per token (list[str])
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

        # For each driver variable, create a temporal sequence (just like lake variables)
        for var_idx, var_id in enumerate(self.variate_ids_1D):
            # Create temporal sequence for this driver variable
            for time_idx, dt in enumerate(datetimes):
                if time_idx < driver_data.shape[0] and var_idx < driver_data.shape[1]:
                    value = driver_data[time_idx, var_idx].item()
                    
                    # Only add if we have a valid observed value
                    if not np.isnan(value):
                        all_values.append(value)
                        all_var_ids.append(var_id)
                        all_depth_vals.append(0.0)  # Surface depth for all driver variables
                        # time_ids are days since first datetime in the window
                        if base_dt is not None:
                            rel_days = (pd.to_datetime(dt) - base_dt).days
                        else:
                            rel_days = 0
                        all_time_ids.append(rel_days)
                        all_time_values.append(dt)  # Use actual datetime
              
        # Convert datetime strings to numeric values (day of year, normalized) for PyTorch tensors
        time_values_numeric = []
        for dt_str in all_time_values:
            # Convert standardized YYYY-MM-DD string to datetime and then to normalized day of year
            dt = pd.to_datetime(dt_str)
            # Use day of year [0-365] instead of days since epoch for better seasonality encoding
            day_of_year = dt.dayofyear - 1  # 0-indexed (0-364 or 0-365 for leap years)
            # Normalize to [0, 1] range for numerical stability
            normalized_day_of_year = day_of_year / 365.25  # 365.25 accounts for leap years
            time_values_numeric.append(normalized_day_of_year)
              
        return (torch.tensor(all_values, dtype=torch.float32),
                torch.tensor(all_var_ids, dtype=torch.long), 
                torch.tensor(all_depth_vals, dtype=torch.float32),
                torch.tensor(all_time_ids, dtype=torch.long),
                torch.tensor(time_values_numeric, dtype=torch.float32),
                all_time_values)

    def __split__(self, flag='train'):
        self.set_type = self.type_map[flag]

        '''
        Split data
        '''
        self.border1_DR = self.border1s_DR[self.set_type]
        self.border2_DR = self.border2s_DR[self.set_type]
        
        self.border1_DF = self.border1s_DF[self.set_type]
        self.border2_DF = self.border2s_DF[self.set_type]

        df_stamp = self.df_ts[self.border1_DR:self.border2_DR]
        # Date column is now standardized to YYYY-MM-DD string format
        
        '''
        Time features generation
        '''
        if self.timeenc == 1:
            # Convert standardized date strings to datetime for feature extraction
            df_stamp_dt = df_stamp.copy()
            df_stamp_dt[self.date_col] = pd.to_datetime(df_stamp_dt[self.date_col])
            df_stamp_dt['month'] = df_stamp_dt[self.date_col].apply(lambda row: row.month, 1)
            df_stamp_dt['day'] = df_stamp_dt[self.date_col].apply(lambda row: row.day, 1)
            df_stamp_dt['weekday'] = df_stamp_dt[self.date_col].apply(lambda row: row.weekday(), 1)
            df_stamp_dt['hour'] = df_stamp_dt[self.date_col].apply(lambda row: row.hour, 1)
            data_stamp = df_stamp_dt.drop([self.date_col], 1).values
        else:
            data_stamp = df_stamp.values

        # Convert to numpy arrays for tensor operations, excluding metadata columns
        driver_cols = [col for col in self.df_driver.columns if col != self.date_col]
        lake_cols = [col for col in self.df_lake.columns if col not in [self.date_col, self.depth_col]]
        
        # Split-scoped slices
        self.data_DR = self.df_driver[driver_cols][self.border1_DR:self.border2_DR].values
        # Keep a split-scoped DataFrame that includes date/depth and vars
        self.split_df_lake = self.df_lake.iloc[self.border1_DF:self.border2_DF].copy()
        # Feature-only view for model inputs
        self.data_DF = self.split_df_lake[lake_cols]
        
        self.data_stamp = data_stamp
        # Build unique datetimes and map from the current split only
        unique_dates = self.split_df_lake[self.date_col].unique()
        self.unique_datetimes = sorted(unique_dates)
        self.datetime_var_depth_map = self._create_datetime_var_depth_map()
        self.__build_valid_idx__()

    def _standardize_date_columns(self):
        """
        Standardize date columns to datetime type with daily granularity (YYYY-MM-DD format).
        This ensures consistency across different datasets during pretraining.
        """
        import pandas as pd
        
        # Standardize lake data date column
        if self.date_col in self.lake_df.columns:
            self.lake_df[self.date_col] = pd.to_datetime(self.lake_df[self.date_col]).dt.date.astype(str)
        
        # Standardize driver data date column  
        if self.date_col in self.driver_df.columns:
            self.driver_df[self.date_col] = pd.to_datetime(self.driver_df[self.date_col]).dt.date.astype(str)
        
        # Standardize param data date column if it exists
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

        # Extract depth values and compute normalization parameters
        self.depth_values = sorted(self.lake_df[self.depth_col].unique())
        self.num_unique_depths = len(self.depth_values)
        self.max_depth = max(self.depth_values) if len(self.depth_values) > 0 else 1.0
        self.min_depth = min(self.depth_values) if len(self.depth_values) > 0 else 0.0

        # Preserve raw depths for plotting before normalization
        # self.depth_raw_col = f"{self.depth_col}_raw"
        # self.lake_df[self.depth_raw_col] = self.lake_df[self.depth_col].copy()
        # Normalize the depth column in lake_df using min-max scaling
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
        
        train_data_DR = df_driver[self.border1s_DR[0]:self.border2s_DR[0]]
        train_data_DF = df_lake[self.border1s_DF[0]:self.border2s_DF[0]]

        use_scaling = self.cfg.get('scaling', True)

        if use_scaling:
            self.norm.fit_scalers(train_data_DR, train_data_DF)
            df_driver_scaled, df_lake_scaled = self.norm.transform_data(df_driver, df_lake)
        else:
            df_driver_scaled = df_driver.values if hasattr(df_driver, 'values') else df_driver
            df_lake_scaled = df_lake.values if hasattr(df_lake, 'values') else df_lake
        
        # Convert back to DataFrames and add back metadata columns
        self.df_driver = pd.DataFrame(df_driver_scaled, columns=df_driver.columns)
        self.df_driver[self.date_col] = self.driver_df[self.date_col].values
        
        self.df_lake = pd.DataFrame(df_lake_scaled, columns=df_lake.columns)
        self.df_lake[self.date_col] = self.lake_df[self.date_col].values
        self.df_lake[self.depth_col] = self.lake_df[self.depth_col].values

        if use_scaling and self.normalization_stats_path:
            lake_stats_dir = os.path.dirname(self.normalization_stats_path)
            global_stats_path = os.path.join(lake_stats_dir, "global_variable_stats.json")
            
            if not self.create_global_stats:
                global_stats_path = None
            
            self.norm.save_normalization_stats(
                self.normalization_stats_path,
                global_stats_path=global_stats_path,
                num_unique_depths=self.num_unique_depths,
                depth_values=self.depth_values,
                max_depth=self.max_depth,
                min_depth=self.min_depth,
                variate_ids_2D=self.variate_ids_2D,
                variate_ids_1D=self.variate_ids_1D
            )

    def __getitem__(self, idx):
        '''
        Updated to handle irregular grid data structure with consistent token format.
        Each token is now (time, depth, variable, value) for consistent embedding.
        Tokens are passed as separate lists instead of stacked tensors.
        '''
        if not self.valid_idx:
            raise RuntimeError("No valid datetime windows available for current context/pred lengths")

        # Generate samples from precomputed valid starts only
        s_begin_lake = self.valid_idx[int(idx)]
        s_end_lake = s_begin_lake + self.current_context_len
        r_begin_lake = s_end_lake
        r_end_lake = r_begin_lake + self.current_prediction_len

        context_datetimes = self.unique_datetimes[s_begin_lake:s_end_lake]
        prediction_datetimes = self.unique_datetimes[r_begin_lake:r_end_lake]

        # Find corresponding driver data indices for these datetimes
        # Driver data should be available for all lake datetimes (regular data)
        s_begin_DR = s_begin_lake
        s_end_DR = s_end_lake
        r_begin_DR = r_begin_lake
        r_end_DR = r_end_lake

        driver1D_x = torch.tensor(self.data_DR[s_begin_DR:s_end_DR])
        driver1D_y = torch.tensor(self.data_DR[r_begin_DR:r_end_DR])
        # Shared base for time ids: first context datetime
        base_dt_ctx = pd.to_datetime(context_datetimes[0]) if len(context_datetimes) > 0 else None
        # Get data for context period
        lake_data_x, lake_var_ids_x, lake_depth_vals_x, lake_time_ids_x, lake_time_values_x, lake_datetime_strs_x = self._extract_irregular_data(
            context_datetimes, base_dt=base_dt_ctx
        )
        # Get data for prediction period with same base
        lake_data_y, lake_var_ids_y, lake_depth_vals_y, lake_time_ids_y, lake_time_values_y, lake_datetime_strs_y = self._extract_irregular_data(
            prediction_datetimes, base_dt=base_dt_ctx
        )

        date_values_x = self.data_stamp[s_begin_DR:s_end_DR]
        date_values_y = self.data_stamp[r_begin_DR:r_end_DR]

        if self.param_df is not None:
            simulation_params = torch.tensor(self.param_df.values)
        else:
            simulation_params = None

        # Extract driver variables as surface (depth=0) variables for context only
        # We don't predict driver variables, only use them as input
        driver_data_x, driver_var_ids_x, driver_depth_vals_x, driver_time_ids_x, driver_time_values_x, driver_datetime_strs_x = self._extract_driver_as_surface_variables(
            driver1D_x, context_datetimes, base_dt=base_dt_ctx
        )
        # Context: Include both driver (surface) and lake data - drivers first since they're at surface (depth=0)
        flat_seq_x = torch.cat([driver_data_x, lake_data_x])

        # Prediction: Only lake variables (no driver variables to predict)
        flat_seq_y = lake_data_y

        var_ids_x = torch.cat([driver_var_ids_x, lake_var_ids_x])
        depth_vals_x = torch.cat([driver_depth_vals_x, lake_depth_vals_x])
        time_values_x = torch.cat([driver_time_values_x, lake_time_values_x])
        time_ids_x = torch.cat([driver_time_ids_x, lake_time_ids_x])

        var_ids_y = lake_var_ids_y
        depth_vals_y = lake_depth_vals_y
        time_values_y = lake_time_values_y
        time_ids_y = torch.cat([lake_time_ids_y])

        sample_ids_x, sample_ids_y = self.get_sample_ids(time_ids_x, time_ids_y)

        # Create masks for all observed tokens (both driver and lake)
        flat_mask_x = (~torch.isnan(flat_seq_x)).float()
        flat_mask_y = (~torch.isnan(flat_seq_y)).float()

        flat_mask_x = torch.nan_to_num(flat_mask_x).float()
        flat_mask_y = torch.nan_to_num(flat_mask_y).float()

        # Updated masking logic for irregular data structure
        # TODO: This masking logic needs to be updated for the new irregular structure
        # For now, we'll skip the complex masking and implement it later if needed
        if self.mask_variable_idx is not None or self.mask_depth is not None:
            print("Warning: Masking logic not yet implemented for irregular data structure")

        sample = {"time_values_x": time_values_x,    # Time values (normalized days)
                  "time_values_y": time_values_y,    # Time values (normalized days)
                  "datetime_strs_x": np.array(lake_datetime_strs_x, dtype='datetime64[ns]'), # Raw datetime strings for context tokens
                  "datetime_strs_y": np.array(lake_datetime_strs_y, dtype='datetime64[ns]'), # Raw datetime strings for prediction tokens
                  "depth_values_x": depth_vals_x,          # Depth values
                  "depth_values_y": depth_vals_y,          # Depth values
                  "var_ids_x": var_ids_x,                  # Variable IDs
                  "var_ids_y": var_ids_y,                  # Variable IDs
                  "time_ids_x": time_ids_x,
                  "time_ids_y": time_ids_y,
                  "flat_seq_x": flat_seq_x,            # Uppercase for loader compatibility
                  "flat_seq_y": flat_seq_y,            # Uppercase for loader compatibility
                  "flat_mask_x": flat_mask_x,
                  "flat_mask_y": flat_mask_y,
                  "lake_id": self.lake_id,
                  "sample_ids_x": sample_ids_x,
                  "sample_ids_y": sample_ids_y,
                  "num2Dvars": len(self.variate_ids_2D),
                  "num1Dvars": len(self.variate_ids_1D),
                  "num_depths": self.num_unique_depths,
                  "lake_name": self.lake_name,
                  "idx": s_begin_lake
                  }

        return sample
    
    def __len__(self):
        return max(0, len(self.valid_idx))

    def set_window_lengths(self, context_len: int, prediction_len: int):
        self.current_context_len = int(context_len)
        self.current_prediction_len = int(prediction_len)
        self.__build_valid_idx__()

    def sample_feasible_window(self, rng: np.random.Generator = None, epoch: int = 0):
        """
        Sample a feasible (context, prediction) pair under split length.
        If not dynamic, returns fixed config lengths.
        """
        if not self.dynamic_windows:
            return self.context_len, self.prediction_len
        
        if rng is None:
            rng = np.random.default_rng()
        
        # get the predefined pairs
        pairs = self.cfg.get("window_pairs", None)
        if pairs is None:
            return self.context_len, self.prediction_len
        
        strategy = self.cfg.get("window_sampling_strategy", "cycle")
        if strategy == "cycle":
            # cycle through the pairs deterministically
            idx = epoch % len(pairs)
        elif strategy == "random":
            idx = int(rng.integers(0, len(pairs)))
        elif strategy == "curriculum":
            # Start with easier (smaller) pairs, progress to harder
            progress = min(epoch / 20, 1.0)  # 20 epochs to full curriculum
            idx = int(progress * (len(pairs) - 1))
        else:
            idx=0

        return int(pairs[idx][0]), int(pairs[idx][1])

        # available = len(self.unique_datetimes)
        # min_ctx, max_ctx = self.context_window_range
        # min_pred, max_pred = self.pred_window_range
        # # clip to feasible maximums given available dates
        # max_ctx = max(min_ctx, min(max_ctx, available))
        # max_pred = max(min_pred, min(max_pred, available))
        # # try a few samples to ensure feasibility
        # for _ in range(10):
        #     ctx = int(rng.integers(min_ctx, max_ctx + 1))
        #     pred = int(rng.integers(min_pred, max_pred + 1))
        #     if ctx + pred <= available:
        #         return ctx, pred
        # # Fallback: shrink to fit
        # total = min(available, max_ctx + max_pred)
        # ctx = min(max_ctx, total - min_pred)
        # pred = total - ctx
        # ctx = max(ctx, min_ctx)
        # pred = max(pred, min_pred)
        # return ctx, pred
    
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
        
        Args:
            time_ids_x: Time IDs for context tokens
            time_ids_y: Time IDs for prediction tokens
            
        Returns:
            torch.Tensor: Sample IDs for each token
        """
        sample_ids_x = []
        sample_ids_y = []
        
        # Process context tokens
        current_sample_id = 1
        for time_id in time_ids_x:
            if time_id == 0:
                current_sample_id += 1
            sample_ids_x.append(current_sample_id)
        
        # Process prediction tokens
        # Continue with the same sample ID logic
        for time_id in time_ids_y:
            if time_id == 0:
                current_sample_id += 1
            sample_ids_y.append(current_sample_id)
        
        
        return torch.tensor(sample_ids_x, dtype=torch.long), torch.tensor(sample_ids_y, dtype=torch.long)

    def __repr__(self):
        return f"LakeDatasetIrregular(id={self.id_str}, len={len(self)}, depths={self.num_unique_depths})"
