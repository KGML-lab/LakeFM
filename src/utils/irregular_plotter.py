import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
import pandas as pd
from datetime import datetime, timedelta
from scipy import stats
import seaborn as sns

class IrregularGridPlotter:
    """
    Plotter for irregular grid data with variable time steps and depths
    """
    def __init__(self, device, pad_val):
        self.device = device
        self.pad_val = pad_val
    
    def _denormalize_time_values(self, time_values, base_year=2020):
        """
        Convert normalized time-of-year values [0, 1] array to list of datetime objects.
        Robust to inputs of shape (T,), (T,1), lists, or arrays with dtype object.
        """
        tv = np.asarray(time_values)
        # Flatten and ensure float
        tv = tv.astype(float).ravel()
        # Map to day indices (1..365)
        day_idx = np.clip((tv * 365.25).astype(int), 1, 365)
        base = datetime(base_year, 1, 1)
        return [base + timedelta(days=int(d) - 1) for d in day_idx]
    
    def _save_context_for_predictions(self, data_points, context_seq_row, context_var_ids_row,
                                      context_depth_vals_row, context_time_vals_row,
                                      context_datetime_strs, context_mask_row,
                                      var_id, depth, lake_name, save_path):
        """
        Save corresponding context (input window) data for the selected predictions.
        """
        try:
            import os
            import json
            
            # Create context data directory
            context_dir = os.path.join(os.path.dirname(save_path), "context_data")
            os.makedirs(context_dir, exist_ok=True)
            
            # Extract unique sample indices from the selected predictions
            sample_indices = [dp['sample_idx'] for dp in data_points]
            unique_samples = sorted(set(sample_indices))
            
            # Prepare data to save
            context_data = {
                'lake_name': str(lake_name) if lake_name else 'unknown',
                'variable_id': int(var_id),
                'depth': float(depth),
                'prediction_dates': [str(dp['time']) for dp in data_points],
                'prediction_values': [float(dp['pred']) for dp in data_points],
                'ground_truth_values': [float(dp['gt']) for dp in data_points],
                # List of context windows (one per unique underlying eval sample)
                'context_windows': []
            }
            
            # Extract context for each unique sample
            for sidx in unique_samples:
                # Convert to numpy
                ctx_seq = context_seq_row[sidx].detach().cpu().numpy() if hasattr(context_seq_row[sidx], 'detach') else context_seq_row[sidx]
                ctx_var_ids = context_var_ids_row[sidx].detach().cpu().numpy() if hasattr(context_var_ids_row[sidx], 'detach') else context_var_ids_row[sidx]
                ctx_depths = context_depth_vals_row[sidx].detach().cpu().numpy() if hasattr(context_depth_vals_row[sidx], 'detach') else context_depth_vals_row[sidx]
                ctx_times = context_time_vals_row[sidx].detach().cpu().numpy() if hasattr(context_time_vals_row[sidx], 'detach') else context_time_vals_row[sidx]
                ctx_datetimes = context_datetime_strs[sidx]
                ctx_mask = context_mask_row[sidx].detach().cpu().numpy() if hasattr(context_mask_row[sidx], 'detach') else context_mask_row[sidx]
                
                # Ensure all arrays have consistent length (handle padding)
                min_len = min(len(ctx_seq), len(ctx_var_ids), len(ctx_depths), 
                             len(ctx_times), len(ctx_datetimes), len(ctx_mask))
                
                if min_len < len(ctx_mask):
                    # Truncate to consistent length
                    ctx_seq = ctx_seq[:min_len]
                    ctx_var_ids = ctx_var_ids[:min_len]
                    ctx_depths = ctx_depths[:min_len]
                    ctx_times = ctx_times[:min_len]
                    ctx_datetimes = ctx_datetimes[:min_len]
                    ctx_mask = ctx_mask[:min_len]
                
                # Filter context tokens to the (var_id, depth) that predictions were filtered to.
                # `depth` here is taken from the prediction tokens (depth_vals_row), so it should be on the same scale.
                is_target_var = (ctx_var_ids == int(var_id))
                depth_tolerance = 1e-6  # keep tight; depths come from the same tensor family as the prediction keys
                is_target_depth = np.isclose(ctx_depths.astype(float), float(depth), atol=depth_tolerance)
                target_all_idx = is_target_var & is_target_depth
                target_obs_idx = target_all_idx & ctx_mask.astype(bool)
                
                # Context window record (target var+depth only)
                # Values are DENORMALIZED (by evaluator) but masked tokens are set to 0.0 with observed_mask=0.
                context_window = {
                    'values': ctx_seq[target_all_idx].tolist(),
                    'observed_mask': ctx_mask[target_all_idx].tolist(),
                    'datetimes': [str(dt) for dt in ctx_datetimes[target_all_idx]],
                    # Optional helpers (kept small; still useful for debugging/plotting)
                    'depths': ctx_depths[target_all_idx].tolist(),
                    'times': ctx_times[target_all_idx].tolist(),
                }
                context_data['context_windows'].append(context_window)
            
            var_name = f"var{var_id}"
            depth_str = f"{depth:.2f}m".replace('.', 'p')
            filename = f"context_{lake_name}_{var_name}_{depth_str}.json"
            filepath = os.path.join(context_dir, filename)
        
            with open(filepath, 'w') as f:
                json.dump(context_data, f, indent=2)
            
            print(f"Saved context data to: {filepath}")
            print(f"  {len(unique_samples)} context window(s) for {len(data_points)} predictions")
        
        except Exception as e:
            print(f"Warning: Failed to save context data: {str(e)}")
            # Don't raise - we don't want to break plotting if context saving fails
    
    def _compute_confidence_intervals(self, distribution, confidence_level=0.95):
        """
        Compute confidence intervals for Student-t distribution
        """
        # For Student-t distribution, we can use the quantile function
        alpha = 1 - confidence_level
        lower_quantile = alpha / 2
        upper_quantile = 1 - alpha / 2
        
        lower_bound = distribution.icdf(torch.tensor(lower_quantile))
        upper_bound = distribution.icdf(torch.tensor(upper_quantile))
        
        return lower_bound, upper_bound
    

    def plot_from_flat_tokens(self,
                              gt_row,
                              preds_row,
                              time_vals_row,
                              depth_vals_row,
                              var_ids_row,
                              mask_row,
                              feature_dict,
                              epoch,
                              train_or_val,
                              title_prefix="Irregular Tokens",
                              plot_type='line',
                              max_features=6,
                              max_depths_per_feature=6,
                              filter_first_pred=False,
                              depth_units=None):
        """
        Plot from flattened token tensors for irregular data.
        Expects 1D arrays for a single sample (tokens along axis 0).
        """
        # to numpy
        gt = gt_row.detach().cpu().numpy() if hasattr(gt_row, 'detach') else np.asarray(gt_row)
        preds = preds_row.detach().cpu().numpy() if hasattr(preds_row, 'detach') else np.asarray(preds_row)
        time_vals = time_vals_row.detach().cpu().numpy() if hasattr(time_vals_row, 'detach') else np.asarray(time_vals_row)
        depth_vals = depth_vals_row.detach().cpu().numpy() if hasattr(depth_vals_row, 'detach') else np.asarray(depth_vals_row)
        var_ids = var_ids_row.detach().cpu().numpy() if hasattr(var_ids_row, 'detach') else np.asarray(var_ids_row)
        mask = mask_row.detach().cpu().numpy() if hasattr(mask_row, 'detach') else np.asarray(mask_row)

        # Flatten and filter valid tokens (mask==1)
        gt = gt.ravel()
        preds = preds.ravel()
        time_vals = time_vals.ravel()
        depth_vals = depth_vals.ravel()
        var_ids = var_ids.ravel()
        mask = mask.ravel().astype(bool)
        # breakpoint()
        valid_idx = np.where(mask)[0]
        if valid_idx.size == 0:
            return

        gt = gt[valid_idx]
        preds = preds[valid_idx]
        time_vals = time_vals[valid_idx]
        depth_vals = depth_vals[valid_idx]
        var_ids = var_ids[valid_idx]
        if filter_first_pred:
            min_t = np.min(time_vals)
            sel_first = (time_vals == min_t)
            gt = gt[sel_first]
            preds = preds[sel_first]
            time_vals = time_vals[sel_first]
            depth_vals = depth_vals[sel_first]
            var_ids = var_ids[sel_first]

        # Unique features to plot (limit)
        unique_feats = []
        for vid in var_ids:
            if vid not in unique_feats:
                unique_feats.append(int(vid))
            if len(unique_feats) >= max_features:
                break
        nfeat = len(unique_feats)
        if nfeat == 0:
            return

        fig, axes = plt.subplots(nfeat, 1, figsize=(12, 3.5 * nfeat))
        if nfeat == 1:
            axes = [axes]

        for ax, feat_id in zip(axes, unique_feats):
            sel = var_ids == feat_id
            if not np.any(sel):
                continue
            t_feat = time_vals[sel]
            p_feat = preds[sel]
            g_feat = gt[sel]
            d_feat = depth_vals[sel]

            # Depths for this feature (limit)
            depths = np.unique(d_feat)
            if len(depths) > max_depths_per_feature:
                depths = depths[:max_depths_per_feature]
            colors = plt.cm.plasma(np.linspace(0, 1, len(depths)))

            # Convert time to dates using provided raw datetime strings if available
            dates_all = np.array(t_feat).astype(str)

            for c, dep in zip(colors, depths):
                sel_d = d_feat == dep
                if not np.any(sel_d):
                    continue
                dates = dates_all[sel_d]
                p_vals = p_feat[sel_d]
                g_vals = g_feat[sel_d]

                # sort by time
                order = np.argsort(dates)
                dates = dates[order]
                p_vals = p_vals[order]
                g_vals = g_vals[order]

                # plot preds with markers (irregular spacing is preserved by datetime x-axis)
                ax.plot(dates,
                        p_vals,
                        color=c,
                        linewidth=1.8,
                        marker='o',
                        markersize=3,
                        label=(f"Pred @ depth={float(dep):.2f}{(' '+depth_units) if depth_units else ''}"))
                # plot gt (sparse)
                if plot_type == 'scatter':
                    ax.scatter(dates, g_vals, color=c, s=16, marker='o', alpha=0.7, label=f"GT @ {float(dep):.1f}m")
                else:
                    ax.plot(dates,
                            g_vals,
                            color=c,
                            linestyle='--',
                            linewidth=1.2,
                            marker='x',
                            markersize=3,
                            alpha=0.8,
                            label=(f"GT @ depth={float(dep):.2f}{(' '+depth_units) if depth_units else ''}"))

            feat_name = feature_dict.get(int(feat_id), f"var_{int(feat_id)}")
            ax.set_title(f"{feat_name} ({train_or_val})")
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        title = f"{title_prefix}: {train_or_val} tokens at epoch {epoch}"
        plt.suptitle(title, fontsize=14, y=1.02)
        wandb.log({title: wandb.Image(plt)})
        plt.close()
    
    def plot_depth_profile(self, 
                          gt, 
                          preds, 
                          time_values,
                          depth_values,
                          var_ids,
                          gt_masks,
                          feature_dict,
                          epoch,
                          train_or_val,
                          title_prefix="Depth Profile",
                          selected_times=None):
        """
        Plot depth profiles at selected time points
        """
        preds = preds.detach()
        gt = gt.detach()
        time_values = time_values.detach()
        depth_values = depth_values.detach()
        gt_masks = gt_masks.detach()
        
        # Convert to numpy
        preds = preds.cpu().numpy()
        gt = gt.cpu().numpy()
        time_values = time_values.cpu().numpy()
        depth_values = depth_values.cpu().numpy()
        gt_masks = gt_masks.cpu().numpy()
        var_ids = var_ids.cpu().numpy()
        
        # Use raw datetime strings directly
        time_dates = np.array(time_values[0, :]).astype(str)
        
        num_features = preds.shape[2]
        
        if selected_times is None:
            # Select 3 time points evenly distributed
            num_times = min(3, preds.shape[1])
            selected_times = np.linspace(0, preds.shape[1]-1, num_times, dtype=int)
        
        fig, axes = plt.subplots(num_features, len(selected_times), 
                                figsize=(5 * len(selected_times), 4 * num_features))
        if num_features == 1:
            axes = axes.reshape(1, -1)
        if len(selected_times) == 1:
            axes = axes.reshape(-1, 1)
        
        for feat_idx in range(num_features):
            feature_name = feature_dict[int(var_ids[0, feat_idx])]
            
            for time_idx, t in enumerate(selected_times):
                ax = axes[feat_idx, time_idx]
                
                # Get depths and values at this time
                depths = depth_values[0, t, :]
                pred_vals = preds[0, t, feat_idx, :]
                gt_vals = gt[0, t, feat_idx, :]
                masks = gt_masks[0, t, feat_idx, :]
                
                # Remove padding values
                valid_mask = depths != self.pad_val
                depths = depths[valid_mask]
                pred_vals = pred_vals[valid_mask]
                gt_vals = gt_vals[valid_mask]
                masks = masks[valid_mask]
                
                # Sort by depth
                sort_idx = np.argsort(depths)
                depths = depths[sort_idx]
                pred_vals = pred_vals[sort_idx]
                gt_vals = gt_vals[sort_idx]
                masks = masks[sort_idx]
                
                # Apply masks
                gt_vals[~masks] = np.nan
                
                # Plot
                ax.plot(pred_vals, depths, 'g-', linewidth=2, label='Prediction', marker='o', markersize=4)
                valid_gt = ~np.isnan(gt_vals)
                if np.any(valid_gt):
                    ax.plot(gt_vals[valid_gt], depths[valid_gt], 'r--', linewidth=1.5, 
                           label='Ground Truth', marker='s', markersize=4)
                
                ax.set_xlabel('Value')
                ax.set_ylabel('Depth (m)')
                ax.set_title(f'{feature_name} at {time_dates[t].strftime("%b %d")}')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.invert_yaxis()  # Depth increases downward
        
        title = f'{title_prefix}: {train_or_val} Depth Profiles at epoch {epoch}'
        plt.suptitle(title, fontsize=14, y=1.02)
        plt.tight_layout()
        wandb.log({title: wandb.Image(plt)})
        plt.close()
    
    def plot_temporal_evolution(self, 
                               gt, 
                               preds, 
                               time_values,
                               depth_values,
                               var_ids,
                               gt_masks,
                               feature_dict,
                               epoch,
                               train_or_val,
                               title_prefix="Temporal Evolution",
                               selected_depths=None):
        """
        Plot temporal evolution for selected depths as heatmaps
        """
        preds = preds.detach()
        gt = gt.detach()
        time_values = time_values.detach()
        depth_values = depth_values.detach()
        gt_masks = gt_masks.detach()
        
        # Convert to numpy
        preds = preds.cpu().numpy()
        gt = gt.cpu().numpy()
        time_values = time_values.cpu().numpy()
        depth_values = depth_values.cpu().numpy()
        gt_masks = gt_masks.cpu().numpy()
        var_ids = var_ids.cpu().numpy()
        
        # Use raw datetime strings directly
        time_dates = np.array(time_values[0, :]).astype(str)
        
        num_features = preds.shape[2]
        
        # Get unique depths
        unique_depths = np.unique(depth_values[0, :, :].flatten())
        unique_depths = unique_depths[unique_depths != self.pad_val]
        
        if selected_depths is None:
            selected_depths = unique_depths
        else:
            selected_depths = np.array(selected_depths)
        
        fig, axes = plt.subplots(num_features, len(selected_depths), 
                                figsize=(6 * len(selected_depths), 4 * num_features))
        if num_features == 1:
            axes = axes.reshape(1, -1)
        if len(selected_depths) == 1:
            axes = axes.reshape(-1, 1)
        
        for feat_idx in range(num_features):
            feature_name = feature_dict[int(var_ids[0, feat_idx])]
            
            for depth_idx, depth_val in enumerate(selected_depths):
                ax = axes[feat_idx, depth_idx]
                
                # Create data matrix for this depth
                time_steps = preds.shape[1]
                data_matrix = np.full((2, time_steps), np.nan)  # [pred, gt]
                
                for t in range(time_steps):
                    depth_locations = np.where(depth_values[0, t, :] == depth_val)[0]
                    if len(depth_locations) > 0:
                        depth_loc = depth_locations[0]
                        data_matrix[0, t] = preds[0, t, feat_idx, depth_loc]  # prediction
                        
                        if gt_masks[0, t, feat_idx, depth_loc]:
                            data_matrix[1, t] = gt[0, t, feat_idx, depth_loc]  # ground truth
                
                # Plot as line plot
                time_vals = time_dates
                valid_pred = ~np.isnan(data_matrix[0, :])
                valid_gt = ~np.isnan(data_matrix[1, :])
                
                if np.any(valid_pred):
                    ax.plot(time_vals[valid_pred], data_matrix[0, valid_pred], 
                           'g-', linewidth=2, label='Prediction', alpha=0.8)
                if np.any(valid_gt):
                    ax.plot(time_vals[valid_gt], data_matrix[1, valid_gt], 
                           'r--', linewidth=1.5, label='Ground Truth', alpha=0.6)
                
                ax.set_xlabel('Time')
                ax.set_ylabel('Value')
                ax.set_title(f'{feature_name} @ {depth_val:.1f}m')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Format x-axis to show dates nicely
                if len(time_vals) > 0:
                    ax.tick_params(axis='x', rotation=45)
                    if len(time_vals) > 30:  # Only if we have enough data points
                        ax.xaxis.set_major_locator(plt.MaxNLocator(6))
        
        title = f'{title_prefix}: {train_or_val} Temporal Evolution at epoch {epoch}'
        plt.suptitle(title, fontsize=14, y=1.02)
        plt.tight_layout()
        wandb.log({title: wandb.Image(plt)})
        plt.close()
    
    def plot_forecast_irregular_grid(self,
                                     gt_row,
                                     preds_row,
                                     time_vals_row,
                                     datetime_raw_vals,
                                     depth_vals_row,
                                     var_ids_row,
                                     mask_row,
                                     feature_dict,
                                     sample_idx,
                                     plt_idx,
                                     epoch,
                                     train_or_val,
                                     title_prefix="T+1 Forecast",
                                     plot_type='line',
                                     max_features=6,
                                     max_depths_per_feature=6,
                                     filter_first_pred=True,
                                     depth_units=None,
                                     confidence_level=0.95,
                                     plot_interval=True,
                                     save_path=None):
        """
        Plot T+1 predictions from multiple samples with stride=1.
        
        For each (variable, depth) combination, extracts the first prediction from each sample
        to create a continuous time series of T+1 forecasts.
    
        """
        # Handle predictions - extract mean if distribution dict
        if isinstance(preds_row, dict):
            has_distribution = True
            preds_mean = preds_row['mean'].detach().cpu().numpy()
            preds_loc = preds_row['loc'].detach().cpu().numpy()
            preds_scale = preds_row['scale'].detach().cpu().numpy()
            preds_df = preds_row['df'].detach().cpu().numpy()
        else:
            has_distribution = False
            preds_mean = preds_row.detach().cpu().numpy() if hasattr(preds_row, 'detach') else np.asarray(preds_row)
        
        # Convert to numpy
        gt = gt_row.detach().cpu().numpy() if hasattr(gt_row, 'detach') else np.asarray(gt_row)
        time_vals = time_vals_row.detach().cpu().numpy() if hasattr(time_vals_row, 'detach') else np.asarray(time_vals_row)
        depth_vals = depth_vals_row.detach().cpu().numpy() if hasattr(depth_vals_row, 'detach') else np.asarray(depth_vals_row)
        var_ids = var_ids_row.detach().cpu().numpy() if hasattr(var_ids_row, 'detach') else np.asarray(var_ids_row)
        mask = mask_row.detach().cpu().numpy() if hasattr(mask_row, 'detach') else np.asarray(mask_row)
        
        num_samples = gt.shape[0]
        
        # Key: (var_id, depth), Value: list of (time, pred_mean, gt, [lower, upper])
        var_depth_data = {}
        for sample_idx in range(num_samples):
            # Get valid tokens for this sample
            sample_mask = mask[sample_idx].astype(bool)
            valid_idx = np.where(sample_mask)[0]
            
            if valid_idx.size == 0:
                continue
            
            # Extract data for valid tokens
            sample_gt = gt[sample_idx, valid_idx]
            sample_preds = preds_mean[sample_idx, valid_idx] # TODO
            # datetime_strs_row is now a 2D array (B, T), index with sample_idx and valid_idx
            sample_times = datetime_raw_vals[sample_idx, valid_idx]
            sample_depths = depth_vals[sample_idx, valid_idx]
            sample_vars = var_ids[sample_idx, valid_idx]
            
            if has_distribution:
                sample_loc = preds_loc[sample_idx, valid_idx]
                sample_scale = preds_scale[sample_idx, valid_idx]
                sample_df = preds_df[sample_idx, valid_idx]
            
            # If filter_first_pred, only take the first prediction (T+1)
            if filter_first_pred and len(sample_times) > 0:
                # Find the first time step
                min_time = np.min(sample_times)
                first_pred_idx = np.where(sample_times == min_time)[0]
                
                sample_gt = sample_gt[first_pred_idx]
                sample_preds = sample_preds[first_pred_idx]
                sample_times = sample_times[first_pred_idx]
                sample_depths = sample_depths[first_pred_idx]
                sample_vars = sample_vars[first_pred_idx]
                
                if has_distribution:
                    sample_loc = sample_loc[first_pred_idx]
                    sample_scale = sample_scale[first_pred_idx]
                    sample_df = sample_df[first_pred_idx]
            
            # Group by (variable, depth) combination
            for i in range(len(sample_times)):
                var_id = int(sample_vars[i])
                depth = float(sample_depths[i])
                key = (var_id, depth)
                
                if key not in var_depth_data:
                    var_depth_data[key] = []
                
                # Compute confidence intervals if distribution available
                if has_distribution:
                    # Use Student-t distribution to compute confidence intervals
                    loc = sample_loc[i]
                    scale = sample_scale[i]
                    df_val = sample_df[i]
                    
                    # Compute quantiles for confidence interval
                    alpha = 1 - confidence_level
                    t_dist = stats.t(df=df_val, loc=loc, scale=scale)
                    lower = t_dist.ppf(alpha / 2)
                    upper = t_dist.ppf(1 - alpha / 2)
                    
                    var_depth_data[key].append({
                        'time': sample_times[i],
                        'pred': sample_preds[i],
                        'gt': sample_gt[i],
                        'lower': lower,
                        'upper': upper
                    })
                else:
                    var_depth_data[key].append({
                        'time': sample_times[i],
                        'pred': sample_preds[i],
                        'gt': sample_gt[i]
                    })
        
        # Plot each variable
        unique_vars = sorted(set([k[0] for k in var_depth_data.keys()]))
        unique_vars = unique_vars[:max_features]
        
        if len(unique_vars) == 0:
            print("No valid data to plot")
            return
        
        fig, axes = plt.subplots(len(unique_vars), 1, figsize=(14, 4 * len(unique_vars)))
        if len(unique_vars) == 1:
            axes = [axes]
        
        for ax, var_id in zip(axes, unique_vars):
            # Get all depths for this variable
            depths_for_var = sorted(set([k[1] for k in var_depth_data.keys() if k[0] == var_id]))
            depths_for_var = depths_for_var[:max_depths_per_feature]
            # Limit to first 2 and last 2 depths to avoid too many plots
            if len(depths_for_var) > 4:
                depths_for_var = depths_for_var[:1] + depths_for_var[-1:]
            else:
                # If 4 or fewer depths, use all of them (up to max_depths_per_feature)
                depths_for_var = depths_for_var[:max_depths_per_feature]
            
            # Use a better color palette - tab10 or Set2 for better differentiation
            if len(depths_for_var) <= 10:
                color_palette = plt.cm.tab10(np.linspace(0, 1, 10))
                colors = [color_palette[i % 10] for i in range(len(depths_for_var))]
            else:
                colors = plt.cm.viridis(np.linspace(0, 1, len(depths_for_var)))
            
            for color, depth in zip(colors, depths_for_var):
                key = (var_id, depth)
                if key not in var_depth_data:
                    continue
                
                data_points = var_depth_data[key]
                
                # Sort by time
                data_points = sorted(data_points, key=lambda x: x['time'])
                
                times = np.array([d['time'] for d in data_points])
                preds = np.array([d['pred'] for d in data_points])
                gts = np.array([d['gt'] for d in data_points])
                
                # Convert to pandas datetime for proper plotting (date only)
                dates = pd.to_datetime(times).date
                
                # Plot predictions
                if plot_type == 'scatter':
                    ax.scatter(dates, preds, color=color, marker='o', s=30, alpha=0.8,
                             edgecolors='white', linewidths=0.5,
                             label=f"Pred @ depth={depth:.2f}{(' '+depth_units) if depth_units else ''}")
                else:
                    ax.plot(dates, preds, color=color, linewidth=2, markersize=4, linestyle='--',
                           label=f"Pred @ depth={depth:.2f}{(' '+depth_units) if depth_units else ''}")
                
                # Plot ground truth
                if plot_type == 'scatter':
                    ax.scatter(dates, gts, color=color, marker='x', s=30, alpha=0.6,
                             edgecolors='white', linewidths=0.5, label=f"GT @ depth={depth:.2f}")
                else:
                    ax.plot(dates, gts, color=color, linewidth=1.5, linestyle='-',
                           markersize=4, alpha=0.7, label=f"GT @ depth={depth:.2f}")
                
                # Plot confidence intervals if available
                if plot_interval and has_distribution and 'lower' in data_points[0]:
                    lowers = np.array([d['lower'] for d in data_points])
                    uppers = np.array([d['upper'] for d in data_points])
                    ax.fill_between(dates, lowers, uppers, color=color, alpha=0.2,
                                   label=f"{int(confidence_level*100)}% CI @ depth={depth:.2f}")
            
            # Set labels and title
            var_name = feature_dict.get(var_id, f"Var_{var_id}")
            ax.set_ylabel(var_name, fontsize=12)
            ax.set_xlabel('Date', fontsize=12)
            ax.legend(loc='best', fontsize=8, framealpha=0.9)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            
            # Format x-axis
            ax.tick_params(axis='x', rotation=90)
            if len(dates) > 30:
                ax.xaxis.set_major_locator(plt.MaxNLocator(10))
        
        # Add note about confidence intervals if they exist
        ci_note = f" (shaded = {int(confidence_level*100)}% CI)" if has_distribution else ""
        title = f'{title_prefix}: {train_or_val} T+1 Forecasts at epoch {epoch}{ci_note}'
        plt.suptitle(title, fontsize=14, y=1.0)
        plt.tight_layout()
        
        # Save to disk if save_path is provided
        if save_path is not None:
            import os
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            plt.savefig(save_path, dpi=180, bbox_inches='tight')
            print(f"Saved irregular grid forecasts to {save_path}")
        
        wandb.log({title: wandb.Image(plt)})
        plt.close()

    def plot_forecast_irregular_grid_single_depth_var(self,
                                                      gt_row,
                                                      preds_row,
                                                      time_vals_row,
                                                      datetime_raw_vals,
                                                      depth_vals_row,
                                                      var_ids_row,
                                                      mask_row,
                                                      feature_dict,
                                                      sample_idx,
                                                      plt_idx,
                                                      epoch,
                                                      train_or_val,
                                                      title_prefix="T+1 Forecast",
                                                      plot_type='line',
                                                      max_features=6,
                                                      max_depths_per_feature=6,
                                                      filter_first_pred=True,
                                                      depth_units=None,
                                                      var_names_subset=None,
                                                      depth_index=1,
                                                      pred_len=None,
                                                      lake_name=None,
                                                      confidence_level=0.95,
                                                      plot_interval=True,
                                                      save_path=None,
                                                      depth_name=1.5,
                                                      pred_offset=0.0,
                                                      plot_full_timeseries=True,
                                                      show_xticks: bool = True,
                                                      show_xlabel: bool = True,
                                                      ymin=None,
                                                      ymax=None,
                                                      axis_label_fontsize: int = 20,
                                                      tick_labelsize: int = 20,
                                                      ytick_step=None,
                                                      show_title: bool = True,
                                                      show_ylabel: bool = True,
                                                      # Context (input X) data for baseline comparison
                                                      context_seq_row=None,
                                                      context_var_ids_row=None,
                                                      context_depth_vals_row=None,
                                                      context_time_vals_row=None,
                                                      context_datetime_strs=None,
                                                      context_mask_row=None):

        pred_color = "tab:orange"
        gt_color = "tab:blue"
        pred_marker = "o"
        gt_marker = "o"

        # Handle predictions - extract mean if distribution dict
        if isinstance(preds_row, dict):
            has_distribution = True
            preds_mean = preds_row['mean'].detach().cpu().numpy()
            preds_loc = preds_row['loc'].detach().cpu().numpy()
            preds_scale = preds_row['scale'].detach().cpu().numpy()
            preds_df = preds_row['df'].detach().cpu().numpy()
        else:
            has_distribution = False
            preds_mean = preds_row.detach().cpu().numpy() if hasattr(preds_row, 'detach') else np.asarray(preds_row)
        
        # Convert to numpy
        gt = gt_row.detach().cpu().numpy() if hasattr(gt_row, 'detach') else np.asarray(gt_row)
        depth_vals = depth_vals_row.detach().cpu().numpy() if hasattr(depth_vals_row, 'detach') else np.asarray(depth_vals_row)
        var_ids = var_ids_row.detach().cpu().numpy() if hasattr(var_ids_row, 'detach') else np.asarray(var_ids_row)
        mask = mask_row.detach().cpu().numpy() if hasattr(mask_row, 'detach') else np.asarray(mask_row)
        
        num_samples = gt.shape[0]
        
        var_depth_data = {}
        for sidx in range(num_samples):
            sample_mask = mask[sidx].astype(bool)
            valid_idx = np.where(sample_mask)[0]
            if valid_idx.size == 0:
                continue
            
            sample_gt = gt[sidx, valid_idx]
            sample_preds = preds_mean[sidx, valid_idx]
            sample_times = datetime_raw_vals[sidx, valid_idx]
            sample_depths = depth_vals[sidx, valid_idx]
            sample_vars = var_ids[sidx, valid_idx]
            
            if has_distribution:
                sample_loc = preds_loc[sidx, valid_idx]
                sample_scale = preds_scale[sidx, valid_idx]
                sample_df = preds_df[sidx, valid_idx]

            if filter_first_pred and len(sample_times) > 0:
                if isinstance(sample_times[0], bytes):
                    times_str = np.array([t.decode("utf-8") for t in sample_times], dtype=object)
                else:
                    times_str = np.array([str(t) for t in sample_times], dtype=object)
                times_dt = pd.to_datetime(times_str)

                best_idx_by_key = {}
                for j in range(len(sample_times)):
                    key = (int(sample_vars[j]), float(sample_depths[j]))
                    if key not in best_idx_by_key:
                        best_idx_by_key[key] = j
                    else:
                        if times_dt[j] < times_dt[best_idx_by_key[key]]:
                            best_idx_by_key[key] = j

                keep_idx = np.array(sorted(best_idx_by_key.values()), dtype=int)

                sample_gt = sample_gt[keep_idx]
                sample_preds = sample_preds[keep_idx]
                sample_times = sample_times[keep_idx]
                sample_depths = sample_depths[keep_idx]
                sample_vars = sample_vars[keep_idx]

                if has_distribution:
                    sample_loc = sample_loc[keep_idx]
                    sample_scale = sample_scale[keep_idx]
                    sample_df = sample_df[keep_idx]

                pass
            
            for i in range(len(sample_times)):
                var_id = int(sample_vars[i])
                depth = float(sample_depths[i])
                key = (var_id, depth)
                if key not in var_depth_data:
                    var_depth_data[key] = []
                
                if has_distribution:
                    loc = sample_loc[i]
                    scale = sample_scale[i]
                    df_val = sample_df[i]
                    
                    alpha = 1 - confidence_level
                    t_dist = stats.t(df=df_val, loc=loc, scale=scale)
                    lower = t_dist.ppf(alpha / 2)
                    upper = t_dist.ppf(1 - alpha / 2)
                    
                    var_depth_data[key].append({
                        'time': sample_times[i],
                        'pred': sample_preds[i],
                        'gt': sample_gt[i],
                        'lower': lower,
                        'upper': upper,
                        'sample_idx': sidx  # Track which sample this prediction came from
                    })
                else:
                    var_depth_data[key].append({
                        'time': sample_times[i],
                        'pred': sample_preds[i],
                        'gt': sample_gt[i],
                        'sample_idx': sidx  # Track which sample this prediction came from
                    })
        
        unique_vars = sorted(set([k[0] for k in var_depth_data.keys()]))
        
        if var_names_subset is None:
            var_names_subset = ["WaterTemp_C"]
        
        if var_names_subset is not None:
            name_to_id = {vname: int(vid) for vid, vname in feature_dict.items()}
            selected_var_ids = [name_to_id[name] for name in var_names_subset if name in name_to_id]
            if len(selected_var_ids) > 0:
                unique_vars = [vid for vid in unique_vars if vid in set(selected_var_ids)]
        
        unique_vars = unique_vars[:max_features]
        
        if len(unique_vars) == 0:
            print("No valid data to plot")
            return
        
        fig, axes = plt.subplots(len(unique_vars), 1, figsize=(10, 4 * len(unique_vars)))
        if len(unique_vars) == 1:
            axes = [axes]

        depth_for_title = None
        
        for ax, var_id in zip(axes, unique_vars):
            depths_for_var = sorted(set([k[1] for k in var_depth_data.keys() if k[0] == var_id]))
            if len(depths_for_var) == 0:
                continue
            requested_depth_m = None
            du = (str(depth_units).lower() if depth_units is not None else "")
            if du in ("m", "meter", "meters"):
                try:
                    requested_depth_m = float(depth_name)
                except Exception:
                    requested_depth_m = None

            if requested_depth_m is not None:
                chosen = min(depths_for_var, key=lambda d: abs(float(d) - requested_depth_m))
                depths_for_var = [chosen]
            else:
                di = int(depth_index)
                if di < 0:
                    di = 0
                if di >= len(depths_for_var):
                    di = len(depths_for_var) - 1
                depths_for_var = [depths_for_var[di]]

            if depth_for_title is None and len(depths_for_var) > 0:
                depth_for_title = float(depths_for_var[0])
            
            for depth in depths_for_var:
                key = (var_id, depth)
                if key not in var_depth_data:
                    continue
                
                data_points = sorted(var_depth_data[key], key=lambda x: x['time'])
                times = np.array([d['time'] for d in data_points])
                preds = np.array([d['pred'] for d in data_points])
                gts = np.array([d['gt'] for d in data_points])
                times_dt = pd.to_datetime(times, utc=True)
                order = np.argsort(times_dt.astype("int64"))
                times_dt = times_dt[order]
                preds = preds[order]
                gts = gts[order]
                data_points = [data_points[i] for i in order.tolist()]

                # Deduplicate by day (keep first occurrence per day)
                dti = pd.DatetimeIndex(times_dt)
                if dti.tz is not None:
                    dti = dti.tz_convert("UTC").tz_localize(None)
                day = dti.normalize().to_numpy(dtype="datetime64[ns]")
                _, first_idx = np.unique(day, return_index=True)
                first_idx = np.sort(first_idx)
                times_dt = times_dt[first_idx]
                preds = preds[first_idx]
                gts = gts[first_idx]
                data_points = [data_points[i] for i in first_idx.tolist()]

                if not plot_full_timeseries:
                    try:
                        pl = int(pred_len)
                    except Exception:
                        pl = None
                    if pl is not None and pl > 0:
                        times_dt = times_dt[:pl]
                        preds = preds[:pl]
                        gts = gts[:pl]
                        data_points = data_points[:pl]
                
                if context_seq_row is not None and save_path is not None:
                    self._save_context_for_predictions(
                        data_points=data_points,
                        context_seq_row=context_seq_row,
                        context_var_ids_row=context_var_ids_row,
                        context_depth_vals_row=context_depth_vals_row,
                        context_time_vals_row=context_time_vals_row,
                        context_datetime_strs=context_datetime_strs,
                        context_mask_row=context_mask_row,
                        var_id=var_id,
                        depth=depth,
                        lake_name=lake_name,
                        save_path=save_path
                    )

                dates = pd.DatetimeIndex(times_dt).date

                # Apply vertical offset to predictions if specified (useful for visual comparison)
                preds_offset = preds + pred_offset

                # Depth label: show chosen depth, and requested depth (if provided) to avoid confusion.
                du = (str(depth_units).lower() if depth_units is not None else "")
                if requested_depth_m is not None and du in ("m", "meter", "meters"):
                    depth_label = f"{float(depth):.2f}m"
                else:
                    depth_label = f"{float(depth):.4f}{(' '+depth_units) if depth_units else ''}"
                
                if plot_type == 'scatter':
                    ax.scatter(dates, preds_offset, color=pred_color, marker=pred_marker, s=34, alpha=0.85,
                               edgecolors='white', linewidths=0.5,
                               label=f"Pred @ depth={depth_label}")
                else:
                    ax.plot(dates, preds_offset, color=pred_color, linewidth=2, markersize=8.5, linestyle='--',
                            marker=pred_marker,
                            label=f"Pred @ depth={depth_label}")
                
                if plot_type == 'scatter':
                    ax.scatter(dates, gts, color=gt_color, marker=gt_marker, s=34, alpha=0.85,
                               edgecolors='white', linewidths=0.5, label=f"GT @ depth={depth_label}")
                else:
                    ax.plot(dates, gts, color=gt_color, linewidth=2, linestyle='-',
                            marker=gt_marker, markersize=8.5, alpha=0.9, label=f"GT @ depth={depth_label}")

                if plot_interval and has_distribution and 'lower' in data_points[0]:
                    lowers = np.array([d['lower'] for d in data_points])
                    uppers = np.array([d['upper'] for d in data_points])
                    ax.fill_between(dates, lowers + pred_offset, uppers + pred_offset, color=pred_color, alpha=0.2,
                                    label=f"{int(confidence_level*100)}% CI @ depth={depth_label}")
            
            var_name = feature_dict.get(var_id, f"Var_{var_id}")
            if show_ylabel:
                ax.set_ylabel(var_name, fontsize=int(axis_label_fontsize))
            else:
                ax.set_ylabel('')

            if ymin is not None or ymax is not None:
                try:
                    y0, y1 = ax.get_ylim()
                    y0 = float(ymin) if ymin is not None else y0
                    y1 = float(ymax) if ymax is not None else y1
                    if y0 < y1:
                        ax.set_ylim(y0, y1)
                except Exception:
                    pass

            if ytick_step is not None and (ymin is not None or ymax is not None):
                try:
                    step = float(ytick_step)
                    if step > 0:
                        y0, y1 = ax.get_ylim()
                        eps = 1e-9
                        ticks = np.arange(float(y0), float(y1) + eps, step, dtype=float)
                        if ticks.size >= 2:
                            ax.set_yticks(ticks)
                except Exception:
                    pass

            if show_xlabel:
                ax.set_xlabel('Date', fontsize=int(axis_label_fontsize))
            else:
                ax.set_xlabel('')
            ax.legend(loc='best', fontsize=8, framealpha=0.9)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            ax.tick_params(axis='y', labelsize=int(tick_labelsize))
            if show_xticks:
                ax.tick_params(axis='x', rotation=90, labelsize=int(tick_labelsize))
                if len(dates) > 30:
                    ax.xaxis.set_major_locator(plt.MaxNLocator(10))
            else:
                ax.set_xticks([])
                ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        
        lake_str = lake_name if lake_name is not None else "Lake"
        pred_len_str = str(pred_len) if pred_len is not None else "?"
        ci_note = f" (shaded = {int(confidence_level*100)}% CI)" if has_distribution else ""
        du = (str(depth_units).lower() if depth_units is not None else "")
        if depth_for_title is None:
            depth_title_str = "?"
        elif du in ("m", "meter", "meters"):
            try:
                req = float(depth_name)
                depth_title_str = f"{float(depth_for_title):.2f}m (req={req:.2f}m)"
            except Exception:
                depth_title_str = f"{float(depth_for_title):.2f}m"
        else:
            depth_title_str = f"{float(depth_for_title):.4f}{(' '+depth_units) if depth_units else ''}"
        offset_note = f" (pred_offset={float(pred_offset):g})" if float(pred_offset) != 0.0 else ""
        title = f"{lake_str} at Depth {depth_title_str}{ci_note}{offset_note}"
        if show_title:
            plt.suptitle(title, fontsize=14, y=1.0)
        plt.tight_layout()
        
        if save_path is not None:
            import os
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            plt.savefig(save_path, dpi=180, bbox_inches='tight')
            print(f"Saved irregular grid forecasts to {save_path}")
        
        # Use a stable, non-empty W&B key even if we suppress the figure title.
        wandb_key = title if (isinstance(title, str) and title.strip()) else f"{title_prefix}: {train_or_val} (no_title)"
        wandb.log({wandb_key: wandb.Image(plt)})
        plt.close()

    def plot_forecast_irregular_grid_single_depth_var_multi_horizon(self,
                                                                    gt_row,
                                                                    preds_row,
                                                                    time_vals_row,
                                                                    datetime_raw_vals,
                                                                    depth_vals_row,
                                                                    var_ids_row,
                                                                    mask_row,
                                                                    feature_dict,
                                                                    sample_idx,
                                                                    plt_idx,
                                                                    epoch,
                                                                    train_or_val,
                                                                    title_prefix="Multi-horizon Forecast",
                                                                    plot_type='line',
                                                                    max_features=6,
                                                                    max_depths_per_feature=6,
                                                                    filter_first_pred=False,
                                                                    depth_units=None,
                                                                    var_names_subset=None,
                                                                    depth_index=1,
                                                                    pred_len=None,
                                                                    lake_name=None,
                                                                    confidence_level=0.95,
                                                                    plot_interval=True,
                                                                    save_path=None,
                                                                    depth_name=1.5,
                                                                    pred_offset=0.0,
                                                                    horizons=(1, 7, 14, 21),
                                                                    use_common_intersection=False,
                                                                    intersection_mode="range",
                                                                    filename_prefix="t+N_preds",
                                                                    # Context (input X) data for baseline comparison (accepted for call-site compatibility)
                                                                    context_seq_row=None,
                                                                    context_var_ids_row=None,
                                                                    context_depth_vals_row=None,
                                                                    context_time_vals_row=None,
                                                                    context_datetime_strs=None,
                                                                    context_mask_row=None):
        pred_color = "tab:orange"
        gt_color = "tab:blue"
        pred_marker = "o"
        gt_marker = "o"

        # Handle predictions - extract mean if distribution dict
        if isinstance(preds_row, dict):
            has_distribution = True
            preds_mean = preds_row['mean'].detach().cpu().numpy()
            preds_loc = preds_row['loc'].detach().cpu().numpy()
            preds_scale = preds_row['scale'].detach().cpu().numpy()
            preds_df = preds_row['df'].detach().cpu().numpy()
        else:
            has_distribution = False
            preds_mean = preds_row.detach().cpu().numpy() if hasattr(preds_row, 'detach') else np.asarray(preds_row)

        # Convert to numpy
        gt = gt_row.detach().cpu().numpy() if hasattr(gt_row, 'detach') else np.asarray(gt_row)
        depth_vals = depth_vals_row.detach().cpu().numpy() if hasattr(depth_vals_row, 'detach') else np.asarray(depth_vals_row)
        var_ids = var_ids_row.detach().cpu().numpy() if hasattr(var_ids_row, 'detach') else np.asarray(var_ids_row)
        mask = mask_row.detach().cpu().numpy() if hasattr(mask_row, 'detach') else np.asarray(mask_row)

        num_samples = gt.shape[0]

        # Collect points per (var, depth)
        var_depth_data = {}
        for sidx in range(num_samples):
            sample_mask = mask[sidx].astype(bool)
            valid_idx = np.where(sample_mask)[0]
            if valid_idx.size == 0:
                continue

            sample_gt = gt[sidx, valid_idx]
            sample_preds = preds_mean[sidx, valid_idx]
            sample_times = datetime_raw_vals[sidx, valid_idx]
            sample_depths = depth_vals[sidx, valid_idx]
            sample_vars = var_ids[sidx, valid_idx]

            if has_distribution:
                sample_loc = preds_loc[sidx, valid_idx]
                sample_scale = preds_scale[sidx, valid_idx]
                sample_df = preds_df[sidx, valid_idx]

            if filter_first_pred and len(sample_times) > 0:
                if isinstance(sample_times[0], bytes):
                    times_str = np.array([t.decode("utf-8") for t in sample_times], dtype=object)
                else:
                    times_str = np.array([str(t) for t in sample_times], dtype=object)
                times_dt = pd.to_datetime(times_str)

                best_idx_by_key = {}
                for j in range(len(sample_times)):
                    key = (int(sample_vars[j]), float(sample_depths[j]))
                    if key not in best_idx_by_key:
                        best_idx_by_key[key] = j
                    else:
                        if times_dt[j] < times_dt[best_idx_by_key[key]]:
                            best_idx_by_key[key] = j

                keep_idx = np.array(sorted(best_idx_by_key.values()), dtype=int)

                sample_gt = sample_gt[keep_idx]
                sample_preds = sample_preds[keep_idx]
                sample_times = sample_times[keep_idx]
                sample_depths = sample_depths[keep_idx]
                sample_vars = sample_vars[keep_idx]

                if has_distribution:
                    sample_loc = sample_loc[keep_idx]
                    sample_scale = sample_scale[keep_idx]
                    sample_df = sample_df[keep_idx]

            for i in range(len(sample_times)):
                var_id = int(sample_vars[i])
                depth = float(sample_depths[i])
                key = (var_id, depth)
                if key not in var_depth_data:
                    var_depth_data[key] = []

                if has_distribution:
                    loc = sample_loc[i]
                    scale = sample_scale[i]
                    df_val = sample_df[i]

                    alpha = 1 - confidence_level
                    t_dist = stats.t(df=df_val, loc=loc, scale=scale)
                    lower = t_dist.ppf(alpha / 2)
                    upper = t_dist.ppf(1 - alpha / 2)

                    var_depth_data[key].append({
                        'time': sample_times[i],
                        'pred': sample_preds[i],
                        'gt': sample_gt[i],
                        'lower': lower,
                        'upper': upper,
                        'sample_idx': sidx
                    })
                else:
                    var_depth_data[key].append({
                        'time': sample_times[i],
                        'pred': sample_preds[i],
                        'gt': sample_gt[i],
                        'sample_idx': sidx
                    })

        if len(var_depth_data) == 0:
            print("No valid data to plot")
            return

        unique_vars = sorted(set([k[0] for k in var_depth_data.keys()]))
        if var_names_subset is None:
            var_names_subset = ["WaterTemp_C"]

        if var_names_subset is not None:
            name_to_id = {vname: int(vid) for vid, vname in feature_dict.items()}
            selected_var_ids = [name_to_id[name] for name in var_names_subset if name in name_to_id]
            if len(selected_var_ids) > 0:
                unique_vars = [vid for vid in unique_vars if vid in set(selected_var_ids)]

        unique_vars = unique_vars[:max_features]
        if len(unique_vars) == 0:
            print("No valid data to plot")
            return

        var_id = unique_vars[0]
        depths_for_var = sorted(set([k[1] for k in var_depth_data.keys() if k[0] == var_id]))
        if len(depths_for_var) == 0:
            print("No valid depths to plot")
            return

        di = int(depth_index)
        if di < 0:
            di = 0
        if di >= len(depths_for_var):
            di = len(depths_for_var) - 1
        depth = float(depths_for_var[di])

        key = (var_id, depth)
        points = var_depth_data.get(key, [])
        if len(points) == 0:
            print("No valid data points for selected (var, depth)")
            return

        def _to_times_dt(times):
            if len(times) == 0:
                return pd.to_datetime([], utc=True)
            if isinstance(times[0], bytes):
                times_str = [t.decode("utf-8") for t in times]
            else:
                times_str = [str(t) for t in times]
            return pd.to_datetime(np.array(times_str, dtype=object), utc=True)

        def _dedupe_by_day(times_dt, idxs):
            dti = pd.DatetimeIndex(times_dt)
            if dti.tz is not None:
                dti = dti.tz_convert("UTC").tz_localize(None)
            day = dti.normalize().to_numpy(dtype="datetime64[ns]")
            _, first_idx = np.unique(day, return_index=True)
            first_idx = np.sort(first_idx)
            return idxs[first_idx]

        def _series_for_horizon(points_list, h):
            # Group by forecast window (sample_idx)
            by_sample = {}
            for p in points_list:
                by_sample.setdefault(int(p['sample_idx']), []).append(p)

            chosen = []
            for sidx, lst in by_sample.items():
                times_dt = _to_times_dt([x['time'] for x in lst])
                if len(times_dt) == 0:
                    continue
                order = np.argsort(pd.DatetimeIndex(times_dt).astype("int64")).astype(int)

                # Dedupe by day within this window, then pick h-th element
                order = _dedupe_by_day(times_dt[order], order)
                if len(order) < int(h):
                    continue
                chosen.append(lst[int(order[int(h) - 1])])

            if len(chosen) == 0:
                return None

            times_dt = _to_times_dt([x['time'] for x in chosen])
            order = np.argsort(pd.DatetimeIndex(times_dt).astype("int64")).astype(int)
            times_dt = times_dt[order]
            chosen = [chosen[i] for i in order.tolist()]

            idxs = np.arange(len(chosen), dtype=int)
            idxs = _dedupe_by_day(times_dt, idxs)
            times_dt = times_dt[idxs]
            chosen = [chosen[i] for i in idxs.tolist()]

            dates = pd.DatetimeIndex(times_dt).date
            preds = np.array([x['pred'] for x in chosen]) + pred_offset
            gts = np.array([x['gt'] for x in chosen])

            lowers = uppers = None
            if has_distribution and plot_interval and 'lower' in chosen[0]:
                lowers = np.array([x['lower'] for x in chosen]) + pred_offset
                uppers = np.array([x['upper'] for x in chosen]) + pred_offset

            return dates, preds, gts, lowers, uppers

        # Build series for each horizon
        series_by_h = {}
        for h in horizons:
            out = _series_for_horizon(points, int(h))
            if out is not None:
                series_by_h[int(h)] = out

        if len(series_by_h) == 0:
            print("No valid horizon series to plot")
            return

        if use_common_intersection and len(series_by_h) > 1:
            mode = str(intersection_mode or "range").lower()
            if mode not in {"range", "set"}:
                mode = "range"

            if mode == "range":
                starts = []
                ends = []
                for h, (dates, *_rest) in series_by_h.items():
                    if dates is None or len(dates) == 0:
                        continue
                    d = pd.to_datetime(np.array(dates, dtype=object)).normalize()
                    starts.append(d.min())
                    ends.append(d.max())
                if len(starts) > 0 and len(ends) > 0:
                    start = max(starts)
                    end = min(ends)
                    if start <= end:
                        for h, (dates, preds, gts, lowers, uppers) in list(series_by_h.items()):
                            if dates is None or len(dates) == 0:
                                continue
                            d = pd.to_datetime(np.array(dates, dtype=object)).normalize()
                            keep = (d >= start) & (d <= end)
                            keep = np.asarray(keep, dtype=bool)
                            series_by_h[h] = (
                                np.asarray(dates, dtype=object)[keep],
                                np.asarray(preds)[keep],
                                np.asarray(gts)[keep],
                                (np.asarray(lowers)[keep] if lowers is not None else None),
                                (np.asarray(uppers)[keep] if uppers is not None else None),
                            )
            else:
                date_sets = [set(map(str, series_by_h[h][0])) for h in series_by_h.keys()]
                common = set.intersection(*date_sets) if len(date_sets) > 0 else set()
                if len(common) > 0:
                    for h, (dates, preds, gts, lowers, uppers) in list(series_by_h.items()):
                        keep = np.array([str(d) in common for d in dates], dtype=bool)
                        series_by_h[h] = (
                            np.asarray(dates, dtype=object)[keep],
                            np.asarray(preds)[keep],
                            np.asarray(gts)[keep],
                            (np.asarray(lowers)[keep] if lowers is not None else None),
                            (np.asarray(uppers)[keep] if uppers is not None else None),
                        )

        requested_order = [1, 7, 14, 21]
        hs = [h for h in requested_order if h in series_by_h]
        for h in sorted(series_by_h.keys()):
            if h not in hs:
                hs.append(h)

        fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
        axes = axes.flatten()

        lake_str = lake_name if lake_name is not None else "Lake"
        var_name = feature_dict.get(var_id, f"Var_{var_id}")
        ci_note = f" (shaded = {int(confidence_level*100)}% CI)" if (has_distribution and plot_interval) else ""
        title = f"{lake_str}: {var_name} @ depth={depth_name:.2f}m{ci_note}"
        plt.suptitle(title, fontsize=14, y=1.02)

        for ax_i in range(4):
            ax = axes[ax_i]
            if ax_i >= len(hs):
                ax.axis("off")
                continue

            h = hs[ax_i]
            dates, preds, gts, lowers, uppers = series_by_h[h]

            if plot_type == 'scatter':
                ax.scatter(dates, preds, color=pred_color, marker=pred_marker, s=28, alpha=0.85,
                           edgecolors='white', linewidths=0.5, label="Pred")
                ax.scatter(dates, gts, color=gt_color, marker=gt_marker, s=28, alpha=0.85,
                           edgecolors='white', linewidths=0.5, label="GT")
            else:
                ax.plot(dates, preds, color=pred_color, linewidth=2, linestyle='--',
                        marker=pred_marker, markersize=8.5, label="Pred")
                ax.plot(dates, gts, color=gt_color, linewidth=2, linestyle='-',
                        marker=gt_marker, markersize=8.5, alpha=0.9, label="GT")

            if lowers is not None and uppers is not None and len(dates) > 0:
                ax.fill_between(dates, lowers, uppers, color=pred_color, alpha=0.2, label="CI")

            ax.set_title(f"T+{h} timestep ahead prediction", fontsize=12)
            # Y label: variable name (same var across all horizons in this plot)
            ax.set_ylabel(var_name, fontsize=11)

            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            # Bigger tick labels + tick marks
            ax.tick_params(axis='x', rotation=90, labelsize=11, length=6, width=1.2)
            ax.tick_params(axis='y', labelsize=11, length=6, width=1.2)
            ax.legend(loc='best', fontsize=8, framealpha=0.9)

        plt.tight_layout()

        # Save with prefix if requested
        if save_path is not None:
            import os
            import json
            save_dir = os.path.dirname(save_path)
            base = os.path.basename(save_path)
            if base == "":
                base = "plot.png"
            if not base.lower().endswith((".png", ".jpg", ".jpeg", ".pdf")):
                base = base + ".png"
            prefixed = f"{filename_prefix}_{base}" if filename_prefix else base
            final_path = os.path.join(save_dir, prefixed) if save_dir else prefixed
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            plt.savefig(final_path, dpi=180, bbox_inches='tight')
            print(f"Saved multi-horizon irregular forecasts to {final_path}")
            
            # Save predictions and dates for each horizon
            try:
                predictions_data = {
                    'lake_name': str(lake_name) if lake_name is not None else 'unknown',
                    'variable_id': int(var_id),
                    'variable_name': str(var_name),
                    'depth': float(depth),
                    'depth_name': float(depth_name),
                    'horizons': {},
                    'metadata': {
                        'pred_offset': float(pred_offset),
                        'confidence_level': float(confidence_level),
                        'has_distribution': bool(has_distribution),
                        'plot_interval': bool(plot_interval),
                        'use_common_intersection': bool(use_common_intersection),
                        'intersection_mode': str(intersection_mode),
                    }
                }
                
                for h, (dates, preds, gts, lowers, uppers) in series_by_h.items():
                    # Convert dates to strings for JSON serialization
                    dates_str = [str(d) for d in dates] if dates is not None and len(dates) > 0 else []
                    
                    horizon_data = {
                        'dates': dates_str,
                        'predictions': preds.tolist() if preds is not None and len(preds) > 0 else [],
                        'ground_truth': gts.tolist() if gts is not None and len(gts) > 0 else [],
                        'num_points': len(dates_str),
                    }
                    
                    # Add confidence intervals if available
                    if lowers is not None and uppers is not None:
                        horizon_data['lower_bound'] = lowers.tolist()
                        horizon_data['upper_bound'] = uppers.tolist()
                    
                    predictions_data['horizons'][f'T+{h}'] = horizon_data
                
                # Save to JSON file
                json_base = os.path.splitext(base)[0]  # Remove extension
                json_filename = f"{filename_prefix}_{json_base}_predictions.json" if filename_prefix else f"{json_base}_predictions.json"
                json_path = os.path.join(save_dir, json_filename) if save_dir else json_filename
                
                with open(json_path, 'w') as f:
                    json.dump(predictions_data, f, indent=2)
                
                print(f"Saved multi-horizon predictions and dates to {json_path}")
                print(f"  Saved data for {len(predictions_data['horizons'])} horizon(s): {list(predictions_data['horizons'].keys())}")
                
            except Exception as e:
                print(f"Warning: Failed to save predictions and dates data: {str(e)}")
                import traceback
                traceback.print_exc()

        wandb.log({title: wandb.Image(plt)})
        plt.close()
    
    def plot_single_sample_forecast(self,
                                    inputs_row,
                                    gt_row,
                                    preds_row,
                                    time_vals_row,
                                    datetime_raw_vals,
                                    depth_vals_row,
                                    var_ids_row,
                                    mask_row,
                                    feature_dict,
                                    sample_idx,
                                    var_name=None,
                                    target_depth=0.0,
                                    context_len=None,
                                    horizons=[14, 21, 30],
                                    confidence_level=0.95,
                                    epoch=None,
                                    train_or_val=None,
                                    lake_name=None,
                                    save_path=None,
                                    depth_name="1.5"):
        """
        Plot context window, ground truth, and predictions for a single sample at a specific depth.
        Shows T+1, T+7, T+14, T+30 predictions in separate rows with uncertainty bands.
        Plots the irregular sequence directly without regular grid assumptions.
        """
        # Handle predictions - extract mean and distribution params if available
        if isinstance(preds_row, dict):
            has_distribution = True
            preds_mean = preds_row['mean'].detach().cpu().numpy()
            preds_loc = preds_row['loc'].detach().cpu().numpy()
            preds_scale = preds_row['scale'].detach().cpu().numpy()
            preds_df = preds_row['df'].detach().cpu().numpy()
        else:
            has_distribution = False
            preds_mean = preds_row.detach().cpu().numpy() if hasattr(preds_row, 'detach') else np.asarray(preds_row)
        
        # Convert to numpy
        inputs = inputs_row.detach().cpu().numpy() if hasattr(inputs_row, 'detach') else np.asarray(inputs_row)
        gt = gt_row.detach().cpu().numpy() if hasattr(gt_row, 'detach') else np.asarray(gt_row)
        depth_vals = depth_vals_row.detach().cpu().numpy() if hasattr(depth_vals_row, 'detach') else np.asarray(depth_vals_row)
        var_ids = var_ids_row.detach().cpu().numpy() if hasattr(var_ids_row, 'detach') else np.asarray(var_ids_row)
        mask = mask_row.detach().cpu().numpy() if hasattr(mask_row, 'detach') else np.asarray(mask_row)
        
        # Get data for the specified sample
        sample_mask = mask[sample_idx].astype(bool)
        valid_idx = np.where(sample_mask)[0]
        
        # Extract data for valid tokens
        sample_inputs = inputs[sample_idx, valid_idx]
        sample_gt = gt[sample_idx, valid_idx]
        sample_preds = preds_mean[sample_idx, valid_idx]
        sample_depths = depth_vals[sample_idx, valid_idx]
        sample_vars = var_ids[sample_idx, valid_idx]
        sample_datetimes = datetime_raw_vals[sample_idx, valid_idx]
        
        if has_distribution:
            sample_loc = preds_loc[sample_idx, valid_idx]
            sample_scale = preds_scale[sample_idx, valid_idx]
            sample_df = preds_df[sample_idx, valid_idx]
        
        # Find variable ID
        target_var_id = None
        if var_name is not None:
            for vid, vname in feature_dict.items():
                if vname == var_name:
                    target_var_id = vid
                    break
        
        if target_var_id is None:
            unique_vars = np.unique(sample_vars)
            target_var_id = int(unique_vars[0])
        
        # Filter by target variable and depth
        var_mask = sample_vars == target_var_id
        depth_mask = np.abs(sample_depths - target_depth) < 0.1
        combined_mask = var_mask & depth_mask
        
        # Extract filtered data
        dts_raw = sample_datetimes[combined_mask]
        inputs_filtered = sample_inputs[combined_mask]
        gt_filtered = sample_gt[combined_mask]
        preds_filtered = sample_preds[combined_mask]
        
        if has_distribution:
            loc_filtered = sample_loc[combined_mask]
            scale_filtered = sample_scale[combined_mask]
            df_filtered = sample_df[combined_mask]
        
        # Convert datetimes to pandas datetime for sorting
        try:
            # Decode bytes to strings if necessary
            if len(dts_raw) > 0 and isinstance(dts_raw[0], bytes):
                dts_raw = np.array([dt.decode('utf-8') if isinstance(dt, bytes) else str(dt) for dt in dts_raw])
        
            dts = pd.to_datetime(dts_raw)
            if isinstance(dts, pd.DatetimeIndex):
                dts_series = pd.Series(dts)
            else:
                dts_series = dts
        except Exception:
            print("Error parsing datetimes")
            return
        
        # Sort by datetime to get temporal sequence
        order = np.argsort(dts.values)
        dates_plot = dts_series.iloc[order].dt.date.values if hasattr(dts_series, 'iloc') else pd.Series(dts)[order].dt.date.values
        inputs_sorted = inputs_filtered[order]
        gt_sorted = gt_filtered[order]
        preds_sorted = preds_filtered[order]
        
        if has_distribution:
            loc_sorted = loc_filtered[order]
            scale_sorted = scale_filtered[order]
            df_sorted = df_filtered[order]
        
        # Determine context length
        if context_len is None:
            context_len = len(inputs_sorted) // 2
        
        context_end_idx = min(context_len, len(inputs_sorted))
        
        # Create figure with subplots for each horizon
        num_horizons = len(horizons)
        fig, axes = plt.subplots(num_horizons, 1, figsize=(16, 4 * num_horizons))
        if num_horizons == 1:
            axes = [axes]
        
        var_display_name = feature_dict.get(target_var_id, f"Variable {target_var_id}")
        
        # Context window data
        dates_context = dates_plot[:context_end_idx]
        inputs_context = inputs_sorted[:context_end_idx]
        gt_context = gt_sorted[:context_end_idx]
        
        # Prediction window data (after context)
        dates_pred = dates_plot[context_end_idx:]
        preds_pred = preds_sorted[context_end_idx:]
        gt_pred = gt_sorted[context_end_idx:]
        
        if has_distribution:
            loc_pred = loc_sorted[context_end_idx:]
            scale_pred = scale_sorted[context_end_idx:]
            df_pred = df_sorted[context_end_idx:]
        
        for hi, horizon in enumerate(sorted(horizons)):
            ax = axes[hi]
            
            # Position in prediction window = horizon - 1 (T+1 -> position 0, T+7 -> position 6, etc.)
            pos_in_pred_window = horizon - 1
            
            if pos_in_pred_window >= len(preds_pred):
                ax.text(0.5, 0.5, f'No data for T+{horizon} (prediction window too short)', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"T+{horizon} - {var_display_name} @ depth {target_depth:.2f}")
                continue
            
            ax.plot(dates_context, inputs_context, 'o-', color='gray', linewidth=1.5, 
                   markersize=3, alpha=0.7, label='Context (Input)', zorder=1)
            ax.plot(dates_context, gt_context, 's-', color='blue', linewidth=1.5, 
                   markersize=3, alpha=0.7, label='Context (GT)', zorder=1)
            
            ax.plot(dates_pred, gt_pred, 's-', color='lightblue', linewidth=1, 
                   markersize=2, alpha=0.5, label='Prediction Window (GT)', zorder=1)
            
            pred_date = dates_pred[pos_in_pred_window]
            pred_val = preds_pred[pos_in_pred_window]
            gt_val_at_horizon = gt_pred[pos_in_pred_window]
            
            ax.plot(pred_date, pred_val, 'o', color='red', markersize=10, 
                   label=f'T+{horizon} Prediction', zorder=3)
            ax.plot(pred_date, gt_val_at_horizon, 's', color='green', markersize=10, 
                   label=f'T+{horizon} GT', zorder=3)
            
            # Plot uncertainty band if available
            if has_distribution and pos_in_pred_window < len(loc_pred):
                loc_val = loc_pred[pos_in_pred_window]
                scale_val = scale_pred[pos_in_pred_window]
                df_val = df_pred[pos_in_pred_window]
                
                # Compute confidence interval
                alpha = 1 - confidence_level
                t_dist = stats.t(df=df_val, loc=loc_val, scale=scale_val)
                lower = t_dist.ppf(alpha / 2)
                upper = t_dist.ppf(1 - alpha / 2)
                
                # Plot uncertainty band as vertical error bar
                ax.errorbar(pred_date, pred_val, yerr=[[pred_val - lower], [upper - pred_val]], 
                           fmt='none', color='red', linewidth=2, capsize=5, capthick=2,
                           alpha=0.6, label=f'{int(confidence_level*100)}% CI', zorder=2)
                # Also fill between for visibility
                ax.fill_between([pred_date, pred_date], [lower, upper], 
                               color='red', alpha=0.15, zorder=2)
            
            # Draw vertical dotted line at context/prediction boundary
            if context_end_idx > 0 and context_end_idx < len(dates_plot):
                boundary_date = dates_plot[context_end_idx - 1]
                ax.axvline(x=boundary_date, color='black', linestyle='--', linewidth=2, 
                          alpha=0.7, zorder=2)
                # Add label only for first subplot
                if hi == 0:
                    ax.text(boundary_date, ax.get_ylim()[1] * 0.95, 'Context/Prediction Boundary',
                           rotation=90, verticalalignment='top', fontsize=8, alpha=0.7)
            
            # Set labels and title
            ax.set_ylabel(f'{var_display_name}', fontsize=12)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_title(f"T+{horizon} - {var_display_name} @ depth {target_depth:.2f}")
            ax.legend(loc='best', fontsize=8, framealpha=0.9, ncol=2)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            
            # Format x-axis
            ax.tick_params(axis='x', rotation=45)
            if len(dates_plot) > 30:
                ax.xaxis.set_major_locator(plt.MaxNLocator(10))
        
        plt.tight_layout()
        
        # Title
        if lake_name is not None:
            title = str(lake_name)
        else:
            title = f'{var_display_name} @ depth {target_depth:.2f}'
        plt.suptitle(title, fontsize=14, y=1.0)
        
        # Save to disk if save_path is provided
        if save_path is not None:
            import os
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            plt.savefig(save_path, dpi=180, bbox_inches='tight')
            print(f"Saved single sample forecast to {save_path}")
        
        # Log to wandb
        if epoch is not None and train_or_val is not None:
            wandb_title = f'Single Sample Forecast ({var_display_name}): {train_or_val} epoch {epoch}'
            if lake_name is not None:
                wandb_title = f'Single Sample Forecast ({lake_name}): {train_or_val} epoch {epoch}'
            wandb.log({wandb_title: wandb.Image(plt)})
        
        plt.close()
        return
    
    def plot_prediction_lengths_comparison_multi(self,
                                                  gt_row,
                                                  preds_by_predlen,
                                                  time_vals_row,
                                                  datetime_raw_vals,
                                                  depth_vals_row,
                                                  var_ids_row,
                                                  mask_row,
                                                  feature_dict,
                                                  sample_idx,
                                                  var_name=None,
                                                  target_depth=0.0,
                                                  prediction_lengths=[14, 21, 30],
                                                  confidence_level=0.95,
                                                  lake_name=None,
                                                  save_path=None):
        """Plot predictions from multiple files (different prediction lengths) for the same sample."""
        max_pred_len = max(prediction_lengths)
        if max_pred_len not in preds_by_predlen:
            print(f"Error: pred_len={max_pred_len} not found")
            return
        
        gt = gt_row.detach().cpu().numpy() if hasattr(gt_row, 'detach') else np.asarray(gt_row)
        depth_vals = depth_vals_row.detach().cpu().numpy() if hasattr(depth_vals_row, 'detach') else np.asarray(depth_vals_row)
        var_ids = var_ids_row.detach().cpu().numpy() if hasattr(var_ids_row, 'detach') else np.asarray(var_ids_row)
        mask = mask_row.detach().cpu().numpy() if hasattr(mask_row, 'detach') else np.asarray(mask_row)
        
        sample_mask = mask[sample_idx].astype(bool)
        valid_idx = np.where(sample_mask)[0]
        
        sample_gt = gt[sample_idx, valid_idx]
        sample_depths = depth_vals[sample_idx, valid_idx]
        sample_vars = var_ids[sample_idx, valid_idx]
        sample_datetimes = datetime_raw_vals[sample_idx, valid_idx]
        
        target_var_id = None
        if var_name is not None:
            for vid, vname in feature_dict.items():
                if vname == var_name:
                    target_var_id = vid
                    break
        
        if target_var_id is None:
            unique_vars = np.unique(sample_vars)
            target_var_id = int(unique_vars[0])
        
        var_mask = sample_vars == target_var_id
        depth_mask = np.abs(sample_depths - target_depth) < 0.1
        combined_mask = var_mask & depth_mask
        
        if not combined_mask.any():
            print(f"No data for variable {target_var_id} at depth {target_depth}")
            return
        
        # Get all unique dates for regular grid (fill missing with NaN)
        dts_raw = sample_datetimes[combined_mask]
        gt_filtered = sample_gt[combined_mask]
        
        # Decode bytes to strings if necessary
        if len(dts_raw) > 0 and isinstance(dts_raw[0], bytes):
            dts_raw = np.array([dt.decode('utf-8') if isinstance(dt, bytes) else str(dt) for dt in dts_raw])
        
        dts = pd.to_datetime(dts_raw)
        if isinstance(dts, pd.DatetimeIndex):
            dts_series = pd.Series(dts)
        else:
            dts_series = dts
        
        order = np.argsort(dts.values)
        dates_plot = dts_series.iloc[order].dt.date.values if hasattr(dts_series, 'iloc') else pd.Series(dts)[order].dt.date.values
        gt_sorted = gt_filtered[order]
        
        context_len = len(gt_sorted) - max_pred_len
        context_end_idx = max(1, context_len)
        
        dates_context = dates_plot[:context_end_idx]
        gt_context = gt_sorted[:context_end_idx]
        dates_pred = dates_plot[context_end_idx:]
        gt_pred = gt_sorted[context_end_idx:]
        
        num_horizons = len(prediction_lengths)
        fig, axes = plt.subplots(1, num_horizons, figsize=(6 * num_horizons, 6))
        if num_horizons == 1:
            axes = [axes]
        
        var_display_name = feature_dict.get(target_var_id, f"Variable {target_var_id}")
        
        for hi, horizon in enumerate(sorted(prediction_lengths)):
            ax = axes[hi]
            
            if horizon not in preds_by_predlen:
                ax.text(0.5, 0.5, f'No data for pred_len={horizon}', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(f"Pred Len {horizon}", fontsize=14, fontweight='bold')
                continue
            
            preds_data = preds_by_predlen[horizon]
            if isinstance(preds_data, dict):
                has_distribution = True
                preds_mean = preds_data['mean'].detach().cpu().numpy() if hasattr(preds_data['mean'], 'detach') else np.asarray(preds_data['mean'])
                preds_loc = preds_data['loc'].detach().cpu().numpy() if preds_data['loc'] is not None and hasattr(preds_data['loc'], 'detach') else None
                preds_scale = preds_data['scale'].detach().cpu().numpy() if preds_data['scale'] is not None and hasattr(preds_data['scale'], 'detach') else None
                preds_df = preds_data['df'].detach().cpu().numpy() if preds_data['df'] is not None and hasattr(preds_data['df'], 'detach') else None
            else:
                has_distribution = False
                preds_mean = preds_data.detach().cpu().numpy() if hasattr(preds_data, 'detach') else np.asarray(preds_data)
            
            preds_sample = preds_mean[sample_idx, valid_idx][combined_mask]
            preds_sorted = preds_sample[order]
            preds_pred = preds_sorted[context_end_idx:]
            
            if has_distribution and preds_loc is not None:
                loc_sample = preds_loc[sample_idx, valid_idx][combined_mask]
                scale_sample = preds_scale[sample_idx, valid_idx][combined_mask] if preds_scale is not None else None
                df_sample = preds_df[sample_idx, valid_idx][combined_mask] if preds_df is not None else None
                loc_sorted = loc_sample[order]
                scale_sorted = scale_sample[order] if scale_sample is not None else None
                df_sorted = df_sample[order] if df_sample is not None else None
                loc_pred = loc_sorted[context_end_idx:]
                scale_pred = scale_sorted[context_end_idx:] if scale_sorted is not None else None
                df_pred = df_sorted[context_end_idx:] if df_sorted is not None else None
            else:
                loc_pred = None
                scale_pred = None
                df_pred = None
            
            if has_distribution and preds_loc is not None:
                loc_sample = preds_loc[sample_idx, valid_idx][combined_mask]
                scale_sample = preds_scale[sample_idx, valid_idx][combined_mask] if preds_scale is not None else None
                df_sample = preds_df[sample_idx, valid_idx][combined_mask] if preds_df is not None else None
                loc_sorted = loc_sample[order]
                scale_sorted = scale_sample[order] if scale_sample is not None else None
                df_sorted = df_sample[order] if df_sample is not None else None
                loc_pred = loc_sorted[context_end_idx:]
                scale_pred = scale_sorted[context_end_idx:] if scale_sorted is not None else None
                df_pred = df_sorted[context_end_idx:] if df_sorted is not None else None
            else:
                loc_pred = None
                scale_pred = None
                df_pred = None
            
            pos_in_pred_window = horizon - 1
            
            if pos_in_pred_window >= len(preds_pred):
                ax.text(0.5, 0.5, f'No data for T+{horizon}\n(prediction window too short)', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(f"Pred Len {horizon}", fontsize=14, fontweight='bold')
                continue
            
            # Plot with NaN handling (gaps for missing data)
            valid_context = ~np.isnan(gt_context)
            valid_pred = ~np.isnan(gt_pred)
            
            if valid_context.sum() > 0:
                ax.plot(dates_context[valid_context], gt_context[valid_context], 'o-', color='blue', linewidth=2, 
                       markersize=8, alpha=0.7, label='Context (GT)', zorder=1)
            if valid_pred.sum() > 0:
                ax.plot(dates_pred[valid_pred], gt_pred[valid_pred], 's-', color='lightblue', linewidth=1.5, 
                       markersize=8, alpha=0.5, label='Prediction Window (GT)', zorder=1)
            
            if pos_in_pred_window < len(dates_pred):
                pred_date = dates_pred[pos_in_pred_window]
                if pos_in_pred_window < len(preds_pred) and not np.isnan(preds_pred[pos_in_pred_window]):
                    pred_val = preds_pred[pos_in_pred_window]
                    ax.plot(pred_date, pred_val, 'o', color='red', markersize=12, 
                           label=f'Pred Len {horizon} Prediction', zorder=3, markeredgecolor='darkred', markeredgewidth=2)
                
                if pos_in_pred_window < len(gt_pred) and not np.isnan(gt_pred[pos_in_pred_window]):
                    gt_val_at_horizon = gt_pred[pos_in_pred_window]
                    ax.plot(pred_date, gt_val_at_horizon, 's', color='green', markersize=12, 
                           label=f'Pred Len {horizon} GT', zorder=3, markeredgecolor='darkgreen', markeredgewidth=2)
            
            if has_distribution and loc_pred is not None and pos_in_pred_window < len(loc_pred):
                if not np.isnan(loc_pred[pos_in_pred_window]):
                    loc_val = loc_pred[pos_in_pred_window]
                    scale_val = scale_pred[pos_in_pred_window] if scale_pred is not None and not np.isnan(scale_pred[pos_in_pred_window]) else None
                    df_val = df_pred[pos_in_pred_window] if df_pred is not None and not np.isnan(df_pred[pos_in_pred_window]) else None
                    
                    if scale_val is not None and df_val is not None and pos_in_pred_window < len(dates_pred):
                        pred_date = dates_pred[pos_in_pred_window]
                        pred_val = preds_pred[pos_in_pred_window] if pos_in_pred_window < len(preds_pred) else None
                        if pred_val is not None and not np.isnan(pred_val):
                            alpha = 1 - confidence_level
                            t_dist = stats.t(df=df_val, loc=loc_val, scale=scale_val)
                            lower = t_dist.ppf(alpha / 2)
                            upper = t_dist.ppf(1 - alpha / 2)
                            
                            ax.errorbar(pred_date, pred_val, yerr=[[pred_val - lower], [upper - pred_val]], 
                                       fmt='none', color='red', linewidth=3, capsize=8, capthick=2,
                                       alpha=0.7, label=f'{int(confidence_level*100)}% CI', zorder=2)
                            ax.fill_between([pred_date, pred_date], [lower, upper], 
                                           color='red', alpha=0.2, zorder=2)
            
            if context_end_idx > 0 and context_end_idx < len(dates_plot):
                boundary_date = dates_plot[context_end_idx - 1]
                ax.axvline(x=boundary_date, color='black', linestyle='--', linewidth=2, 
                          alpha=0.7, zorder=2)
                if hi == 0:
                    ax.text(boundary_date, ax.get_ylim()[1] * 0.95, 'Context/Prediction\nBoundary',
                           rotation=90, verticalalignment='top', fontsize=9, alpha=0.7,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_ylabel(f'{var_display_name}', fontsize=12)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_title(f"Pred Len {horizon}", fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=9, framealpha=0.9, ncol=1)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            ax.tick_params(axis='x', rotation=45)
            if len(dates_plot) > 30:
                ax.xaxis.set_major_locator(plt.MaxNLocator(10))
        
        plt.tight_layout()
        
        title_parts = [f'Prediction Length Comparison: {var_display_name}']
        if target_depth is not None:
            title_parts.append(f'@ depth {target_depth:.2f}')
        if lake_name is not None:
            title_parts.append(f'({lake_name})')
        title = ' - '.join(title_parts)
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        
        if save_path is not None:
            import os
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            plt.savefig(save_path, dpi=180, bbox_inches='tight')
            print(f"Saved prediction lengths comparison to {save_path}")
        
        plt.close()
    
    def plot_prediction_error_heatmaps(self,
                                      gt_row,
                                      preds_row,
                                      time_vals_row,
                                      datetime_raw_vals,
                                      depth_vals_row,
                                      var_ids_row,
                                      mask_row,
                                      feature_dict,
                                      sample_idx,
                                      epoch,
                                      train_or_val,
                                      title_prefix="Prediction Error Heatmaps",
                                      max_features=6,
                                      max_depths=20,
                                      depth_min=None,
                                      depth_max=None,
                                      save_path=None,
                                      pred_len=None):
        """
        Plot prediction error heatmaps (prediction - ground truth) for a single forecast origin.
        """
        # Handle predictions - extract mean if distribution dict
        if isinstance(preds_row, dict):
            preds_mean = preds_row['mean'].detach().cpu().numpy()
        else:
            preds_mean = preds_row.detach().cpu().numpy() if hasattr(preds_row, 'detach') else np.asarray(preds_row)

        # Convert to numpy
        gt = gt_row.detach().cpu().numpy() if hasattr(gt_row, 'detach') else np.asarray(gt_row)
        time_vals = time_vals_row.detach().cpu().numpy() if hasattr(time_vals_row, 'detach') else np.asarray(time_vals_row)
        depth_vals = depth_vals_row.detach().cpu().numpy() if hasattr(depth_vals_row, 'detach') else np.asarray(depth_vals_row)
        var_ids = var_ids_row.detach().cpu().numpy() if hasattr(var_ids_row, 'detach') else np.asarray(var_ids_row)
        mask = mask_row.detach().cpu().numpy() if hasattr(mask_row, 'detach') else np.asarray(mask_row)

        B = gt.shape[0]
        valid_mask = mask.astype(bool)

        # Find a sample during summer months (April to October)
        dt_array = np.asarray(datetime_raw_vals)
        summer_sample_idx = None

        for b in range(B):
            sel = valid_mask[b]
            if not sel.any():
                continue
            try:
                dts = pd.to_datetime(dt_array[b][sel])
                months = dts.month
                if np.any((months >= 4) & (months <= 10)):
                    summer_sample_idx = b
                    break
            except Exception:
                continue

        if summer_sample_idx is None:
            print("No sample found during summer months (April-October) for error heatmap")
            return

        # Use the first valid summer sample
        b = summer_sample_idx
        sel = valid_mask[b]
        if not sel.any():
            print("No valid data for selected summer sample")
            return

        try:
            dts_b = pd.to_datetime(dt_array[b][sel])
        except Exception as e:
            print(f"Error parsing datetime information: {e}")
            return

        time_vals_b = time_vals[b][sel]
        depth_vals_b = depth_vals[b][sel]
        var_ids_b = var_ids[b][sel]
        gt_b = gt[b][sel]
        pred_b = preds_mean[b][sel]

        # Sort by datetime to give a consistent temporal order
        sort_idx = np.argsort(dts_b.values)
        dts_sorted = dts_b.iloc[sort_idx] if hasattr(dts_b, "iloc") else pd.Series(dts_b)[sort_idx]
        time_vals_sorted = time_vals_b[sort_idx]
        depth_vals_sorted = depth_vals_b[sort_idx]
        var_ids_sorted = var_ids_b[sort_idx]
        gt_sorted = gt_b[sort_idx]
        pred_sorted = pred_b[sort_idx]

        unique_vars = np.unique(var_ids_sorted)
        plot_vars = unique_vars[:max_features]
        if len(plot_vars) == 0:
            print("No valid variables to plot in error heatmap")
            return

        all_unique_times = np.unique(time_vals_sorted)
        all_unique_times_sorted = np.sort(all_unique_times)

        if pred_len is not None:
            all_unique_times_sorted = all_unique_times_sorted[:pred_len]
        H = len(all_unique_times_sorted)
        if H == 0:
            print("No unique horizons found for error heatmap")
            return

        fig, axes = plt.subplots(len(plot_vars), 1, figsize=(16, 4 * len(plot_vars)), squeeze=False)

        for vi, var_id in enumerate(plot_vars):
            ax = axes[vi, 0]

            var_mask = (var_ids_sorted == var_id)
            if not var_mask.any():
                continue

            times_v = time_vals_sorted[var_mask]
            depths_v = depth_vals_sorted[var_mask]
            gt_v = gt_sorted[var_mask]
            pred_v = pred_sorted[var_mask]
            errors_v = pred_v - gt_v

            unique_depths_var = np.unique(depths_v)
            unique_depths_var = np.sort(unique_depths_var)[:max_depths]
            Dn = len(unique_depths_var)
            if Dn == 0:
                continue

            time_to_col = {t: j for j, t in enumerate(all_unique_times_sorted)}

            err_mat = np.full((Dn, H), np.nan, dtype=float)
            for t_val, d_val, e_val in zip(times_v, depths_v, errors_v):
                if t_val in time_to_col:
                    h_idx = time_to_col[t_val]
                else:
                    closest_idx = int(np.argmin(np.abs(all_unique_times_sorted - t_val)))
                    h_idx = closest_idx
                d_idx_arr = np.where(np.abs(unique_depths_var - d_val) < 1e-6)[0]
                if d_idx_arr.size == 0:
                    continue
                d_idx = int(d_idx_arr[0])
                if np.isnan(err_mat[d_idx, h_idx]):
                    err_mat[d_idx, h_idx] = e_val

            cmap = plt.cm.get_cmap('RdBu_r').copy()
            cmap.set_bad(color='green')  # green = no data at that (depth, horizon)
            im = ax.imshow(err_mat,
                           aspect='auto',
                           cmap=cmap,
                           origin='lower',
                           interpolation='none')

            var_name = feature_dict.get(int(var_id), f"Variable {int(var_id)}")
            ax.set_title(f"Prediction Error (Pred - GT) - {var_name}")
            ax.set_ylabel("Depth (m)")
            ax.set_xlabel("Prediction horizon (T+n)")

            try:
                if depth_min is not None and depth_max is not None:
                    depth_labels = [
                        f"{(float(d) * (depth_max - depth_min) + depth_min):.2f}"
                        for d in unique_depths_var
                    ]
                else:
                    depth_labels = [f"{float(d):.2f}" for d in unique_depths_var]
            except Exception:
                depth_labels = [str(d) for d in unique_depths_var]

            depth_tick_step = max(1, Dn // 10)
            ax.set_yticks(np.arange(Dn)[::depth_tick_step])
            ax.set_yticklabels([depth_labels[i] for i in range(0, Dn, depth_tick_step)])

            tick_step = max(1, H // 20)  # show ~20 ticks max
            x_ticks = np.arange(0, H, tick_step)
            x_labels = [f"T+{i+1}" for i in x_ticks]
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_labels, rotation=0)

            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Prediction Error (Pred - GT)', rotation=270, labelpad=20)

        plt.tight_layout()
        title = f'{title_prefix}: {train_or_val} Prediction Error Heatmaps at epoch {epoch}'
        plt.suptitle(title, fontsize=14, y=1.0)

        if save_path is not None:
            import os
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            plt.savefig(save_path, dpi=180, bbox_inches='tight')
            print(f"Saved prediction error heatmap to {save_path}")

        wandb.log({title: wandb.Image(plt)})
        plt.close()
        return


    def plot_Tn_predictions_for_day(self,
                                    preds_row,
                                    gt_row,
                                    datetime_raw_vals,
                                    depth_vals_row,
                                    var_ids_row,
                                    mask_row,
                                    feature_dict,
                                    epoch,
                                    train_or_val,
                                    var_name=None,
                                    lake_name=None,
                                    start_date=None,
                                    end_date=None,
                                    horizons=[1, 7, 14, 30],
                                    max_depths=20,
                                    depth_min=None,
                                    depth_max=None,
                                    save_path=None):
        """
        Plot T+n prediction errors (pred - gt) for a date range, showing specific horizons (T+1, T+7, T+14, T+30).
        """
        # Handle predictions - extract mean if distribution dict
        if isinstance(preds_row, dict):
            preds_mean = preds_row['mean'].detach().cpu().numpy()
            if 'scale' in preds_row:
                preds_std = preds_row['scale'].detach().cpu().numpy()
            elif 'std' in preds_row:
                preds_std = preds_row['std'].detach().cpu().numpy()
            else:
                preds_std = None
        else:
            preds_mean = preds_row.detach().cpu().numpy() if hasattr(preds_row, 'detach') else np.asarray(preds_row)
            preds_std = None
        
        # Handle ground truth
        if hasattr(gt_row, 'detach'):
            gt_mean = gt_row.detach().cpu().numpy()
        else:
            gt_mean = np.asarray(gt_row)

        # Convert to numpy
        depth_vals = depth_vals_row.detach().cpu().numpy() if hasattr(depth_vals_row, 'detach') else np.asarray(depth_vals_row)
        var_ids = var_ids_row.detach().cpu().numpy() if hasattr(var_ids_row, 'detach') else np.asarray(var_ids_row)
        mask = mask_row.detach().cpu().numpy() if hasattr(mask_row, 'detach') else np.asarray(mask_row)

        B = preds_mean.shape[0]
        valid_mask = mask.astype(bool)

        if datetime_raw_vals is None:
            print("No valid datetimes available for T+n predictions plot: datetime_raw_vals is None")
            return
        
        dt_array = np.asarray(datetime_raw_vals)
        
        if dt_array.size == 0:
            print("No valid datetimes available for T+n predictions plot: datetime_raw_vals is empty")
            return
        
        if dt_array.ndim != 2 or dt_array.shape[0] != B:
            print(f"No valid datetimes available for T+n predictions plot: datetime_raw_vals shape {dt_array.shape} doesn't match batch size {B}")
            return

        sample_observation_counts = []
        for b in range(B):
            sel = valid_mask[b]
            count = sel.sum() if sel.any() else 0
            sample_observation_counts.append(count)
        
        if max(sample_observation_counts) == 0:
            print("No valid observations found in any sample")
            return
        
        selected_batch_idx = np.argmax(sample_observation_counts)
        max_observations = sample_observation_counts[selected_batch_idx]
        print(f"Selecting sample {selected_batch_idx} with {max_observations} observations (most dense)")

        all_dates = []
        sel = valid_mask[selected_batch_idx]
        if sel.any():
            try:
                dts_b = pd.to_datetime(dt_array[selected_batch_idx][sel])
                if isinstance(dts_b, pd.DatetimeIndex):
                    dates_b = pd.Series(dts_b).dt.date.values
                else:
                    dates_b = dts_b.dt.date.values
                all_dates.extend(list(dates_b))
            except Exception:
                pass

        if len(all_dates) == 0:
            print("No valid datetimes available for T+n predictions plot")
            return

        all_dates_series = pd.Series(all_dates)
        unique_dates = sorted(all_dates_series.unique())
        
        if start_date is None:
            summer_dates = [d for d in unique_dates if d.month in [5, 6]]
            if len(summer_dates) > 0:
                start_date = min(summer_dates)
            else:
                start_date = unique_dates[0] if len(unique_dates) > 0 else None
        else:
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date).date()
        
        if start_date is None:
            print("No dates available in data")
            return
        
        if end_date is None:
            target_end = start_date + pd.Timedelta(days=30)
            available_after_start = [d for d in unique_dates if d >= start_date and d <= target_end]
            if len(available_after_start) > 0:
                end_date = max(available_after_start)
            else:
                dates_after = [d for d in unique_dates if d > start_date]
                if len(dates_after) > 0:
                    end_date = min(dates_after)
                else:
                    end_date = start_date
        else:
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date).date()

        print(f"T+n predictions: date range {start_date} to {end_date}, horizons {horizons}")

        target_var_id = None
        if var_name is not None:
            for vid, vname in feature_dict.items():
                if vname == var_name:
                    target_var_id = vid
                    break
            if target_var_id is None:
                print(f"Variable '{var_name}' not found in feature_dict. Available: {list(feature_dict.values())}")
                target_var_id = list(feature_dict.keys())[0] if len(feature_dict) > 0 else None
        else:
            target_var_id = list(feature_dict.keys())[0] if len(feature_dict) > 0 else None

        if target_var_id is None:
            print("No variables available to plot")
            return

        horizon_data = {h: [] for h in horizons}
        
        all_depths_all_data = set()
        
        b = selected_batch_idx
        sel = valid_mask[b]
        if not sel.any():
            print(f"Selected batch {b} has no valid observations")
            return
        
        try:
            dts_b = pd.to_datetime(dt_array[b][sel])
        except Exception:
            print(f"Error parsing datetimes for batch {b}")
            return

        # Convert DatetimeIndex to Series to use .dt accessor
        if isinstance(dts_b, pd.DatetimeIndex):
            dts_b_series = pd.Series(dts_b)
        else:
            dts_b_series = dts_b
        dates_b = dts_b_series.dt.date.values
        
        # Get data for this batch
        batch_depths = depth_vals[b][sel]
        batch_vars = var_ids[b][sel]
        batch_preds = preds_mean[b][sel]
        batch_gt = gt_mean[b][sel]
        if preds_std is not None:
            batch_std = preds_std[b][sel]
        else:
            batch_std = None
        
        var_depth_groups = {}
        for idx in range(len(dates_b)):
            var_id = int(batch_vars[idx])
            depth_val = batch_depths[idx]
            date_val = dates_b[idx]
            key = (var_id, depth_val)
            
            if key not in var_depth_groups:
                var_depth_groups[key] = []
            
            if hasattr(dts_b, 'iloc'):
                dt_val = dts_b.iloc[idx]
            elif isinstance(dts_b, pd.DatetimeIndex):
                dt_val = dts_b[idx]
            else:
                dt_val = pd.to_datetime(dates_b[idx])
            
            var_depth_groups[key].append({
                'date': date_val,
                'datetime': dt_val,
                'pred': batch_preds[idx],
                'gt': batch_gt[idx],
                'std': batch_std[idx] if batch_std is not None else None
            })
        
        for (var_id, depth_val), group_data in var_depth_groups.items():
            if var_id != target_var_id:
                continue
            
            all_depths_all_data.add(depth_val)
            
            group_data_sorted = sorted(group_data, key=lambda x: x['datetime'])
            
            for horizon in horizons:
                pos_in_sequence = horizon - 1  # T+1 -> position 0, T+7 -> position 6, etc.
                
                if pos_in_sequence < len(group_data_sorted):
                    item = group_data_sorted[pos_in_sequence]
                    date_val = item['date']
                    
                    # Check if date is in range
                    if date_val < start_date or date_val > end_date:
                        continue
                    
                    pred_val = item['pred']
                    gt_val = item['gt']
                    error_val = pred_val - gt_val
                    std_val = item.get('std', None)
                    
                    horizon_data[horizon].append((date_val, depth_val, pred_val, gt_val, error_val, std_val))

        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        sorted_dates = [d.date() for d in date_range]
        Dt = len(sorted_dates)
        
        sorted_depths = sorted(all_depths_all_data)[:max_depths]
        Dn = len(sorted_depths)
        
        if Dn == 0:
            print("No depths found in data")
            return
        
        total_records = sum(len(records) for records in horizon_data.values())
        if total_records == 0 and Dt == 0:
            return

        num_horizons = len(horizons)
        fig, axes = plt.subplots(num_horizons, 2, figsize=(32, 4 * num_horizons), squeeze=False)

        var_display_name = feature_dict.get(target_var_id, f"Variable {target_var_id}")

        for hi, horizon in enumerate(sorted(horizons)):
            records = horizon_data[horizon]
            
            error_sum_mat = np.zeros((Dn, Dt), dtype=float)
            error_count_mat = np.zeros((Dn, Dt), dtype=float)
            std_sum_mat = np.zeros((Dn, Dt), dtype=float)
            std_count_mat = np.zeros((Dn, Dt), dtype=float)

            for record in records:
                if len(record) == 6:
                    date_val, depth_val, pred_val, gt_val, error_val, std_val = record
                else:
                    date_val, depth_val, pred_val, gt_val, error_val = record[:5]
                    std_val = record[5] if len(record) > 5 else None
                
                depth_idx = None
                for di, d in enumerate(sorted_depths):
                    if abs(d - depth_val) < 1e-6:
                        depth_idx = di
                        break
                if depth_idx is None:
                    continue
                
                try:
                    date_idx = sorted_dates.index(date_val)
                except ValueError:
                    continue
                
                error_sum_mat[depth_idx, date_idx] += error_val
                error_count_mat[depth_idx, date_idx] += 1.0
                
                if std_val is not None:
                    std_sum_mat[depth_idx, date_idx] += std_val
                    std_count_mat[depth_idx, date_idx] += 1.0

            with np.errstate(invalid="ignore", divide="ignore"):
                error_mat = np.where(error_count_mat > 0, error_sum_mat / error_count_mat, np.nan)
                std_mat = np.where(std_count_mat > 0, std_sum_mat / std_count_mat, np.nan) if preds_std is not None else np.full((Dn, Dt), np.nan)

            ax_error = axes[hi, 0]
            cmap_error = plt.cm.get_cmap('RdBu_r').copy()  # Red-Blue reversed
            cmap_error.set_bad(color='green')  # green = no data
            
            vmax_error = np.nanmax(np.abs(error_mat)) if not np.isnan(error_mat).all() else 1.0
            vmin_error = -vmax_error if vmax_error > 0 else -1.0
            
            im_error = ax_error.imshow(error_mat,
                          aspect='auto',
                          cmap=cmap_error,
                          origin='lower',
                          interpolation='none',
                          vmin=vmin_error,
                          vmax=vmax_error)

            ax_error.set_title(f"T+{horizon} Error - {var_display_name}")
            ax_error.set_ylabel("Depth (m)")
            ax_error.set_xlabel("Date")

            try:
                if depth_min is not None and depth_max is not None:
                    depth_labels = [
                        f"{(float(d) * (depth_max - depth_min) + depth_min):.2f}"
                        for d in sorted_depths
                    ]
                else:
                    depth_labels = [f"{float(d):.2f}" for d in sorted_depths]
            except Exception:
                depth_labels = [str(d) for d in sorted_depths]

            depth_tick_step = max(1, Dn // 10)
            ax_error.set_yticks(np.arange(Dn)[::depth_tick_step])
            ax_error.set_yticklabels([depth_labels[i] for i in range(0, Dn, depth_tick_step)])

            date_tick_step = max(1, Dt // 15)  # show ~15 date ticks max
            date_ticks = np.arange(0, Dt, date_tick_step)
            date_labels = [sorted_dates[i].strftime('%Y-%m-%d') for i in date_ticks]
            ax_error.set_xticks(date_ticks)
            ax_error.set_xticklabels(date_labels, rotation=45, ha='right')

            cbar_error = fig.colorbar(im_error, ax=ax_error, fraction=0.046, pad=0.04)
            cbar_error.set_label('Error (Pred - GT)', rotation=270, labelpad=20)

            ax_std = axes[hi, 1]
            if preds_std is not None and not np.isnan(std_mat).all():
                cmap_std = plt.cm.get_cmap('viridis').copy()
                cmap_std.set_bad(color='green')  # green = no data
                
                vmin_std = 0.0
                vmax_std = np.nanmax(std_mat) if not np.isnan(std_mat).all() else 1.0
                
                im_std = ax_std.imshow(std_mat,
                              aspect='auto',
                              cmap=cmap_std,
                              origin='lower',
                              interpolation='none',
                              vmin=vmin_std,
                              vmax=vmax_std)
                
                ax_std.set_title(f"T+{horizon} Std - {var_display_name}")
                ax_std.set_ylabel("Depth (m)")
                ax_std.set_xlabel("Date")
                
                ax_std.set_yticks(np.arange(Dn)[::depth_tick_step])
                ax_std.set_yticklabels([depth_labels[i] for i in range(0, Dn, depth_tick_step)])
                ax_std.set_xticks(date_ticks)
                ax_std.set_xticklabels(date_labels, rotation=45, ha='right')
                
                cbar_std = fig.colorbar(im_std, ax=ax_std, fraction=0.046, pad=0.04)
                cbar_std.set_label('Std', rotation=270, labelpad=20)
            else:
                ax_std.text(0.5, 0.5, 'No std data available', 
                           ha='center', va='center', transform=ax_std.transAxes, fontsize=12)
                ax_std.set_title(f"T+{horizon} Std - {var_display_name}")
                ax_std.set_ylabel("Depth (m)")
                ax_std.set_xlabel("Date")

        plt.tight_layout()
        if lake_name is not None:
            title = str(lake_name)
        else:
            title = var_display_name
        plt.suptitle(title, fontsize=14, y=1.0)

        if save_path is not None:
            import os
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            plt.savefig(save_path, dpi=180, bbox_inches='tight')
            print(f"Saved T+n error maps to {save_path}")

        # Log to wandb
        wandb_title = f'T+n Error Maps ({var_display_name}): {train_or_val} epoch {epoch}'
        if lake_name is not None:
            wandb_title = f'T+n Error Maps ({lake_name}): {train_or_val} epoch {epoch}'
        wandb.log({wandb_title: wandb.Image(plt)})
        plt.close()
        return

class SpatioTemporalHeatmapPlotter:
    """
    Plotter for spatio-temporal heatmaps with regular depth-time grids
    """
    def __init__(self, device, pad_val):
        self.device = device
        self.pad_val = pad_val
    
    def plot_heatmap_forecast(self, 
                            gt_data, pred_data, 
                            datetime_vals, depth_vals,
                            var_ids, masks,
                            feature_dict, sample_idx, 
                            epoch, train_or_val,
                            title_prefix="Spatio-Temporal Forecast",
                            max_features=6,
                            confidence_level=0.95,
                            depth_units="m"):
        """
        Plot spatio-temporal heatmaps for ground truth and predictions
        
        """
        if len(gt_data.shape) == 3:  # (B, T, D)
            B, T, D = gt_data.shape
            gt_flat = gt_data.view(B, -1)
            pred_flat = pred_data['mean'] if isinstance(pred_data, dict) else pred_data
            pred_flat = pred_flat.view(B, -1) if len(pred_flat.shape) == 3 else pred_flat
            var_ids_flat = var_ids.view(B, -1) if len(var_ids.shape) == 3 else var_ids
            masks_flat = masks.view(B, -1) if len(masks.shape) == 3 else masks
        else:  # (B, T*D)
            B, TD = gt_data.shape
            T = len(datetime_vals)
            D = len(depth_vals)
            gt_flat = gt_data
            pred_flat = pred_data['mean'] if isinstance(pred_data, dict) else pred_data
            var_ids_flat = var_ids
            masks_flat = masks
        
        # Get unique variables
        unique_vars = torch.unique(var_ids_flat[sample_idx])
        unique_vars = unique_vars[unique_vars != self.pad_val]
        
        if len(unique_vars) == 0:
            print("No valid variables found for heatmap plotting")
            return
        
        # Limit number of features
        if len(unique_vars) > max_features:
            unique_vars = unique_vars[:max_features]
        
        n_vars = len(unique_vars)
        fig, axes = plt.subplots(2, n_vars, figsize=(4*n_vars, 8))
        if n_vars == 1:
            axes = axes.reshape(2, 1)
        
        for i, var_id in enumerate(unique_vars):
            var_id_int = int(var_id)
            var_name = feature_dict.get(var_id_int, f"Variable {var_id_int}")
            
            # Extract data for this variable across all samples
            var_mask = (var_ids_flat[sample_idx] == var_id)
            if not var_mask.any():
                continue
            
            # Get valid positions for this variable
            valid_positions = torch.where(var_mask)
            sample_indices = valid_positions[0]
            position_indices = valid_positions[1]
            
            # Create grids for this variable
            gt_grid = np.full((len(sample_idx), T, D), np.nan)
            pred_grid = np.full((len(sample_idx), T, D), np.nan)
            mask_grid = np.full((len(sample_idx), T, D), False)
            
            # Fill grids with data
            for j, (s_idx, pos_idx) in enumerate(zip(sample_indices, position_indices)):
                if s_idx < len(sample_idx):
                    t_idx = pos_idx // D
                    d_idx = pos_idx % D
                    if t_idx < T and d_idx < D:
                        gt_grid[s_idx, t_idx, d_idx] = gt_flat[sample_idx[s_idx], pos_idx].item()
                        pred_grid[s_idx, t_idx, d_idx] = pred_flat[sample_idx[s_idx], pos_idx].item()
                        mask_grid[s_idx, t_idx, d_idx] = masks_flat[sample_idx[s_idx], pos_idx].item()
            
            # Average across samples for visualization
            gt_mean = np.nanmean(gt_grid, axis=0)
            pred_mean = np.nanmean(pred_grid, axis=0)
            mask_mean = np.any(mask_grid, axis=0)
            
            # Apply mask
            gt_mean[~mask_mean] = np.nan
            pred_mean[~mask_mean] = np.nan
            
            # Plot ground truth heatmap
            ax_gt = axes[0, i]
            im_gt = ax_gt.imshow(gt_mean.T, aspect='auto', origin='lower', 
                               extent=[0, len(datetime_vals)-1, depth_vals[0], depth_vals[-1]],
                               cmap='viridis', interpolation='none')
            ax_gt.set_title(f'{var_name} - Ground Truth')
            ax_gt.set_xlabel('Time Step')
            ax_gt.set_ylabel(f'Depth ({depth_units})')
            ax_gt.set_xticks(range(0, len(datetime_vals), max(1, len(datetime_vals)//10)))
            ax_gt.set_xticklabels([datetime_vals[j].strftime('%m/%d') for j in range(0, len(datetime_vals), max(1, len(datetime_vals)//10))], 
                                rotation=45)
            plt.colorbar(im_gt, ax=ax_gt, label='Value')
            
            # Plot prediction heatmap
            ax_pred = axes[1, i]
            im_pred = ax_pred.imshow(pred_mean.T, aspect='auto', origin='lower',
                                   extent=[0, len(datetime_vals)-1, depth_vals[0], depth_vals[-1]],
                                   cmap='viridis', interpolation='none')
            ax_pred.set_title(f'{var_name} - Predictions')
            ax_pred.set_xlabel('Time Step')
            ax_pred.set_ylabel(f'Depth ({depth_units})')
            ax_pred.set_xticks(range(0, len(datetime_vals), max(1, len(datetime_vals)//10)))
            ax_pred.set_xticklabels([datetime_vals[j].strftime('%m/%d') for j in range(0, len(datetime_vals), max(1, len(datetime_vals)//10))], 
                                  rotation=45)
            plt.colorbar(im_pred, ax=ax_pred, label='Value')
        
        # Remove empty subplots
        for i in range(n_vars, axes.shape[1]):
            for j in range(2):
                axes[j, i].remove()
        
        title = f'{title_prefix}: {train_or_val} Spatio-Temporal Heatmaps at epoch {epoch}'
        plt.suptitle(title, fontsize=14, y=1.0)
        plt.tight_layout()
        wandb.log({title: wandb.Image(plt)})
        plt.close()
    
    def plot_simple_heatmap(self, 
                           gt_data, pred_data, 
                           datetime_vals, depth_vals,
                           var_ids, masks,
                           feature_dict, sample_idx, 
                           epoch, train_or_val,
                           title_prefix="Simple Regular Grid Forecast",
                           max_features=6,
                           depth_units="m"):
        """
        Plot simple spatio-temporal heatmaps for regular grid data
        """
        if len(gt_data.shape) == 4:  # (B, T, D, V)
            B, T, D, V = gt_data.shape
        elif len(gt_data.shape) == 2:  # (B, T*D*V) - flattened
            B, TDV = gt_data.shape
            print(f"Warning: Expected 4D data but got 2D with shape {gt_data.shape}")
            return
        else:
            print(f"Warning: Unexpected data shape {gt_data.shape}")
            return
        
        # Get unique variables
        unique_vars = np.unique(var_ids)
        unique_vars = unique_vars[unique_vars != self.pad_val]
        
        if len(unique_vars) == 0:
            print("No valid variables found for heatmap plotting")
            return
        
        if len(unique_vars) > max_features:
            unique_vars = unique_vars[:max_features]
        
        n_vars = len(unique_vars)
        fig, axes = plt.subplots(2, n_vars, figsize=(4*n_vars, 8))
        if n_vars == 1:
            axes = axes.reshape(2, 1)
        
        for i, var_id in enumerate(unique_vars):
            var_id_int = int(var_id)
            var_name = feature_dict.get(var_id_int, f"Variable {var_id_int}")
            
            var_idx = np.where(var_ids == var_id)[0]
            if len(var_idx) == 0:
                continue
            var_idx = var_idx[0]
            
            gt_var = gt_data[:, :, :, var_idx]  # (B, T, D)
            pred_var = pred_data[:, :, :, var_idx]  # (B, T, D)
            mask_var = masks[:, :, :, var_idx]  # (B, T, D)
            
            gt_mean = np.nanmean(gt_var, axis=0)  # (T, D)
            pred_mean = np.nanmean(pred_var, axis=0)  # (T, D)
            mask_mean = np.any(mask_var, axis=0)  # (T, D)
            
            gt_mean[~mask_mean] = np.nan
            pred_mean[~mask_mean] = np.nan
            
            ax_gt = axes[0, i]
            im_gt = ax_gt.imshow(gt_mean.T, aspect='auto', origin='lower', 
                               extent=[0, len(datetime_vals)-1, depth_vals[0], depth_vals[-1]],
                               cmap='viridis', interpolation='none')
            ax_gt.set_title(f'{var_name} - Ground Truth')
            ax_gt.set_xlabel('Time Step')
            ax_gt.set_ylabel(f'Depth ({depth_units})')
            ax_gt.set_xticks(range(0, len(datetime_vals), max(1, len(datetime_vals)//10)))
            ax_gt.set_xticklabels([datetime_vals[j].strftime('%m/%d') for j in range(0, len(datetime_vals), max(1, len(datetime_vals)//10))], 
                                rotation=45)
            plt.colorbar(im_gt, ax=ax_gt, label='Value')
            
            ax_pred = axes[1, i]
            im_pred = ax_pred.imshow(pred_mean.T, aspect='auto', origin='lower',
                                   extent=[0, len(datetime_vals)-1, depth_vals[0], depth_vals[-1]],
                                   cmap='viridis', interpolation='none')
            ax_pred.set_title(f'{var_name} - Predictions')
            ax_pred.set_xlabel('Time Step')
            ax_pred.set_ylabel(f'Depth ({depth_units})')
            ax_pred.set_xticks(range(0, len(datetime_vals), max(1, len(datetime_vals)//10)))
            ax_pred.set_xticklabels([datetime_vals[j].strftime('%m/%d') for j in range(0, len(datetime_vals), max(1, len(datetime_vals)//10))], 
                                  rotation=45)
            plt.colorbar(im_pred, ax=ax_pred, label='Value')
        
        for i in range(n_vars, axes.shape[1]):
            for j in range(2):
                axes[j, i].remove()
        
        title = f'{title_prefix}: {train_or_val} Spatio-Temporal Heatmaps at epoch {epoch}'
        plt.suptitle(title, fontsize=14, y=1.0)
        plt.tight_layout()
        wandb.log({title: wandb.Image(plt)})
        plt.close()
        plt.close()
    
    def plot_prediction_lengths_comparison(self,
                                          gt_row,
                                          preds_row,
                                          time_vals_row,
                                          datetime_raw_vals,
                                          depth_vals_row,
                                          var_ids_row,
                                          mask_row,
                                          feature_dict,
                                          sample_idx,
                                          var_name=None,
                                          target_depth=0.0,
                                          context_len=None,
                                          prediction_lengths=[14, 21, 30],
                                          confidence_level=0.95,
                                          epoch=None,
                                          train_or_val=None,
                                          lake_name=None,
                                          save_path=None):
        """
        Plot predictions for different prediction lengths (horizons) in a single row.
        Shows T+14, T+21, T+30 predictions side by side with uncertainty bands.
        Uses a fixed context length and shows how predictions change at different horizons.
        
        """
        if isinstance(preds_row, dict):
            has_distribution = True
            preds_mean = preds_row['mean'].detach().cpu().numpy()
            preds_loc = preds_row['loc'].detach().cpu().numpy()
            preds_scale = preds_row['scale'].detach().cpu().numpy()
            preds_df = preds_row['df'].detach().cpu().numpy()
        else:
            has_distribution = False
            preds_mean = preds_row.detach().cpu().numpy() if hasattr(preds_row, 'detach') else np.asarray(preds_row)
        
        # Convert to numpy
        gt = gt_row.detach().cpu().numpy() if hasattr(gt_row, 'detach') else np.asarray(gt_row)
        time_vals = time_vals_row.detach().cpu().numpy() if hasattr(time_vals_row, 'detach') else np.asarray(time_vals_row)
        depth_vals = depth_vals_row.detach().cpu().numpy() if hasattr(depth_vals_row, 'detach') else np.asarray(depth_vals_row)
        var_ids = var_ids_row.detach().cpu().numpy() if hasattr(var_ids_row, 'detach') else np.asarray(var_ids_row)
        mask = mask_row.detach().cpu().numpy() if hasattr(mask_row, 'detach') else np.asarray(mask_row)
        
        # Get data for the specified sample
        sample_mask = mask[sample_idx].astype(bool)
        valid_idx = np.where(sample_mask)[0]
        
        if valid_idx.size == 0:
            print(f"No valid data for sample {sample_idx}")
            return
        
        sample_gt = gt[sample_idx, valid_idx]
        sample_preds = preds_mean[sample_idx, valid_idx]
        sample_depths = depth_vals[sample_idx, valid_idx]
        sample_vars = var_ids[sample_idx, valid_idx]
        sample_datetimes = datetime_raw_vals[sample_idx, valid_idx]
        
        if has_distribution:
            sample_loc = preds_loc[sample_idx, valid_idx]
            sample_scale = preds_scale[sample_idx, valid_idx]
            sample_df = preds_df[sample_idx, valid_idx]
        
        target_var_id = None
        if var_name is not None:
            for vid, vname in feature_dict.items():
                if vname == var_name:
                    target_var_id = vid
                    break
        
        if target_var_id is None:
            unique_vars = np.unique(sample_vars)
            target_var_id = int(unique_vars[0])
        
        var_mask = sample_vars == target_var_id
        depth_mask = np.abs(sample_depths - target_depth) < 0.1
        combined_mask = var_mask & depth_mask
        
        if not combined_mask.any():
            print(f"No data for variable {target_var_id} at depth {target_depth}")
            return
        
        dts_raw = sample_datetimes[combined_mask]
        gt_filtered = sample_gt[combined_mask]
        preds_filtered = sample_preds[combined_mask]
        
        if has_distribution:
            loc_filtered = sample_loc[combined_mask]
            scale_filtered = sample_scale[combined_mask]
            df_filtered = sample_df[combined_mask]
        
        try:
            dts = pd.to_datetime(dts_raw)
            if isinstance(dts, pd.DatetimeIndex):
                dts_series = pd.Series(dts)
            else:
                dts_series = dts
        except Exception:
            print("Error parsing datetimes")
            return
        
        order = np.argsort(dts.values)
        dates_plot = dts_series.iloc[order].dt.date.values if hasattr(dts_series, 'iloc') else pd.Series(dts)[order].dt.date.values
        gt_sorted = gt_filtered[order]
        preds_sorted = preds_filtered[order]
        
        if has_distribution:
            loc_sorted = loc_filtered[order]
            scale_sorted = scale_filtered[order]
            df_sorted = df_filtered[order]
        
        # Determine context length
        if context_len is None:
            context_len = len(gt_sorted) // 2
        
        context_end_idx = min(context_len, len(gt_sorted))
        
        # Context window data
        dates_context = dates_plot[:context_end_idx]
        gt_context = gt_sorted[:context_end_idx]
        
        # Prediction window data (after context)
        dates_pred = dates_plot[context_end_idx:]
        preds_pred = preds_sorted[context_end_idx:]
        gt_pred = gt_sorted[context_end_idx:]
        
        if has_distribution:
            loc_pred = loc_sorted[context_end_idx:]
            scale_pred = scale_sorted[context_end_idx:]
            df_pred = df_sorted[context_end_idx:]
        
        num_horizons = len(prediction_lengths)
        fig, axes = plt.subplots(1, num_horizons, figsize=(6 * num_horizons, 6))
        if num_horizons == 1:
            axes = [axes]
        
        var_display_name = feature_dict.get(target_var_id, f"Variable {target_var_id}")
        
        for hi, horizon in enumerate(sorted(prediction_lengths)):
            ax = axes[hi]
            
            pos_in_pred_window = horizon - 1
            
            if pos_in_pred_window >= len(preds_pred):
                ax.text(0.5, 0.5, f'No data for T+{horizon}\n(prediction window too short)', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(f"T+{horizon}", fontsize=14, fontweight='bold')
                ax.set_xlabel('Date', fontsize=12)
                ax.set_ylabel(var_display_name, fontsize=12)
                continue
            
            ax.plot(dates_context, gt_context, 'o-', color='blue', linewidth=2, 
                   markersize=8, alpha=0.7, label='Context (GT)', zorder=1)
            
            ax.plot(dates_pred, gt_pred, 's-', color='lightblue', linewidth=1.5, 
                   markersize=8, alpha=0.5, label='Prediction Window (GT)', zorder=1)
            
            pred_date = dates_pred[pos_in_pred_window]
            pred_val = preds_pred[pos_in_pred_window]
            gt_val_at_horizon = gt_pred[pos_in_pred_window]
            
            ax.plot(pred_date, pred_val, 'o', color='red', markersize=12, 
                   label=f'T+{horizon} Prediction', zorder=3, markeredgecolor='darkred', markeredgewidth=2)
            ax.plot(pred_date, gt_val_at_horizon, 's', color='green', markersize=12, 
                   label=f'T+{horizon} GT', zorder=3, markeredgecolor='darkgreen', markeredgewidth=2)
            
            if has_distribution and pos_in_pred_window < len(loc_pred):
                loc_val = loc_pred[pos_in_pred_window]
                scale_val = scale_pred[pos_in_pred_window]
                df_val = df_pred[pos_in_pred_window]
                
                alpha = 1 - confidence_level
                t_dist = stats.t(df=df_val, loc=loc_val, scale=scale_val)
                lower = t_dist.ppf(alpha / 2)
                upper = t_dist.ppf(1 - alpha / 2)
                
                ax.errorbar(pred_date, pred_val, yerr=[[pred_val - lower], [upper - pred_val]], 
                           fmt='none', color='red', linewidth=3, capsize=8, capthick=2,
                           alpha=0.7, label=f'{int(confidence_level*100)}% CI', zorder=2)
                ax.fill_between([pred_date, pred_date], [lower, upper], 
                               color='red', alpha=0.2, zorder=2)
            
            if context_end_idx > 0 and context_end_idx < len(dates_plot):
                boundary_date = dates_plot[context_end_idx - 1]
                ax.axvline(x=boundary_date, color='black', linestyle='--', linewidth=2, 
                          alpha=0.7, zorder=2)
                if hi == 0:
                    ax.text(boundary_date, ax.get_ylim()[1] * 0.95, 'Context/Prediction\nBoundary',
                           rotation=90, verticalalignment='top', fontsize=9, alpha=0.7,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_ylabel(f'{var_display_name}', fontsize=12)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_title(f"T+{horizon}", fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=9, framealpha=0.9, ncol=1)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            
            ax.tick_params(axis='x', rotation=45)
            if len(dates_plot) > 30:
                ax.xaxis.set_major_locator(plt.MaxNLocator(10))
        
        plt.tight_layout()
        
        title_parts = [f'Prediction Length Comparison: {var_display_name}']
        if target_depth is not None:
            title_parts.append(f'@ depth {target_depth:.2f}')
        if lake_name is not None:
            title_parts.append(f'({lake_name})')
        title = ' - '.join(title_parts)
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        
        if save_path is not None:
            import os
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            plt.savefig(save_path, dpi=180, bbox_inches='tight')
            print(f"Saved prediction lengths comparison to {save_path}")
        
        if epoch is not None and train_or_val is not None:
            wandb_title = f'Prediction Lengths Comparison ({var_display_name}): {train_or_val} epoch {epoch}'
            wandb.log({wandb_title: wandb.Image(plt)})
        
        plt.close()