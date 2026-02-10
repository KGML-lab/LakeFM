import math
import torch
import numpy as np
import torch.nn as nn
import pandas as pd
import wandb
import math
import sys
import os
import copy
import time
import torch.nn.functional as F
import copy
import torch.distributed as dist
from torch.distributions import StudentT

from functools import partial
from tqdm import trange, tqdm
from data.loader import build_dataloader
from lakefm.model import LakeFMModule
from utils.util import ModelPlugins, Plotter, DTWCacheMixin
from utils.irregular_plotter import IrregularGridPlotter
from hydra.utils import instantiate
from torch.utils.data import Dataset
from utils.exp_utils import pretty_print
from omegaconf import OmegaConf
from collections import Counter
from utils.exp_utils import print_model_and_data_info


def check_contrastive_batch_stats(batch, p_pos, print_freq=100, batch_idx=0):
    if batch_idx % print_freq != 0:
        return

    if 'lake_id' not in batch:
        print(f"[Batch {batch_idx} Check] 'lake_id' not found in batch. Skipping check.")
        return

    lake_ids = batch['lake_id']
    if isinstance(lake_ids, torch.Tensor):
        lake_ids = lake_ids.cpu().numpy()

    batch_size = len(lake_ids)
    counts = Counter(lake_ids)

    positive_pairs = 0
    for count in counts.values():
        positive_pairs += count * (count - 1) // 2

    total_possible_pairs = batch_size * (batch_size - 1) // 2
    negative_pairs = total_possible_pairs - positive_pairs

    print("\n" + "="*50)
    print(f"[Contrastive Batch Sanity Check at Batch {batch_idx}]")
    print(f"  - Batch Size: {batch_size}")
    print(f"  - Unique Lakes in Batch: {len(counts)}")
    print(f"  - Samples per Lake (P_pos): {list(counts.values())}")
    
    is_balanced = all(c == p_pos for c in counts.values())
    print(f"  - Is Batch Balanced (all lakes have {p_pos} samples)? {'Yes' if is_balanced else 'No'}")

    print(f"\n  - Total Possible Pairs: {total_possible_pairs}")
    print(f"  - Positive Pairs (same lake): {positive_pairs}")
    print(f"  - Negative Pairs (different lakes): {negative_pairs}")
    print("="*50 + "\n")

    if not is_balanced:
        print(f"WARNING: Batch is not balanced as expected. Counts: {counts}")

def init_wandb(cfg, task_name):
    config=OmegaConf.to_container(cfg, 
                                  resolve=True, 
                                  throw_on_missing=True)
    wandb.init(project=config['wandb_project'], 
               name="_".join([task_name, config['wandb_name']]), 
               config=config, 
               save_code=config['save_code'])
    config = wandb.config
    return config

def _updated_mask_(mask_Y, 
                   padding_mask, 
                   pred_len, 
                   seq_len):
    valid_mask = mask_Y.bool() & padding_mask
    
    # Convert to float for loss computation
    mask = valid_mask.float()

    return mask

def reduce_mean(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt

def all_gather_tensor(tensor):
    world_size = dist.get_world_size()
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor)
    return torch.cat(tensor_list, dim=0)

def all_gather_object_list(obj_list):
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, obj_list)
    return sum(gathered, [])  # flatten the list of lists

def safe_cat_and_gather(tensor_list, dim):
    local_tensor = torch.cat(tensor_list, dim=dim)
    return all_gather_tensor(local_tensor)

class Trainer():
    
    def __init__(self, cfg, model, rank=0):
        self.cfg = cfg
        self.model = model
        self.seq_len = self.cfg.seq_len
        self.pred_len = self.cfg.pred_len
        self.rank = rank
        self.trainer = self.cfg.trainer
        self.data = self.cfg.data
        self.max_epochs = self.trainer.max_epochs
        self.pad_value_id = self.cfg.PAD_VAL_ID
        self.pad_value_default = self.cfg.PAD_VAL_DEFAULT
        self.use_lr_scheduler = self.trainer.use_lr_scheduler
        self.warmup_epochs = self.trainer.warmup_epochs
        self.optimization_mode = self.trainer.optimization_mode # not used currently
        
        # cl
        self.use_lake_cl = self.trainer.use_lake_cl
        self.use_temporal_cl = self.trainer.use_temporal_cl
        self.cl_mode = self.trainer.cl_mode

        self.device = f"cuda:{self.rank}" #self.trainer.device

        self.plotter = Plotter(self.device, self.pad_value_default)
        self.irregular_plotter = IrregularGridPlotter(self.device, self.pad_value_default)
        self.dtw = DTWCacheMixin()
        
        # Add exponential moving average for contrastive loss stability
        self.cl_loss_ema = None
        self.cl_ema_momentum = getattr(self.trainer, 'cl_ema_momentum', 0.9)

    def compute_nll_loss(self, forecasts, targets, mask=None):

        if isinstance(forecasts, dict) and 'distribution' in forecasts:
            # Probabilistic forecasting case
            distribution = forecasts['distribution']
            nll = -distribution.log_prob(targets)  # (B, S_t)
            
            if mask is not None:
                nll = nll * mask  # Mask out invalid predictions
                loss = nll.sum() / mask.sum()  # Average over valid predictions
            else:
                loss = nll.mean()  # Average over all predictions
                
            return loss
        else:
            # Fallback to MSE for non-probabilistic forecasts
            if mask is not None:
                mse = F.mse_loss(forecasts, targets, reduction='none')
                mse = mse * mask
                return mse.sum() / mask.sum()
            else:
                return F.mse_loss(forecasts, targets)

    def reshape_vectors(self, x, num_2d_vars, num_1d_vars, num_depths):
        """
        Reshape the input tensor to separate 2D and 1D variables
        """
        size_2d = self.pred_len * num_2d_vars * num_depths

        # Split the flattened tensor into the 2D and 1D portions
        x_2d_flat = x[:, :size_2d]  # First part corresponds to 2D variables
        x_1d_flat = x[:, size_2d:]  # Remaining part corresponds to 1D variables

        x_2d = x_2d_flat.view(-1, num_2d_vars, num_depths, self.pred_len)
        x_2d = x_2d.permute(0, 3, 1, 2)
        try:
            x_1d = x_1d_flat.view(-1, num_1d_vars, self.pred_len).permute(0, 2, 1)
        except:
            x_1d = x_1d_flat.reshape(-1, num_1d_vars, self.pred_len).permute(0, 2, 1)
        return x_2d, x_1d
    
    def reshape_inp_ids(self, ids, num_2d_vars, num_1d_vars, num_depths):
        
        indices_1 = [self.seq_len * num_depths * i for i in range(num_2d_vars)]
        base = self.seq_len * num_depths * num_2d_vars
        indices_2 = [base + self.seq_len * i for i in range(num_1d_vars)]

        return ids[:, indices_1], ids[:, indices_2]
        

    def select_optimizer_(self):
        optimizer = torch.optim.Adam(self.model.parameters(), 
                                     lr=self.trainer.lr, 
                                     betas=(0.9, 0.95), 
                                     weight_decay=self.trainer.weight_decay)
        return optimizer


    def soft_info_nce(self, z, w):
        """
        z : (B,d)  L2‑normalised embeddings
        w : (B,B)  weights in [0,1]   (zero on diag if self excluded)
        """
        z = F.normalize(z, dim=1)
        sim = z @ z.T / self.trainer.tau_emb                       # (B,B)
        logp = sim - torch.logsumexp(sim, 1, keepdim=True)
        return -(w * logp).sum() / w.sum().clamp_min(1e-8)

    def cl_loss(self,
            z, 
            lake_ids, 
            global_idx,
            raw_window
            ):
        B, device = z.size(0), z.device
        eye = torch.eye(B, dtype=torch.bool, device=device)
        lake_ids = torch.tensor(lake_ids, device=z.device)
        same = lake_ids[:, None].eq(lake_ids[None, :])  # (B,B)
        
        if self.cl_mode == "hard":                            # hard CL 
            w = same.float()
        else:                                      # soft core
            dtw_matrix = self.dtw.get_dtw_matrix(global_idx, raw_window)
            w = 2 * self.trainer.alpha * torch.sigmoid(-self.trainer.tau_I * dtw_matrix)
            if self.cl_mode == "hardnegative":                        # hard negatives
                w.masked_fill_(~same, 1.0)

        w.masked_fill_(eye, 0.)                    # no self-pairs

        loss = self.soft_info_nce(z, w) if w.sum() > 0 else 0
        return loss

    def combined_loss(self, 
                    seq_Y, 
                    pred, 
                    mask_out, 
                    z, 
                    global_idx, 
                    raw_window, 
                    lake_ids,
                    epoch
                    ):
        
        # Use NLL loss for probabilistic forecasting, MSE for fallback
        forecasting_loss = self.compute_nll_loss(forecasts=pred, targets=seq_Y, mask=mask_out)

        # Add Student-t parameter regularizers if probabilistic forecasting
        reg_loss = 0.0
        if isinstance(pred, dict) and 'scale' in pred and 'df' in pred:
            scale = pred['scale']  # (B, S_t)
            df = pred['df']  # (B, S_t)
            

            # Apply mask to parameters if available
            if mask_out is not None:
                valid_scale = scale[mask_out.bool()]
                valid_df = df[mask_out.bool()]
            else:
                valid_scale = scale
                valid_df = df
            

            # Scale regularization: L2 on log(scale) to prevent runaway scales
            reg_scale_weight = getattr(self.trainer, 'reg_scale_weight', 1e-4)
            if reg_scale_weight > 0:
                # Ensure scale values are positive before taking log
                valid_scale_safe = torch.clamp(valid_scale, min=1e-6)
                reg_scale = reg_scale_weight * torch.mean(torch.log(valid_scale_safe) ** 2)
                reg_loss += reg_scale
            

            # Degrees of freedom regularization: L2 on (df - df_target) or log(df)
            reg_df_weight = getattr(self.trainer, 'reg_df_weight', 1e-4)
            df_target = getattr(self.trainer, 'df_target', 5.0)  # Target df (default 5)
            
            if reg_df_weight > 0:
                if not getattr(self.trainer, 'use_log_reg', True):
                    valid_df_safe = torch.clamp(valid_df, min=1e-6)
                    reg_df = reg_df_weight * torch.mean(torch.log(valid_df_safe) ** 2)
                else:
                    reg_df = reg_df_weight * torch.mean((valid_df - df_target) ** 2)
                reg_loss += reg_df
            
            if self.rank == 0 and hasattr(self, 'step_count'):
                if self.step_count % 50 == 0:  # Log every 50 steps
                    mean_scale = torch.mean(valid_scale).item()
                    mean_df = torch.mean(valid_df).item()
                    print(f"Step {self.step_count}: mean_scale={mean_scale:.4f}, mean_df={mean_df:.4f}")
        
        forecasting_loss = forecasting_loss + reg_loss

        if self.use_lake_cl:
            
            lake_cl_loss = self.cl_loss(z=z,
                                        lake_ids=lake_ids,
                                        global_idx=global_idx,
                                        raw_window=raw_window
                                        )
            if self.cl_loss_ema is None:
                self.cl_loss_ema = lake_cl_loss.item()
            else:
                self.cl_loss_ema = self.cl_ema_momentum * self.cl_loss_ema + (1 - self.cl_ema_momentum) * lake_cl_loss.item()
            
            warmup_weight = self.get_cl_warmup_weight(epoch)

            use_ema = getattr(self.trainer, 'cl_use_ema_for_weighting', True)
            cl_ref_mag = (self.cl_loss_ema if (use_ema and self.cl_loss_ema is not None)
                          else lake_cl_loss.item())

            loss_ratio = forecasting_loss.item() / (cl_ref_mag + 1e-8)
            adaptive_weight = self.trainer.lake_weight * min(1.0, loss_ratio * 0.1)  # Cap the weight
            
            if hasattr(self.trainer, 'cl_adaptive_scaling') and self.trainer.cl_adaptive_scaling:
                target_cl_magnitude = forecasting_loss.item() * 0.5  # Target CL to be 50% of forecast loss
                current_cl_magnitude = cl_ref_mag
                if current_cl_magnitude > 0:
                    adaptive_weight = self.trainer.lake_weight * (target_cl_magnitude / current_cl_magnitude)
                    adaptive_weight = torch.clamp(torch.tensor(adaptive_weight), min=0.01, max=2.0).item()
                else:
                    adaptive_weight = self.trainer.lake_weight
            else:
                adaptive_weight = self.trainer.lake_weight
            
            # Apply warmup weight
            final_weight = adaptive_weight * warmup_weight
            cl_loss_term = final_weight * lake_cl_loss
            
            cl_components = {
                'lake_cl_loss': lake_cl_loss.item(),
                'adaptive_weight': adaptive_weight,
                'warmup_weight': warmup_weight,
                'final_weight': final_weight,
                'cl_loss_term': cl_loss_term.item()
            }
        else:
            lake_cl_loss = None
            cl_loss_term = 0
        
        total_loss = forecasting_loss + cl_loss_term
        return total_loss, forecasting_loss, cl_loss_term, cl_components if self.use_lake_cl else None
            
    def get_avg_loss(self, batch_loss, num_batches):
        
        avg_batch_loss = batch_loss/num_batches
        avg_batch_loss_tensor = torch.tensor(avg_batch_loss, device=self.device)
        avg_batch_loss_global = reduce_mean(avg_batch_loss_tensor, dist.get_world_size()).item()

        return avg_batch_loss_global

    def _pad_and_concat_tensors(self, tensor_list, pad_value=0.0):

        if not tensor_list:
            return None
            
        seq_lengths = [t.shape[1] for t in tensor_list]
        if len(set(seq_lengths)) == 1:
            # All tensors have the same length, no padding needed
            return torch.cat(tensor_list, dim=0)
        
        # Find max sequence length
        max_seq_len = max(seq_lengths)
        
        # Pad and concatenate
        padded_tensors = []
        for t in tensor_list:
            if t.shape[1] < max_seq_len:
                pad_size = max_seq_len - t.shape[1]
                if len(t.shape) == 2:
                    padding = torch.full((t.shape[0], pad_size), pad_value,
                                       device=t.device, dtype=t.dtype)
                else:
                    # 3D tensor: (batch_size, sequence_length, features)
                    padding = torch.full((t.shape[0], pad_size, t.shape[2]), pad_value,
                                       device=t.device, dtype=t.dtype)
                padded_tensors.append(torch.cat([t, padding], dim=1))
            else:
                padded_tensors.append(t)
        
        return torch.cat(padded_tensors, dim=0)

    def _pad_and_concat_numpy_arrays(self, array_list, pad_value=''):
        if not array_list:
            return np.array([])
        
        # Check if all arrays have the same sequence length
        seq_lengths = [arr.shape[1] for arr in array_list]
        if len(set(seq_lengths)) == 1:
            # All arrays have the same length, no padding needed
            return np.concatenate(array_list, axis=0)
        
        # Find max sequence length
        max_seq_len = max(seq_lengths)
        
        # Pad and concatenate
        padded_arrays = []
        for arr in array_list:
            if arr.shape[1] < max_seq_len:
                pad_size = max_seq_len - arr.shape[1]
                if len(arr.shape) == 2:
                    padding = np.full((arr.shape[0], pad_size), pad_value, dtype=arr.dtype)
                else:
                    padding = np.full((arr.shape[0], pad_size, arr.shape[2]), pad_value, dtype=arr.dtype)
                padded_arrays.append(np.concatenate([arr, padding], axis=1))
            else:
                padded_arrays.append(arr)
        
        return np.concatenate(padded_arrays, axis=0)

    def val_one_epoch(self, dataloader, epoch, plot=True):
        
        num_plot_batches=self.cfg.num_plot_batches

        if plot and self.rank == 0:
            var_ids_2d_list = []
            depth_val_list = []
            time_val_list = []
            datetime_list = []
            preds_list_2d = []
            labels_list_2d = []
            masks_list_2d = []
            lake_names_list = []
            lake_id_list = []
        else:
            var_ids_2d_list = None
            depth_val_list = None
            time_val_list = None
            datetime_list = None
            preds_list_2d = None
            labels_list_2d = None
            masks_list_2d = None
            lake_names_list = []
            lake_id_list = []
        
        batch_loss = 0
        batch_pred_loss = 0
        batch_lake_contrastive_loss = 0
        batch_df_sum = 0.0
        batch_df_count = 0

        mse_sum = 0.0
        mae_sum = 0.0
        crps_sum = 0.0
        metric_count = 0

        batch_df_base_sums = None
        batch_df_base_counts = None
        global_num_variates = None  # size of variate vocabulary across all datasets
        df_base_by_var_mean = None
        df_base_mean = None

        avg_batch_loss = 0
        avg_batch_pred_loss = 0
        avg_batch_lake_contrastive_loss = None
        avg_batch_df_mean = None

        self.model.eval()

        if self.rank==0:
            pbar=tqdm(total=len(dataloader), desc=f'Epoch {epoch + 1}/{self.max_epochs}', unit='batch')

        for iteration, sample in enumerate(dataloader):
            seq_X = sample["flat_seq_x"].to(self.device)
            mask_X = sample["flat_mask_x"].to(self.device) 
            sample_ids_x = sample["sample_ids_x"].to(self.device) 
            time_ids_x = sample["time_ids_x"].to(self.device)
            var_ids_x = sample["var_ids_x"].to(self.device)
            padding_mask_x = sample["padding_mask_x"].to(self.device)
            depth_values_x = sample["depth_values_x"].to(self.device)
            time_values_x = sample["time_values_x"].to(self.device)
            
            lake_ids = sample["lake_id"] 
            lake_names = sample["lake_name"]
            idx = sample["idx"]
            num_2d_vars = sample['num2Dvars']
            num_1d_vars = sample['num1Dvars']
            num_depths = sample['num_depths']

            seq_Y = sample["flat_seq_y"].to(self.device)
            mask_Y = sample["flat_mask_y"].to(self.device)
            sample_ids_y = sample["sample_ids_y"].to(self.device)
            time_ids_y = sample["time_ids_y"].to(self.device)
            var_ids_y = sample["var_ids_y"].to(self.device)
            padding_mask_y = sample["padding_mask_y"].to(self.device)
            depth_values_y = sample["depth_values_y"].to(self.device)
            time_values_y = sample["time_values_y"].to(self.device)
            datetime_y = sample["datetime_strs_y"]

            mask_out = _updated_mask_(mask_Y=mask_Y, 
                                        padding_mask=padding_mask_y,
                                        pred_len=self.pred_len,
                                        seq_len=self.seq_len)

            model_ref = self.model.module if (plot and hasattr(self.model, "module")) else self.model
            with torch.autocast(device_type='cuda'):
                x_enc, pred, z = model_ref(data=seq_X,
                                             observed_mask=mask_X,
                                             sample_ids=sample_ids_x,
                                             variate_ids=var_ids_x,
                                             padding_mask=padding_mask_x,
                                             depth_values=depth_values_x,
                                             pred_len=self.pred_len,
                                             seq_len=self.seq_len,
                                             time_values=time_values_x,
                                             time_ids=time_ids_x,
                                             tgt_variate_ids=var_ids_y,
                                             tgt_time_values=time_values_y,
                                             tgt_time_ids=time_ids_y,
                                             tgt_depth_values=depth_values_y,
                                             tgt_padding_mask=padding_mask_y)

                if not plot:
                    loss, pred_loss, lake_contrastive_loss, cl_components = self.combined_loss(seq_Y=seq_Y, 
                                                                                pred=pred, 
                                                                                mask_out=mask_out, 
                                                                                z=z, 
                                                                                global_idx=idx, 
                                                                                raw_window=seq_X, 
                                                                                lake_ids=lake_ids,
                                                                                epoch=epoch)                                                                                       
                                                                                                        
                    loss_value = loss.item()
                    batch_loss += loss_value

                    batch_pred_loss += pred_loss.item()

                    with torch.no_grad():
                        valid_mask_bool = mask_out.bool()
                        if isinstance(pred, dict) and 'distribution' in pred:
                            pred_mean_tensor = pred['distribution'].mean
                            dist_obj = pred['distribution']
                        else:
                            pred_mean_tensor = pred
                            dist_obj = None

                        y_valid = seq_Y[valid_mask_bool]
                        yhat_valid = pred_mean_tensor[valid_mask_bool]
                        if y_valid.numel() > 0:
                            mse_sum += F.mse_loss(yhat_valid, y_valid, reduction='sum').item()
                            mae_sum += F.l1_loss(yhat_valid, y_valid, reduction='sum').item()
                            metric_count += y_valid.numel()

                        if dist_obj is not None and y_valid.numel() > 0:
                            K = 16
                            try:
                                samples = dist_obj.rsample((K,))  # (K, B, S)
                            except Exception:
                                samples = dist_obj.sample((K,))
                            samples = samples[:, valid_mask_bool]
                            if samples.numel() > 0:
                                yv = y_valid.unsqueeze(0)
                                term1 = (samples - yv).abs().mean(dim=0)  # (N,)
                                if K >= 2:
                                    s1 = samples[:K//2]
                                    s2 = samples[K//2:K]
                                    pair = (s1.unsqueeze(1) - s2.unsqueeze(0)).abs().mean(dim=(0,1))  # (N,)
                                else:
                                    pair = torch.zeros_like(term1)
                                crps_sum += (term1 - 0.5 * pair).sum().item()

                    if lake_contrastive_loss:
                        batch_lake_contrastive_loss += lake_contrastive_loss

                    if isinstance(pred, dict) and 'df' in pred:
                        valid_df = pred['df'][mask_out.bool()]
                        if len(valid_df) > 0:
                            batch_df_sum += valid_df.sum().item()
                            batch_df_count += len(valid_df)
                    if isinstance(pred, dict) and 'df_base' in pred:
                        valid_df_base = pred['df_base'][mask_out.bool()].float()
                        valid_var_ids = var_ids_y[mask_out.bool()].long()
                        if len(valid_df_base) > 0:
                            if global_num_variates is None:
                                try:
                                    global_num_variates = max(int(k) for k in self.data.id_to_var.keys()) + 1
                                except Exception:
                                    global_num_variates = valid_var_ids.max().item() + 1

                            if batch_df_base_sums is None:
                                batch_df_base_sums = torch.zeros(global_num_variates, device=self.device, dtype=torch.float32)
                                batch_df_base_counts = torch.zeros_like(batch_df_base_sums)
                            else:
                                current_size = batch_df_base_sums.numel()
                                need_size = max(global_num_variates, (valid_var_ids.max().item() + 1))
                                if need_size > current_size:
                                    new_sums = torch.zeros(need_size, device=self.device, dtype=batch_df_base_sums.dtype)
                                    new_counts = torch.zeros(need_size, device=self.device, dtype=batch_df_base_counts.dtype)
                                    new_sums[:current_size] = batch_df_base_sums
                                    new_counts[:current_size] = batch_df_base_counts
                                    batch_df_base_sums = new_sums
                                    batch_df_base_counts = new_counts

                            batch_df_base_sums.scatter_add_(0, valid_var_ids, valid_df_base)
                            batch_df_base_counts.scatter_add_(0, valid_var_ids, torch.ones_like(valid_df_base, dtype=torch.float32))

                    if not math.isfinite(loss_value):
                        if self.rank==0:
                            print("Loss is {}, stopping validation".format(loss_value))

                    loss /= self.trainer.accum_iter
            
            if iteration<num_plot_batches and self.rank==0 and plot:
                num_2d_vars=num_2d_vars[0] # assuming uniform sampling in a batch
                num_1d_vars=num_1d_vars[0]
                num_depths=num_depths[0]

                if isinstance(pred, dict) and 'distribution' in pred:
                    pred_to_store = {
                        'loc': pred['loc'].detach(),
                        'scale': pred['scale'].detach(),
                        'df': pred['df'].detach(),
                        'mean': pred['distribution'].mean.detach()
                    }
                else:
                    pred_to_store = pred.detach()
                
                if isinstance(pred_to_store, dict):
                    preds_list_2d.append({k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v)
                                          for k, v in pred_to_store.items()})
                else:
                    preds_list_2d.append(pred_to_store.detach().cpu() if isinstance(pred_to_store, torch.Tensor) else pred_to_store)

                labels_list_2d.append(seq_Y.detach().cpu())

                masks_list_2d.append(mask_out.detach().cpu())

                var_ids_2d_list.append(var_ids_y.detach().cpu())
                
                depth_val_list.append(depth_values_y.detach().cpu())
                time_val_list.append(time_values_y.detach().cpu())
                datetime_list.append(datetime_y)

                lake_names_list.append(lake_names)
                lake_id_list.append(lake_ids)
            
            if self.rank==0:
                pbar.update(1)

        if not plot:
            avg_batch_loss = self.get_avg_loss(batch_loss, len(dataloader))
            avg_batch_pred_loss = self.get_avg_loss(batch_pred_loss, len(dataloader))
            
            avg_batch_df_mean = None
            if batch_df_count > 0:
                df_sum_tensor = torch.tensor(batch_df_sum, device=self.device)
                df_count_tensor = torch.tensor(batch_df_count, device=self.device, dtype=torch.float32)
                
                world_size = dist.get_world_size()
                global_df_sum = reduce_mean(df_sum_tensor, world_size).item()
                global_df_count = reduce_mean(df_count_tensor, world_size).item()
                
                total_df_sum = global_df_sum * world_size
                total_df_count = global_df_count * world_size
                avg_batch_df_mean = total_df_sum / total_df_count if total_df_count > 0 else None

            df_base_by_var_mean = None
            df_base_mean = None
            if batch_df_base_sums is not None:
                world_size = dist.get_world_size()
                global_df_base_sums = reduce_mean(batch_df_base_sums, world_size) * world_size
                global_df_base_counts = reduce_mean(batch_df_base_counts, world_size) * world_size
                df_base_by_var_mean = (global_df_base_sums / global_df_base_counts.clamp_min(1e-6)).detach().cpu()
                total_counts = global_df_base_counts.clone()
                total_sums = global_df_base_sums.clone()
                if total_counts.numel() > 1:
                    nonpad_count = total_counts[1:].sum().item()
                    if nonpad_count > 0:
                        df_base_mean = (total_sums[1:].sum() / nonpad_count).item()

            if metric_count > 0:
                mse_tensor = torch.tensor(mse_sum, device=self.device)
                mae_tensor = torch.tensor(mae_sum, device=self.device)
                crps_tensor = torch.tensor(crps_sum, device=self.device)
                cnt_tensor = torch.tensor(metric_count, device=self.device, dtype=torch.float32)
                world_size = dist.get_world_size()
                mse_total = reduce_mean(mse_tensor, world_size) * world_size
                mae_total = reduce_mean(mae_tensor, world_size) * world_size
                crps_total = reduce_mean(crps_tensor, world_size) * world_size
                cnt_total = reduce_mean(cnt_tensor, world_size) * world_size
                val_mse = (mse_total / cnt_total).item()
                val_mae = (mae_total / cnt_total).item()
                val_crps = (crps_total / cnt_total).item()
            else:
                val_mse = None
                val_mae = None
                val_crps = None

            if self.use_lake_cl:
                avg_batch_lake_contrastive_loss = self.get_avg_loss(batch_lake_contrastive_loss, len(dataloader))
            else:
                avg_batch_lake_contrastive_loss = None
            

        if self.rank==0 and plot:
            if len(preds_list_2d) > 0:
                if isinstance(preds_list_2d[0], dict):
                    preds_list_2d = {
                        'loc': self._pad_and_concat_tensors([p['loc'] for p in preds_list_2d]),
                        'scale': self._pad_and_concat_tensors([p['scale'] for p in preds_list_2d]),
                        'df': self._pad_and_concat_tensors([p['df'] for p in preds_list_2d]),
                        'mean': self._pad_and_concat_tensors([p['mean'] for p in preds_list_2d])
                    }
                else:
                    preds_list_2d = self._pad_and_concat_tensors(preds_list_2d)
                
                labels_list_2d = self._pad_and_concat_tensors(labels_list_2d)
                masks_list_2d = self._pad_and_concat_tensors(masks_list_2d)
                var_ids_2d_list = self._pad_and_concat_tensors(var_ids_2d_list)
                depth_val_list = self._pad_and_concat_tensors(depth_val_list)
                time_val_list = self._pad_and_concat_tensors(time_val_list)
                if len(datetime_list) > 0:
                    datetime_list = self._pad_and_concat_numpy_arrays(datetime_list)
                else:
                    datetime_list = np.array([])

                lake_names_list = [name for batch in lake_names_list for name in batch]
                lake_id_list = [lid for batch in lake_id_list for lid in batch]
            else:
                preds_list_2d = None
                labels_list_2d = None
                masks_list_2d = None
                var_ids_2d_list = None
                depth_val_list = None
                time_val_list = None
                datetime_list = None
                lake_names_list = []
                lake_id_list = []

        
        return {'loss': avg_batch_loss,
                'pred_loss': avg_batch_pred_loss,
                'lake_contrastive_loss': avg_batch_lake_contrastive_loss,
                'preds2d': preds_list_2d,
                'labels2d': labels_list_2d,
                'masks2d': masks_list_2d,
                'var_ids_2d': var_ids_2d_list,
                'depth_vals': depth_val_list,
                'time_vals': time_val_list,
                'datetime_strs': datetime_list,
                'lake_names': lake_names_list,
                'lake_ids': lake_id_list,
                'df_mean': avg_batch_df_mean,
                'val_mse': val_mse if not plot else None,
                'val_mae': val_mae if not plot else None,
                'val_crps': val_crps if not plot else None,
                'df_base_by_var_mean': df_base_by_var_mean}
    
    def train_one_epoch(self, 
                        dataloader,
                        optimizer, 
                        scheduler,
                        scaler,
                        epoch):
    
        avg_batch_loss = 0
        batch_loss = 0
        batch_pred_loss = 0
        batch_lake_contrastive_loss = 0
        batch_gradient_norm = 0
        
        mse_sum = 0.0
        mae_sum = 0.0
        crps_sum = 0.0
        metric_count = 0
        
        batch_df_sum = 0.0
        batch_df_count = 0
        batch_df_base_sums = None
        batch_df_base_counts = None
        global_num_variates = None
        df_base_by_var_mean = None
        df_base_mean = None
        
        cl_components_accum = {
            'lake_cl_loss': 0.0,
            'adaptive_weight': 0.0,
            'warmup_weight': 0.0,
            'final_weight': 0.0,
            'cl_loss_term': 0.0
        }
        cl_batch_count = 0

        optimizer.zero_grad()
        
        self.model.train()

        if self.rank==0:
            pbar=tqdm(total=len(dataloader), desc=f'Epoch {epoch + 1}/{self.max_epochs}', unit='batch')

        for iteration, sample in enumerate(dataloader):
            seq_X = sample["flat_seq_x"].to(self.device)
            mask_X = sample["flat_mask_x"].to(self.device) 
            sample_ids_x = sample["sample_ids_x"].to(self.device) 
            time_ids_x = sample["time_ids_x"].to(self.device)
            var_ids_x = sample["var_ids_x"].to(self.device)
            padding_mask_x = sample["padding_mask_x"].to(self.device)
            depth_values_x = sample["depth_values_x"].to(self.device)
            time_values_x = sample["time_values_x"].to(self.device)

            lake_ids = sample["lake_id"] 
            lake_names = sample["lake_name"]
            idx = sample["idx"]

            # target data
            seq_Y = sample["flat_seq_y"].to(self.device)
            mask_Y = sample["flat_mask_y"].to(self.device)
            sample_ids_y = sample["sample_ids_y"].to(self.device)
            time_ids_y = sample["time_ids_y"].to(self.device)
            var_ids_y = sample["var_ids_y"].to(self.device)
            padding_mask_y = sample["padding_mask_y"].to(self.device)
            depth_values_y = sample["depth_values_y"].to(self.device)
            time_values_y = sample["time_values_y"].to(self.device)

            mask_out = _updated_mask_(mask_Y=mask_Y, 
                                        padding_mask=padding_mask_y,
                                        pred_len=self.pred_len,
                                        seq_len=self.seq_len)
                                        
            with torch.autocast(device_type='cuda'):
                x_enc, pred, z = self.model(data=seq_X,
                                            observed_mask=mask_X,
                                            sample_ids=sample_ids_x,
                                            variate_ids=var_ids_x,
                                            padding_mask=padding_mask_x,
                                            depth_values=depth_values_x,
                                            pred_len=self.pred_len,
                                            seq_len=self.seq_len,
                                            time_values=time_values_x,
                                            time_ids=time_ids_x,
                                            tgt_variate_ids=var_ids_y,
                                            tgt_time_values=time_values_y,
                                            tgt_time_ids=time_ids_y,
                                            tgt_depth_values=depth_values_y,
                                            tgt_padding_mask=padding_mask_y)

                '''
                compute loss
                '''
                loss, pred_loss, lake_contrastive_loss, cl_components = self.combined_loss(seq_Y=seq_Y, 
                                                                            pred=pred, 
                                                                            mask_out=mask_out, 
                                                                            z=z, 
                                                                            global_idx=idx, 
                                                                            raw_window=seq_X, 
                                                                            lake_ids=lake_ids,
                                                                            epoch=epoch)

                loss_value = loss.item()
                batch_loss += loss_value

                batch_pred_loss += pred_loss.item()
                if lake_contrastive_loss:
                    batch_lake_contrastive_loss += lake_contrastive_loss
                
                # Track distribution parameters if available
                if isinstance(pred, dict) and 'df' in pred:
                    valid_df = pred['df'][mask_out.bool()]
                    if len(valid_df) > 0:
                        batch_df_sum += valid_df.sum().item()
                        batch_df_count += len(valid_df)
                # Track variate-wise base df if available
                if isinstance(pred, dict) and 'df_base' in pred:
                    valid_df_base = pred['df_base'][mask_out.bool()].float()
                    valid_var_ids = var_ids_y[mask_out.bool()].long()
                    if len(valid_df_base) > 0:
                        # Determine global variate vocabulary size once
                        if global_num_variates is None:
                            try:
                                global_num_variates = max(int(k) for k in self.data.id_to_var.keys()) + 1
                            except Exception:
                                global_num_variates = valid_var_ids.max().item() + 1

                        # Initialize or grow accumulators as needed
                        if batch_df_base_sums is None:
                            batch_df_base_sums = torch.zeros(global_num_variates, device=self.device, dtype=torch.float32)
                            batch_df_base_counts = torch.zeros_like(batch_df_base_sums)
                        else:
                            current_size = batch_df_base_sums.numel()
                            need_size = max(global_num_variates, (valid_var_ids.max().item() + 1))
                            if need_size > current_size:
                                new_sums = torch.zeros(need_size, device=self.device, dtype=batch_df_base_sums.dtype)
                                new_counts = torch.zeros(need_size, device=self.device, dtype=batch_df_base_counts.dtype)
                                new_sums[:current_size] = batch_df_base_sums
                                new_counts[:current_size] = batch_df_base_counts
                                batch_df_base_sums = new_sums
                                batch_df_base_counts = new_counts

                        # Accumulate df_base sums and counts per variable
                        batch_df_base_sums.scatter_add_(0, valid_var_ids, valid_df_base)
                        batch_df_base_counts.scatter_add_(0, valid_var_ids, torch.ones_like(valid_df_base, dtype=torch.float32))
                
                if cl_components is not None:
                    for key in cl_components_accum:
                        cl_components_accum[key] += cl_components[key]
                    cl_batch_count += 1
                
                if not math.isfinite(loss_value):
                    if self.rank==0:
                        print("Loss is {}, stopping training".format(loss_value))
                    # sys.exit(1)

                loss /= self.trainer.accum_iter
                
                scaler.scale(loss).backward()

                if (iteration + 1) % self.trainer.accum_iter == 0:
                    if self.rank == 0 and iteration % 50 == 0:  # Print every 50 iterations
                        with torch.no_grad():
                            total_norm = torch.sqrt(sum(p.grad.norm()**2 for p in self.model.parameters() if p.grad is not None))
                            batch_gradient_norm += total_norm
                        print(f"Gradient norm: {total_norm:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

                    # Apply gradient clipping
                    if self.trainer.gradient_clip_val > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.trainer.gradient_clip_val,  # Use config value
                            norm_type=2.0
                        )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad() # reset grads after update
                
                with torch.no_grad():
                    valid_mask_bool = mask_out.bool()
                    if isinstance(pred, dict) and 'distribution' in pred:
                        pred_mean_tensor = pred['distribution'].mean
                        dist_obj = pred['distribution']
                    else:
                        pred_mean_tensor = pred
                        dist_obj = None

                    y_valid = seq_Y[valid_mask_bool]
                    yhat_valid = pred_mean_tensor[valid_mask_bool]
                    if y_valid.numel() > 0:
                        mse_sum += F.mse_loss(yhat_valid, y_valid, reduction='sum').item()
                        mae_sum += F.l1_loss(yhat_valid, y_valid, reduction='sum').item()
                        metric_count += y_valid.numel()

                    if dist_obj is not None and y_valid.numel() > 0:
                        K = 16
                        try:
                            samples = dist_obj.rsample((K,))  # (K, B, S)
                        except Exception:
                            samples = dist_obj.sample((K,))
                        samples = samples[:, valid_mask_bool]
                        if samples.numel() > 0:
                            yv = y_valid.unsqueeze(0)
                            term1 = (samples - yv).abs().mean(dim=0)  # (N,)
                            if K >= 2:
                                s1 = samples[:K//2]
                                s2 = samples[K//2:K]
                                pair = (s1.unsqueeze(1) - s2.unsqueeze(0)).abs().mean(dim=(0,1))  # (N,)
                            else:
                                pair = torch.zeros_like(term1)
                            crps_sum += (term1 - 0.5 * pair).sum().item()

            if self.rank==0:
                pbar.update(1)

        avg_batch_loss = self.get_avg_loss(batch_loss, len(dataloader))
        avg_batch_pred_loss = self.get_avg_loss(batch_pred_loss, len(dataloader))
        avg_batch_gradient_norm = self.get_avg_loss(batch_gradient_norm, len(dataloader))

        avg_batch_df_mean = None
        if batch_df_count > 0:
            df_sum_tensor = torch.tensor(batch_df_sum, device=self.device)
            df_count_tensor = torch.tensor(batch_df_count, device=self.device, dtype=torch.float32)
            
            world_size = dist.get_world_size()
            global_df_sum = reduce_mean(df_sum_tensor, world_size).item()
            global_df_count = reduce_mean(df_count_tensor, world_size).item()
            
            total_df_sum = global_df_sum * world_size
            total_df_count = global_df_count * world_size
            avg_batch_df_mean = total_df_sum / total_df_count if total_df_count > 0 else None

        if self.use_lake_cl:
            avg_batch_lake_contrastive_loss = self.get_avg_loss(batch_lake_contrastive_loss, len(dataloader))
        else:
            avg_batch_lake_contrastive_loss = None

        # Compute variate-wise df_base means across ranks
        df_base_by_var_mean = None
        df_base_mean = None
        if batch_df_base_sums is not None:
            world_size = dist.get_world_size()
            global_df_base_sums = reduce_mean(batch_df_base_sums, world_size) * world_size
            global_df_base_counts = reduce_mean(batch_df_base_counts, world_size) * world_size
            df_base_by_var_mean = (global_df_base_sums / global_df_base_counts.clamp_min(1e-6)).detach().cpu()
            # Overall mean excluding PAD id 0
            total_counts = global_df_base_counts.clone()
            total_sums = global_df_base_sums.clone()
            if total_counts.numel() > 1:
                nonpad_count = total_counts[1:].sum().item()
                if nonpad_count > 0:
                    df_base_mean = (total_sums[1:].sum() / nonpad_count).item()

        if metric_count > 0:
            mse_tensor = torch.tensor(mse_sum, device=self.device)
            mae_tensor = torch.tensor(mae_sum, device=self.device)
            crps_tensor = torch.tensor(crps_sum, device=self.device)
            cnt_tensor = torch.tensor(metric_count, device=self.device, dtype=torch.float32)
            world_size = dist.get_world_size()
            mse_total = reduce_mean(mse_tensor, world_size) * world_size
            mae_total = reduce_mean(mae_tensor, world_size) * world_size
            crps_total = reduce_mean(crps_tensor, world_size) * world_size
            cnt_total = reduce_mean(cnt_tensor, world_size) * world_size
            train_mse = (mse_total / cnt_total).item()
            train_mae = (mae_total / cnt_total).item()
            train_crps = (crps_total / cnt_total).item()
        else:
            train_mse = None
            train_mae = None
            train_crps = None

        if self.rank == 0 and self.use_lake_cl and cl_batch_count > 0:
            avg_cl_components = {key: value / cl_batch_count for key, value in cl_components_accum.items()}
            print(f"Epoch {epoch + 1} - Contrastive Learning Summary:")
            print(f"  lake_cl_loss: {avg_cl_components['lake_cl_loss']:.6f}")
            print(f"  Weight components - adaptive_weight: {avg_cl_components['adaptive_weight']:.6f}, warmup_weight: {avg_cl_components['warmup_weight']:.6f}, final_weight: {avg_cl_components['final_weight']:.6f}")
            print(f"  Final CL term - cl_loss_term: {avg_cl_components['cl_loss_term']:.6f}")

        return {'loss': avg_batch_loss,
                'pred_loss': avg_batch_pred_loss,
                'lake_contrastive_loss': avg_batch_lake_contrastive_loss,
                'gradient_norm': avg_batch_gradient_norm,
                'df_mean': avg_batch_df_mean,
                'train_mse': train_mse,
                'train_mae': train_mae,
                'train_crps': train_crps,
                'df_base_by_var_mean': df_base_by_var_mean}
    
    def update_split(self, datasets, flag='train'):
        updated_ds = copy.deepcopy(datasets)
        for ds in updated_ds:
            ds.__split__(flag=flag)
        return updated_ds

    def get_finetune_lr_scheduler(self, 
                                optimizer, 
                                warmup_epochs,
                                max_epochs):
        base_lr = optimizer.param_groups[0]['lr']
        start_lr = base_lr / 2
        max_warmup_lr = base_lr

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                warmup_fraction = epoch / float(max(1, warmup_epochs))
                current_lr = start_lr + (max_warmup_lr - start_lr) * warmup_fraction
                return current_lr / base_lr
            else:
                cosine_factor = 0.5 * (1.0 + math.cos(math.pi * (epoch - warmup_epochs) / (max_epochs - warmup_epochs)))
                return (max_warmup_lr / base_lr) * cosine_factor
        
        if warmup_epochs is not None:
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        else:
            # No warmup, use cosine annealing from base_lr
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    def get_lr_scheduler(self, optimizer, max_epochs, warmup_epochs=None, max_warmup_lr=None):
        base_lr = optimizer.param_groups[0]['lr']
        max_warmup_lr = max_warmup_lr if max_warmup_lr is not None else base_lr
        
        def lr_lambda(epoch):
            if warmup_epochs is not None and epoch < warmup_epochs:
                warmup_fraction = float(epoch) / float(max(1, warmup_epochs))
                return warmup_fraction * (max_warmup_lr / base_lr)
            else:
                cosine_factor = 0.5 * (1.0 + math.cos(math.pi * (epoch - warmup_epochs) / (max_epochs - warmup_epochs)))
                return (max_warmup_lr / base_lr) * cosine_factor
        
        if warmup_epochs is not None:
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        else:
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
        
    def get_multistage_scheduler(self, optimizer, stage_config):
        """
        Create a multi-stage scheduler based on stage configuration.
        
        Args:
            optimizer: PyTorch optimizer
            stage_config: Dict with stage definitions
                Example: {
                    'warmup_epochs': 20,
                    'cosine_epochs': 30, 
                    'linear_epochs': 30,
                    'max_warmup_lr': 3e-4,
                    'cosine_start_lr': 3e-4,
                    'final_lr': 1e-6
                }
        """
        warmup_epochs = stage_config.get('warmup_epochs', 0)
        cosine_epochs = stage_config.get('cosine_epochs', 0)
        linear_epochs = stage_config.get('linear_epochs', 0)
        total_epochs = warmup_epochs + cosine_epochs + linear_epochs
        
        max_warmup_lr = stage_config.get('max_warmup_lr', optimizer.param_groups[0]['lr'])
        cosine_start_lr = stage_config.get('cosine_start_lr', max_warmup_lr)
        final_lr = stage_config.get('final_lr', 1e-6)
        
        base_lr = optimizer.param_groups[0]['lr']
        
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                # Stage 1: Linear warmup
                return (epoch / warmup_epochs) * (max_warmup_lr / base_lr)
            elif epoch < warmup_epochs + cosine_epochs:
                # Stage 2: Cosine annealing
                cosine_epoch = epoch - warmup_epochs
                cosine_factor = 0.5 * (1 + math.cos(math.pi * cosine_epoch / cosine_epochs))
                return (cosine_start_lr / base_lr) * cosine_factor
            else:
                # Stage 3: Linear decay
                linear_epoch = epoch - warmup_epochs - cosine_epochs
                decay_factor = 1 - (linear_epoch / linear_epochs)
                current_lr = max(final_lr, cosine_start_lr * 0.5 * decay_factor)  # Don't go below final_lr
                return current_lr / base_lr
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def pretrain(self, datasets, plot_dataset=None, resume_states=None):
        if self.rank==0:
            config = init_wandb(self.trainer, self.cfg.task_name)
        
        print_model_and_data_info(self.model, datasets, self.rank)
        
        self.model.to(self.device)

        optimizer = self.select_optimizer_()
        if self.trainer.use_lr_scheduler:
            if resume_states is not None:
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']/2
                model_scheduler = self.get_finetune_lr_scheduler(optimizer=optimizer, 
                                                                warmup_epochs=self.trainer.warmup_epochs,
                                                                max_epochs=self.max_epochs)
                print("Last LR: ", model_scheduler.get_last_lr())
            else:
                model_scheduler = self.get_lr_scheduler(optimizer=optimizer, 
                                                    max_epochs=self.max_epochs,
                                                    warmup_epochs=self.trainer.warmup_epochs)
        else:
            model_scheduler = None       
        scaler = torch.amp.GradScaler()

        # data loader
        train_dataloader = build_dataloader(datasets=datasets, 
                                            cfg=self.cfg.dataloader, 
                                            pad_value_id=self.pad_value_id, 
                                            pad_value_default=self.pad_value_default,
                                            distributed=True,
                                            use_cl=self.trainer.use_lake_cl)   
        
        val_datasets = self.update_split(datasets, flag='val')
        val_dataloader = build_dataloader(datasets=val_datasets, 
                                            cfg=self.cfg.dataloader, 
                                            pad_value_id=self.pad_value_id, 
                                            pad_value_default=self.pad_value_default,
                                            distributed=True,
                                            use_cl=self.trainer.use_lake_cl)

        # plot data loader
        if plot_dataset:
            plot_train_dataloader = build_dataloader(datasets=plot_dataset, 
                                                     cfg=self.cfg.dataloader, 
                                                     pad_value_id=self.pad_value_id, 
                                                     pad_value_default=self.pad_value_default,
                                                     distributed=False,
                                                     use_cl=self.trainer.use_lake_cl,
                                                     plot=True)
            plot_val_datasets = self.update_split([plot_dataset], flag='val')
            plot_val_dataloader = build_dataloader(datasets=plot_val_datasets, 
                                                   cfg=self.cfg.dataloader, 
                                                   pad_value_id=self.pad_value_id, 
                                                   pad_value_default=self.pad_value_default,
                                                   distributed=False,
                                                   use_cl=self.trainer.use_lake_cl,
                                                   plot=True)   
        else:
            plot_train_dataloader = None
            plot_val_dataloader = None

        min_vali_loss = float("inf")
        val_loss = 0
        losses = np.full(self.max_epochs, np.nan)

        ckpt_path=os.path.join(self.trainer.pretrain_ckpts_dir, self.trainer.wandb_name)
        if not os.path.exists(ckpt_path) and self.rank==0:
            os.makedirs(ckpt_path)

        # Resume from checkpoint if specified
        start_epoch = 0
        
        # Resume from state_dicts from the main function (that handles ckpt paths)
        if resume_states is not None:
            _, _ = self.resume_training_from_checkpoint(resume_states, 
                                                        optimizer, 
                                                        model_scheduler, 
                                                        scaler)

        with trange(start_epoch, self.max_epochs) as tr:
            for epoch in tr:
                # Ensure distributed samplers reshuffle differently each epoch
                if hasattr(train_dataloader, 'sampler') and hasattr(train_dataloader.sampler, 'set_epoch'):
                    train_dataloader.sampler.set_epoch(epoch)
                if hasattr(train_dataloader, 'batch_sampler') and hasattr(train_dataloader.batch_sampler, 'set_epoch'):
                    train_dataloader.batch_sampler.set_epoch(epoch)

                if hasattr(val_dataloader, 'sampler') and hasattr(val_dataloader.sampler, 'set_epoch'):
                    val_dataloader.sampler.set_epoch(epoch)
                if hasattr(val_dataloader, 'batch_sampler') and hasattr(val_dataloader.batch_sampler, 'set_epoch'):
                    val_dataloader.batch_sampler.set_epoch(epoch)
                
                epoch_time = time.time()
                # Per-epoch dynamic window selection for TRAIN
                if getattr(self.cfg.data, 'dynamic_windows', True):
                    train_ds_list = getattr(train_dataloader.dataset, 'datasets', [train_dataloader.dataset])
                    chosen_ctx, chosen_pred = None, None
                    for ds in train_ds_list:
                        if hasattr(ds, 'sample_feasible_window'):
                            chosen_ctx, chosen_pred = ds.sample_feasible_window(epoch=epoch)
                            break
                    if chosen_ctx is not None and chosen_pred is not None:
                        for ds in train_ds_list:
                            if hasattr(ds, 'set_window_lengths'):
                                ds.set_window_lengths(chosen_ctx, chosen_pred)
                        self.seq_len = chosen_ctx
                        self.pred_len = chosen_pred
                        if self.rank == 0:
                            print(f"[TRAIN] Using window ctx={chosen_ctx}, pred={chosen_pred} for epoch {epoch}")
                train_elements = self.train_one_epoch(dataloader=train_dataloader,
                                                      optimizer=optimizer, 
                                                      scheduler=model_scheduler,
                                                      scaler=scaler,
                                                      epoch=epoch)

                if model_scheduler is not None:
                    model_scheduler.step()
                else:
                    pass

                train_loss = train_elements['loss']
                train_pred_loss = train_elements['pred_loss']
                train_lake_contrastive_loss = train_elements['lake_contrastive_loss']
                grad_norm = train_elements['gradient_norm']
                train_df_mean = train_elements.get('df_mean', None)
                train_df_base_mean = train_elements.get('df_base_mean', None)
                train_mse = train_elements.get('train_mse', None)
                train_mae = train_elements.get('train_mae', None)
                train_crps = train_elements.get('train_crps', None)
                
                if self.rank==0:
                    print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

                if (epoch % self.cfg.eval_freq==0):
                    if self.rank==0:
                        pretty_print("Validating")
                    val_ds_list = getattr(val_dataloader.dataset, 'datasets', [val_dataloader.dataset])
                    for ds in val_ds_list:
                        ds.set_window_lengths(self.cfg.seq_len, self.cfg.pred_len)
                    self.seq_len = self.cfg.seq_len
                    self.pred_len = self.cfg.pred_len

                    with torch.no_grad():
                        t_one_e = time.time()
                        val_elements = self.val_one_epoch(dataloader=val_dataloader,
                                                          epoch=epoch,
                                                          plot=False)
                        val_loss = val_elements['loss']
                        val_pred_loss = val_elements['pred_loss']
                        val_lake_contrastive_loss = val_elements['lake_contrastive_loss']
                        val_df_mean = val_elements.get('df_mean', None)
                        val_mse = val_elements.get('val_mse', None)
                        val_mae = val_elements.get('val_mae', None)
                        val_crps = val_elements.get('val_crps', None)

                        t_one_e_ = time.time()
                        
                        if self.rank==0:
                            print(f"Time taken for 1 validation epoch = {t_one_e_ - t_one_e}")

                        if self.rank==0:
                            pretty_print("Done Validating")
                losses[epoch] = train_loss
                
                if epoch%self.cfg.eval_freq==0:
                    metrics = {
                        "train_loss":train_loss,
                        "train_pred_loss":train_pred_loss,
                        "train_lake_CL":train_lake_contrastive_loss,
                        "train_grad_norm": grad_norm,
                        "learning_rate": optimizer.param_groups[0]['lr'],
                        "val_loss": val_loss,
                        "val_pred_loss": val_pred_loss,
                        "val_lake_CL": val_lake_contrastive_loss,
                    }
                    # Add train df if available
                    if train_df_mean is not None:
                        metrics["train_df_mean"] = train_df_mean
                    
                    # Add val df if available
                    if val_df_mean is not None:
                        metrics["val_df_mean"] = val_df_mean
                    
                    # train metrics
                    if train_mse is not None:
                        metrics["train_mse"] = train_mse
                    if train_mae is not None:
                        metrics["train_mae"] = train_mae
                    if train_crps is not None:
                        metrics["train_crps"] = train_crps

                    # Validation metrics
                    if val_mse is not None:
                        metrics["val_mse"] = val_mse
                    if val_mae is not None:
                        metrics["val_mae"] = val_mae
                    if val_crps is not None:
                        metrics["val_crps"] = val_crps

                    # Per-variable df_base logging (train and val) if available
                    train_df_base_by_var = train_elements.get('df_base_by_var_mean', None)
                    val_df_base_by_var = val_elements.get('df_base_by_var_mean', None) if 'val_elements' in locals() else None
                    id_to_var = getattr(self.data, 'id_to_var', None)
                    max_vars_to_log = getattr(self.trainer, 'max_vars_to_log', 50)
                    
                    if id_to_var is not None:
                        if train_df_base_by_var is not None:
                            counts_mask = train_df_base_by_var.numpy() > 0  # zeros imply no observations
                            for var_id in range(1, min(len(train_df_base_by_var), max_vars_to_log+1)):
                                if counts_mask[var_id-1]:
                                    var_name = id_to_var.get(var_id, f"var_{var_id}")
                                    metrics[f"train_df_base/{var_name}"] = float(train_df_base_by_var[var_id])
                        
                        if val_df_base_by_var is not None:
                            counts_mask = val_df_base_by_var.numpy() > 0
                            for var_id in range(1, min(len(val_df_base_by_var), max_vars_to_log+1)):
                                if counts_mask[var_id-1]:
                                    var_name = id_to_var.get(var_id, f"var_{var_id}")
                                    metrics[f"val_df_base/{var_name}"] = float(val_df_base_by_var[var_id])
                else:
                    metrics = {
                        "epoch": epoch,
                        "train_loss":train_loss,
                        "train_pred_loss":train_pred_loss,
                        "train_lake_CL":train_lake_contrastive_loss,
                        "train_grad_norm": grad_norm,
                        "learning_rate": optimizer.param_groups[0]['lr'],
                    }
                    # Add train df if available
                    if train_df_mean is not None:
                        metrics["train_df_mean"] = train_df_mean
                    if train_df_base_mean is not None:
                        metrics["train_df_base_mean"] = train_df_base_mean
                if (epoch % self.cfg.plot_freq==0):
                    # torch.distributed.barrier()

                    if self.rank==0:
                        with self.model.no_sync():
                            self.plot_predictions(loader=plot_val_dataloader, flag='val', it=epoch)
                            self.plot_predictions(loader=plot_train_dataloader, flag='train', it=epoch)
                    
                    # torch.distributed.barrier()

                if self.rank==0:
                    tr.set_postfix(metrics)
                    wandb.log(metrics)

                # checkpoint saving
                if (not min_vali_loss or val_loss <= min_vali_loss) and self.rank==0:
                    
                    print(
                        "Validation loss decreased ({0:.4f} --> {1:.4f}).  Saving model epoch{2} ...".format(min_vali_loss, val_loss, epoch))

                    min_vali_loss = val_loss
                    model_ckpt = {
                        'epoch': epoch, 
                        'model_state_dict': self.model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': model_scheduler.state_dict() if model_scheduler is not None else None,
                        'scaler_state_dict': scaler.state_dict(),
                        'min_vali_loss': min_vali_loss,
                        'training_stage': getattr(self.trainer, 'training_stage', 'stage1')
                    }
                    path = os.path.join(ckpt_path, self.trainer.best_ckpt)
                    torch.save(model_ckpt, path)

                if (epoch + 1) % self.cfg.trainer.save_freq == 0 and self.rank==0:
                    print("Saving model at epoch {}...".format(epoch + 1))
                    model_ckpt = {
                        'epoch': epoch, 
                        'model_state_dict': self.model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': model_scheduler.state_dict() if model_scheduler is not None else None,
                        'scaler_state_dict': scaler.state_dict(),
                        'min_vali_loss': min_vali_loss,
                        'training_stage': getattr(self.trainer, 'training_stage', 'stage1')
                    }
                    ckpt_name = self.trainer.ckpt_filename.format(epoch)
                    path = os.path.join(ckpt_path, ckpt_name)
                    torch.save(model_ckpt, path)

        if self.rank==0:
            wandb.summary['train_nll_loss'] = train_pred_loss
            wandb.summary['val_nll_loss'] = val_pred_loss
            wandb.summary['train_mse'] = train_mse
            wandb.summary['train_mae'] = train_mae
            wandb.summary['train_crps'] = train_crps
            wandb.summary['val_mse'] = val_mse
            wandb.summary['val_mae'] = val_mae
            wandb.summary['val_crps'] = val_crps
            wandb.finish()
            pretty_print("Model Pre-trained")
        return losses, self.model


    def plot_predictions(self, loader, flag, it):
    
        pretty_print(f"Prediction Visualization :: {flag}")
        elements = self.val_one_epoch(dataloader=loader, epoch=it)
        preds=elements['preds2d']
        gt=elements['labels2d']
        var_ids=elements["var_ids_2d"]
        depth_vals=elements['depth_vals']
        time_vals=elements['time_vals']
        datetime_raw_vals = elements['datetime_strs']
        gt_mask=elements['masks2d']
        
        # Handle total_samples for both dict and tensor predictions
        if isinstance(preds, dict):
            total_samples = preds['mean'].shape[0]
        else:
            total_samples = preds.shape[0] if preds is not None else 0

        id_to_var = self.data.id_to_var

        num_windows = self.cfg.num_plot_batches
        num_samples = self.cfg.plot_num_samples

        # Check if we have data to plot
        if preds is not None and gt is not None:
            start = 0
            idx = np.arange(start, start+total_samples, 1)
            plt_idx = np.floor(np.linspace(0, 1, 1))
            try:
                self.irregular_plotter.plot_forecast_irregular_grid(
                    gt_row=gt,
                    preds_row=preds,
                    time_vals_row=time_vals,
                    datetime_raw_vals=datetime_raw_vals,
                    depth_vals_row=depth_vals,
                    var_ids_row=var_ids,
                    mask_row=gt_mask,
                    feature_dict=id_to_var,
                    sample_idx=idx,
                    plt_idx=plt_idx,
                    epoch=it,
                    train_or_val=flag,
                    plot_type=self.cfg.forecast_plot_type,
                    max_features=14,
                    max_depths_per_feature=14,
                    filter_first_pred=True,
                    depth_units="norm",
                    plot_interval=self.cfg.plot_interval
                )
            except Exception as e:
                if self.rank == 0:
                    print(f"Skipping token plot due to error: {e}")
        else:
            print("No data available for plotting")

    def resume_training_from_checkpoint(self, resume_states, optimizer, scheduler, scaler):
        """
        Restore optimizer, scheduler, and scaler states from checkpoint.
        Model state should already be loaded in main().
        """

        # Load optimizer state
        if 'optimizer_state_dict' in resume_states:
            optimizer.load_state_dict(resume_states['optimizer_state_dict'])
            if self.rank == 0:
                print("Restored optimizer state")
        
        # Load scaler state
        if 'scaler_state_dict' in resume_states:
            scaler.load_state_dict(resume_states['scaler_state_dict'])
            if self.rank == 0:
                print("Restored scaler state")
        
        start_epoch = resume_states['epoch'] + 1
        min_vali_loss = resume_states.get('min_vali_loss', float('inf'))
        training_stage = resume_states.get('training_stage', 'stage1')
        
        return start_epoch, min_vali_loss

    def init_weights(self, module):
        """
        Initialize model weights with proper scaling for transformers.
        Maintains activation and gradient scales across layers and model width.
        """
        if isinstance(module, nn.Linear):
            fan_in = module.weight.size(1)
            std = (2.0 / fan_in) ** 0.5
            
            if hasattr(self.trainer, 'init_scale'):
                std *= self.trainer.init_scale
            else:
                std *= 0.02  # Default transformer scale
                
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.0)
                
        elif isinstance(module, nn.Embedding):
            # Embedding initialization scaled by model dimension
            embed_dim = module.weight.size(1)
            std = (1.0 / embed_dim) ** 0.5
            
            if hasattr(self.trainer, 'embed_init_scale'):
                std *= self.trainer.embed_init_scale
            else:
                std *= 0.02  # Default scale
                
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
            # Standard normalization layer initialization
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.0)
            if module.weight is not None:
                torch.nn.init.constant_(module.weight, 1.0)
                
        elif isinstance(module, nn.Conv1d):
            # Scaled Kaiming initialization for conv layers
            fan_in = module.weight.size(1) * module.weight.size(2)  # in_channels * kernel_size
            std = (2.0 / fan_in) ** 0.5
            
            if hasattr(self.trainer, 'conv_init_scale'):
                std *= self.trainer.conv_init_scale
            else:
                std *= 1.0  # No additional scaling for conv by default
                
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.0)
                
        elif isinstance(module, nn.MultiheadAttention):
            # Special initialization for attention layers
            if hasattr(module, 'in_proj_weight') and module.in_proj_weight is not None:
                # Query, Key, Value projection
                fan_in = module.in_proj_weight.size(1)
                std = (2.0 / fan_in) ** 0.5 * 0.02
                torch.nn.init.normal_(module.in_proj_weight, mean=0.0, std=std)
                
            if hasattr(module, 'out_proj') and module.out_proj.weight is not None:
                # Output projection - scale down for residual connections
                fan_in = module.out_proj.weight.size(1)
                std = (2.0 / fan_in) ** 0.5 * 0.02
                
                # Additional scaling for output projections in transformers
                num_layers = getattr(self.trainer, 'num_layers', 6)  # Default 6 layers
                std /= (2 * num_layers) ** 0.5  # Scale by sqrt(2*num_layers)
                
                torch.nn.init.normal_(module.out_proj.weight, mean=0.0, std=std)
                
        # Handle custom transformer blocks if they exist
        elif hasattr(module, '__class__') and 'TransformerBlock' in str(module.__class__):
            # If this is a transformer block, apply scaled initialization to its components
            for name, child in module.named_children():
                if 'ffn' in name.lower() or 'mlp' in name.lower():
                    # Scale down feed-forward network outputs
                    for submodule in child.modules():
                        if isinstance(submodule, nn.Linear) and 'output' in str(submodule):
                            fan_in = submodule.weight.size(1)
                            std = (2.0 / fan_in) ** 0.5 * 0.02
                            num_layers = getattr(self.trainer, 'num_layers', 6)
                            std /= (2 * num_layers) ** 0.5
                            torch.nn.init.normal_(submodule.weight, mean=0.0, std=std)

    def apply_weight_initialization(self):
        """
        Apply weight initialization to the model.
        """
        if hasattr(self.trainer, 'init_weights') and self.trainer.init_weights:
            print("Initializing model weights...")
            self.model.apply(self.init_weights)
            print("Model weights initialized successfully!")
        else:
            print("Using PyTorch default initialization")

    def get_cl_warmup_weight(self, epoch, warmup_epochs=None):
        """
        Get contrastive loss weight with warmup schedule for stability.
        Gradually increase CL weight from 0 to full weight over warmup_epochs.
        """
        if warmup_epochs is None:
            warmup_epochs = getattr(self.trainer, 'cl_warmup_epochs', 10)
        
        if epoch < warmup_epochs:
            # Linear warmup
            warmup_factor = epoch / warmup_epochs
            # Smooth warmup using cosine
            warmup_factor = 0.5 * (1 - math.cos(math.pi * warmup_factor))
            return warmup_factor
        else:
            return 1.0