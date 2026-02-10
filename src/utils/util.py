import numpy as np
import os
import pandas as pd
import math
import argparse
import pickle
import json
from contextlib import contextmanager

from sklearn.preprocessing import StandardScaler
from collections import OrderedDict
from scipy.interpolate import interp1d

# Optional imports for compatibility
try:
    import torch
    from torch import nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import fcntl
    FCNTL_AVAILABLE = True
except ImportError:
    FCNTL_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from fastdtw import fastdtw
    FASTDTW_AVAILABLE = True
except ImportError:
    FASTDTW_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


@contextmanager
def _file_lock(lock_path: str):
    """
    Lightweight file-based lock using fcntl. Falls back to a no-op on platforms
    without fcntl support.
    """
    if not lock_path or not FCNTL_AVAILABLE:
        yield
        return

    directory = os.path.dirname(lock_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _atomic_write_json(path: str, payload: dict):
    """
    Write JSON atomically by dumping to a temporary file and replacing.
    """
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as tmp_file:
        json.dump(payload, tmp_file, indent=2)
    os.replace(tmp_path, path)


def _default_global_stats() -> dict:
    return {
        "variables": {},
        "metadata": {
            "total_datasets_processed": 0,
            "datasets_processed": [],
            "last_updated": None,
        },
    }



def apply_standard_scaling_per_sample(data, seq_len, observed_mask, padding_mask):
    """
    Apply standard scaling to each sample in the batch independently using sklearn's StandardScaler.
    sklearn handles null values automatically
    
    Args:
        data: (B, L) - Input data tensor
        seq_len: int - Length of context window (unused, kept for compatibility)
        observed_mask: (B, L) - Mask indicating observed values
        padding_mask: (B, L) - Mask indicating valid (non-padded) positions
    
    Returns:
        scaled_data: (B, L) - Scaled data
        scalers: list - sklearn StandardScaler objects for each sample (for inverse transform)
    """
    B, L = data.shape
    device = data.device
    dtype = data.dtype
    
    scaled_data = torch.zeros_like(data)
    scalers = []
    
    for i in range(B):
        # Get sample data and create mask for invalid values
        sample_data = data[i].cpu().numpy()  # (L,)
        valid_mask = (observed_mask[i].bool() & padding_mask[i].bool()).cpu().numpy()
        
        # Set invalid values to NaN - sklearn handles this automatically
        sample_with_nans = sample_data.copy()
        sample_with_nans[~valid_mask] = np.nan
        
        # Fit and transform using sklearn
        scaler = StandardScaler()
        scaled_sample = scaler.fit_transform(sample_with_nans.reshape(-1, 1)).flatten()
        
        # Convert back to tensor
        scaled_data[i] = torch.from_numpy(scaled_sample).to(device=device, dtype=dtype)
        scalers.append(scaler)
    
    scaled_data = torch.nan_to_num(scaled_data, nan=0.0)
    
    return scaled_data, scalers

def apply_standard_scaling_per_sample_wrapper(data, 
                                              seq_len=None, 
                                              observed_mask=None, 
                                              padding_mask=None, 
                                              scaler=None,
                                              mask_out=None, 
                                              **kwargs):
    """
    Simplified wrapper for standard scaling using sklearn StandardScaler.
    
    Args:
        data: (B, L) - Input data tensor
        seq_len: int - Length of context window (kept for compatibility)
        observed_mask: (B, L) - Mask indicating observed values
        padding_mask: (B, L) - Mask indicating valid (non-padded) positions
        scaler: list of sklearn StandardScaler objects or None - If provided, use these scalers; if None, fit new ones
        **kwargs: Additional arguments (ignored for simplicity)
    
    Returns:
        scaled_data: (B, L) - Scaled data
        scalers: list - sklearn StandardScaler objects for inverse transform
    """
    if scaler is None:
        return apply_standard_scaling_per_sample(data, seq_len, observed_mask, padding_mask)
    else:
        # Apply existing scalers
        B, L = data.shape
        device = data.device
        dtype = data.dtype
        
        scaled_data = torch.zeros_like(data)
        
        for i in range(B):
            # Get sample data and create mask for invalid values
            sample_data = data[i].cpu().numpy()  # (L,)
            
            # Set invalid values to NaN
            sample_with_nans = sample_data.copy()
            sample_with_nans[~mask_out[i].cpu().bool()] = np.nan
            
            # Transform using provided scaler
            scaled_sample = scaler[i].transform(sample_with_nans.reshape(-1, 1)).flatten()
            
            # Convert back to tensor
            scaled_data[i] = torch.from_numpy(scaled_sample).to(device=device, dtype=dtype)
        
        scaled_data = torch.nan_to_num(scaled_data, nan=0.0)
        
        return scaled_data, scaler

def inverse_standard_scaling(scaled_data, scalers):
    """
    Simple inverse transform using sklearn scalers.
    
    Args:
        scaled_data: (B, L) - Scaled data tensor
        scalers: list - sklearn StandardScaler objects from scaling step
        
    Returns:
        original_data: (B, L) - Data transformed back to original scale
    """
    B, L = scaled_data.shape
    device = scaled_data.device
    dtype = scaled_data.dtype
    
    original_data = torch.zeros_like(scaled_data)
    
    for i in range(B):
        # Use sklearn's inverse_transform
        sample_numpy = scaled_data[i].cpu().numpy().reshape(-1, 1)
        original_sample = scalers[i].inverse_transform(sample_numpy).flatten()
        original_data[i] = torch.from_numpy(original_sample).to(device=device, dtype=dtype)
    
    return original_data

class ActiveEmbed(nn.Module):
    """ 
    record to mask embedding
    """
    def __init__(self, embed_dim=64, norm_layer=None):
        
        super().__init__()
        # self.rec_len = rec_len
        self.proj = nn.Conv1d(1, embed_dim, kernel_size=1, stride=1)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, _, L = x.shape
        # assert(L == self.rec_len, f"Input data width ({L}) doesn't match model ({self.rec_len}).")
        x = self.proj(x)
        x = torch.sin(x)
        x = x.transpose(1, 2)
        #   x = torch.cat((torch.sin(x), torch.cos(x + math.pi/2)), -1)
        x = self.norm(x)
        return x

class ModelPlugins():
    
    def __init__(self, 
                 window_len, 
                 enc_embed_dim,
                 dec_embed_dim,
                 task_name,
                 num_feats,
                 n2one,
                 batch_size,
                 device):
        
        self.enc_embed_dim = enc_embed_dim
        self.dec_embed_dim = dec_embed_dim
        self.window_len = window_len 
        self.device = device
        self.batch_size = batch_size
        self.num_feats = num_feats
        
        self.pos_embed = nn.Parameter(torch.zeros(1, self.window_len, self.enc_embed_dim), requires_grad=False).to(self.device)
        # self.pos_embed = PositionalEncoding2D(enc_embed_dim).to(self.device)
        
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.window_len, self.dec_embed_dim), requires_grad=False).to(self.device)
        # self.decoder_pos_embed = PositionalEncoding2D(dec_embed_dim).to(self.device)
        
        if n2one==True:
            self.decoder_pred = nn.Linear(self.dec_embed_dim, 1, bias=True).to(self.device)  # decoder to patch
        else:
            self.decoder_pred = nn.Linear(self.dec_embed_dim, num_feats, bias=True).to(self.device)  # decoder to patch
        
        self.initialize_embeddings()
    
    def initialize_embeddings(self):
        
        # enc_z = torch.rand((1, self.window_len, self.num_feats, self.enc_embed_dim)).to(self.device)
        # self.pos_embed = self.pos_embed(enc_z)
        encoder_pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.window_len, cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(encoder_pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_1d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.window_len, cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))


def get_1d_sincos_pos_embed(embed_dim, pos, cls_token=False):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """

    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = np.arange(pos)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    pos_embed = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)

    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)

    return pos_embed


def adjust_learning_rate(optimizer, epoch, lr, min_lr, max_epochs, warmup_epochs):
    """Decay the learning rate with half-cycle cosine after warmup"""
    
    if epoch < warmup_epochs:
        tmp_lr = lr * epoch / warmup_epochs 
    else:
        tmp_lr = min_lr + (lr - min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - warmup_epochs) / (max_epochs - warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = tmp_lr * param_group["lr_scale"]
        else:
            param_group["lr"] = tmp_lr
    return tmp_lr


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == np.inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


class DTWCacheMixin:
    def __init__(self, max_cache=128):
        super().__init__()
        self._dtw_cache = OrderedDict()
        self._max_cache = max_cache

    @torch.no_grad()
    def get_dtw_matrix(self, batch_idx, raw_windows):
        """
        batch_idx    : (B,) global indices of the samples in dataset
        raw_windows  : (B, L) cpu/gpu tensor  (will be moved to cpu)
        returns      : (B,B) torch.float32  min‑max normalised DTW distances
        """
        key = frozenset(int(i) for i in batch_idx)
        if key in self._dtw_cache:
            self._dtw_cache.move_to_end(key)
            return self._dtw_cache[key].to(raw_windows.device)

        win = raw_windows.detach().cpu().numpy()
        B, D = len(win), np.zeros((len(win), len(win)), dtype=np.float32)
        for i in range(B):
            for j in range(i+1, B):
                d, _ = fastdtw(win[i], win[j])
                D[i, j] = D[j, i] = d

        tri = D[np.triu_indices(B, 1)]
        scale = np.median(tri) if np.median(tri) > 0 else tri.mean() + 1e-6
        D /= scale

        Dt = torch.from_numpy(D)              # stays on CPU
        self._dtw_cache[key] = Dt
        if len(self._dtw_cache) > self._max_cache:
            self._dtw_cache.popitem(last=False)
        return Dt.to(raw_windows.device)

class Plotter():
    def __init__(self, device, pad_val):
        self.device = device
        self.pad_val = pad_val
    
    def plot_forecast(self, 
                        df, 
                        preds, 
                        sample_idx, 
                        plt_idx, 
                        feature_dict,
                        var_ids,
                        gt_masks,
                        epoch, 
                        depth_id, 
                        depth_val, 
                        train_or_val, 
                        seq_len,
                        title_prefix,
                        plot_type='line'):
        """
        This function plots t+plt_idx horizon plots
        """
        preds = preds.detach()
        df = df.detach()

        sample_time_series = df[sample_idx] # GT : samples, L, feat
        predictions = preds[sample_idx]
        var_ids = var_ids[sample_idx].squeeze()
        mask = gt_masks[sample_idx]

        sample_time_series = sample_time_series.cpu().numpy()                          
        predictions = predictions.cpu().numpy()
        mask = mask.cpu().numpy()
    
        # Create a figure
        num_feats = predictions.shape[2]
        num_samples = len(plt_idx)
        
        fig, axes = plt.subplots(num_feats, num_samples, figsize=(6*num_samples, num_feats*3))

        for i, horizon in enumerate(plt_idx):
            horizon = int(horizon)

            for idx in range(num_feats):
                if num_samples == 1:
                    ax = axes[idx]
                else:
                    ax = axes[idx, i]      
                
                feature_name = feature_dict[int(var_ids[0, idx])]
                dates = np.arange(predictions.shape[0])
                
                pred_curve = predictions[:, horizon, idx, depth_id]
                gt_curve = sample_time_series[:, horizon, idx, depth_id]
                gt_mask = mask[:, horizon, idx, depth_id]
                # Replace missing values with NaN to create gaps
                gt_values_for_plot = gt_curve.copy()
                gt_values_for_plot[~gt_mask.astype(bool)] = np.nan

                ax.plot(dates, pred_curve, label='Predictions TS', linestyle='-', markersize=3, color='green')
                if plot_type=='line':
                    ax.plot(dates, gt_values_for_plot, label='Ground-Truth TS', linestyle='-', markersize=3, color='red')
                elif plot_type=='scatter':
                    ax.scatter(dates, gt_values_for_plot, label='Ground-Truth TS', s=3, color='red')

                subtitle = f'{feature_name}: Plot at t+{horizon + 1}'
                ax.set_title(subtitle)
                ax.set_xlabel('Time Step')
                ax.set_ylabel('Values')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
                ax.legend(loc="best")
        
        title = f'{title_prefix}: {train_or_val} t+N Forecasts at epoch: {epoch} and depth: {depth_val} m'
        plt.tight_layout()
        plt.suptitle(title, y=1.02)
        wandb.log({title: wandb.Image(plt)})
        plt.close()

    def plot_context_window_grid(self, 
                                    df, 
                                    preds, 
                                    sample_index, 
                                    depth_dict, 
                                    feature_dict,
                                    var_ids,
                                    gt_masks, 
                                    epoch, 
                                    train_or_val, 
                                    title_prefix,
                                    plot_type='line'):
        """
        This function creates a grid of size Number of features X Num_Samples
        where num_samples = len(sample_index)
        """
        preds = preds.detach()
        df = df.detach()
        
        sample_time_series = df[sample_index] # GT
        predictions = preds[sample_index]
        var_ids = var_ids[sample_index].squeeze()
        mask = gt_masks[sample_index]

        sample_time_series = sample_time_series.cpu().numpy()                          
        predictions = predictions.cpu().numpy()
        mask = mask.cpu().numpy()

        # Create a figure
        num_feats = preds.shape[2]
        num_depths = preds.shape[3]

        a, b = max(num_depths, num_feats), min(num_depths, num_feats)
        fig, axes = plt.subplots(a, b, figsize=(3 * a, 4 * b))

        # Ensure axes is 2D even if a or b is 1
        axes = np.atleast_2d(axes)

        for k in range(a):
            for j in range(b):
                ax = axes[k, j]
                dates = np.arange(predictions.shape[1])      

                # Assign idx/depth based on layout
                if a == num_depths:
                    depth_id = k
                    idx = j
                else:
                    depth_id = j
                    idx = k
                
                pred_values=predictions[0, :, idx, depth_id]
                gt_values_for_plot=sample_time_series[0, :, idx, depth_id]
                gt_mask=mask[0, :, idx, depth_id]
                
                # Replace missing values with NaN to create gaps
                gt_values_for_plot[~gt_mask.astype(bool)] = np.nan

                # Plot predictions as a line
                ax.plot(dates, pred_values, label='Predictions TS', linestyle='-', markersize=3, color='green')
                
                if plot_type=='line':
                    ax.plot(dates, gt_values_for_plot, label='Ground-Truth TS', linestyle='-', markersize=3, color='blue')
                elif plot_type=='scatter':
                    ax.scatter(dates, gt_values_for_plot, label='Ground-Truth TS', s=6, color='red')

                ax.set_xlabel('Time Step')
                ax.set_ylabel('Values')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
                ax.legend(loc="best")

                # Set column title only for top row
                if k == 0:
                    if a == num_depths:
                        col_label_type = feature_dict[int(var_ids[j])]
                    else:
                        col_label_type = "Depth " + str(depth_dict[j].cpu().numpy()) + " m"
                    ax.set_title(f"{col_label_type}", fontsize=12)

            # Set row label on the first column
            if a == num_depths:
                row_label_type = "Depth " + str(depth_dict[k].cpu().numpy()) + " m"
            else:
                row_label_type = feature_dict[int(var_ids[k])]
            axes[k, 0].set_ylabel(f"{row_label_type}", rotation=0, labelpad=80, fontsize=10, va='center', ha='right')
        
        title = '{}: {} Individual Context-Windows at epoch: {}'.format(title_prefix, train_or_val, epoch)
        fig.suptitle(title, fontsize=14, y=1.02)
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        wandb.log({title: wandb.Image(plt)})
        plt.close()

class NativeScaler:

    state_dict_key = "amp_scaler"
    def __init__(self):
        self._scaler = torch.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)



class MAEDataset(Dataset):

    def __init__(self, X, M):
        self.X = X
        self.M = M

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.M[idx]


class Utils:
    
    def __init__(self, inp_cols, date_col, args, stride=1):
        
        self.inp_cols = inp_cols
        self.date_col = date_col
        self.seq_len = args.seq_len
        self.stride = stride
        self.y_mean = None
        self.y_std = None
        self.device = args.device
        self.windowed_dataset_path = ""
        self.task_name = args.task_name
        self.n2one = args.n2one
        self.pred_len = args.pred_len

        if args.task_name=='finetune':
            self.pre_train_window = self.seq_len + self.pred_len
        else:
            self.pre_train_window = self.seq_len
        
        self.target_index = -1
    
    def load_pickle(self, path):
        
        with open(path, 'rb') as pickle_file:
            arr = pickle.load(pickle_file)
        
        return arr
        
    def split_data(self, df, ratios):
        '''
        For ETT we follow 6:2:2 ratio, and for other datasets, it is usually. 7:1:1
        '''
        total_rows = len(df)
        train_ratio = ratios['train']
        val_ratio = ratios['val']
        test_ratio = ratios['test']
        
        train_end_pt = int(train_ratio*total_rows)
        train_df = df[:train_end_pt].reset_index(drop='true')

        val_end_pt = train_end_pt + int(val_ratio*total_rows)
        val_df = df[train_end_pt:val_end_pt].reset_index(drop='true')

        test_df = df[val_end_pt:].reset_index(drop='true')
        
        return train_df, val_df, test_df
        
    
    def normalize_tensor(self, tensor, use_stat=False):
        
        eps = 1e-5 # epsilon for zero std
        
        '''
        use this when working on masked data
        '''
        if not use_stat:
            self.feat_mean = tensor.nanmean(dim=(0, 1))[None, None, :]
            mask = torch.isnan(tensor)
            filtered_data = tensor.clone()
            filtered_data[mask] = 0
            
            rev_mask = 1-(mask*1)
            
            sqred_values = rev_mask*((filtered_data - self.feat_mean)**2)
            sqred_sum = sqred_values.sum(dim=(0, 1))
            variance = sqred_sum/torch.sum(rev_mask, dim=(0, 1))
            
            self.feat_std = torch.sqrt(variance)[None, None, :]   
        
        tensor[:, :, :len(self.inp_cols)] = (tensor[:, :, :len(self.inp_cols)]-self.feat_mean)/(self.feat_std+eps)
        return tensor
        
    def normalize_pd(self, df, use_stat=False):
        '''
        Normalize data
        '''
        if use_stat:
            df[self.inp_cols] = (df[self.inp_cols]-self.feat_mean)/self.feat_std
            return df
                
        # compute mean and std of target variable - to be used for unnormalizing
        self.feat_std = df[self.inp_cols].std(skipna=True)
        self.feat_mean = df[self.inp_cols].mean(skipna=True)
        
        df[self.inp_cols] = (df[self.inp_cols]-self.feat_mean)/self.feat_std
            
        return df
    
    def perform_windowing(self, df, path, name, split='train'):
        '''
        create a windowed dataset
    
        : param y:                time series feature (array)
        : param input_window:     number of y samples to give model
        : param output_window:    number of future y samples to predict
        : param stide:            spacing between windows
        : param num_features:     number of features (i.e., 1 for us, but we could have multiple features)
        : return X, Y:            arrays with correct dimensions for LSTM
        :                         (i.e., [input/output window size # examples, # features])
        '''
        
        filename=name + '_' + split + '_' + str(self.pre_train_window) +'.pkl'
        save_path=os.path.join(path, filename)
        
        if os.path.exists(save_path):
            print("Window dataset already exists")
            X = self.load_pickle(save_path)
            return X
        
        else:
            L = df.shape[0]
            num_samples = (L - self.pre_train_window) // self.stride + 1

            X = [] #np.zeros([num_samples, self.pre_train_window, self.num_features])
            
            for ii in tqdm(np.arange(num_samples)):
                start_x = self.stride * ii
                end_x = start_x + self.pre_train_window

                subset_df = df.iloc[start_x:end_x, :].copy(deep=True)
                
                X.append(np.expand_dims(subset_df, axis=0))
            

            X = np.concatenate(X, axis=0)
            
            return X
    
    def plot_context_window_grid_with_original_masks(self, df, preds, og_masks, sample_index, epoch, train_or_val, title_prefix):
        """
        This function creates a grid of size Number of features X Num_Samples
        where num_samples = len(sample_index)
        """
        preds = preds.detach()
        df = df.detach()
        og_masks = og_masks.detach()
        
        sample_time_series = df[sample_index] # GT
        predictions = preds[sample_index]
        og_mask = og_masks[sample_index]
        
        og_masked_ts = og_mask*sample_time_series
        og_masked_ts = torch.where(og_masked_ts==0, torch.tensor(float('nan')).to(self.device), og_masked_ts)
        
        masked_ts = og_masked_ts.cpu().numpy()
        og_masked_ts = og_masked_ts.cpu().numpy()
        sample_time_series = sample_time_series.cpu().numpy()                          
        predictions = predictions.cpu().numpy()
        
        # Create a figure
        num_feats = preds.shape[2]
        num_samples = len(sample_index)
        
        fig, axes = plt.subplots(num_feats, num_samples, figsize=(4*num_samples, num_feats*3))
        
        if self.n2one=="True":
            
            for sample_id in range(num_samples):
                # for idx in range(num_feats):
                # ax = axes[idx, sample_id]      
                ax = axes[sample_id]
                
                feature_name = self.inp_cols[self.chloro_index]#[idx]
                dates = np.arange(predictions.shape[1])
                plt.tight_layout()
                ax.plot(dates, predictions[sample_id, :, 0], label='Predictions TS', marker='o', linestyle='-', markersize=1,
                        color='green')
                ax.plot(dates, masked_ts[sample_id, :, 0], label='Original TS', marker='o', linestyle='-', markersize=1, 
                         color='blue')
                ax.axvline(x=self.seq_len, color='black', linestyle='-', linewidth=2)
                
                subtitle = '{}'.format(feature_name)
                ax.set_title(subtitle)
                ax.set_xlabel('Time Step')
                ax.set_ylabel('Values')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
                ax.legend(loc="best")
        else:
            for sample_id in range(num_samples):
                
                for idx in range(num_feats):
                    ax = axes[idx, sample_id]      

                    feature_name = self.inp_cols[idx]
                    dates = np.arange(predictions.shape[1])
                    ax.plot(dates, predictions[sample_id, :, idx], label='Predictions TS', marker='o', linestyle='-', markersize=1,
                            color='green')
                    ax.plot(dates, masked_ts[sample_id, :, idx], label='Original TS', marker='o', linestyle='-', markersize=1, 
                         color='blue')
                    
                    if self.task_name=='finetune' or self.task_name=='zeroshot':
                        plt.tight_layout()
                        ax.axvline(x=self.seq_len, color='black', linestyle='-', linewidth=2)
                        
                    subtitle = '{}'.format(feature_name)
                    ax.set_title(subtitle)
                    ax.set_xlabel('Time Step')
                    ax.set_ylabel('Values')
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
                    ax.legend(loc="best")
        
        title = '{}: {} Single Windows w/Original Masks at epoch: {}'.format(title_prefix, train_or_val, epoch)
        plt.tight_layout()
        plt.suptitle(title, y=1.02)
        wandb.log({title: wandb.Image(plt)})
        plt.close()


class Normalizer:
    
    def __init__(self, lake_id=None, lake_name=None, id2var=None, variate_ids_2D=None, variate_ids_1D=None, run_name=None, ckpt_name=None):
        """
        Initialize normalization manager.
        
        Args:
            lake_id: Unique identifier for the lake
            lake_name: Human-readable name for the lake
            id2var: Mapping from variable ID to variable name
            variate_ids_2D: List of 2D variable IDs
            variate_ids_1D: List of 1D variable IDs
            run_name: Name of the run
        """
        self.lake_id = lake_id
        self.lake_name = lake_name
        self.id2var = id2var or {}
        self.variate_ids_2D = variate_ids_2D or []
        self.variate_ids_1D = variate_ids_1D or []
        self.run_name = run_name
        self.ckpt_name = ckpt_name
        self.scaler_DR = None
        self.scaler_DF = None
        self.data_DR = None
        self.data_DF = None
        
    def fit_scalers(self, df_driver, df_lake):
        """
        Fit StandardScaler instances to driver and lake data.
        
        Args:
            df_driver: DataFrame with driver variables
            df_lake: DataFrame with lake variables
        """
        # Store data for later use in aggregation
        self.data_DR = df_driver
        self.data_DF = df_lake
        
        # Fit scalers
        self.scaler_DR = StandardScaler()
        self.scaler_DF = StandardScaler()
        
        self.scaler_DR.fit(df_driver)
        self.scaler_DF.fit(df_lake)
        
        print(f"Fitted scalers for lake {self.lake_id or self.lake_name}")
        print(f"  - Driver variables: {df_driver.shape[1]} features")
        print(f"  - Lake variables: {df_lake.shape[1]} features")
    
    def transform_data(self, df_driver, df_lake):
        """
        Transform data using fitted scalers.
        
        Args:
            df_driver: DataFrame with driver variables
            df_lake: DataFrame with lake variables
            
        Returns:
            tuple: (scaled_driver_data, scaled_lake_data)
        """
        if self.scaler_DR is None or self.scaler_DF is None:
            raise ValueError("Scalers not fitted. Call fit_scalers() first.")
        
        scaled_driver = self.scaler_DR.transform(df_driver)
        scaled_lake = self.scaler_DF.transform(df_lake)
        
        return scaled_driver, scaled_lake
    
    def inverse_transform_data(self, scaled_driver, scaled_lake):
        """
        Inverse transform scaled data back to original scale.
        
        Args:
            scaled_driver: Scaled driver data
            scaled_lake: Scaled lake data
            
        Returns:
            tuple: (original_driver_data, original_lake_data)
        """
        if self.scaler_DR is None or self.scaler_DF is None:
            raise ValueError("Scalers not fitted. Call fit_scalers() first.")
        
        original_driver = self.scaler_DR.inverse_transform(scaled_driver)
        original_lake = self.scaler_DF.inverse_transform(scaled_lake)
        
        return original_driver, original_lake
    
    @staticmethod
    def denormalize_by_var_ids(values, var_ids, scaler_DF, variate_ids_2D, scaler_DR=None, variate_ids_1D=None):
        """
        Denormalize irregular/flattened token values using provided scalers and variable id lists.
        
        This function applies inverse transform per variable by mapping variable IDs to the
        appropriate mean/std entry in the provided sklearn StandardScaler objects.
        
        Args:
            values: torch.Tensor of normalized values (any shape, will be flattened)
            var_ids: torch.Tensor of variable IDs, same shape as values
            scaler_DF: sklearn StandardScaler for lake (2D) variables (required for lake vars)
            variate_ids_2D: list of variable IDs that correspond to lake (2D) variables
            scaler_DR: optional sklearn StandardScaler for driver (1D) variables
            variate_ids_1D: optional list of variable IDs for driver (1D) variables
        
        Returns:
            torch.Tensor with same shape as values, denormalized per variable.
        """
        import torch
        if values is None or var_ids is None:
            return values
        # Flatten
        flat_vals = values.flatten()
        flat_vids = var_ids.flatten()
        out = flat_vals.clone()
        unique_vids = torch.unique(flat_vids)
        # Handle driver variables if scalers provided
        if scaler_DR is not None and variate_ids_1D is not None:
            for i, vid in enumerate(variate_ids_1D):
                if torch.is_tensor(vid):
                    vid_int = int(vid.item())
                else:
                    vid_int = int(vid)
                if vid_int in unique_vids.tolist():
                    mask = (flat_vids == vid_int)
                    mean = scaler_DR.mean_[i] if i < len(scaler_DR.mean_) else 0.0
                    std = scaler_DR.scale_[i] if i < len(scaler_DR.scale_) else 1.0
                    out[mask] = flat_vals[mask] * std + mean
        # Handle lake variables
        for i, vid in enumerate(variate_ids_2D):
            if torch.is_tensor(vid):
                vid_int = int(vid.item())
            else:
                vid_int = int(vid)
            if vid_int in unique_vids.tolist():
                mask = (flat_vids == vid_int)
                mean = scaler_DF.mean_[i] if i < len(scaler_DF.mean_) else 0.0
                std = scaler_DF.scale_[i] if i < len(scaler_DF.scale_) else 1.0
                out[mask] = flat_vals[mask] * std + mean
        return out.reshape(values.shape)
    
    @staticmethod
    def denormalize_scale_by_var_ids(values, var_ids, scaler_DF, variate_ids_2D, scaler_DR=None, variate_ids_1D=None):
        import torch
        if values is None or var_ids is None:
            return values
        # Flatten
        flat_vals = values.flatten()
        flat_vids = var_ids.flatten()
        out = flat_vals.clone()
        unique_vids = torch.unique(flat_vids)
        # Handle driver variables if scalers provided
        if scaler_DR is not None and variate_ids_1D is not None:
            for i, vid in enumerate(variate_ids_1D):
                if torch.is_tensor(vid):
                    vid_int = int(vid.item())
                else:
                    vid_int = int(vid)
                if vid_int in unique_vids.tolist():
                    mask = (flat_vids == vid_int)
                    std = scaler_DR.scale_[i] if i < len(scaler_DR.scale_) else 1.0
                    out[mask] = flat_vals[mask] * std  # Only multiply by std, no mean shift
        # Handle lake variables
        for i, vid in enumerate(variate_ids_2D):
            if torch.is_tensor(vid):
                vid_int = int(vid.item())
            else:
                vid_int = int(vid)
            if vid_int in unique_vids.tolist():
                mask = (flat_vids == vid_int)
                std = scaler_DF.scale_[i] if i < len(scaler_DF.scale_) else 1.0
                out[mask] = flat_vals[mask] * std  # Only multiply by std, no mean shift
        return out.reshape(values.shape)
    
    def save_normalization_stats(self, save_path: str, global_stats_path: str = None, 
                                num_unique_depths=None, depth_values=None, 
                                max_depth=None, min_depth=None,
                                variate_ids_2D=None, variate_ids_1D=None):
        """
        Save normalization statistics to JSON file.
        Optionally update global per-variable statistics.
        
        Args:
            save_path: Path where to save the per-lake normalization stats JSON file
            global_stats_path: Path to global per-variable stats file (optional)
            num_unique_depths: Number of unique depths (for metadata)
            depth_values: Array of depth values (for metadata)
            variate_ids_2D: 2D variable IDs (for metadata)
            variate_ids_1D: 1D variable IDs (for metadata)
        """
        if self.scaler_DR is None or self.scaler_DF is None:
            raise ValueError("Scalers not fitted. Call fit_scalers() first.")
        
        # Ensure directory exists and construct full path with run_name
        save_dir = os.path.dirname(save_path)
        save_dir = os.path.join(save_dir, self.run_name)
        save_filename = os.path.basename(save_path)
        save_path = os.path.join(save_dir, save_filename)
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # Extract scaler statistics (per-lake stats)
        stats = {
            "scaler_DR": {
                "mean": self.scaler_DR.mean_.tolist() if hasattr(self.scaler_DR, 'mean_') else None,
                "scale": self.scaler_DR.scale_.tolist() if hasattr(self.scaler_DR, 'scale_') else None,
                "var": self.scaler_DR.var_.tolist() if hasattr(self.scaler_DR, 'var_') else None,
            },
            "scaler_DF": {
                "mean": self.scaler_DF.mean_.tolist() if hasattr(self.scaler_DF, 'mean_') else None,
                "scale": self.scaler_DF.scale_.tolist() if hasattr(self.scaler_DF, 'scale_') else None,
                "var": self.scaler_DF.var_.tolist() if hasattr(self.scaler_DF, 'var_') else None,
            },
            "metadata": {
                "lake_id": self.lake_id,
                "lake_name": self.lake_name,
                "num_unique_depths": num_unique_depths,
                "depth_values": depth_values.tolist() if hasattr(depth_values, 'tolist') else (list(depth_values) if depth_values is not None else None),
                "max_depth": float(max_depth) if max_depth is not None else None,
                "min_depth": float(min_depth) if min_depth is not None else None,
                "variate_ids_2D": variate_ids_2D,
                "variate_ids_1D": variate_ids_1D
            }
        }
        
        # Save to JSON atomically with a file lock to avoid concurrent truncation
        lock_path = f"{save_path}.lock"
        with _file_lock(lock_path):
            _atomic_write_json(save_path, stats)
        
        print(f"Saved per-lake normalization stats to: {save_path}")
        
        # Update global per-variable statistics
        if global_stats_path:
            self._update_global_variable_stats(global_stats_path)
    
    def _update_global_variable_stats(self, global_stats_path: str):
        """
        Update global per-variable statistics with proper aggregation.
        
        For each variable, computes true aggregated mean and std across all datasets.
        Uses the mathematical formula for combining means and standard deviations.
        
        Args:
            global_stats_path: Path to global variable statistics file
        """
        global_stats_dir = os.path.dirname(global_stats_path)
        global_stats_dir = os.path.join(global_stats_dir, self.run_name)
        global_stats_filename = os.path.basename(global_stats_path)
        global_stats_path = os.path.join(global_stats_dir, global_stats_filename)
        
        if global_stats_dir:
            os.makedirs(global_stats_dir, exist_ok=True)
        
        lock_path = f"{global_stats_path}.lock"
        with _file_lock(lock_path):
            # Load existing global stats or create new if file is missing/corrupt
            if os.path.exists(global_stats_path) and os.path.getsize(global_stats_path) > 0:
                try:
                    with open(global_stats_path, 'r') as f:
                        global_stats = json.load(f)
                except json.JSONDecodeError:
                    print(f"Warning: Global stats file at {global_stats_path} is corrupted. Reinitializing.")
                    global_stats = _default_global_stats()
            else:
                if os.path.exists(global_stats_path):
                    print(f"Warning: Global stats file at {global_stats_path} is empty. Reinitializing.")
                global_stats = _default_global_stats()
        
            # Get dataset identifier - use actual lake name if available
            dataset_name = self.lake_name if self.lake_name else (f"lake_{self.lake_id}" if self.lake_id else "unknown_lake")

            # Check if this dataset has already been processed
            if dataset_name in global_stats["metadata"]["datasets_processed"]:
                print(f"Warning: Dataset '{dataset_name}' has already been processed. Skipping to avoid double-counting.")
                return

            # Update driver variable stats with proper aggregation
            if hasattr(self.scaler_DR, 'mean_') and self.scaler_DR.mean_ is not None:
                for i, (mean, std, var) in enumerate(zip(self.scaler_DR.mean_, self.scaler_DR.scale_, self.scaler_DR.var_)):
                    # Use id2var mapping for 1D variables (driver variables)
                    var_id = self.variate_ids_1D[i] if i < len(self.variate_ids_1D) else i
                    var_name = self.id2var[var_id]
                    n_samples = len(self.data_DR)  # Number of samples in current dataset

                    self._aggregate_variable_stats(
                        global_stats, var_name, float(mean), float(std), float(var), n_samples, dataset_name
                    )

            # Update lake variable stats with proper aggregation
            if hasattr(self.scaler_DF, 'mean_') and self.scaler_DF.mean_ is not None:
                for i, (mean, std, var) in enumerate(zip(self.scaler_DF.mean_, self.scaler_DF.scale_, self.scaler_DF.var_)):
                    # Use id2var mapping for 2D variables (lake variables)
                    var_id = self.variate_ids_2D[i] if i < len(self.variate_ids_2D) else i
                    var_name = self.id2var[var_id]
                    n_samples = len(self.data_DF)  # Number of samples in current dataset

                    self._aggregate_variable_stats(
                        global_stats, var_name, float(mean), float(std), float(var), n_samples, dataset_name
                    )

            # Update metadata
            if dataset_name not in global_stats["metadata"]["datasets_processed"]:
                global_stats["metadata"]["datasets_processed"].append(dataset_name)

            global_stats["metadata"]["total_datasets_processed"] = len(global_stats["metadata"]["datasets_processed"])
            global_stats["metadata"]["last_updated"] = pd.Timestamp.now().isoformat()

            # Ensure directory exists
            global_dir = os.path.dirname(global_stats_path)
            if global_dir:
                os.makedirs(global_dir, exist_ok=True)

            # Save updated global stats atomically
            _atomic_write_json(global_stats_path, global_stats)
        
        print(f"Updated global variable stats at: {global_stats_path}")
        print(f"Processed {global_stats['metadata']['total_datasets_processed']} datasets total")
    
    def _aggregate_variable_stats(self, global_stats, var_name, new_mean, new_std, new_var, n_new, dataset_name):
        """
        Aggregate statistics for a single variable using proper mathematical formulas.
        
        Combines existing global stats with new dataset stats to compute true aggregated mean and std.
        """
        if var_name not in global_stats["variables"]:
            # First time seeing this variable
            global_stats["variables"][var_name] = {
                "mean": new_mean,
                "std": new_std,
                "var": new_var,
                "total_samples": n_new,
                "num_datasets": 1,
                "datasets": [dataset_name]
            }
        else:
            # Aggregate with existing statistics
            existing = global_stats["variables"][var_name]
            
            # Get existing values
            old_mean = existing["mean"]
            old_var = existing["var"] 
            n_old = existing["total_samples"]
            
            # Compute aggregated statistics using mathematical formulas
            n_total = n_old + n_new
            
            # Aggregated mean: weighted average
            agg_mean = (n_old * old_mean + n_new * new_mean) / n_total
            
            # Aggregated variance using the formula for combining variances
            # Var(combined) = (n1*var1 + n2*var2 + n1*n2*(mean1-mean2)^2/n_total) / n_total
            mean_diff_squared = (old_mean - new_mean) ** 2
            agg_var = (n_old * old_var + n_new * new_var + n_old * n_new * mean_diff_squared / n_total) / n_total
            agg_std = np.sqrt(agg_var)
            
            # Update global stats
            global_stats["variables"][var_name] = {
                "mean": float(agg_mean),
                "std": float(agg_std), 
                "var": float(agg_var),
                "total_samples": int(n_total),
                "num_datasets": existing["num_datasets"] + 1,
                "datasets": existing["datasets"] + [dataset_name] if dataset_name not in existing["datasets"] else existing["datasets"]
            }
    
    def apply_global_normalization(self, global_stats_path: str):
        """
        Apply global per-variable normalization to this dataset.
        Useful for zero-shot inference on unseen lakes.
        
        Args:
            global_stats_path: Path to global variable statistics file
        """
        # Construct path with run_name (checkpoint name) in stats dir
        global_stats_dir = os.path.dirname(global_stats_path)
        global_stats_dir = os.path.join(global_stats_dir, self.ckpt_name)
        global_stats_filename = os.path.basename(global_stats_path)
        global_stats_path = os.path.join(global_stats_dir, global_stats_filename)
        
        # Load global stats
        global_stats = self.load_global_variable_stats(global_stats_path)
        
        # Create new scalers with global statistics
        self.scaler_DR = StandardScaler()
        self.scaler_DF = StandardScaler()
        
        # Set global statistics for driver variables
        driver_means = []
        driver_stds = []
        
        for i in range(len(self.variate_ids_1D)):
            # Use id2var mapping for 1D variables (driver variables)
            var_id = self.variate_ids_1D[i] if i < len(self.variate_ids_1D) else i
            var_name = self.id2var[var_id]
            if var_name in global_stats["variables"]:
                var_stats = global_stats["variables"][var_name]
                driver_means.append(var_stats["mean"])
                driver_stds.append(max(var_stats["std"], 1e-8))  # Avoid division by zero
            else:
                # Fallback to per-lake stats if global stats not available
                driver_means.append(0.0)
                driver_stds.append(1.0)
        
        # Manually set scaler parameters
        self.scaler_DR.mean_ = np.array(driver_means)
        self.scaler_DR.scale_ = np.array(driver_stds)
        self.scaler_DR.var_ = np.array(driver_stds) ** 2
        self.scaler_DR.n_features_in_ = len(driver_means)
        
        # Set global statistics for lake variables
        lake_means = []
        lake_stds = []
        
        for i in range(len(self.variate_ids_2D)):
            # Use id2var mapping for 2D variables (lake variables)
            var_id = self.variate_ids_2D[i] if i < len(self.variate_ids_2D) else i
            var_name = self.id2var[var_id]
            if var_name in global_stats["variables"]:
                var_stats = global_stats["variables"][var_name]
                lake_means.append(var_stats["mean"])
                lake_stds.append(max(var_stats["std"], 1e-8))  # Avoid division by zero
            else:
                # Fallback to per-lake stats if global stats not available
                lake_means.append(0.0)
                lake_stds.append(1.0)
            
            # Manually set scaler parameters
            self.scaler_DF.mean_ = np.array(lake_means)
            self.scaler_DF.scale_ = np.array(lake_stds)
            self.scaler_DF.var_ = np.array(lake_stds) ** 2
            self.scaler_DF.n_features_in_ = len(lake_means)
        
        print(f"Applied global normalization from: {global_stats_path}")
    
    def apply_normalization_from_stats(self, stats_path: str):
        """
        Loads pre-computed normalization stats from a file and applies them
        to this dataset.
        
        This function assumes the stats file is a dictionary with keys
        'scaler_DR' and 'scaler_DF', which in turn contain 'mean', 'scale', 
        and 'var' as numpy arrays.
        
        Args:
            stats_path: Path to the file containing the stats dictionary.
        """
        stats_dir = os.path.dirname(stats_path)
        stats_dir = os.path.join(stats_dir, self.ckpt_name)
        stats_filename = os.path.basename(stats_path)
        stats_path = os.path.join(stats_dir, stats_filename)
        
        # 1. Load the stats file
        stats = self.load_normalization_stats(stats_path)
        if not stats:
            print(f"Could not load or apply stats from {stats_path}")
            return

        # 2. Apply stats for Driver (DR) variables
        self.scaler_DR = StandardScaler()
        
        # Check if stats for DR are available
        if 'scaler_DR' in stats and len(stats['scaler_DR'].get('mean', [])) > 0:
            dr_stats = stats['scaler_DR']
            mean = np.array(dr_stats['mean'])
            scale = np.array(dr_stats['scale'])
            var = np.array(dr_stats['var'])
            
            # Manually set the scaler attributes
            self.scaler_DR.mean_ = mean
            self.scaler_DR.scale_ = scale
            self.scaler_DR.var_ = var
            self.scaler_DR.n_features_in_ = len(mean)
        else:
            raise RuntimeError(f"'scaler_DR' not found in {stats_path} or was empty. Cannot apply normalization for DR variables. Please check your normalization stats file.")

        self.scaler_DF = StandardScaler()
        
        # Check if stats for DF are available
        if 'scaler_DF' in stats and len(stats['scaler_DF'].get('mean', [])) > 0:
            df_stats = stats['scaler_DF']
            mean = np.array(df_stats['mean'])
            scale = np.array(df_stats['scale'])
            var = np.array(df_stats['var'])

            # Manually set the scaler attributes
            self.scaler_DF.mean_ = mean
            self.scaler_DF.scale_ = scale
            self.scaler_DF.var_ = var
            self.scaler_DF.n_features_in_ = len(mean)
        else:
            raise RuntimeError(f"'scaler_DF' not found in {stats_path} or was empty. Cannot apply normalization for DF variables. Please check your normalization stats file.")
        
        print(f"Successfully applied normalization from: {stats_path}")

    @staticmethod
    def load_normalization_stats(load_path: str):
        """
        Load normalization statistics from JSON file.
        
        Args:
            load_path: Path to normalization stats file
            
        Returns:
            dict: Dictionary with normalization statistics
        """
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Normalization stats file not found: {load_path}")
        
        with open(load_path, 'r') as f:
            stats = json.load(f)
        
        print(f"Loaded normalization stats from: {load_path}")
        return stats
    
    def load_global_variable_stats(self, global_stats_path: str):
        """
        Load global per-variable statistics from JSON file.
        
        Args:
            global_stats_path: Path to global variable stats file
            
        Returns:
            dict: Dictionary with aggregated per-variable statistics
        """
        if not os.path.exists(global_stats_path):
            raise FileNotFoundError(f"Global variable stats file not found: {global_stats_path}")
        
        with open(global_stats_path, 'r') as f:
            stats = json.load(f)
        
        # Print summary
        if "metadata" in stats:
            print(f"Loaded global variable stats from: {global_stats_path}")
            print(f"  - Total datasets: {stats['metadata']['total_datasets_processed']}")
            print(f"  - Variables: {len(stats.get('variables', {}))}")
            print(f"  - Last updated: {stats['metadata'].get('last_updated', 'Unknown')}")
        
        return stats

    @staticmethod
    def get_variable_normalization_params(global_stats_path: str, run_name: str = None):
        """
        Get normalization parameters (mean, std) for each variable from global stats.
        
        Args:
            global_stats_path: Path to global variable stats file
            run_name: Optional checkpoint/run name to use in path construction
            
        Returns:
            dict: {var_name: {"mean": float, "std": float}} for each variable
        """
        # Construct path with run_name if provided
        if run_name:
            global_stats_dir = os.path.dirname(global_stats_path)
            global_stats_dir = os.path.join(global_stats_dir, run_name)
            global_stats_filename = os.path.basename(global_stats_path)
            global_stats_path = os.path.join(global_stats_dir, global_stats_filename)
        
        if not os.path.exists(global_stats_path):
            raise FileNotFoundError(f"Global variable stats file not found: {global_stats_path}")
        
        with open(global_stats_path, 'r') as f:
            stats = json.load(f)
        
        norm_params = {}
        for var_name, var_stats in stats.get("variables", {}).items():
            norm_params[var_name] = {
                "mean": var_stats["mean"],
                "std": max(var_stats["std"], 1e-8)  # Avoid division by zero
            }
        
        return norm_params