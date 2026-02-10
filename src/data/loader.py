import torch
import numpy as np
import pandas as pd
import random

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Sampler, SequentialSampler
from collections import defaultdict

PAD_VAL_ID=0
PAD_VAL_DEFAULT=0

class LakeBalancedBatchSampler(DistributedSampler):
    """
    Updated batch sampler for irregular lake data structure.
    Handles variable sequence lengths that may result from irregular depth structure.
    """
    def __init__(self, meta_df, 
                P_pos=4, 
                batch_size=64, 
                num_replicas=None, 
                rank=None, 
                seed=0):

        # Create a dummy dataset with length equal to number of samples
        dummy_dataset = list(range(len(meta_df)))
        super().__init__(dataset=dummy_dataset, 
                        num_replicas=num_replicas, 
                        rank=rank, 
                        shuffle=True, 
                        seed=seed)

        if num_replicas is None:
            if not torch.distributed.is_available():
                num_replicas = 1
            else:
                num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                rank = 0
            else:
                rank = torch.distributed.get_rank()
                
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.seed = seed
        
        self.meta = meta_df
        self.P_pos = P_pos
        self.B = batch_size

        # Pre-compute lake_id lookup dictionary
        self.idx_to_lake = dict(zip(meta_df.idx, meta_df.lake_id))  # Faster than DataFrame lookup
        
        # Group samples by lake ID for fast positive sampling
        self.by_lake = defaultdict(list)
        for _, row in meta_df.iterrows():
            self.by_lake[row.lake_id].append(row.idx)

        # Flattened list of all indices
        self.indices = list(meta_df.idx)
        
        # Partition indices across processes
        self.num_samples = len(self.indices) // self.num_replicas
        self.total_size = self.num_samples * self.num_replicas
        
        # Pre-compute everything possible
        self.lake_ids = list(self.by_lake.keys())
        self.lake_sizes = {lake: len(indices) for lake, indices in self.by_lake.items()}
        
        # Pre-compute negative pools
        self.negative_pools = {}
        for lake_id in self.lake_ids:
            self.negative_pools[lake_id] = np.array(list(set(self.indices) - set(self.by_lake[lake_id])))
        
        # Convert positive pools to numpy arrays for faster sampling
        self.positive_pools = {lake: np.array(indices) for lake, indices in self.by_lake.items()}
        
        self.n_pos = P_pos
        self.n_neg = batch_size - P_pos - 1  # anchor + positives + negatives = batch_size

        # Track samples used per lake
        self.samples_per_lake = {lake: len(indices) for lake, indices in self.by_lake.items()}
        
        # Instead of dividing batch_size, use full batch_size
        self.batch_size = batch_size
        
        # Number of anchor-positive sets that can fit in a batch
        # Each anchor-positive set takes (1 + P_pos) spots
        self.sets_per_batch = batch_size // (P_pos + 1)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        
        # shuffle global indices then take this rank's slice 
        indices = torch.randperm(len(self.indices), generator=g).tolist()
        indices = indices[self.rank:self.total_size:self.num_replicas]
        
        # PAD to full batches so every rank yields the same number of batches
        num_full = (len(indices) + self.B - 1) // self.B  # ceil
        needed = num_full * self.B - len(indices)
        if needed > 0:
            indices.extend(indices[:needed])

        # process full batch_size chunks
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
                
            # Randomly select which samples will be anchors
            anchor_indices = np.random.choice(
                len(batch_indices), 
                self.sets_per_batch, 
                replace=False
            )
            
            batch = []
            # Use selected samples as anchors
            for idx in anchor_indices:
                anchor = batch_indices[idx]
                lake_id = self.idx_to_lake[anchor]
                
                # Add anchor
                batch.append(anchor)
                
                # Add positives
                pos_pool = self.positive_pools[lake_id]
                pos_mask = pos_pool != anchor
                pos_indices = pos_pool[pos_mask]
                positives = np.random.choice(
                    pos_indices,
                    min(self.P_pos, len(pos_indices)),
                    replace=False
                )
                batch.extend(positives)
            
            # Fill remaining slots with other samples
            remaining = [idx for idx in batch_indices if idx not in batch]
            batch.extend(remaining[:self.batch_size - len(batch)])
            
            yield batch

    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size
        
    def set_epoch(self, epoch):
        self.epoch = epoch

def get_padding_mask(padded_tensor: 
                     torch.Tensor, 
                     pad_value: float = 0.0) -> torch.BoolTensor:
    if padded_tensor.ndim == 3:  # (B, T, V)
        return ~(padded_tensor == pad_value).all(dim=-1)
    elif padded_tensor.ndim == 2:  # (B, T)
        return ~(padded_tensor == pad_value)
    else:
        raise ValueError(f"Unsupported shape: {padded_tensor.shape}")

def collate_fn(sample_dict):
    collated = {}
    keys = sample_dict[0].keys()

    for key in keys:
        values = [sample[key] for sample in sample_dict]
        
        if key == "simulation_params":
            if values[0] is not None:
                collated[key] = torch.stack(values)
            else:
                collated[key] = None
        # Keep metadata/non-tensor fields as lists (do not pad/stack)
        elif key in [
            # current names in dataset
            "lake_id", "num2Dvars", "num1Dvars", "num_depths", "lake_name", "idx"
        ]:
            collated[key] = values
        else:
            # convert numpy arrays to tensors when applicable
            if isinstance(values[0], np.ndarray):
                # handle datetime64 arrays specially - pad and stack them
                if values[0].dtype.kind == "M":  # datetime64
                    max_len = max(v.shape[0] for v in values)
                    padded = []
                    for v in values:
                        if v.shape[0] < max_len:
                            # Pad with NaT (Not a Time)
                            pad = np.array([np.datetime64('NaT')] * (max_len - v.shape[0]), dtype='datetime64[ns]')
                            v = np.concatenate([v, pad])
                        padded.append(v)
                    collated[key] = np.stack(padded, axis=0)  # (B, T) datetime64 array
                    continue
                # If object/string dtype, keep as list (cannot convert/pad)
                elif values[0].dtype.kind in ("U", "S", "O"):
                    collated[key] = values
                    continue
                values = [torch.from_numpy(v) for v in values]

            # Determine if padding is needed
            lengths = [v.shape[0] for v in values]
            needs_padding = len(set(lengths)) > 1

            # Use ID padding value for variable id sequences
            pad_value = PAD_VAL_ID if key in ("var_ids", "var_ids_x", "var_ids_y") else PAD_VAL_DEFAULT

            if needs_padding:
                collated[key] = pad_sequence(
                    values,
                    batch_first=True,
                    padding_value=pad_value
                )
            else:
                collated[key] = torch.stack(values)
                  

    # Create padding masks for irregular data
    collated["padding_mask_x"] = get_padding_mask(
        padded_tensor=collated["flat_seq_x"], 
        pad_value=PAD_VAL_DEFAULT)
    
    collated["padding_mask_y"] = get_padding_mask(
        padded_tensor=collated["flat_seq_y"], 
        pad_value=PAD_VAL_DEFAULT)

    return collated

def get_meta_df(datasets):
    if isinstance(datasets, ConcatDataset):
        ds_list = datasets.datasets
    elif isinstance(datasets, list):
        ds_list = datasets
    else:
        ds_list = [datasets]
    
    # Pre-allocate rows list with total size
    total_size = sum(len(ds) for ds in ds_list)
    rows = []
    rows.extend(
        {"idx": idx, "lake_id": ds.lake_id}
        for ds_idx, ds in enumerate(ds_list)
        for idx in range(sum(len(d) for d in ds_list[:ds_idx]), 
                        sum(len(d) for d in ds_list[:ds_idx+1]))
    )
    
    return pd.DataFrame(rows)

def build_dataloader(
    datasets,
    cfg,
    pad_value_id,
    pad_value_default,
    distributed=True,
    use_cl=False,
    plot=False
):
    if isinstance(datasets, list):
        if len(datasets) == 1:
            dataset = datasets[0]
        else:
            dataset = ConcatDataset(datasets)
    else:
        dataset = datasets
    
    if distributed:
        batch_size=cfg.batch_size
    else:
        batch_size=cfg.plot_batch_size
        
    P_pos=cfg.P_pos
    shuffle=cfg.shuffle
    drop_last=cfg.drop_last
    num_workers=cfg.num_workers
    pin_memory=cfg.pin_memory
    persistent_workers=cfg.persistent_workers
    parallel_mode=cfg.sharding_mode
    prefetch_factor=cfg.prefetch_factor
    global PAD_VAL_ID, PAD_VAL_DEFAULT
    PAD_VAL_ID=pad_value_id
    PAD_VAL_DEFAULT=pad_value_default

    if use_cl and not plot:
        meta_df = get_meta_df(dataset)
        is_ddp = (parallel_mode == "ddp")
        sampler = LakeBalancedBatchSampler(meta_df=meta_df,
                                        P_pos=P_pos,
                                        batch_size=batch_size,
                                        num_replicas=(torch.distributed.get_world_size() if is_ddp else 1),
                                        rank=(torch.distributed.get_rank() if is_ddp else 0))
        loader = DataLoader(
                        dataset,
                        batch_sampler=sampler,
                        num_workers=num_workers,
                        pin_memory=pin_memory,
                        persistent_workers=persistent_workers,
                        prefetch_factor=prefetch_factor,
                        multiprocessing_context='fork',
                        collate_fn=collate_fn
                    )

    else:
        if distributed:
            if parallel_mode=='ddp':
                sampler = DistributedSampler(
                    dataset,
                    num_replicas=torch.distributed.get_world_size(),
                    rank=torch.distributed.get_rank(),
                    shuffle=shuffle,
                    drop_last=drop_last
                )
            else:
                sampler=None
                shuffle=shuffle
        else:
            sampler=SequentialSampler(dataset)
            shuffle=False

        loader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                drop_last=drop_last,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                collate_fn=collate_fn)
    
    return loader

# Backward compatibility - keep original function names as aliases
get_padding_mask = get_padding_mask
collate_fn_ = collate_fn
get_meta_df = get_meta_df   
build_dataloader = build_dataloader
LakeBalancedBatchSampler = LakeBalancedBatchSampler
