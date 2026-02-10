import os
import torch
from utils.exp_utils import pretty_print

def resolve_from_dir(cfg, dir_path):
    # pick “best” or “last”
    name = cfg.trainer.best_ckpt if cfg.evaluator.which_ckpt == 'best' else cfg.trainer.last_filename
    candidate = os.path.join(dir_path, name)
    if os.path.isfile(candidate):
        return candidate
    # fallback to the other
    alt_name = cfg.trainer.last_filename if cfg.evaluator.which_ckpt == 'best' else cfg.trainer.best_ckpt
    alt = os.path.join(dir_path, alt_name)
    if os.path.isfile(alt):
        return alt
    return None

def load_model_from_ckpt(cfg, model, ckpt_path, local_rank):
    """
    Load model weights from checkpoint.
    """
    if os.path.isdir(ckpt_path):
        found = resolve_from_dir(cfg,  ckpt_path)
        if not found:
            raise FileNotFoundError(f"No checkpoint (best or last) found in directory: {ckpt_path}")
        ckpt_path = found
    map_loc = {"cuda:%d" % 0: f"cuda:{local_rank}"} if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=map_loc)
    state_dict = ckpt.get("model_state_dict", ckpt)
    raw_epoch = ckpt.get("epoch", None) or ckpt.get("global_step", None)
    model.module.load_state_dict(state_dict, strict=False)   
    # allow mismatches
    missing_keys, unexpected_keys = model.module.load_state_dict(state_dict, strict=False)
    if local_rank == 0:
        pretty_print(f"Ignored keys in checkpoint that didn’t match model:\n  {unexpected_keys}")
        pretty_print(f"Model params missing in checkpoint:\n  {missing_keys}")
    if local_rank == 0:
            pretty_print(f"Loaded weights from {ckpt_path}")
    return model, raw_epoch

def load_checkpoint_states(cfg, ckpt_path, local_rank):
    """
    Load all checkpoint states and return as dict.
    Returns None if no checkpoint found.
    """
    if os.path.isdir(ckpt_path):
        found = resolve_from_dir(cfg, ckpt_path)
        if not found:
            print(f"No checkpoint found in directory: {ckpt_path}")
            return None
        ckpt_path = found
    elif not os.path.exists(ckpt_path):
        print(f"Checkpoint file not found: {ckpt_path}")
        return None
    
    map_loc = {"cuda:%d" % 0: f"cuda:{local_rank}"} if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(ckpt_path, map_location=map_loc)
    
    if local_rank == 0:
        print(f"Loaded checkpoint from {ckpt_path}")
        epoch = checkpoint.get('epoch', 'unknown')
        min_loss = checkpoint.get('min_vali_loss', 'unknown')
        print(f"Checkpoint info - Epoch: {epoch}, Min validation loss: {min_loss}")
    
    return checkpoint