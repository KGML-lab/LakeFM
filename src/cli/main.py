import os
import sys
import warnings
import random
import numpy as np
import torch
import hydra

from dotenv import load_dotenv 
load_dotenv()
from typing import cast
from torch.utils.data import Dataset
from functools import partial
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch import nn
from utils.exp_utils import pretty_print
from utils.eval_utils import load_model_from_ckpt, load_checkpoint_states
from lakefm.trainer import Trainer
from lakefm.evaluator import Evaluator
from torch.nn.parallel import DistributedDataParallel as DDP
from data.builder.base import BaseLakeBuilder
from hydra.experimental import initialize, compose
from lakefm.extract_embeddings import run_extract

warnings.filterwarnings('ignore')

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using multi-GPU

    # for deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@hydra.main(version_base="1.3", config_name="default.yaml", config_path="conf/pretrain/")
def main(cfg: DictConfig):
    # Get DDP env vars (set by torchrun)
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Set up device and DDP
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl")

    print(f"[Rank {local_rank}] Using device: cuda:{local_rank} = {torch.cuda.get_device_name(local_rank)}")

    if cfg.tf32:
        assert cfg.trainer.precision == 32
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    seed_everything(cfg.seed + rank)

    model: nn.Module = instantiate(cfg.model, _convert_="all").to(local_rank)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # Load checkpoint states if resuming
    resume_states = None
    if cfg.task_name in ("pretrain", "finetune") and cfg.trainer.resume_checkpoint:
        # model, epoch = load_model_from_ckpt(cfg, model, cfg.trainer.resume_checkpoint, local_rank)
        resume_states = load_checkpoint_states(cfg, cfg.trainer.resume_checkpoint, local_rank)
        # Load model state immediately since model is already created
        if resume_states and 'model_state_dict' in resume_states:
            model.module.load_state_dict(resume_states['model_state_dict'])
            if rank == 0:
                print(f"Loaded model state from checkpoint")

    if rank==0:
        pretty_print("Model instantiated")

    builder: BaseLakeBuilder = instantiate(cfg.data)
    datasets, plot_dataset = builder.load_dataset(server_prefix=cfg.server_prefix, 
                                                    rank=rank, 
                                                    world_size=world_size, 
                                                    root_cfg=cfg)
    trainer = Trainer(cfg, model, rank=rank)

    if cfg.task_name == 'pretrain':
        if rank == 0:
            pretty_print("Starting Pre-training")
        trainer.pretrain(datasets, plot_dataset, resume_states=resume_states)
    elif cfg.task_name == 'evaluate':
        if rank == 0:
            pretty_print("Starting Evaluation")
            
        model, epoch = load_model_from_ckpt(cfg, 
                                            model, 
                                            cfg.evaluator.ckpt_path, 
                                            local_rank)
        evaluator = Evaluator(cfg, model, trainer, rank=rank, epoch=epoch)
        evaluator.run(datasets=datasets, flag='test', plot_datasets=plot_dataset, scaling=cfg.data.norm_override)
    torch.distributed.destroy_process_group()

if __name__=='__main__':
    main()