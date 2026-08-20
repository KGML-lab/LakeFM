import os
import sys
import socket
# ensures 'src/' is in PYTHONPATH for imports and Hydra _target_
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# sys.path.append(os.path.abspath(os.path.join('..', os.path.dirname(__name__))))
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

def _get_env_int(name: str, default: int | None = None) -> int | None:
    value = os.environ.get(name, None)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"Environment variable {name} must be an integer, got: {value!r}") from exc


def _validate_cuda_ddp_env(world_size: int, local_rank: int, local_world_size: int):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, but DDP/NCCL launch was requested.")

    visible_gpu_count = torch.cuda.device_count()
    if visible_gpu_count < 1:
        raise RuntimeError("No visible CUDA devices found for this process.")

    if local_rank >= visible_gpu_count:
        raise RuntimeError(
            f"Invalid LOCAL_RANK={local_rank} for visible_gpu_count={visible_gpu_count}. "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}"
        )

    if local_world_size > visible_gpu_count:
        raise RuntimeError(
            f"LOCAL_WORLD_SIZE={local_world_size} exceeds visible_gpu_count={visible_gpu_count}. "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}"
        )

    expected_local_world_size = _get_env_int("EXPECTED_LOCAL_WORLD_SIZE", None)
    if expected_local_world_size is not None and local_world_size != expected_local_world_size:
        raise RuntimeError(
            f"Expected LOCAL_WORLD_SIZE={expected_local_world_size}, got {local_world_size}. "
            "Check --nproc_per_node and launcher arguments."
        )

    expected_world_size = _get_env_int("EXPECTED_WORLD_SIZE", None)
    if expected_world_size is not None and world_size != expected_world_size:
        raise RuntimeError(
            f"Expected WORLD_SIZE={expected_world_size}, got {world_size}. "
            "Check torchrun launch arguments."
        )


def _print_rank_device_summary(rank: int, world_size: int, local_rank: int):
    payload = {
        "rank": rank,
        "local_rank": local_rank,
        "host": socket.gethostname(),
        "device": torch.cuda.current_device(),
        "device_name": torch.cuda.get_device_name(torch.cuda.current_device()),
    }
    gathered = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(gathered, payload)

    if rank == 0:
        print("DDP sanity summary:")
        for item in sorted(gathered, key=lambda x: x["rank"]):
            print(
                f"  rank={item['rank']}/{world_size - 1} "
                f"host={item['host']} local_rank={item['local_rank']} "
                f"device={item['device']} name={item['device_name']}"
            )


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
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    print(f"Rank {rank}: Using device: cuda:{local_rank} = {torch.cuda.get_device_name(local_rank)}")
    _validate_cuda_ddp_env(world_size, local_rank, local_world_size)

    # Set up device and DDP
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl")

    print(f"[Rank {local_rank}] Using device: cuda:{local_rank} = {torch.cuda.get_device_name(local_rank)}")
    _print_rank_device_summary(rank, world_size, local_rank)

    # if cfg.task_name == 'evaluate':
    #     cfg = get_final_eval_cfg(cfg)
    
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
    # elif cfg.task_name == 'finetune':
    #     # not implemented
    #     if rank == 0:
    #         pretty_print("Starting Fine-tuning")
    #     trainer.finetune(datasets, plot_dataset, resume_states=resume_states)
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