import os
import sys
import logging
import optuna
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner, HyperbandPruner
import torch
import hydra
from hydra import initialize, compose
import torch.distributed as dist
from hydra.core.global_hydra import GlobalHydra
import json
import numpy as np
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from pathlib import Path

from cli.main import seed_everything
from lakefm.optuna_trainer import OptunaTrainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_base_config():
    """Load the base configuration with proper path resolution"""
    # Clear any existing Hydra instance
    GlobalHydra.instance().clear()
    
    # Get current working directory
    cwd = Path.cwd()
    print(f"Current working directory: {cwd}")
    
    # Try different relative paths based on where we might be running from
    possible_paths = [
        "cli/conf/pretrain",  # If running from src/
        "conf/pretrain",      # If running from src/cli/
        "../cli/conf/pretrain",  # If running from some other location
    ]
    
    config_path = None
    for path in possible_paths:
        full_path = cwd / path
        print(f"Checking path: {full_path}")
        if full_path.exists() and (full_path / "default.yaml").exists():
            config_path = path
            print(f"Found config at: {full_path}")
            break
    
    if config_path is None:
        raise FileNotFoundError(f"Could not find pretrain config. Tried paths: {[str(cwd / p) for p in possible_paths]}")
    
    print(f"Using relative config path: {config_path}")
    # For Hydra, remove the leading "cli/" because CWD is already src/
    hydra_config_path = config_path
    if hydra_config_path.startswith("cli/"):
        hydra_config_path = hydra_config_path[len("cli/"):]
    print(f"Using relative config path for hydra: {hydra_config_path}")
    
    try:
        # Use the relative path for initialization
        with initialize(version_base="1.3", config_path=hydra_config_path):
            base_cfg = compose(config_name="default.yaml")
        
        print("Base configuration loaded successfully!")
        print(f"Available config keys: {list(base_cfg.keys())}")
        
        # Print model and data info if available
        if 'model' in base_cfg:
            print(f"Model keys: {list(base_cfg.model.keys())}")
            print(f"Model target: {base_cfg.model._target_}")
        
        if 'data' in base_cfg:
            print(f"Data keys: {list(base_cfg.data.keys())}")
            print(f"Data target: {base_cfg.data._target_}")
            
        return base_cfg
        
    except Exception as e:
        print(f"Error loading base config: {e}")
        import traceback
        traceback.print_exc()
        raise

def create_pruner(cfg):
    """Create the appropriate pruner based on configuration"""
    if not hasattr(cfg.optimization, 'pruner'):
        return MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=5)
    
    pruner_config = cfg.optimization.pruner
    
    if pruner_config.type == "median":
        return MedianPruner(
            n_startup_trials=pruner_config.get("n_startup_trials", 5),
            n_warmup_steps=pruner_config.get("n_warmup_steps", 10),
            interval_steps=pruner_config.get("interval_steps", 5)
        )
    elif pruner_config.type == "successive_halving":
        return SuccessiveHalvingPruner(
            min_resource=pruner_config.get("n_warmup_steps", 10),
            reduction_factor=2
        )
    elif pruner_config.type == "hyperband":
        return HyperbandPruner(
            min_resource=pruner_config.get("n_warmup_steps", 10),
            max_resource=cfg.resources.max_epochs,
            reduction_factor=3
        )
    else:
        return MedianPruner()

def suggest_parameter(trial, param_spec):
    """Suggest a parameter value based on its specification"""
    name = param_spec.name
    
    if param_spec.type == "float":
        return trial.suggest_float(
            name, param_spec.low, param_spec.high, 
            log=param_spec.get("log", False)
        )
    elif param_spec.type == "int":
        return trial.suggest_int(
            name, param_spec.low, param_spec.high,
            step=param_spec.get("step", 1)
        )
    elif param_spec.type == "categorical":
        return trial.suggest_categorical(name, param_spec.choices)
    else:
        raise ValueError(f"Unknown parameter type: {param_spec.type}")

def define_search_space(trial, cfg):
    """Define the search space based on config"""
    params = {}
    
    for param_spec in cfg.search_space:
        params[param_spec.path] = suggest_parameter(trial, param_spec)
    
    return params

def validate_model_dimensions(trial, params):
    """Validate that model dimensions are compatible"""
    d_model = params.get("model.d_model")
    n_heads = params.get("model.n_heads")
    
    if d_model is not None and n_heads is not None:
        if d_model % n_heads != 0:
            divisors = [i for i in range(2, 17) if d_model % i == 0]
            if divisors:
                params["model.n_heads"] = min(divisors, key=lambda x: abs(x - n_heads))
                print(f"Adjusted n_heads to {params['model.n_heads']} to be compatible with d_model={d_model}")
    trial.set_user_attr("final_n_heads", n_heads)
    return params

def apply_hyperparameters_to_config(base_cfg, params, study_cfg):
    """Apply hyperparameters to the base configuration"""
    trial_cfg = OmegaConf.create(base_cfg)
    
    OmegaConf.set_struct(trial_cfg, False)
    
    print(f"Applying hyperparameters: {params}")
    
    for path, value in params.items():
        try:
            OmegaConf.update(trial_cfg, path, value, merge=True)
            print(f"Successfully updated {path} = {value}")
        except Exception as e:
            print(f"Warning: Failed to update {path} = {value}: {e}")
    
    # Handle seq_len and pred_len combinations
    if "seq_pred_combo" in params:
        combo = params["seq_pred_combo"]
        # seq_len, pred_len = map(int, combo.split("_"))
        # Add type checking to handle both string and non-string cases
        if isinstance(combo, str):
            print(f"Using seq_pred_combo as string: {combo}")
            if "_" in combo:
                seq_len, pred_len = map(int, combo.split("_"))
            elif ":" in combo:
                seq_len, pred_len = map(int, combo.split(":"))
        else:
            # Try to convert to string if it's a numeric value
            print(f"WARNING: seq_pred_combo is not a string: {combo}, type: {type(combo)}")
            combo_str = str(combo)
            if "_" in combo_str:
                seq_len, pred_len = map(int, combo_str.split("_"))
            elif ":" in combo:
                seq_len, pred_len = map(int, combo.split(":"))
            else:
                # Fall back to default values
                seq_len = int(combo)[:2]
                pred_len = seq_len // 2
                print(f"Using fallback: seq_len={seq_len}, pred_len={pred_len}")
        
        # If using patch tokenization, ensure seq_len is divisible by patch_size
        if trial_cfg.model.tokenization == "patch":
            patch_size = trial_cfg.model.patch_size
            
            # Verify divisibility (should already be satisfied by our preset combinations)
            if seq_len % patch_size != 0:
                print(f"Warning: seq_len {seq_len} not divisible by patch_size {patch_size}")
                # Adjust if needed (though our presets should avoid this)
                seq_len = ((seq_len // patch_size) + 1) * patch_size
                print(f"Adjusted seq_len to {seq_len}")
        
        # Set the final values in the trial config
        trial_cfg.seq_len = seq_len
        trial_cfg.pred_len = pred_len
        
        print(f"Set seq_len={seq_len}, pred_len={pred_len}")
    
    # Handle embedding dimensions based on add_or_concat mode
    if trial_cfg.model.add_or_concat == "add":
        # In 'add' mode, set all embedding dims equal to d_model.
        trial_cfg.model.var_embed_dim = trial_cfg.model.d_model
        trial_cfg.model.depth_embed_dim = trial_cfg.model.d_model
        trial_cfg.model.inp_embed_dim = trial_cfg.model.d_model
        print(f"[Add mode] Set all embedding dims to d_model: {trial_cfg.model.d_model}")
    
    # Apply study-specific dataset configuration
    if hasattr(study_cfg, "study_dataset"):
        # Update the top-level study_dataset so later trainer code finds it
        OmegaConf.update(trial_cfg, "study_dataset", study_cfg.study_dataset, merge=True)
        print(f"Overrode top-level study_dataset = {trial_cfg.study_dataset}")
        # Override data keys with study_dataset values
        # Update with merge=True
        OmegaConf.update(trial_cfg, "data.pretrain_dataset", study_cfg.study_dataset.pretrain_dataset, merge=True)
        OmegaConf.update(trial_cfg, "data.lake_ids",  study_cfg.study_dataset.lake_ids, merge=True)
        OmegaConf.update(trial_cfg, "data.lake_ids_format", study_cfg.study_dataset.lake_ids_format, merge=True)
        OmegaConf.update(trial_cfg, "data.use_global_lake_filter", True, merge=True)  # Explicitly enable filter
        
        # Add debug prints
        print(f"Updated lake_ids = {trial_cfg.data.lake_ids}")
        print(f"Updated lake_ids_format = {trial_cfg.data.lake_ids_format}")
        print(f"Updated lake_ids_format = {trial_cfg.data.lake_ids_format}")

        # After updating config
    if trial_cfg.data.lake_ids != study_cfg.study_dataset.lake_ids:
        print("WARNING: Lake IDs not properly updated!")
        print(f"Expected: {study_cfg.study_dataset.lake_ids}")
        print(f"Got: {trial_cfg.data.lake_ids}")
    
    # Apply resource limits
    if hasattr(study_cfg, 'resources'):
        try:
            if hasattr(study_cfg.resources, 'max_epochs'):
                trial_cfg.trainer.max_epochs = study_cfg.resources.max_epochs
                print(f"Updated trainer.max_epochs = {study_cfg.resources.max_epochs}")
                
            if hasattr(study_cfg.resources, 'batch_size'):
                trial_cfg.dataloader.batch_size = study_cfg.resources.batch_size
                print(f"Updated dataloader.batch_size = {study_cfg.resources.batch_size}")
                
            if hasattr(study_cfg.resources, 'num_workers'):
                trial_cfg.dataloader.num_workers = study_cfg.resources.num_workers
                print(f"Updated dataloader.num_workers = {study_cfg.resources.num_workers}")
        except Exception as e:
            print(f"Warning: Failed to update resource limits: {e}")
    
    # Overwrite server prefix from hyperparameter config
    if hasattr(study_cfg.optimization, 'server_prefix'):
        try:
            OmegaConf.update(trial_cfg, "server_prefix", study_cfg.optimization.server_prefix, merge=True)
            print(f"Updated server_prefix = {trial_cfg.server_prefix}")
        except Exception as e:
            print(f"Warning: Failed to update server_prefix: {e}")
    
    # Disable normal W&B for trials
    try:
        trial_cfg.trainer.wandb_project = None
        trial_cfg.trainer.save_code = False
        print("Disabled W&B for trial")
    except Exception as e:
        print(f"Warning: Failed to disable W&B: {e}")
    
    # Merge optimization settings from the hyperparameter config
    if hasattr(study_cfg, "optimization"):
        OmegaConf.update(trial_cfg, "optimization", study_cfg.optimization, merge=True)
        print(f"Updated optimization config: {trial_cfg.optimization}")
    
    # Re-enable struct mode
    OmegaConf.set_struct(trial_cfg, True)
    
    return trial_cfg

def objective(trial, study_cfg, base_cfg):
    """Objective function for Optuna (with DDP init)"""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    # set device
    torch.cuda.set_device(local_rank)
    
    if not dist.is_initialized():
        if world_size > 1:
            dist.init_process_group(backend="nccl", init_method="env://")
        else:
            # For single GPU, use gloo backend or nccl with single rank
            dist.init_process_group(backend="gloo", rank=0, world_size=1)
    # barrier to sync before trial starts (only if multi-GPU)
    if world_size > 1:
        dist.barrier()
    
    try:
        # 1) Sample hyperparams
        params = define_search_space(trial, study_cfg)
        params = validate_model_dimensions(trial, params)
        print(f"Generated parameters for trial {trial.number}: {params}")
        
        # 2) Apply to base config
        trial_cfg = apply_hyperparameters_to_config(base_cfg, params, study_cfg)
        
        # Record final values as trial attributes
        trial.set_user_attr("final_d_model", trial_cfg.model.d_model)
        trial.set_user_attr("final_var_embed_dim", trial_cfg.model.var_embed_dim)
        trial.set_user_attr("final_depth_embed_dim", trial_cfg.model.depth_embed_dim)
        trial.set_user_attr("final_inp_embed_dim", trial_cfg.model.inp_embed_dim)
        trial.set_user_attr("final_seq_len", trial_cfg.seq_len)
        trial.set_user_attr("final_pred_len", trial_cfg.pred_len)
        
        logger.info(f"Trial {trial.number} parameters: {params}")
        logger.info(f"Study dataset: {study_cfg.study_dataset.pretrain_dataset}")
        logger.info(f"Lake IDs: {study_cfg.study_dataset.lake_ids if hasattr(study_cfg.study_dataset, 'lake_ids') else 'Not specified'}")
        print(f"Model config: d_model={trial_cfg.model.d_model}, num_layers={trial_cfg.model.num_layers}")
        print(f"Trainer config: lr={trial_cfg.trainer.lr}, max_epochs={trial_cfg.trainer.max_epochs}")
        
        # Seed everything for reproducibility
        seed_everything(trial_cfg.get("seed", 2025) + trial.number)
        
        # Instantiate model with updated parameters
        print(f"Instantiating model...")
        model = instantiate(trial_cfg.model, _convert_="all").to(local_rank)
        
        # Wrap model in DDP if multi-GPU
        if world_size > 1:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank], output_device=local_rank
            )
 
        # Load datasets
        # In optimize.py, before loading datasets, add:
        print("Before loading datasets:")
        print(f"trial_cfg.data.lake_ids = {trial_cfg.data.lake_ids}")
        print(f"trial_cfg.data.lake_ids_format = {trial_cfg.data.lake_ids_format}")
        print(f"trial_cfg.study_dataset = {trial_cfg.study_dataset}")

        print(f"Loading datasets...")
        builder = instantiate(trial_cfg.data)
        datasets, plot_dataset = builder.load_dataset(
            server_prefix=trial_cfg.server_prefix,
            rank=rank,
            world_size=world_size,
            root_cfg=trial_cfg,
        )

        print("Datasets loaded successfully")
        # 5) Create trainer and run
        trainer = OptunaTrainer(trial_cfg, model, trial=trial)

        best_val_loss = trainer.pretrain_for_tuning(datasets, plot_dataset)
        return best_val_loss
        
    except optuna.exceptions.TrialPruned:
        print(f"Trial {trial.number} was pruned")
        raise
    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}")
        import traceback
        traceback.print_exc()
        return float('inf')
    finally:
        torch.cuda.empty_cache()

@hydra.main(version_base="1.3", config_name="default.yaml", config_path="conf/hparam_search/")
def main(cfg: DictConfig):
    """Main entry point for hyperparameter optimization"""
    # rank = int(os.environ.get("RANK", 0))
    # world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    logger.info("Starting hyperparameter optimization")
    logger.info(f"Study name: {cfg.optimization.study_name}")
    logger.info(f"Number of trials: {cfg.optimization.n_trials}")
    # logger.info(f"World size: {world_size}")
    
    # Load the base configuration
    print("Loading base configuration...")
    try:
        base_cfg = load_base_config()
        print("Base configuration loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load base configuration: {e}")
        return
        
    logger.info(f"Database storage: {cfg.optimization.storage}")
    logger.info(f"Output directory: {cfg.optimization.output_dir}")
    logger.info(f"Study dataset: {cfg.study_dataset.pretrain_dataset}")
    logger.info(f"Lake IDs: {cfg.study_dataset.lake_ids if hasattr(cfg.study_dataset, 'lake_ids') else 'Not specified'}")
    
    if hasattr(cfg.optimization, 'wandb'):
        logger.info(f"W&B Project: {cfg.optimization.wandb.project}")
        logger.info(f"W&B Username: {cfg.optimization.wandb.username}")
        
    # Create output directory
    os.makedirs(cfg.optimization.output_dir, exist_ok=True)
    
    # Create database directory
    db_dir = os.path.dirname(cfg.optimization.storage.replace("sqlite:///", ""))
    os.makedirs(db_dir, exist_ok=True)
    logger.info(f"Created database directory: {db_dir}")
    
    # Create pruner
    pruner = create_pruner(cfg)
    
    # Create optuna study
    study = optuna.create_study(
        study_name=cfg.optimization.study_name,
        storage=cfg.optimization.storage,
        direction=cfg.optimization.direction,
        load_if_exists=cfg.optimization.load_if_exists,
        pruner=pruner
    )
        
        # Run optimization with both configs
    try:
        study.optimize(
            lambda trial: objective(trial, cfg, base_cfg),
            n_trials=cfg.optimization.n_trials,
            n_jobs=1,
            show_progress_bar=True
        )
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Print and save results
    if len(study.trials) > 0 and study.best_trial is not None:
        logger.info("="*50)
        logger.info("OPTIMIZATION COMPLETE")
        logger.info("="*50)
        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best validation loss: {study.best_trial.value:.6f}")
        logger.info(f"Best parameters:")
        for key, value in study.best_trial.params.items():
            logger.info(f"  {key}: {value}")
        
        # Save results
        wandb_info = {}
        if hasattr(cfg.optimization, 'wandb'):
            wandb_info = {
                "wandb_project": cfg.optimization.wandb.project,
                "wandb_username": cfg.optimization.wandb.username,
                "wandb_url": f"https://wandb.ai/{cfg.optimization.wandb.username}/{cfg.optimization.wandb.project}"
            }
            
        results = {
            "study_name": cfg.optimization.study_name,
            "study_dataset": {
                "pretrain_dataset": list(cfg.study_dataset.pretrain_dataset),
                "lake_ids": list(cfg.study_dataset.lake_ids) if hasattr(cfg.study_dataset, 'lake_ids') else [],
                "lake_ids_format": cfg.study_dataset.lake_ids_format
            },
            "best_trial": int(study.best_trial.number),
            "best_value": float(study.best_trial.value),
            "best_params": dict(study.best_trial.params),
            "n_trials": len(study.trials),
            "pruned_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            "completed_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            **wandb_info
        }
        
        try:
            with open(os.path.join(cfg.optimization.output_dir, "best_params.json"), "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to: {cfg.optimization.output_dir}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    else:
        logger.warning("No successful trials completed")

if __name__ == "__main__":
    main()