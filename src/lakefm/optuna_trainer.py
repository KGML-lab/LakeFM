import torch
import optuna
import wandb

import math
import numpy as np
import time
import os
from tqdm import trange
from lakefm.trainer import Trainer
import torch.nn as nn

from utils.exp_utils import pretty_print

class OptunaTrainer(Trainer):
    """Enhanced Trainer with Optuna pruning capabilities for hyperparameter tuning"""
    
    def __init__(self, cfg, model, trial=None):
        super().__init__(cfg, model)
        self.trial = trial
        self.best_val_loss = float('inf')
        self.best_pred_loss = float('inf')
        self.best_lake_cl_loss = float('inf')
        self.epoch_val_losses = []
        self.should_prune_trial = False

    def init_wandb_for_trial(self):
        """Initialize W&B for this specific trial"""
        if self.rank == 0 and hasattr(self.cfg.optimization, 'wandb'):
            wandb_cfg = self.cfg.optimization.wandb
            
            # Prepare config for W&B
            config = {
                "trial_number": self.trial.number if self.trial else "unknown",
                "trial_params": self.trial.params if self.trial else {},
                "study_name": self.cfg.optimization.study_name,
                "dataset_config": {
                    "pretrain_dataset": self.cfg.study_dataset.pretrain_dataset,
                    "lake_ids": self.cfg.study_dataset.lake_ids,
                    "lake_ids_format": self.cfg.study_dataset.lake_ids_format
                }
            }
            
            # Add all config values
            config.update(self.cfg)
            
            wandb.init(
                project=wandb_cfg.project,
                entity=wandb_cfg.get("entity", wandb_cfg.username),
                name=f"trial_{self.trial.number}_{self.cfg.optimization.study_name}",
                config=config,
                tags=[
                    "hyperparameter_tuning", 
                    "optuna",
                    f"study_{self.cfg.optimization.study_name}",
                    f"dataset_{'-'.join(self.cfg.study_dataset.pretrain_dataset)}"
                ],
                save_code=self.cfg.trainer.get("save_code", False)
            )
            
            print(f"W&B initialized for trial {self.trial.number}")
            print(f"W&B URL: https://wandb.ai/{wandb_cfg.username}/{wandb_cfg.project}")
    
    def val_one_epoch(self, dataloader, epoch, plot=True):
        """Override to add trial reporting and pruning"""
        val_results = super().val_one_epoch(dataloader, epoch, plot)
        
        if not plot:  # Only process metrics when not plotting
            val_loss = val_results.get("loss", float('inf'))
            pred_loss = val_results.get("pred_loss", float('inf'))
            lake_cl_loss = val_results.get("lake_contrastive_loss", float('inf'))

            # Handle None values for lake contrastive loss (when disabled)
            lake_cl_loss_raw = val_results.get("lake_contrastive_loss", 100.0)
            lake_cl_loss = lake_cl_loss_raw if lake_cl_loss_raw is not None else 0.0
        
            
            # Store validation losses for this epoch
            self.epoch_val_losses.append({
                'total_loss': val_loss,
                'pred_loss': pred_loss,
                'lake_cl_loss': lake_cl_loss
            })

            # Log to W&B if initialized
            if self.rank == 0 and wandb.run is not None:
                metrics = {
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_pred_loss": pred_loss,
                    "val_lake_CL": lake_cl_loss,
                    "best_val_loss": min(self.best_val_loss, val_loss),
                    "best_pred_loss": min(self.best_pred_loss, pred_loss),
                    "best_lake_cl_loss": min(self.best_lake_cl_loss, lake_cl_loss)
                }
                # wandb.log(metrics)
            
            # Report to Optuna for pruning (using total validation loss)
            if self.trial and self.rank == 0:
                self.trial.report(val_loss, epoch)
                
                # Check if the trial should be pruned
                if self.trial.should_prune():
                    if self.rank == 0:
                        print(f"Trial {self.trial.number} pruned at epoch {epoch} with val_loss: {val_loss:.6f}")
                        if wandb.run is not None:
                            wandb.log({"trial_pruned": True, "pruned_at_epoch": epoch})
                            wandb.finish()
                    self.should_prune_trial = True
                    raise optuna.exceptions.TrialPruned()
            
            # Track best validation losses
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
            if pred_loss < self.best_pred_loss:
                self.best_pred_loss = pred_loss
            if lake_cl_loss < self.best_lake_cl_loss:
                self.best_lake_cl_loss = lake_cl_loss
        
        return val_results
    
    def get_lr_scheduler(self, optimizer, max_epochs, warmup_epochs=10, min_lr=1e-6):
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                # Gentler warmup - don't go to full learning rate
                return float(epoch) / float(max(1, warmup_epochs)) * 0.5  # Peak at 50% of base lr
            
            # More gradual cosine decay
            progress = float(epoch - warmup_epochs) / float(max(1, max_epochs - warmup_epochs))
            return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def pretrain_for_tuning(self, datasets, plot_dataset=None):
        """Modified pretraining function for hyperparameter tuning - returns all three losses"""
        if self.rank == 0:
            print("="*60)
            print(f"STARTING TRIAL {self.trial.number if self.trial else 'N/A'}")
            print("="*60)
            print(f"Trial parameters: {self.trial.params if self.trial else 'N/A'}")
            print(f"Dataset: {self.cfg.study_dataset.pretrain_dataset}")
            print(f"Lake IDs: {self.cfg.study_dataset.lake_ids}")
            print(f"Max epochs: {self.max_epochs}")
            print(f"Batch size: {self.cfg.dataloader.batch_size}")
            print(f"Learning rate: {self.trainer.lr}")
            print(f"Weight decay: {self.trainer.weight_decay}")
            print("="*60)
        
        # # Initialize W&B for this trial
            self.init_wandb_for_trial()
        
        if self.rank == 0:
            print("Moving model to device...")
        self.model.to(self.device)
        
        # Initialize W&B
        # self.init_wandb_for_trial()
        
        print("Setting up optimizer and scheduler...")
        optimizer = self.select_optimizer_()
        if self.lr_scheduling_or_not:
            model_scheduler = self.get_lr_scheduler(
                            optimizer=optimizer, 
                            max_epochs=self.max_epochs,
                            warmup_epochs=self.num_warmup_epochs)
        else:
            model_scheduler = None
        
        scaler = torch.amp.GradScaler()
        
        # Build dataloaders
        if self.rank == 0:
            print("Building dataloaders...")
            print(f"Dataset type: {type(datasets)}")
            if isinstance(datasets, list):
                print(f"Number of datasets: {len(datasets)}")
                for i, ds in enumerate(datasets):
                    print(f"  Dataset {i}: {len(ds)} samples")
            else:
                print(f"Single dataset: {len(datasets)} samples")
        
        from data.loader import build_dataloader
        
        if self.rank == 0:
            print("Building train dataloader...")
        train_dataloader = build_dataloader(
            datasets=datasets, 
            cfg=self.cfg.dataloader, 
            pad_value_id=self.pad_value_id, 
            pad_value_default=self.pad_value_default,
            distributed=True,
            use_cl=self.trainer.use_lake_cl
        )   
        
        if self.rank == 0:
            print(f"Train dataloader created: {len(train_dataloader)} batches")
            print("Building validation dataloader...")
        
        val_datasets = self.update_split(datasets, flag='val')
        val_dataloader = build_dataloader(
            datasets=val_datasets, 
            cfg=self.cfg.dataloader, 
            pad_value_id=self.pad_value_id, 
            pad_value_default=self.pad_value_default,
            distributed=True,
            use_cl=self.trainer.use_lake_cl
        )
        
        if self.rank == 0:
            print(f"Validation dataloader created: {len(val_dataloader)} batches")
            print("Starting training...")
        
        # Initialize tracking variables
        val_loss = 0
        losses = np.full(self.max_epochs, np.nan)
        
        try:
            if self.rank == 0:
                print("Entering training loop...")
                
            with trange(self.max_epochs, disable=(self.rank != 0)) as tr:
                for epoch in tr:
                    if self.rank == 0:
                        epoch_time = time.time()
                        if epoch == 0:
                            print("Starting first epoch...")
                    
                    # Training
                    train_results = self.train_one_epoch(
                        dataloader=train_dataloader,
                        optimizer=optimizer, 
                        scheduler=model_scheduler,
                        scaler=scaler,
                        epoch=epoch
                    )
                    if self.lr_scheduling_or_not:
                        model_scheduler.step()

                    print(f"Epoch {epoch + 1}/{self.max_epochs} - Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
                    
                    train_loss = train_results['loss']
                    train_pred_loss = train_results['pred_loss']
                    train_lake_contrastive_loss = train_results['lake_contrastive_loss']
                    
                    losses[epoch] = train_loss
                    
                    if self.rank == 0:
                        print(f"Epoch {epoch + 1}/{self.max_epochs} - Train Loss: {train_loss:.6f}")
                        print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
                    
                    # Validation every eval_freq epochs
                    if epoch % self.cfg.eval_freq == 0:
                        if self.rank == 0:
                            print("Starting validation...")
                            
                        with torch.no_grad():
                            val_time = time.time()
                            val_results = self.val_one_epoch(
                                dataloader=val_dataloader,
                                epoch=epoch,
                                plot=False  # No plotting during tuning
                            )
                            val_loss = val_results['loss']
                            val_pred_loss = val_results['pred_loss']
                            val_lake_contrastive_loss = val_results['lake_contrastive_loss']
                            
                            if self.rank == 0:
                                print(f"Validation Loss: {val_loss:.6f}")
                                print(f"Time taken for validation = {time.time() - val_time:.2f}s")
                    
                    # Prepare metrics for logging
                    if epoch % self.cfg.eval_freq == 0:
                        metrics = {
                            "train_loss": train_loss,
                            "train_pred_loss": train_pred_loss,
                            "train_lake_CL": train_lake_contrastive_loss,
                            "val_loss": val_loss,
                            "val_pred_loss": val_pred_loss,
                            "val_lake_CL": val_lake_contrastive_loss,
                            "learning_rate": optimizer.param_groups[0]['lr'],
                            "epoch": epoch
                        }
                    
                    # Log to W&B
                    if self.rank == 0 and wandb.run is not None:
                        wandb.log(metrics)
                    
                    # Update progress bar
                    if self.rank == 0:
                        display_metrics = {
                            "train_loss": f"{train_loss:.4f}",
                            "val_loss": f"{val_loss:.4f}" if epoch % self.cfg.eval_freq == 0 else "N/A",
                            "best_val": f"{self.best_val_loss:.4f}"
                        }
                        tr.set_postfix(display_metrics)
                    
                    # Early stopping if pruned
                    if self.should_prune_trial:
                        if self.rank == 0:
                            print(f"Trial pruned at epoch {epoch}")
                        break
        
        except optuna.exceptions.TrialPruned:
            # Trial was pruned, log final metrics and re-raise
            if self.rank == 0:
                print(f"Trial {self.trial.number} was pruned at epoch {epoch}")
                if wandb.run is not None:
                    wandb.summary.update({
                        "trial_pruned": True,
                        "final_train_loss": train_loss,
                        "final_val_loss": val_loss,
                        "best_val_loss": self.best_val_loss,
                        "epochs_completed": epoch + 1
                    })
                    wandb.finish()
            raise
        except Exception as e:
            if self.rank == 0:
                print(f"Trial failed with error: {e}")
                import traceback
                traceback.print_exc()
                if wandb.run is not None:
                    wandb.log({"trial_failed": True, "error": str(e)})
                    wandb.finish()
            return float('inf')
        finally:
            # Clean up W&B for successful completion
            if self.rank == 0 and wandb.run is not None and not self.should_prune_trial:
                print(f"Trial {self.trial.number} completed successfully")
                wandb.summary.update({
                    "final_train_loss": train_loss,
                    "final_val_loss": val_loss,
                    "best_val_loss": self.best_val_loss,
                    "epochs_completed": self.max_epochs,
                    "trial_completed": True
                })
                wandb.finish()
        
        if self.rank == 0:
            print("="*60)
            print(f"TRIAL {self.trial.number} COMPLETED")
            print(f"Best validation loss: {self.best_val_loss:.6f}")
            print(f"Best prediction loss: {self.best_pred_loss:.6f}")
            print(f"Best lake CL loss: {self.best_lake_cl_loss:.6f}")
            print("="*60)
        
        return self.best_val_loss