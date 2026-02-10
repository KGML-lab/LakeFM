import numpy as np
import os

def pretty_print(text):
    width = len(text) + 2  # Adding one space on each side

    # Print the box
    print("+" + "-" * width + "+")
    print("| " + text + " |")
    print("+" + "-" * width + "+")

def print_model_and_data_info(model, datasets, rank=0):
    """
    Print model size and data size information before training.
    
    """
    if rank != 0:
        return
        
    print("="*60)
    print("MODEL AND DATA INFORMATION")
    print("="*60)
    
    # Model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB (assuming float32)")
    
    # Data size
    if isinstance(datasets, list):
        total_samples = sum(len(ds) for ds in datasets)
        print(f"Pretraining Data:")
        print(f"  Number of datasets: {len(datasets)}")
        for i, ds in enumerate(datasets):
            print(f"    Dataset {i}: {len(ds):,} samples")
        print(f"  Total samples: {total_samples:,}")
    else:
        print(f"Pretraining Data:")
        print(f"  Total samples: {len(datasets):,}")
    
    print("="*60)