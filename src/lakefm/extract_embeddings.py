import os
import sys
import argparse
import json
import math
from datetime import datetime as dt

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader
from hydra.utils import instantiate
from hydra.experimental import initialize, compose
from omegaconf import OmegaConf

from data.builder.base import BaseLakeBuilder
from data.loader import build_dataloader
from utils.exp_utils import pretty_print

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def _resolve_ckpt_path(cfg, ckpt_path):
    if os.path.isdir(ckpt_path):
        name = cfg.trainer.best_ckpt if cfg.evaluator.which_ckpt == 'best' else cfg.trainer.last_filename
        candidate = os.path.join(ckpt_path, name)
        if os.path.isfile(candidate):
            return candidate
        alt_name = cfg.trainer.last_filename if cfg.evaluator.which_ckpt == 'best' else cfg.trainer.best_ckpt
        alt = os.path.join(ckpt_path, alt_name)
        if os.path.isfile(alt):
            return alt
        raise FileNotFoundError(f"No checkpoint (best/last) found in directory: {ckpt_path}")
    return ckpt_path


def _strip_module_prefix(state_dict):
    return {k.replace('module.', '', 1): v for k, v in state_dict.items()}


@torch.no_grad()
def extract_embeddings_from_loader(cfg,
                                   model: nn.Module,
                                   loader: DataLoader,
                                   output_dir: str,
                                   device: str = "cuda:0",
                                   winter_months=(11, 12, 1, 2),
                                   summer_months=(6, 7, 8)):
    """
    Iterate an existing loader with an existing model to save per-sample and per-lake representations.
    Directory layout per-lake:
      {output_dir}/{lake_name_or_id}/sample_embeddings/sample_{k}.npy
      {output_dir}/{lake_name_or_id}/average_embedding.npy
      {output_dir}/{lake_name_or_id}/average_winter.npy
      {output_dir}/{lake_name_or_id}/average_summer.npy
    """
    os.makedirs(output_dir, exist_ok=True)
    device_t = torch.device(device if torch.cuda.is_available() else "cpu")

    model_ref = model.module if hasattr(model, "module") else model
    model_ref.eval()

    # Running stats per lake_id
    lake_stats = {}
    sample_counters = {}

    def get_lake_dirname(lake_id, lake_name):
        # Prefer name if non-empty, else id
        if lake_name is None or str(lake_name).strip() == "":
            return f"lake_{int(lake_id)}"
        # sanitize name
        name = str(lake_name)
        if isinstance(lake_name, bytes):
            try:
                name = lake_name.decode('utf-8')
            except Exception:
                name = str(lake_name)
        safe = "".join(c if c.isalnum() or c in ("_", "-", ".") else "_" for c in name.strip())
        return f"{safe}_{int(lake_id)}"

    def month_from_batch_row(dt_row):
        # dt_row: numpy array (T,) of datetime64 or strings/NaT
        if dt_row is None or len(dt_row) == 0:
            return None
        for d64 in dt_row:
            s = str(d64)
            if s != 'NaT' and len(s) >= 10:
                try:
                    return dt.strptime(s.split('T')[0], "%Y-%m-%d").month
                except Exception:
                    continue
        return None

    for batch in loader:
        # Move tensors to device
        seq_X = batch["flat_seq_x"].to(device_t)
        mask_X = batch["flat_mask_x"].to(device_t)
        sample_ids_x = batch["sample_ids_x"].to(device_t)
        time_ids_x = batch["time_ids_x"].to(device_t)
        var_ids_x = batch["var_ids_x"].to(device_t)
        padding_mask_x = batch["padding_mask_x"].to(device_t)
        depth_values_x = batch["depth_values_x"].to(device_t)
        time_values_x = batch["time_values_x"].to(device_t)

        # Targets metadata (decoder inputs)
        tgt_variate_ids = batch["var_ids_y"].to(device_t)
        tgt_time_values = batch["time_values_y"].to(device_t)
        tgt_time_ids = batch["time_ids_y"].to(device_t)
        tgt_depth_values = batch["depth_values_y"].to(device_t)
        tgt_padding_mask = batch["padding_mask_y"].to(device_t)

        # Forward pass to get pooled z (static representation)
        with torch.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
            _, _, z = model_ref(data=seq_X,
                                observed_mask=mask_X,
                                sample_ids=sample_ids_x,
                                variate_ids=var_ids_x,
                                padding_mask=padding_mask_x,
                                depth_values=depth_values_x,
                                pred_len=cfg.pred_len,
                                seq_len=cfg.seq_len,
                                time_values=time_values_x,
                                time_ids=time_ids_x,
                                tgt_variate_ids=tgt_variate_ids,
                                tgt_time_values=tgt_time_values,
                                tgt_time_ids=tgt_time_ids,
                                tgt_depth_values=tgt_depth_values,
                                tgt_padding_mask=tgt_padding_mask)
        Z_cpu = z.detach().cpu().numpy()  # (B, d)

        # Metadata
        lake_ids = batch["lake_id"]
        lake_names = batch["lake_name"]
        dt_x = batch.get("datetime_strs_x", None)  # (B, T) numpy datetime64
        B = Z_cpu.shape[0]
        for i in range(B):
            lid = int(lake_ids[i])
            lname = lake_names[i]
            ldir = get_lake_dirname(lid, lname)
            lake_dir = os.path.join(output_dir, ldir)
            sample_dir = os.path.join(lake_dir, "sample_embeddings")
            os.makedirs(sample_dir, exist_ok=True)

            # Per-sample save
            k = sample_counters.get(lid, 0) + 1
            sample_counters[lid] = k
            np.save(os.path.join(sample_dir, f"sample_{k}.npy"), Z_cpu[i])

            # Running stats
            stats = lake_stats.get(lid)
            if stats is None:
                d = Z_cpu.shape[1]
                stats = {
                    "sum": np.zeros((d,), dtype=np.float64),
                    "count": 0,
                    "winter_sum": np.zeros((d,), dtype=np.float64),
                    "winter_count": 0,
                    "summer_sum": np.zeros((d,), dtype=np.float64),
                    "summer_count": 0,
                    "lake_dir": lake_dir
                }
                lake_stats[lid] = stats
            stats["sum"] += Z_cpu[i]
            stats["count"] += 1

            # Seasonal bucket
            m = None
            if dt_x is not None:
                m = month_from_batch_row(dt_x[i])
            if m in set(winter_months):
                stats["winter_sum"] += Z_cpu[i]
                stats["winter_count"] += 1
            if m in set(summer_months):
                stats["summer_sum"] += Z_cpu[i]
                stats["summer_count"] += 1

    # Write per-lake averages
    for lid, stats in lake_stats.items():
        lake_dir = stats["lake_dir"]
        if stats["count"] > 0:
            avg = stats["sum"] / max(1, stats["count"])
            np.save(os.path.join(lake_dir, "average_embedding.npy"), avg)
        if stats["winter_count"] > 0:
            wavg = stats["winter_sum"] / max(1, stats["winter_count"])
            np.save(os.path.join(lake_dir, "average_winter.npy"), wavg)
        if stats["summer_count"] > 0:
            savg = stats["summer_sum"] / max(1, stats["summer_count"])
            np.save(os.path.join(lake_dir, "average_summer.npy"), savg)

    # Save variable embedding table once at the root output_dir
    model_ref = model.module if hasattr(model, "module") else model
    try:
        var_weight = model_ref.var_id_embed.weight.detach().cpu().numpy()
    except Exception:
        var_weight = None
    if var_weight is not None:
        raw_id_to_var = OmegaConf.to_container(cfg.data.id_to_var, resolve=True)
        id_to_var = {int(k): v for k, v in raw_id_to_var.items()} if raw_id_to_var else {}
        np.savez_compressed(os.path.join(output_dir, "variable_embeddings.npz"),
                            E=var_weight,
                            id_to_var=np.array(list(id_to_var.items()), dtype=object))

    pretty_print(f"Saved sample and averaged embeddings under: {output_dir}")


@torch.no_grad()
def run_extract(cfg, 
                output_dir, 
                device_id=0, 
                seasons=("winter", "summer"), 
                winter_months=(11, 12, 1, 2), 
                summer_months=(6, 7, 8)):
    device = torch.device("cuda:"+str(device_id) if torch.cuda.is_available() else "cpu")

    # Instantiate model
    model: nn.Module = instantiate(cfg.model, _convert_="all").to(device)
    model.eval()

    # Load checkpoint (robust to DDP-saved keys)
    ckpt_path = _resolve_ckpt_path(cfg, cfg.evaluator.ckpt_path)
    map_loc = {"cuda:%d" % 0: "cuda:0"} if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=map_loc)
    state_dict = ckpt.get("model_state_dict", ckpt)
    try:
        model.load_state_dict(state_dict, strict=False)
    except Exception:
        model.load_state_dict(_strip_module_prefix(state_dict), strict=False)
    pretty_print(f"Loaded weights from {ckpt_path}")

    # Datasets and loader
    builder: BaseLakeBuilder = instantiate(cfg.data)
    datasets, _ = builder.load_dataset(server_prefix=cfg.server_prefix, rank=0, world_size=1, root_cfg=cfg)
    loader: DataLoader = build_dataloader(datasets=datasets,
                                          cfg=cfg.dataloader,
                                          pad_value_id=cfg.PAD_VAL_ID,
                                          pad_value_default=cfg.PAD_VAL_DEFAULT,
                                          distributed=False,
                                          use_cl=False,
                                          plot=True)  # plot=True uses small batch size/sequential sampler

    # Collect per-sample pooled embeddings (z) and metadata
    sample_Z = []
    sample_lake_ids = []
    sample_lake_names = []
    sample_first_dates = []

    for batch in loader:
        # Move tensors to device
        seq_X = batch["flat_seq_x"].to(device)
        mask_X = batch["flat_mask_x"].to(device)
        sample_ids_x = batch["sample_ids_x"].to(device)
        time_ids_x = batch["time_ids_x"].to(device)
        var_ids_x = batch["var_ids_x"].to(device)
        padding_mask_x = batch["padding_mask_x"].to(device)
        depth_values_x = batch["depth_values_x"].to(device)
        time_values_x = batch["time_values_x"].to(device)

        # Targets are required by forward signature; pass context metadata as targets (no decode usage)
        tgt_variate_ids = batch["var_ids_y"].to(device)
        tgt_time_values = batch["time_values_y"].to(device)
        tgt_time_ids = batch["time_ids_y"].to(device)
        tgt_depth_values = batch["depth_values_y"].to(device)
        tgt_padding_mask = batch["padding_mask_y"].to(device)

        _, _, z = model(data=seq_X,
                        observed_mask=mask_X,
                        sample_ids=sample_ids_x,
                        variate_ids=var_ids_x,
                        padding_mask=padding_mask_x,
                        depth_values=depth_values_x,
                        pred_len=cfg.pred_len,
                        seq_len=cfg.seq_len,
                        time_values=time_values_x,
                        time_ids=time_ids_x,
                        tgt_variate_ids=tgt_variate_ids,
                        tgt_time_values=tgt_time_values,
                        tgt_time_ids=tgt_time_ids,
                        tgt_depth_values=tgt_depth_values,
                        tgt_padding_mask=tgt_padding_mask)
        # z: (B, d_static)
        Z_cpu = z.detach().cpu().numpy()
        sample_Z.append(Z_cpu)

        # metadata
        lake_ids = batch["lake_id"]
        lake_names = batch["lake_name"]
        dt_x = batch["datetime_strs_x"]  # (B, T) numpy datetime64 (possibly padded)
        first_dates = []
        for row in dt_x:
            if row.size == 0:
                first_dates.append("")
                continue
            # coerce to string, handle NaT
            val = None
            for d64 in row:
                if str(d64) != 'NaT':
                    val = str(d64).split('T')[0]
                    break
            first_dates.append(val or "")

        sample_lake_ids.extend([int(l) if not isinstance(l, (list, tuple)) else int(l[0]) for l in lake_ids])
        sample_lake_names.extend([str(n) if not isinstance(n, (list, tuple)) else str(n[0]) for n in lake_names])
        sample_first_dates.extend(first_dates)

    if len(sample_Z) == 0:
        pretty_print("No samples collected; exiting.")
        return

    Z = np.vstack(sample_Z)  # (N, d_static)
    lake_ids_np = np.array(sample_lake_ids, dtype=np.int64)
    lake_names_np = np.array(sample_lake_names, dtype=object)
    first_dates_np = np.array(sample_first_dates, dtype=object)

    os.makedirs(output_dir, exist_ok=True)
    np.savez_compressed(os.path.join(output_dir, "sample_embeddings.npz"),
                        Z=Z, lake_ids=lake_ids_np, lake_names=lake_names_np, first_dates=first_dates_np)
    pretty_print(f"Saved per-sample embeddings: {os.path.join(output_dir, 'sample_embeddings.npz')}")

    # Per-lake overall mean
    lake_means = {}
    for lid in np.unique(lake_ids_np):
        sel = (lake_ids_np == lid)
        lake_means[int(lid)] = Z[sel].mean(axis=0)
    # Save
    np.savez_compressed(os.path.join(output_dir, "lake_mean_embeddings.npz"),
                        **{f"lake_{k}": v for k, v in lake_means.items()})

    # Seasonal means
    def month_of(s):
        try:
            if not s:
                return None
            return dt.strptime(s, "%Y-%m-%d").month
        except Exception:
            return None

    months = np.array([month_of(s) for s in first_dates_np])
    winter_mask = np.array([m in set(winter_months) if m is not None else False for m in months])
    summer_mask = np.array([m in set(summer_months) if m is not None else False for m in months])

    winter_means = {}
    summer_means = {}
    for lid in np.unique(lake_ids_np):
        lid_sel = (lake_ids_np == lid)
        if winter_mask.any():
            sel_w = lid_sel & winter_mask
            if sel_w.any():
                winter_means[int(lid)] = Z[sel_w].mean(axis=0)
        if summer_mask.any():
            sel_s = lid_sel & summer_mask
            if sel_s.any():
                summer_means[int(lid)] = Z[sel_s].mean(axis=0)
    np.savez_compressed(os.path.join(output_dir, "lake_seasonal_embeddings.npz"),
                        winter={f"lake_{k}": v for k, v in winter_means.items()},
                        summer={f"lake_{k}": v for k, v in summer_means.items()})

    # Variable embeddings (from embedding table)
    try:
        var_weight = model.var_id_embed.weight.detach().cpu().numpy()
    except Exception:
        # In case model is wrapped or attribute name changes
        var_weight = model.module.var_id_embed.weight.detach().cpu().numpy()  # type: ignore

    # id_to_var mapping from cfg
    raw_id_to_var = OmegaConf.to_container(cfg.data.id_to_var, resolve=True)
    id_to_var = {int(k): v for k, v in raw_id_to_var.items()} if raw_id_to_var else {}
    np.savez_compressed(os.path.join(output_dir, "variable_embeddings.npz"),
                        E=var_weight, id_to_var=np.array(list(id_to_var.items()), dtype=object))

    # Optional: quick TSNE plots
    def tsne_plot(mat, labels, title, out_png):
        if mat.shape[0] < 2:
            return
        perplexity = max(2, min(30, mat.shape[0] // 3))
        X2 = TSNE(n_components=2, init="random", perplexity=perplexity, learning_rate="auto").fit_transform(mat)
        plt.figure(figsize=(7, 6))
        plt.scatter(X2[:, 0], X2[:, 1], s=10, alpha=0.7)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_png, dpi=180)
        plt.close()

    # TSNE for variable embeddings
    tsne_plot(var_weight, np.arange(var_weight.shape[0]), "Variable embeddings (TSNE)",
              os.path.join(output_dir, "tsne_variables.png"))

    # TSNE for per-lake mean embeddings
    if len(lake_means) >= 2:
        mat = np.vstack([lake_means[k] for k in sorted(lake_means)])
        tsne_plot(mat, list(sorted(lake_means)), "Lake mean embeddings (TSNE)",
                  os.path.join(output_dir, "tsne_lake_means.png"))

    pretty_print(f"Embedding extraction complete. Outputs saved under: {output_dir}")


