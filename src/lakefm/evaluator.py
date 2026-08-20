import os
import re
import wandb
import json
from omegaconf import OmegaConf
from tqdm import tqdm
import torch
import h5py
import math
import sys 
from h5py import string_dtype  
from collections import defaultdict
import torch.nn.functional as F
from datetime import datetime as dt
from lakefm.trainer import Trainer
from lakefm.trainer import _updated_mask_
import numpy as np
from lakefm.trainer import reduce_mean
import torch.distributed as dist
from data.loader import build_dataloader
from utils.exp_utils import pretty_print
from lakefm.trainer import init_wandb
from utils.util import apply_standard_scaling_per_sample_wrapper
from utils.util import Normalizer
from utils.irregular_plotter import IrregularGridPlotter, SpatioTemporalHeatmapPlotter
import pandas as pd
import matplotlib.pyplot as plt
from lakefm.extract_embeddings import extract_embeddings_from_loader

from dataclasses import dataclass
from typing import Optional, Any

class Evaluator:
    """
    Standalone evaluator with embedded validation loop that logs metrics to W&B under cfg.evaluator.
    Supports multiple trials and incremental saving of predictions to disk.
    """
    @staticmethod
    def _tplusn_update_date_horizon_acc(
        acc: dict,
        *,
        dates: list,
        time_ids: torch.Tensor,
        preds: torch.Tensor,
        gts: torch.Tensor,
        var_ids: torch.Tensor,
        target_var_id: int,
        origin_time_id: int,
        max_horizon: int = 14,
    ) -> None:
        """
        Update a per-variate accumulator that maps:
          date (YYYY-MM-DD) -> length-(max_horizon) vectors of pred sums and counts,
          plus a single GT value aggregated *by date* (independent of horizon).

        - Only tokens for `target_var_id` are used.
        - Horizon is computed as: horizon = time_id - origin_time_id (1..max_horizon).
        - If multiple tokens land in the same (date, horizon) cell (e.g., multiple depths),
          we average via sum/count.
        - GT is aggregated by date only (so GT for a date is identical across all horizons).
        """
        if max_horizon <= 0:
            return

        vmask = (var_ids == int(target_var_id))
        if not bool(vmask.any().item()):
            return

        tids = time_ids[vmask].detach().cpu().numpy().astype(int)
        pr = preds[vmask].detach().cpu().numpy().astype(float)
        gt = gts[vmask].detach().cpu().numpy().astype(float)

        vmask_np = vmask.detach().cpu().numpy().astype(bool)
        dts = [dates[i] for i, ok in enumerate(vmask_np) if ok]

        def _normalize_date_key(dt_val) -> str:
            """
            Return a YYYY-MM-DD string from a datetime-like input.
            Supports:
              - numpy datetime64
              - ISO datetime strings
              - Unix epoch seconds (int/float or numeric string)
            Falls back to the first 10 chars of `str(dt_val)` if parsing fails.
            """
            try:
                # numpy datetime64 path
                if isinstance(dt_val, np.datetime64):
                    if np.isnat(dt_val):
                        return ""
                    return str(np.datetime_as_string(dt_val, unit="D"))

                # bytes -> str
                if isinstance(dt_val, (bytes, bytearray)):
                    dt_val = dt_val.decode("utf-8")

                s = str(dt_val).strip()
                if s == "" or s.lower() == "nat":
                    return ""

                # numeric epoch seconds (common in some LakeBeD exports)
                # e.g. "1593993600" -> 2020-07-06
                if re.fullmatch(r"[-+]?\d+(\.\d+)?", s):
                    x = float(s)
                    ax = abs(x)
                    # Heuristic: seconds ~ 1e9, ms ~ 1e12, us ~ 1e15, ns ~ 1e18
                    if ax >= 1e17:
                        ts = pd.to_datetime(int(x), unit="ns", utc=True, errors="coerce")
                    elif ax >= 1e14:
                        ts = pd.to_datetime(int(x), unit="us", utc=True, errors="coerce")
                    elif ax >= 1e11:
                        ts = pd.to_datetime(int(x), unit="ms", utc=True, errors="coerce")
                    else:
                        ts = pd.to_datetime(int(x), unit="s", utc=True, errors="coerce")
                else:
                    ts = pd.to_datetime(s, utc=True, errors="coerce")

                if pd.isna(ts):
                    return s[:10]
                # normalize to date in UTC, drop tz
                ts = pd.Timestamp(ts)
                if ts.tzinfo is not None:
                    ts = ts.tz_convert("UTC").tz_localize(None)
                return ts.normalize().date().isoformat()
            except Exception:
                return str(dt_val)[:10]

        for dt_str, tid, pr_i, gt_i in zip(dts, tids, pr, gt):
            date = _normalize_date_key(dt_str)
            if not date:
                continue
            horizon = int(tid) - int(origin_time_id)
            if horizon <= 0 or horizon > int(max_horizon):
                continue
            hi = horizon - 1

            if date not in acc:
                acc[date] = {
                    "pred_sum": np.zeros((max_horizon,), dtype=float),
                    "pred_cnt": np.zeros((max_horizon,), dtype=int),
                    "gt_sum": 0.0,
                    "gt_cnt": 0,
                }

            if np.isfinite(pr_i):
                acc[date]["pred_sum"][hi] += float(pr_i)
                acc[date]["pred_cnt"][hi] += 1
            if np.isfinite(gt_i):
                acc[date]["gt_sum"] += float(gt_i)
                acc[date]["gt_cnt"] += 1

    def __init__(self, cfg, model, trainer, rank=0, epoch=0):
        self.cfg = cfg
        self.model = model
        self.rank = rank
        self.device = f"cuda:{rank}"
        model_ref = self.model.module if hasattr(self.model, "module") else self.model
        self.total_params = sum(p.numel() for p in model_ref.parameters())
        self.trainable_params = sum(p.numel() for p in model_ref.parameters() if p.requires_grad)
        self.trainer = trainer
        self.pred_len = cfg.pred_len
        self.seq_len = cfg.seq_len
        self.data = self.cfg.data
        self.do_variate_mse = True
        self.do_depthwise_metrics = True
        self.depth_bin_size_m = getattr(cfg.evaluator, "depth_bin_size_m", None)
        self.depth_round_decimals = getattr(cfg.evaluator, "depth_round_decimals", 3)
        self.save_preds = getattr(cfg.evaluator, "save_predictions", False)
        self.save_per_sample_metrics = getattr(cfg.evaluator, "save_per_sample_metrics", True)
        self.denorm_eval = getattr(cfg.evaluator, "denorm_eval", False)
        self.pad_value_id = self.cfg.PAD_VAL_ID
        self.pad_value_default = self.cfg.PAD_VAL_DEFAULT
        self.model_epoch = epoch
        self.irregular_plotter = IrregularGridPlotter(self.device, self.pad_value_default)
        self.heatmap_plotter = SpatioTemporalHeatmapPlotter(self.device, self.pad_value_default)
        self.plot = getattr(cfg.evaluator, "plot", True)

    

        # Thermocline experiment (Water Temp heatmap + inversion metrics)
        # - Can be enabled via cfg.evaluator.thermocline=true
        # - Supports optional overrides for the target variable and season window
        self.thermocline = bool(getattr(cfg.evaluator, "thermocline", False))
        if self.thermocline:
            self.thermocline_var_name_substr = str(getattr(cfg.evaluator, "thermocline_var_name_substr", "WaterTemp_C")).strip()
            self.thermocline_start_date = getattr(cfg.evaluator, "thermocline_start_date", None)
            self.thermocline_end_date = getattr(cfg.evaluator, "thermocline_end_date", None)
            self.thermocline_season_start = str(getattr(cfg.evaluator, "thermocline_season_start", "06-01")).strip()
            self.thermocline_season_end = str(getattr(cfg.evaluator, "thermocline_season_end", "09-30")).strip()
            # Plot-only MM-DD fallback (defaults to Apr 1 – Nov 30)
            self.thermocline_max_depths = int(getattr(cfg.evaluator, "thermocline_max_depths", 50))
            self.thermocline_depth_round_decimals = int(getattr(cfg.evaluator, "thermocline_depth_round_decimals", 6))
            self.thermocline_skip_irregular_plots = bool(getattr(cfg.evaluator, "thermocline_skip_irregular_plots", True))
            # Evaluation years for thermocline metrics (default: 2018, 2019 and 2020)
            self.thermocline_eval_years = getattr(cfg.evaluator, "thermocline_eval_years", [2018, 2019, 2020])

        # Beer-lambert experiment
        self.BL = bool(getattr(cfg.evaluator, "BL", False))
        if self.BL:
            self.bl_start_date = getattr(cfg.evaluator, "bl_start_date", None)
            self.bl_plot_dates = getattr(cfg.evaluator, "bl_plot_dates", None)
            self.bl_max_scatter_points = int(getattr(cfg.evaluator, "bl_max_scatter_points", 200_000))
            self.bl_lake_id = getattr(cfg.evaluator, "bl_lake_id", None)
            try:
                if self.bl_lake_id is not None:
                    self.bl_lake_id = int(self.bl_lake_id)
            except Exception:
                self.bl_lake_id = None
        
        # Regular grid forecasting options
        self.regular_grid_forecasting = getattr(cfg, "regular_grid_forecasting", False)
        self.regular_grid_depths = getattr(cfg, "regular_grid_depths", 20)
        self.regular_grid_max_depth = getattr(cfg, "regular_grid_max_depth", None)
        self.enable_heatmap_plotting = getattr(cfg, "enable_heatmap_plotting", True)
        self.plot_both_heatmap_types = getattr(cfg, "plot_both_heatmap_types", True)
        
        # Prediction error heatmap plotting option
        self.enable_prediction_error_heatmaps = getattr(cfg.evaluator, "enable_prediction_error_heatmaps", True)
        
        # Loss function configuration
        self.reg_scale_weight = getattr(cfg.evaluator, "reg_scale_weight", 1e-4)
        self.reg_df_weight = getattr(cfg.evaluator, "reg_df_weight", 1e-4)
        self.df_target = getattr(cfg.evaluator, "df_target", 5.0)
        
        self.id_to_var = self.data.id_to_var
        self.var_to_id = self.data.var_to_id
        # Depth denormalization parameters (per dataset)
        self.depth_min = None
        self.depth_max = None
        
        # Lake embedding trajectory extraction
        self.lake_embed_traj = getattr(cfg, "lake_embed_traj", False)

    @staticmethod
    def _parse_mmdd(mmdd: str, default=(6, 1)) -> tuple:
        """
        Parse "MM-DD" into (month, day).
        """
        try:
            mm, dd = mmdd.split("-")
            m = int(mm)
            d = int(dd)
            if 1 <= m <= 12 and 1 <= d <= 31:
                return (m, d)
        except Exception:
            pass
        return default

    @staticmethod
    def _resolve_var_id_by_substr(id_to_var: dict, prefer_substr: str) -> Optional[int]:
        """
        Best-effort: find a variable id whose name contains prefer_substr (case-insensitive).
        If not found, fall back to a WaterTemp-like heuristic.
        """
        if not isinstance(id_to_var, dict) or not id_to_var:
            return None
        prefer = (prefer_substr or "").lower().strip()
        candidates = []
        for k, v in id_to_var.items():
            try:
                vid = int(k)
            except Exception:
                continue
            name = str(v) if v is not None else ""
            lname = name.lower()
            if prefer and prefer in lname:
                candidates.append((0, len(name), vid, name))
            # heuristic candidates
            elif ("water" in lname and "temp" in lname) or ("watertemp" in lname):
                candidates.append((1, len(name), vid, name))
            elif lname in {"watertemp", "water_temp", "watertemp_c", "water_temp_c"}:
                candidates.append((0, len(name), vid, name))
        if not candidates:
            return None
        # Prefer exact/shorter matches
        candidates.sort(key=lambda x: (x[0], x[1], x[2]))
        return int(candidates[0][2])

    def _resolve_var_id(self, *, var_name: Optional[str], id_to_var: dict) -> Optional[int]:
        """
        Resolve a variable id from a variable name.

        Primary path (preferred): cfg.data.var_to_id exact lookup, e.g. "WaterTemp_C" -> 7.
        Fallback: substring/heuristic match over id_to_var (kept for robustness when var_to_id is absent).
        """
        name = str(var_name).strip() if var_name is not None else ""
        if name:
            try:
                v2id = OmegaConf.to_container(getattr(self.cfg.data, "var_to_id", None), resolve=True)
                if isinstance(v2id, dict) and name in v2id:
                    return int(v2id[name])
            except Exception:
                pass
        # Fallback: substring match on id_to_var
        return self._resolve_var_id_by_substr(id_to_var, name)

    @dataclass
    class _BLConfig:
        enabled: bool
        start_date: Optional[str]
        plot_dates: Optional[list]
        max_scatter_points: int
        lake_id: Optional[int]
        chla_var_id: int
        att_var_id: int
        depth_round_decimals: int

    @dataclass
    class _BLState:
        out_dir: Optional[str]
        scatter_x: list
        scatter_y: list
        scatter_depth: list
        scatter_seen: int
        r2_sum: float
        r2_n: int
        selected_sample: Optional[dict]
        profiles_by_date: dict

    # ---------------- Thermocline experiment (WaterTemp inversion + heatmap) ----------------
    @dataclass
    class _ThermoConfig:
        enabled: bool
        var_name: str
        var_id: int
        eval_years: list
        season_start_mmdd: str
        season_end_mmdd: str
        max_depths: int
        depth_round_decimals: int

    @dataclass
    class _ThermoState:
        out_dir: Optional[str]
        # profiles_metric[(lake_id, day_str)][depth_key] = {"dt": Timestamp, "depth": float, "gt": float, "pred": float}
        profiles_metric: dict
        # profiles_plot[(lake_id, day_str)][depth_key] = {"dt": Timestamp, "depth": float, "gt": float, "pred": float}
        profiles_plot: dict
        lake_id_to_name: dict

    def _thermo_make_config(self) -> Optional["_ThermoConfig"]:
        if not self.thermocline or self.rank != 0:
            return None
        # Resolve var id from cfg.data.var_to_id (preferred) with fallback
        var_id = self._resolve_var_id(var_name=self.thermocline_var_name_substr, id_to_var=self.data.id_to_var)
        if var_id is None:
            if self.rank == 0:
                print(f"[thermocline] Could not resolve var id for {self.thermocline_var_name_substr!r}")
            return None
        var_name = str(self.data.id_to_var.get(int(var_id), self.thermocline_var_name_substr))

        # Season window for metrics as MM-DD, applied within specific years.
        # If user provided YYYY-MM-DD start/end, we take their MM-DD portion and apply to all eval years.
        season_start_mmdd = str(self.thermocline_season_start)
        season_end_mmdd = str(self.thermocline_season_end)
        if self.thermocline_start_date is not None and self.thermocline_end_date is not None:
            try:
                season_start_mmdd = pd.to_datetime(self.thermocline_start_date).strftime("%m-%d")
                season_end_mmdd = pd.to_datetime(self.thermocline_end_date).strftime("%m-%d")
            except Exception:
                season_start_mmdd = str(self.thermocline_season_start)
                season_end_mmdd = str(self.thermocline_season_end)

        eval_years = getattr(self, "thermocline_eval_years", [2018, 2019, 2020])
        try:
            if isinstance(eval_years, (tuple, list, set)):
                eval_years = [int(x) for x in list(eval_years)]
            else:
                eval_years = [int(eval_years)]
        except Exception:
            eval_years = [2018, 2019, 2020]

        return Evaluator._ThermoConfig(
            enabled=True,
            var_name=str(var_name),
            var_id=int(var_id),
            eval_years=eval_years,
            season_start_mmdd=str(season_start_mmdd),
            season_end_mmdd=str(season_end_mmdd),
            max_depths=int(self.thermocline_max_depths),
            depth_round_decimals=int(self.thermocline_depth_round_decimals),
        )

    def _thermo_init_state(self, save_dir: str, cfg: "_ThermoConfig") -> "_ThermoState":
        out_dir = None
        if cfg.enabled:
            out_dir = os.path.join(save_dir, "THERMOCLINE")
            os.makedirs(out_dir, exist_ok=True)
        return Evaluator._ThermoState(out_dir=out_dir, profiles_metric={}, profiles_plot={}, lake_id_to_name={})

    # NOTE: plotting uses ALL dates; metrics are filtered by (year, MM-DD season) in _thermo_finalize.

    def _thermo_update(
        self,
        cfg: "_ThermoConfig",
        state: "_ThermoState",
        *,
        lake_id: int,
        lake_name: Any,
        dt_row: np.ndarray,
        depth_m: np.ndarray,
        var_ids: np.ndarray,
        gt: np.ndarray,
        pred: np.ndarray,
    ) -> None:
        if not cfg.enabled or state.out_dir is None:
            return
        lid = int(lake_id)
        if lake_name is not None:
            if isinstance(lake_name, bytes):
                lake_name = lake_name.decode("utf-8")
            state.lake_id_to_name[lid] = str(lake_name)

        # Filter to target variable id
        vids = np.asarray(var_ids).astype(int)
        sel_var = (vids == int(cfg.var_id))
        if not np.any(sel_var):
            return

        dts = pd.to_datetime(np.asarray(dt_row)[sel_var], errors="coerce")
        depths = np.asarray(depth_m, dtype=float)[sel_var]
        gtv = np.asarray(gt, dtype=float)[sel_var]
        prv = np.asarray(pred, dtype=float)[sel_var]

        # Flatten overlapping windows by keeping earliest timestamp per (lake, day, depth_key)
        for ts, dep, gv, pv in zip(dts, depths, gtv, prv):
            if pd.isna(ts):
                continue
            ts = pd.Timestamp(ts)
            if not (math.isfinite(dep) and math.isfinite(gv) and math.isfinite(pv)):
                continue
            day_key = ts.normalize().strftime("%Y-%m-%d")
            depth_key = round(float(dep), int(cfg.depth_round_decimals))

            # Store ALL dates (plot uses all; metrics filter by (year, season) later)
            key = (lid, day_key)
            if key not in state.profiles_metric:
                state.profiles_metric[key] = {}
            if key not in state.profiles_plot:
                state.profiles_plot[key] = {}

            existing_m = state.profiles_metric[key].get(depth_key)
            if existing_m is None or ts < existing_m["dt"]:
                state.profiles_metric[key][depth_key] = {"dt": ts, "depth": float(depth_key), "gt": float(gv), "pred": float(pv)}

            existing_p = state.profiles_plot[key].get(depth_key)
            if existing_p is None or ts < existing_p["dt"]:
                state.profiles_plot[key][depth_key] = {"dt": ts, "depth": float(depth_key), "gt": float(gv), "pred": float(pv)}

    def _thermo_finalize(self, cfg: "_ThermoConfig", state: "_ThermoState") -> None:
        if not cfg.enabled or state.out_dir is None:
            return

        def profile_inversions(values_by_depth: list) -> tuple:
            """
            Returns (inversions, total_pairs, inversion_depths)
            where inversion_depths is a list of the deeper depth values where inversions occur
            """
            inv = 0
            tot = 0
            inversion_depths = []
            # DEBUG: Log the input
            # if len(values_by_depth) > 0:
            #     print(f"[DEBUG profile_inversions] Called with {len(values_by_depth)} depths")
            #     print(f"[DEBUG profile_inversions] Depth values: {[d for d, v in values_by_depth]}")
            #     print(f"[DEBUG profile_inversions] Temperature values: {[v for d, v in values_by_depth]}")
            for i in range(len(values_by_depth) - 1):
                v0 = values_by_depth[i][1]
                v1 = values_by_depth[i + 1][1]
                if not (math.isfinite(v0) and math.isfinite(v1)):
                    continue
                tot += 1
                if v1 > v0:
                    inv += 1
                    # Record the deeper depth (i+1) where the inversion occurs
                    inversion_depths.append(float(values_by_depth[i + 1][0]))
            # DEBUG: Log the result
            # print(f"[DEBUG profile_inversions] Counted {tot} pairs, {inv} inversions")
            # if inv > 0:
            #     print(f"[DEBUG profile_inversions] Inversion depths: {inversion_depths}")
            # if tot != len(values_by_depth) - 1:
            #     print(f"[WARNING] Expected {len(values_by_depth) - 1} pairs but got {tot} pairs!")
            return inv, tot, inversion_depths

        def profile_from_depthmap(depthmap: dict, field: str) -> list:
            pts = [(rec["depth"], rec[field]) for rec in depthmap.values() if field in rec]
            pts.sort(key=lambda x: x[0])
            if len(pts) > int(cfg.max_depths):
                pts = pts[: int(cfg.max_depths)]
            return pts

        # Helper: compute metrics for a filtered subset of profiles_metric
        def compute_metrics(profile_items):
            per_lake = {}
            overall_daily = {}
            overall_agg = {
                "gt": {"inversions": 0, "total_pairs": 0, "n_days": 0, "mean_daily_rate": None, "weighted_rate": None},
                "pred": {"inversions": 0, "total_pairs": 0, "n_days": 0, "mean_daily_rate": None, "weighted_rate": None},
            }

            def _acc(obj: dict, inv: int, tot: int):
                obj["inversions"] += int(inv)
                obj["total_pairs"] += int(tot)
                obj.setdefault("_rates", [])
                if tot > 0:
                    obj["_rates"].append(float(inv) / float(tot))
                obj["n_days"] += 1

            for (lid, day), depthmap in profile_items:
                pts_gt = profile_from_depthmap(depthmap, "gt")
                pts_pr = profile_from_depthmap(depthmap, "pred")
                if len(pts_gt) < 2 or len(pts_pr) < 2:
                    continue
                
               
                inv_g, tot_g, inv_depths_g = profile_inversions(pts_gt)
                # print(f"[DEBUG] Calling profile_inversions for Pred:")
                inv_p, tot_p, inv_depths_p = profile_inversions(pts_pr)
                rate_g = (float(inv_g) / float(tot_g)) if tot_g > 0 else None
                rate_p = (float(inv_p) / float(tot_p)) if tot_p > 0 else None

                if lid not in per_lake:
                    per_lake[lid] = {"by_day": {}, "overall": {"gt": {"inversions": 0, "total_pairs": 0, "n_days": 0},
                                                              "pred": {"inversions": 0, "total_pairs": 0, "n_days": 0}}}
                per_lake[lid]["by_day"][day] = {
                    "gt": {"inversions": int(inv_g), "total_pairs": int(tot_g), "inversion_rate": rate_g, "inversion_depths": inv_depths_g},
                    "pred": {"inversions": int(inv_p), "total_pairs": int(tot_p), "inversion_rate": rate_p, "inversion_depths": inv_depths_p},
                }
                _acc(per_lake[lid]["overall"]["gt"], inv_g, tot_g)
                _acc(per_lake[lid]["overall"]["pred"], inv_p, tot_p)

                if day not in overall_daily:
                    overall_daily[day] = {"gt": {"inversions": 0, "total_pairs": 0}, "pred": {"inversions": 0, "total_pairs": 0}}
                overall_daily[day]["gt"]["inversions"] += int(inv_g)
                overall_daily[day]["gt"]["total_pairs"] += int(tot_g)
                overall_daily[day]["pred"]["inversions"] += int(inv_p)
                overall_daily[day]["pred"]["total_pairs"] += int(tot_p)

                _acc(overall_agg["gt"], inv_g, tot_g)
                _acc(overall_agg["pred"], inv_p, tot_p)

            def _finalize(obj: dict) -> dict:
                rates = obj.pop("_rates", [])
                inv = int(obj.get("inversions", 0))
                tot = int(obj.get("total_pairs", 0))
                obj["weighted_rate"] = (float(inv) / float(tot)) if tot > 0 else None
                obj["mean_daily_rate"] = (float(np.mean(rates)) if len(rates) > 0 else None)
                return obj

            overall_agg["gt"] = _finalize(overall_agg["gt"])
            overall_agg["pred"] = _finalize(overall_agg["pred"])
            for lid, payload in per_lake.items():
                payload["overall"]["gt"] = _finalize(payload["overall"]["gt"])
                payload["overall"]["pred"] = _finalize(payload["overall"]["pred"])

            overall_daily_out = {}
            for day, payload in sorted(overall_daily.items(), key=lambda kv: kv[0]):
                g_inv = int(payload["gt"]["inversions"])
                g_tot = int(payload["gt"]["total_pairs"])
                p_inv = int(payload["pred"]["inversions"])
                p_tot = int(payload["pred"]["total_pairs"])
                overall_daily_out[day] = {
                    "gt": {"inversions": g_inv, "total_pairs": g_tot, "inversion_rate": (g_inv / g_tot) if g_tot > 0 else None},
                    "pred": {"inversions": p_inv, "total_pairs": p_tot, "inversion_rate": (p_inv / p_tot) if p_tot > 0 else None},
                }

            per_lake_out = {str(int(lid)): payload for lid, payload in sorted(per_lake.items(), key=lambda kv: kv[0])}
            daily_counts = {
                "gt": {day: int(v["gt"]["inversions"]) for day, v in overall_daily_out.items()},
                "pred": {day: int(v["pred"]["inversions"]) for day, v in overall_daily_out.items()},
            }
            daily_rates = {
                "gt": {day: v["gt"]["inversion_rate"] for day, v in overall_daily_out.items()},
                "pred": {day: v["pred"]["inversion_rate"] for day, v in overall_daily_out.items()},
            }
            return {
                "overall": overall_agg,
                "daily": overall_daily_out,
                "daily_counts": daily_counts,
                "daily_rates": daily_rates,
                "by_lake": per_lake_out,
            }

        # Compute metrics per year over the season MM-DD window
        start_md = self._parse_mmdd(cfg.season_start_mmdd, default=(6, 1))
        end_md = self._parse_mmdd(cfg.season_end_mmdd, default=(9, 30))

        def in_season_mmdd(ts: pd.Timestamp) -> bool:
            try:
                md = (int(ts.month), int(ts.day))
                return bool(start_md <= md <= end_md)
            except Exception:
                return False

        metrics_by_year = {}
        for year in cfg.eval_years:
            filtered = []
            for (lid, day), depthmap in state.profiles_metric.items():
                try:
                    ts = pd.to_datetime(day, errors="coerce")
                    if pd.isna(ts):
                        continue
                    ts = pd.Timestamp(ts)
                except Exception:
                    continue
                if int(ts.year) != int(year):
                    continue
                if not in_season_mmdd(ts):
                    continue
                filtered.append(((lid, day), depthmap))
            filtered.sort(key=lambda kv: (kv[0][0], kv[0][1]))
            metrics_by_year[str(int(year))] = compute_metrics(filtered)

        payload = {
            "metric": "thermocline_inversion_rate",
            "definition": "Plots use the full available timeline. Metrics are computed per-year over the same MM-DD season window: inversions are adjacent depth pairs where T(deeper) > T(shallower).",
            "config": {
                "var_id": int(cfg.var_id),
                "var_name": str(cfg.var_name),
                "season_start_mmdd": str(cfg.season_start_mmdd),
                "season_end_mmdd": str(cfg.season_end_mmdd),
                "eval_years": [int(y) for y in cfg.eval_years],
                "depth_round_decimals": int(cfg.depth_round_decimals),
                "max_depths_per_day": int(cfg.max_depths),
            },
            "metrics_by_year": metrics_by_year,
            "lake_id_to_name": {str(int(k)): str(v) for k, v in state.lake_id_to_name.items()},
        }

        out_path = os.path.join(state.out_dir, "thermocline_metrics.json")
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)

        # Plot: WaterTemp heatmap (GT vs Pred) over the FULL available timeline, per lake_id
        try:
            lake_ids = sorted({int(lid) for (lid, _day) in state.profiles_plot.keys()})
            for lid in lake_ids:
                # All days for this lake
                days = sorted({day for (lid2, day) in state.profiles_plot.keys() if int(lid2) == int(lid)})
                if not days:
                    continue

                # Build a continuous daily axis for plotting (matches "days from <start>")
                try:
                    day_ts = pd.to_datetime(days, errors="coerce")
                    day_ts = pd.DatetimeIndex([d for d in day_ts if not pd.isna(d)])
                except Exception:
                    day_ts = pd.DatetimeIndex([])
                if len(day_ts) == 0:
                    continue

                plot_start = day_ts.min().normalize()
                plot_end = day_ts.max().normalize()
                try:
                    full_days = pd.date_range(plot_start, plot_end, freq="D")
                except Exception:
                    full_days = day_ts.sort_values().unique()
                # Guard against accidentally huge ranges (e.g., MM-DD fallback across many years)
                if len(full_days) > 2000:
                    full_days = day_ts.sort_values().unique()
                full_day_strs = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in full_days]

                # Union depths across all days (from stored profiles)
                depth_union = set()
                for day in full_day_strs:
                    dm = state.profiles_plot.get((lid, day), {})
                    for dk, rec in dm.items():
                        depth_union.add(float(rec["depth"]))
                depths = sorted(depth_union)[: max(2, int(cfg.max_depths))]
                if len(depths) < 2:
                    continue

                Dn = len(depths)
                Tn = len(full_day_strs)
                gt_mat = np.full((Dn, Tn), np.nan)
                pr_mat = np.full((Dn, Tn), np.nan)
                depth_to_i = {d: i for i, d in enumerate(depths)}
                day_to_j = {d: j for j, d in enumerate(full_day_strs)}

                for day in full_day_strs:
                    dm = state.profiles_plot.get((lid, day), {})
                    j = day_to_j[day]
                    for rec in dm.values():
                        d = float(rec["depth"])
                        if d in depth_to_i:
                            i = depth_to_i[d]
                            gt_mat[i, j] = float(rec["gt"])
                            pr_mat[i, j] = float(rec["pred"])

                # Interpolate to get a smooth seasonal heatmap (like the reference)
                try:
                    gt_df = pd.DataFrame(gt_mat, index=depths, columns=full_day_strs, dtype=float)
                    pr_df = pd.DataFrame(pr_mat, index=depths, columns=full_day_strs, dtype=float)
                    # Interpolate along time (axis=1) then depth (axis=0)
                    gt_df = gt_df.interpolate(axis=1, limit_direction="both").interpolate(axis=0, limit_direction="both")
                    pr_df = pr_df.interpolate(axis=1, limit_direction="both").interpolate(axis=0, limit_direction="both")
                    gt_mat_plot = gt_df.values
                    pr_mat_plot = pr_df.values
                except Exception:
                    gt_mat_plot = gt_mat
                    pr_mat_plot = pr_mat

                gt_masked = np.ma.masked_invalid(gt_mat_plot)
                pr_masked = np.ma.masked_invalid(pr_mat_plot)
                if np.all(gt_masked.mask) and np.all(pr_masked.mask):
                    continue
                vmin = float(min(np.nanmin(gt_masked), np.nanmin(pr_masked)))
                vmax = float(max(np.nanmax(gt_masked), np.nanmax(pr_masked)))

                # X axis in "days from start" (reference style)
                x_vals = np.arange(Tn, dtype=int)
                try:
                    x_vals = (pd.to_datetime(full_day_strs) - pd.Timestamp(plot_start)).days.values.astype(int)
                except Exception:
                    pass

                fig, axes = plt.subplots(2, 1, figsize=(18, 10), squeeze=False)
                ax_gt = axes[0, 0]
                ax_pr = axes[1, 0]

                cmap = plt.cm.get_cmap("coolwarm").copy()
                try:
                    cmap.set_bad(color="#ffcccc")
                except Exception:
                    pass

                im1 = ax_gt.imshow(
                    gt_masked,
                    aspect="auto",
                    origin="upper",  # depth increases downward
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    interpolation="bilinear",
                )
                ax_gt.set_title(f"GT - {cfg.var_name}")
                ax_gt.set_ylabel("Depth (m)")
                ax_gt.set_xlabel("Date")
                tick_idx = np.linspace(0, Tn - 1, num=min(11, Tn), dtype=int)
                ax_gt.set_xticks(tick_idx)
                ax_gt.set_xticklabels([full_day_strs[i] for i in tick_idx], rotation=90)
                ax_gt.set_yticks(np.arange(Dn))
                ax_gt.set_yticklabels([f"{d:.2f}" for d in depths])

                im2 = ax_pr.imshow(
                    pr_masked,
                    aspect="auto",
                    origin="upper",
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    interpolation="bilinear",
                )
                ax_pr.set_title(f"Pred - {cfg.var_name}")
                ax_pr.set_ylabel("Depth (m)")
                ax_pr.set_xlabel("Date")
                ax_pr.set_xticks(tick_idx)
                ax_pr.set_xticklabels([full_day_strs[i] for i in tick_idx], rotation=90)
                ax_pr.set_yticks(np.arange(Dn))
                ax_pr.set_yticklabels([f"{d:.2f}" for d in depths])

                # One shared colorbar (reference style), placed in a dedicated axis to avoid overlap.
                fig.subplots_adjust(right=0.90)
                cax = fig.add_axes([0.92, 0.15, 0.02, 0.70])  # [left, bottom, width, height]
                cbar = fig.colorbar(im2, cax=cax)
                cbar.set_label("Water Temperature (°C)")

                # Leave room for colorbar axis on the right.
                fig.tight_layout(rect=[0, 0, 0.90, 1])
                lake_name = state.lake_id_to_name.get(lid, None)
                suffix = str(lake_name) if lake_name else f"lake_{lid}"
                save_path = os.path.join(state.out_dir, f"thermocline_watertemp_heatmap_{suffix}.png")
                fig.savefig(save_path, dpi=200, bbox_inches="tight")
                plt.close(fig)
        except Exception as e:
            if self.rank == 0:
                print(f"[thermocline] Warning: failed to write heatmap plot(s): {e}")

    def _bl_make_config(self) -> "_BLConfig":
        # resolve var ids from config (fallback to base mapping)
        var_to_id = None
        try:
            var_to_id = OmegaConf.to_container(self.cfg.data.var_to_id, resolve=True)
        except Exception:
            var_to_id = None
        chla_var_id = int(var_to_id.get("Chla_ugL", 5)) if isinstance(var_to_id, dict) else 5
        att_var_id = int(var_to_id.get("LightAttenuation_Kd", 4)) if isinstance(var_to_id, dict) else 4
        # Parse plot dates
        plot_dates = None
        try:
            pdv = self.bl_plot_dates
            if pdv is None:
                plot_dates = None
            elif isinstance(pdv, (list, tuple)):
                plot_dates = [str(x) for x in pdv]
            else:
                s = str(pdv)
                if s.startswith("[") and s.endswith("]"):
                    # hydra stringified list
                    plot_dates = [x.strip().strip('"').strip("'") for x in s[1:-1].split(",") if x.strip()]
                else:
                    plot_dates = [x.strip() for x in s.split(",") if x.strip()]
        except Exception:
            plot_dates = None
        return Evaluator._BLConfig(
            enabled=bool(self.BL and self.rank == 0),
            start_date=str(self.bl_start_date) if self.bl_start_date is not None else None,
            plot_dates=plot_dates,
            max_scatter_points=int(self.bl_max_scatter_points),
            lake_id=int(self.bl_lake_id) if self.bl_lake_id is not None else None,
            chla_var_id=int(chla_var_id),
            att_var_id=int(att_var_id),
            depth_round_decimals=int(self.depth_round_decimals),
        )

    def _bl_init_state(self, save_dir: str, bl_cfg: "_BLConfig") -> "_BLState":
        out_dir = None
        if bl_cfg.enabled:
            out_dir = os.path.join(save_dir, "BL")
            os.makedirs(out_dir, exist_ok=True)
        return Evaluator._BLState(
            out_dir=out_dir,
            scatter_x=[],
            scatter_y=[],
            scatter_depth=[],
            scatter_seen=0,
            r2_sum=0.0,
            r2_n=0,
            selected_sample=None,
            profiles_by_date={},
        )

    def _bl_update(
        self,
        bl_cfg: "_BLConfig",
        bl_state: "_BLState",
        *,
        lake_id: int,
        lake_name: Any,
        dt_row: Optional[np.ndarray],
        depth_m: np.ndarray,
        var_ids: np.ndarray,
        pred: np.ndarray,
    ) -> None:
        if not bl_cfg.enabled or dt_row is None:
            return
        if bl_cfg.lake_id is not None and lake_id != bl_cfg.lake_id:
            return

        dt64 = np.array(dt_row, dtype="datetime64[ns]")
        is_nat = np.isnat(dt64)
        if np.all(is_nat):
            return

        sample_start = str(dt64[~is_nat].min())[:10]

        # capture one sample for Plot 1
        if bl_cfg.start_date is not None and bl_state.selected_sample is None:
            if bl_cfg.start_date == sample_start:
                bl_state.selected_sample = {
                    "lake_id": int(lake_id),
                    "lake_name": lake_name,
                    "start_date": sample_start,
                    "dt64": dt64.copy(),
                    "depth_m": depth_m.copy(),
                    "var_ids": var_ids.copy(),
                    "pred": pred.copy(),
                }

        # group by day and compute r^2 across depths; collect scatter pairs
        day_strs = np.array([str(x)[:10] if not np.isnat(x) else "" for x in dt64])
        p = float(10 ** int(bl_cfg.depth_round_decimals))

        for day in np.unique(day_strs):
            if not day:
                continue
            idx_day = np.where(day_strs == day)[0]
            if idx_day.size == 0:
                continue

            vids = var_ids[idx_day]
            depths = depth_m[idx_day]
            preds = pred[idx_day]
            depths_r = np.round(depths * p) / p

            chla_map = {float(d): float(v) for d, v, vid in zip(depths_r, preds, vids) if int(vid) == bl_cfg.chla_var_id}
            att_map = {float(d): float(v) for d, v, vid in zip(depths_r, preds, vids) if int(vid) == bl_cfg.att_var_id}
            common_depths = sorted(set(chla_map.keys()) & set(att_map.keys()))

            # If user provided explicit plot dates, collect a profile for those dates across the entire eval stream.
            if bl_cfg.plot_dates is not None and day in set(bl_cfg.plot_dates):
                if day not in bl_state.profiles_by_date and len(common_depths) > 0:
                    bl_state.profiles_by_date[day] = {
                        "depths": common_depths,
                        "chla": [chla_map[d] for d in common_depths],
                        "att": [att_map[d] for d in common_depths],
                        "lake_id": int(lake_id),
                        "lake_name": lake_name,
                    }

            if len(common_depths) >= 2:
                x = np.array([chla_map[d] for d in common_depths], dtype=float)
                y = np.array([att_map[d] for d in common_depths], dtype=float)
                if np.std(x) > 0 and np.std(y) > 0:
                    r = float(np.corrcoef(x, y)[0, 1])
                    if math.isfinite(r):
                        bl_state.r2_sum += float(r * r)
                        bl_state.r2_n += 1

            for d in common_depths:
                bl_state.scatter_seen += 1
                xpt = chla_map[d]
                ypt = att_map[d]
                dpt = float(d)
                if len(bl_state.scatter_x) < bl_cfg.max_scatter_points:
                    bl_state.scatter_x.append(xpt)
                    bl_state.scatter_y.append(ypt)
                    bl_state.scatter_depth.append(dpt)
                else:
                    j = np.random.randint(0, bl_state.scatter_seen)
                    if j < bl_cfg.max_scatter_points:
                        bl_state.scatter_x[j] = xpt
                        bl_state.scatter_y[j] = ypt
                        bl_state.scatter_depth[j] = dpt

    def _bl_finalize(self, bl_cfg: "_BLConfig", bl_state: "_BLState", ret: dict) -> None:
        if not bl_cfg.enabled or bl_state.out_dir is None:
            return
        out_dir = bl_state.out_dir

        # Plot 2: Pred(Chla) vs Pred(LightAttenuation) (this was the original diagnostic scatter).
        try:
            if len(bl_state.scatter_x) > 0:
                fig = plt.figure(figsize=(7, 6))
                ax = fig.add_subplot(111)
                chla = np.array(bl_state.scatter_x, dtype=float)
                att = np.array(bl_state.scatter_y, dtype=float)
                if len(bl_state.scatter_depth) == len(bl_state.scatter_x):
                    depths = np.array(bl_state.scatter_depth, dtype=float)
                    sc = ax.scatter(chla, att, c=depths, cmap="viridis", s=6, alpha=0.35, edgecolors="none")
                    cb = fig.colorbar(sc, ax=ax)
                    cb.set_label("Depth (m)")
                else:
                    ax.scatter(chla, att, s=6, alpha=0.35, edgecolors="none")
                ax.set_xlabel("Predicted Chla_ugL")
                ax.set_ylabel("Predicted LightAttenuation_Kd")
                ax.set_title("Predicted Chla vs LightAttenuation (all depths/dates)")
                ax.grid(True, alpha=0.25)
                out_scatter = os.path.join(out_dir, "pred_chla_vs_pred_lightattenuation_scatter.png")
                fig.tight_layout()
                fig.savefig(out_scatter, dpi=200)
                plt.close(fig)
        except Exception as e:
            print(f"[BL] Warning: failed to write scatter plot: {e}")

        # Plot 1: single figure with multiple requested dates (solid=Chla, dashed=Attenuation)
        try:
            if bl_cfg.plot_dates is not None:
                fig = plt.figure(figsize=(8.5, 7.5))
                ax_chla = fig.add_subplot(111)
                ax_att = ax_chla.twiny()

                cmap = plt.get_cmap("tab10")
                found_any = False
                for i, day in enumerate(bl_cfg.plot_dates):
                    prof = bl_state.profiles_by_date.get(day)
                    if not prof:
                        continue
                    found_any = True
                    color = cmap(i % 10)
                    depths = prof["depths"]
                    chla = prof["chla"]
                    att = prof["att"]
                    ax_chla.plot(chla, depths, linewidth=1.6, linestyle="-", color=color, label=f"{day} (Chla)")
                    ax_att.plot(att, depths, linewidth=1.6, linestyle="--", color=color, label=f"{day} (Light)")

                if not found_any:
                    raise RuntimeError("No plot dates found in evaluation stream.")

                ax_chla.set_xlabel("Predicted Chla_ugL")
                ax_att.set_xlabel("Predicted LightAttenuation_Kd")
                ax_chla.set_ylabel("Depth (m)")
                ax_chla.invert_yaxis()
                ax_chla.grid(True, alpha=0.50)
                fig.suptitle("Light intensity vs Chlorophyll-a depth profiles")

                h1, l1 = ax_chla.get_legend_handles_labels()
                h2, l2 = ax_att.get_legend_handles_labels()
                ax_chla.legend(h1 + h2, l1 + l2, fontsize=8, ncol=2, loc="best")

                out_profile = os.path.join(out_dir, "depth_profiles_chla_vs_attenuation_multi_date.png")
                fig.tight_layout()
                fig.savefig(out_profile, dpi=200)
                plt.close(fig)

            elif bl_cfg.start_date is not None and bl_state.selected_sample is not None:
                ss = bl_state.selected_sample
                lake_id_b = int(ss["lake_id"])
                lake_name_b = ss.get("lake_name", None)
                if isinstance(lake_name_b, bytes):
                    lake_name_b = lake_name_b.decode("utf-8")
                lake_name_b = str(lake_name_b) if lake_name_b is not None else str(lake_id_b)

                dt64 = np.array(ss["dt64"], dtype="datetime64[ns]")
                depths_m = np.array(ss["depth_m"], dtype=float)
                vids = np.array(ss["var_ids"], dtype=int)
                preds = np.array(ss["pred"], dtype=float)

                p = float(10 ** int(bl_cfg.depth_round_decimals))
                depths_r = np.round(depths_m * p) / p
                day_strs = np.array([str(x)[:10] if not np.isnat(x) else "" for x in dt64])

                days_sorted = sorted([d for d in np.unique(day_strs) if d])
                if len(days_sorted) == 0:
                    raise RuntimeError("No plot dates found in selected sample.")

                fig = plt.figure(figsize=(8.5, 7.5))
                ax_chla = fig.add_subplot(111)
                ax_att = ax_chla.twiny()

                cmap = plt.get_cmap("tab10")

                for i, day in enumerate(days_sorted):
                    idx_day = np.where(day_strs == day)[0]
                    if idx_day.size == 0:
                        continue
                    vids_d = vids[idx_day]
                    depths_d = depths_r[idx_day]
                    preds_d = preds[idx_day]

                    chla_pts = [(d, v) for d, v, vid in zip(depths_d, preds_d, vids_d) if int(vid) == bl_cfg.chla_var_id]
                    att_pts = [(d, v) for d, v, vid in zip(depths_d, preds_d, vids_d) if int(vid) == bl_cfg.att_var_id]
                    if len(chla_pts) == 0 and len(att_pts) == 0:
                        continue

                    chla_pts.sort(key=lambda x: x[0])
                    att_pts.sort(key=lambda x: x[0])

                    color = cmap(i % 10)
                    if len(chla_pts) > 0:
                        d_chla = [x[0] for x in chla_pts]
                        v_chla = [x[1] for x in chla_pts]
                        ax_chla.plot(v_chla, d_chla, linewidth=1.6, linestyle="-", color=color, label=f"{day} (Chla)")

                    if len(att_pts) > 0:
                        d_att = [x[0] for x in att_pts]
                        v_att = [x[1] for x in att_pts]
                        ax_att.plot(v_att, d_att, linewidth=1.6, linestyle="--", color=color, label=f"{day} (Light)")

                ax_chla.set_xlabel("Predicted Chla_ugL")
                ax_att.set_xlabel("Predicted LightAttenuation_Kd")
                ax_chla.set_ylabel("Depth (m)")
                ax_chla.invert_yaxis()
                ax_chla.grid(True, alpha=0.50)
                fig.suptitle(f"Light intensity vs Chlorophyll-a depth profiles")

                # Combine legends from both axes
                h1, l1 = ax_chla.get_legend_handles_labels()
                h2, l2 = ax_att.get_legend_handles_labels()
                ax_chla.legend(h1 + h2, l1 + l2, fontsize=8, ncol=2, loc="best")

                out_profile = os.path.join(out_dir, "depth_profiles_chla_vs_attenuation_multi_date.png")
                fig.tight_layout()
                fig.savefig(out_profile, dpi=200)
                plt.close(fig)
        except Exception as e:
            print(f"[BL] Warning: failed to write depth-profile plots: {e}")

        # Metrics JSON
        try:
            bl_r2_mean = float(bl_state.r2_sum / bl_state.r2_n) if bl_state.r2_n > 0 else None
            metrics_path = os.path.join(out_dir, "bl_metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(
                    {
                        "metric": "r2_depthwise_correlation",
                        "definition": "Mean of per-day Pearson r^2 between predicted Chla_ugL and predicted LightAttenuation_Kd across matched depths.",
                        "chla_var_id": int(bl_cfg.chla_var_id),
                        "attenuation_var_id": int(bl_cfg.att_var_id),
                        "mean_r2": bl_r2_mean,
                        "n_days_used": int(bl_state.r2_n),
                        "bl_start_date": bl_cfg.start_date,
                        "bl_lake_id": bl_cfg.lake_id,
                        "scatter_points_used": int(len(bl_state.scatter_x)),
                        "scatter_points_seen": int(bl_state.scatter_seen),
                        "bl_plot_dates": bl_cfg.plot_dates,
                    },
                    f,
                    indent=2,
                )
            ret["bl_metrics"] = {"mean_r2": bl_r2_mean, "n_days_used": int(bl_state.r2_n)}
        except Exception as e:
            print(f"[BL] Warning: failed to write bl_metrics.json: {e}")

    def compute_nll_loss(self, forecasts, targets, mask=None):
        """
        Compute Negative Log Likelihood loss for Student-t distribution.
        
        Args:
            forecasts: Dict with 'distribution' key containing StudentT distribution
            targets: Ground truth values (B, S_t)
            mask: Optional mask for valid predictions (B, S_t)
            
        Returns:
            NLL loss
        """
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

    def forecasting_loss(self, seq_Y, pred, mask_out, epoch=0):
        """
        Compute forecasting loss (prediction loss) without contrastive learning.
        This is a simplified version of the combined_loss from trainer.py.
        
        Args:
            seq_Y: Ground truth targets (B, S_t)
            pred: Model predictions (dict with 'distribution' or tensor)
            mask_out: Mask for valid predictions (B, S_t)
            epoch: Current epoch (for compatibility)
            
        Returns:
            total_loss: Combined forecasting + regularization loss
            forecasting_loss: Pure forecasting loss
            reg_loss: Regularization loss
        """
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
            if self.reg_scale_weight > 0:
                # Ensure scale values are positive before taking log
                valid_scale_safe = torch.clamp(valid_scale, min=1e-6)
                reg_scale = self.reg_scale_weight * torch.mean(torch.log(valid_scale_safe) ** 2)
                reg_loss += reg_scale
            
            # Degrees of freedom regularization: L2 on (df - df_target)
            if self.reg_df_weight > 0:
                # Regularize (df - df_target)
                reg_df = self.reg_df_weight * torch.mean((valid_df - self.df_target) ** 2)
                reg_loss += reg_df
        
        # Add regularization to forecasting loss
        total_loss = forecasting_loss + reg_loss
        
        return total_loss, forecasting_loss, reg_loss

    def test_once(
        self,
        dataloader,
        trial_idx=None,
        save_dir=None,
        scaling=None,
        lake_id_to_scalers=None,
        lake_id_to_depth_minmax=None,
    ):
        """
        Runs one pass of evaluation, streaming predictions and labels into a single HDF5 file to avoid large memory usage.

        Returns:
            - global masked MSE ('loss')
            - mse_by_lake:    { lake_id -> masked MSE }
            - mse_by_variate: { variate_id -> masked MSE }
        """
        num_plot_batches=self.cfg.num_plot_batches
        os.makedirs(save_dir, exist_ok=True)

        def _to_int_lake_id(x):
            """
            Robustly coerce a lake id to int.
            Handles torch/numpy scalars, strings like "1", and bytes.
            Falls back to returning `x` unchanged if conversion fails.
            """
            try:
                if isinstance(x, (torch.Tensor, np.ndarray)) and np.ndim(x) == 0:
                    x = x.item()
            except Exception:
                pass
            try:
                if isinstance(x, (bytes, bytearray)):
                    x = x.decode("utf-8")
            except Exception:
                pass
            try:
                return int(x)
            except Exception:
                return x
        
        # Helper function to get scalers for a lake_id
        def get_scalers_for_lake(lake_id):
            if lake_id_to_scalers is None:
                return None
            lake_id_int = _to_int_lake_id(lake_id)
            # Prefer stable int key lookup, but keep backward compatibility if dict was keyed by str.
            scalers = lake_id_to_scalers.get(lake_id_int, None)
            if scalers is None:
                scalers = lake_id_to_scalers.get(str(lake_id_int), None)
            return scalers

        # Depth de-normalization helper: dataset normalizes depth to [0, 1] per lake.
        # For JSON depth-wise metrics we want *actual depth in meters* as keys.
        def denormalize_depths_for_lake(depth_vals: torch.Tensor, lake_id) -> torch.Tensor:
            if lake_id_to_depth_minmax is None:
                return depth_vals
            try:
                lake_id_int = int(lake_id) if not isinstance(lake_id, int) else lake_id
            except Exception:
                return depth_vals
            mm = lake_id_to_depth_minmax.get(lake_id_int, None)
            if not mm:
                return depth_vals
            try:
                dmin = float(mm.get("min_depth"))
                dmax = float(mm.get("max_depth"))
            except Exception:
                return depth_vals
            if not math.isfinite(dmin) or not math.isfinite(dmax) or dmax <= dmin:
                return depth_vals
            return depth_vals * (dmax - dmin) + dmin

        # Beer-Lambert (BL) experiment: keep state separate from the main eval loop.
        if self.BL:
            bl_cfg = self._bl_make_config()
            bl_state = self._bl_init_state(save_dir=save_dir, bl_cfg=bl_cfg)

        # Thermocline experiment: keep state separate and update over full eval stream (rank 0 only).
        thermo_cfg = self._thermo_make_config()
        thermo_state = None
        if thermo_cfg is not None and thermo_cfg.enabled and self.rank == 0:
            thermo_state = self._thermo_init_state(save_dir=save_dir, cfg=thermo_cfg)

        def _depth_key(depth_val: float) -> str:
            """
            Returns a stable JSON-friendly depth key.
            If self.depth_bin_size_m is set, we bucket into [lo, hi) bins; else we round to depth_round_decimals.
            """
            if self.depth_bin_size_m is not None and self.depth_bin_size_m > 0:
                lo = math.floor(depth_val / self.depth_bin_size_m) * self.depth_bin_size_m
                hi = lo + self.depth_bin_size_m
                return f"{lo:.3f}-{hi:.3f}"
            return f"{round(depth_val, int(self.depth_round_decimals)):.{int(self.depth_round_decimals)}f}"

        var_ids_2d_list = []
        depth_val_list = []

        batch_loss = 0
        avg_batch_loss = 0

        preds_list_2d = []
        labels_list_2d = []
        masks_list_2d = []
        lake_names_list = []
        lake_id_list = []
        time_val_list = []
        datetime_list = []
        
        # Context (input X) data for baseline comparison
        context_seq_list = []
        context_var_ids_list = []
        context_depth_vals_list = []
        context_time_vals_list = []
        context_datetime_list = []
        context_mask_list = []
        
        # Lake embedding trajectory data (temporal embeddings over time)
        lake_embed_traj_list = []  # List of pooled temporal embeddings
        lake_embed_traj_dates = []  # List of corresponding last dates in context window
        lake_embed_traj_lake_ids = []  # List of corresponding lake IDs
        lake_embed_traj_lake_names = []  # List of corresponding lake names


        lake_preds, lake_labels   = defaultdict(list), defaultdict(list)
        # Variate-wise metrics are computed directly in the main loop, no need for separate storage
        
        # Initialize metrics accumulators
        mse_sum = 0.0
        mae_sum = 0.0
        crps_sum = 0.0
        crps_count = 0

        # --- T+N date×horizon accumulators (variate-level; union across depths) ---
        # We keep these as sums+counts so duplicates (e.g., multiple depths) are averaged per cell.
        # Structure:
        #   tplusn_acc[var_id][date] = {pred_sum,pred_cnt,gt_sum,gt_cnt} where each is length=max_horizon.
        max_horizon = int(getattr(self.cfg.evaluator, "tplusn_max_horizon", 14))
        # If unset, we compute for all variates observed in Y; you can restrict via cfg.evaluator.tplusn_var_ids.
        tplusn_var_ids_cfg = getattr(self.cfg.evaluator, "tplusn_var_ids", None)
        tplusn_target_var_ids = None
        if tplusn_var_ids_cfg is not None:
            tplusn_target_var_ids = [int(v) for v in list(tplusn_var_ids_cfg)]
        tplusn_acc = defaultdict(dict)

        # WQL accumulators (overall + per-lake). WQL is computed from quantile (pinball) loss:
        #   WQL(q) = 2 * sum_t pinball_q(y_t, yhat_q,t) / (sum_t |y_t| + eps)
        # and averaged across quantiles.
        #
        # We compute quantiles analytically for Student-t via PPF (inverse CDF) if available, else fall back
        # to an MC quantile estimate (approximation) as a last resort.
        default_wql_quantiles = [
            0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
            0.50,
            0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99
        ]
        wql_quantiles = getattr(self.cfg.evaluator, "wql_quantiles", None) or default_wql_quantiles
        try:
            wql_quantiles = [float(q) for q in wql_quantiles]
        except Exception:
            wql_quantiles = default_wql_quantiles
        wql_quantiles = [q for q in wql_quantiles if (q is not None and 0.0 < float(q) < 1.0)]
        # keep stable ordering for JSON + reproducibility
        wql_quantiles = sorted(wql_quantiles)
        wql_eps = float(getattr(self.cfg.evaluator, "wql_denom_eps", 1e-8))
        wql_mc_samples = int(getattr(self.cfg.evaluator, "wql_mc_samples", 1024))
        wql_pinball_sums = [0.0 for _ in wql_quantiles]
        wql_denom_sum = 0.0
        lake_wql_pinball_sums = defaultdict(lambda: [0.0 for _ in wql_quantiles])
        lake_wql_denom_sums = defaultdict(float)
        # Var-wise WQL accumulators (overall per variable, and per-lake per-variable)
        var_wql_pinball_sums = defaultdict(lambda: [0.0 for _ in wql_quantiles])
        var_wql_denom_sums = defaultdict(float)
        lake_var_wql_pinball_sums = defaultdict(lambda: defaultdict(lambda: [0.0 for _ in wql_quantiles]))
        lake_var_wql_denom_sums = defaultdict(lambda: defaultdict(float))
        warned_wql_ppf_fallback = False
        metric_count = 0
        saw_distribution = False

        # Token counts for weighted/unweighted aggregation
        # NOTE: counts represent number of valid target tokens (after masks), not number of lake "samples".
        lake_token_counts = defaultdict(int)

        # Lake-wise CRPS accumulators (computed from distribution samples during the main loop)
        lake_crps_sums = defaultdict(float)
        lake_crps_counts = defaultdict(int)
        
        # Initialize variate-wise metrics accumulators (for irregular grids, no 2D/1D distinction)
        var_mse_sums = defaultdict(float)
        var_mae_sums = defaultdict(float)
        var_crps_sums = defaultdict(float)
        var_counts = defaultdict(int)

        # Initialize per-lake per-variate accumulators
        lake_var_mse_sums = defaultdict(lambda: defaultdict(float))
        lake_var_mae_sums = defaultdict(lambda: defaultdict(float))
        lake_var_crps_sums = defaultdict(lambda: defaultdict(float))
        lake_var_counts = defaultdict(lambda: defaultdict(int))

        # Depth-wise (per lake, per variate, per depth/bin) accumulators
        lake_var_depth_mse_sums = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        lake_var_depth_mae_sums = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        lake_var_depth_crps_sums = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        lake_var_depth_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        # Depth-wise (per lake, aggregated across variates, per depth/bin) accumulators
        lake_depth_mse_sums = defaultdict(lambda: defaultdict(float))
        lake_depth_mae_sums = defaultdict(lambda: defaultdict(float))
        lake_depth_crps_sums = defaultdict(lambda: defaultdict(float))
        lake_depth_counts = defaultdict(lambda: defaultdict(int))
        
        # Track unique variables per lake across all batches (for debugging)
        lake_to_variates = defaultdict(set)

        # For irregular data, collect all data first, then pad and save
        # Group data by lake_id for separate saving
        collected_data_by_lake = defaultdict(list)  # {lake_id: [data_map1, data_map2, ...]}
        total_loss = 0.0
        n_batches = len(dataloader)
        self.model.eval()
        device_obj = torch.device(self.device)
        infer_step_time_s = 0.0
        infer_samples = 0
        infer_valid_tokens = 0
        # Peak VRAM during model forward only (per-batch reset_peak + max across batches).
        infer_forward_peak_vram_gb = 0.0
        
        # For irregular data, collect all data first, then pad and save
        collected_data = []
        
        # Track lake names for later use in saving
        lake_id_to_name_all = {}

        if self.rank == 0:
            pbar = tqdm(total=n_batches, desc=f"Eval trial {trial_idx}", unit='batch')
        #
        # Requested structure (minimal):
        #   sample_<id>: {
        #     "dates": [YYYY-MM-DD, ...],
        #     "variates": {
        #       "<variable_name>": {
        #         "mean": {"depth_<m>": [..], ...},
        #         "std":  {"depth_<m>": [..], ...}
        #       }, ...
        #     }
        #   }
        # Always write per-sample predictions JSON (rank 0 only) to a deterministic path under save_dir.
        # Write per-sample predictions as JSONL (one JSON object per line) to avoid building one huge JSON
        # and to keep the file valid if the run stops mid-way.
        sample_pred_fh = None
        sample_pred_written = 0
        sample_pred_path = None
        # Also write per-sample ground-truth as JSONL in the same schema to a separate file.
        sample_gt_fh = None
        sample_gt_written = 0
        sample_gt_path = None
        if self.rank == 0:
            sample_pred_path = os.path.join(save_dir, f"sample_predictions_trial_{trial_idx}.jsonl")
            sample_pred_fh = open(sample_pred_path, "w")
            sample_gt_path = os.path.join(save_dir, f"sample_ground_truth_trial_{trial_idx}.jsonl")
            sample_gt_fh = open(sample_gt_path, "w")

        def _day_key(dt_val) -> str:
            """Return YYYY-MM-DD key from datetime-like input; '' if invalid/padding."""
            try:
                if isinstance(dt_val, np.datetime64):
                    if np.isnat(dt_val):
                        return ""
                    return str(np.datetime_as_string(dt_val, unit="D"))
                if isinstance(dt_val, (bytes, bytearray)):
                    dt_val = dt_val.decode("utf-8")
                s = str(dt_val).strip()
                if s == "" or s.lower() == "nat":
                    return ""
                # Handle numeric epoch timestamps (common in some exports).
                # Heuristic: seconds ~ 1e9, ms ~ 1e12, us ~ 1e15, ns ~ 1e18.
                if re.fullmatch(r"[-+]?\d+(\.\d+)?", s):
                    x = float(s)
                    ax = abs(x)
                    if ax >= 1e17:
                        ts = pd.to_datetime(int(x), unit="ns", utc=True, errors="coerce")
                    elif ax >= 1e14:
                        ts = pd.to_datetime(int(x), unit="us", utc=True, errors="coerce")
                    elif ax >= 1e11:
                        ts = pd.to_datetime(int(x), unit="ms", utc=True, errors="coerce")
                    else:
                        ts = pd.to_datetime(int(x), unit="s", utc=True, errors="coerce")
                else:
                    ts = pd.to_datetime(s, utc=True, errors="coerce")
                if pd.isna(ts):
                    return s[:10]
                ts = pd.Timestamp(ts)
                if ts.tzinfo is not None:
                    ts = ts.tz_convert("UTC").tz_localize(None)
                return ts.normalize().date().isoformat()
            except Exception:
                try:
                    return str(dt_val)[:10]
                except Exception:
                    return ""
        # breakpoint()
        for iteration, sample in enumerate(dataloader):
            seq_X = sample["flat_seq_x"].to(self.device)
            mask_X = sample["flat_mask_x"].to(self.device) 
            sample_ids_x = sample["sample_ids_x"].to(self.device) 
            time_ids_x = sample["time_ids_x"].to(self.device)
            var_ids_x = sample["var_ids_x"].to(self.device)
            padding_mask_x = sample["padding_mask_x"].to(self.device)
            depth_values_x = sample["depth_values_x"].to(self.device)
            time_values_x = sample["time_values_x"].to(self.device)
            
            lake_ids = sample["lake_id"] # lake id of each lake in batch
            lake_names = sample["lake_name"]
            idx = sample["idx"]
            num_2d_vars = sample['num2Dvars']
            num_1d_vars = sample['num1Dvars']
            num_depths = sample['num_depths']

            # target data
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

            # Track unique variates per lake from target data
            B = seq_Y.shape[0]
            for b in range(B):
                lake_id_b = int(lake_ids[b])
                # Get valid variates from target (what we're evaluating)
                valid_mask = mask_out[b].bool()
                if valid_mask.any():
                    valid_var_ids = var_ids_y[b][valid_mask]
                    unique_vars = torch.unique(valid_var_ids)
                    for var_id in unique_vars:
                        var_id_int = int(var_id)
                        lake_to_variates[lake_id_b].add(var_id_int)

            # Skip iteration if the whole batch has no valid values
            if not mask_X.any():
                if self.rank == 0:
                    pbar.update(1)
                continue
            
            model_ref = self.model.module if hasattr(self.model, "module") else self.model
            tgt_variate_ids = var_ids_y
            tgt_time_values = time_values_y
            tgt_time_ids = time_ids_y
            tgt_depth_values = depth_values_y
            tgt_padding_mask = padding_mask_y.bool()

            # Throughput / VRAM: model forward only (excludes loss, CRPS/WQL, etc.).
            infer_samples += int(seq_X.shape[0])
            infer_valid_tokens += int(mask_out.sum().item())
            if torch.cuda.is_available():
                torch.cuda.synchronize(device_obj)
                torch.cuda.reset_peak_memory_stats(device_obj)
            infer_t0 = dt.now().timestamp()
            with torch.inference_mode(), torch.autocast(device_type='cuda'):
                x_enc, pred, z = model_ref(
                    data=seq_X,
                    observed_mask=mask_X,
                    sample_ids=sample_ids_x,
                    variate_ids=var_ids_x,
                    padding_mask=padding_mask_x,
                    depth_values=depth_values_x,
                    pred_len=self.pred_len,
                    seq_len=self.seq_len,
                    time_values=time_values_x,
                    time_ids=time_ids_x,
                    tgt_variate_ids=tgt_variate_ids,
                    tgt_time_values=tgt_time_values,
                    tgt_time_ids=tgt_time_ids,
                    tgt_depth_values=tgt_depth_values,
                    tgt_padding_mask=tgt_padding_mask,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize(device_obj)
                infer_forward_peak_vram_gb = max(
                    infer_forward_peak_vram_gb,
                    float(torch.cuda.max_memory_allocated(device_obj) / (1024 ** 3)),
                )
            infer_step_time_s += dt.now().timestamp() - infer_t0

            pred_loss, forecasting_loss, reg_loss = self.forecasting_loss(
                seq_Y=seq_Y,
                pred=pred,
                mask_out=mask_out,
                epoch=self.model_epoch,
            )

            # Extract lake embedding trajectories if enabled
            if self.lake_embed_traj:
                # x_enc is already z_temporal_tokens (B, S, d_temporal) from model forward
                # Model returns: z_temporal_tokens, forecasts, z
                z_temporal_tokens = x_enc  # Already projected to d_temporal

                # Apply mean pooling over valid tokens (using padding_mask_x)
                valid_mask = padding_mask_x.bool()  # (B, S)
                masked_temporal = z_temporal_tokens * valid_mask.unsqueeze(-1)  # (B, S, d_temporal)
                pooled_temporal = torch.sum(masked_temporal, dim=1) / torch.sum(valid_mask, dim=1, keepdim=True)  # (B, d_temporal)

                # Extract last date from context window for each sample
                B = seq_X.shape[0]
                datetime_x = sample["datetime_strs_x"]

                for b in range(B):
                    # Get valid mask for this sample
                    valid_mask_b = valid_mask[b]  # (S,)
                    if not valid_mask_b.any():
                        continue

                    # Get last valid datetime string for this sample
                    datetime_x_b = datetime_x[b]  # typically numpy datetime64[ns] array (may contain NaT)
                    valid_indices = torch.where(valid_mask_b)[0].detach().cpu().numpy()

                    # Select the last valid index whose datetime is not NaT (NaT breaks DOY/date coloring)
                    try:
                        dt_arr = np.array(datetime_x_b)
                        # Guard against mismatch in lengths
                        valid_indices = valid_indices[valid_indices < len(dt_arr)]
                        if valid_indices.size == 0:
                            continue
                        dt_valid = dt_arr[valid_indices]
                        if np.issubdtype(dt_valid.dtype, np.datetime64):
                            non_nat = ~np.isnat(dt_valid)
                        else:
                            # String/object fallback: treat empty/NaT strings as invalid
                            dt_valid_str = dt_valid.astype(str)
                            non_nat = (dt_valid_str != "NaT") & (dt_valid_str != "") & (dt_valid_str != "None")

                        candidate = valid_indices[non_nat]
                        if candidate.size == 0:
                            # No usable datetime for this sample; skip storing this trajectory point
                            continue
                        last_valid_idx = int(candidate[-1])
                        last_date_val = dt_arr[last_valid_idx]
                        if isinstance(last_date_val, np.datetime64):
                            last_date = np.datetime_as_string(last_date_val, unit="s")
                        else:
                            last_date = str(last_date_val)
                    except Exception:
                        # Fallback: keep previous behavior (may yield NaT)
                        valid_indices_t = torch.where(valid_mask_b)[0]
                        last_valid_idx = min(valid_indices_t[-1].item(), len(datetime_x_b) - 1)
                        if last_valid_idx < 0 or last_valid_idx >= len(datetime_x_b):
                            continue
                        last_date = str(datetime_x_b[last_valid_idx])

                    # Store pooled embedding, date, and lake info
                    lake_embed_traj_list.append(pooled_temporal[b].detach().cpu())
                    lake_embed_traj_dates.append(last_date)
                    lake_embed_traj_lake_ids.append(int(lake_ids[b]))
                    lake_embed_traj_lake_names.append(lake_names[b])

            # Log loss components for monitoring (every 10 batches)
            if self.rank == 0 and iteration % 10 == 0:
                print(f"Batch {iteration}: total_loss={pred_loss.item():.6f}, "
                      f"forecasting_loss={forecasting_loss.item():.6f}, "
                      f"reg_loss={reg_loss:.6f}")
                
            loss_value = pred_loss.item()
            batch_loss += loss_value
            
            # ---------------- Metrics: MSE, MAE, CRPS (+ depth-wise) ----------------
            with torch.no_grad():
                # Always use irregular ground truth for metrics computation
                valid_mask_bool = mask_out.bool()
                
                if isinstance(pred, dict) and 'distribution' in pred:
                    pred_mean_tensor = pred['distribution'].mean
                    dist_obj = pred['distribution']
                else:
                    pred_mean_tensor = pred
                    dist_obj = None

                # Draw CRPS Monte Carlo samples once per batch (if probabilistic)
                samples = None
                K = 16
                if dist_obj is not None:
                    saw_distribution = True
                    try:
                        samples = dist_obj.rsample((K,))  # (K, B, S)
                    except Exception:
                        samples = dist_obj.sample((K,))

                B = seq_Y.shape[0]
                for b in range(B):
                    sample_mask = mask_out[b].bool()
                    if not sample_mask.any():
                        continue

                    lake_id_b = _to_int_lake_id(lake_ids[b])
                    scalers = get_scalers_for_lake(lake_id_b)

                    y_valid = seq_Y[b][sample_mask]
                    yhat_valid = pred_mean_tensor[b][sample_mask]
                    var_ids_valid = var_ids_y[b][sample_mask]
                    time_ids_valid = time_ids_y[b][sample_mask]
                    # Align datetime strings with the same valid mask (list[str] or numpy array-like)
                    dt_y_row = datetime_y[b]
                    mask_np = sample_mask.detach().cpu().numpy().astype(bool)
                    dt_valid = list(np.array(dt_y_row, dtype=object)[mask_np])

                    # depth_values_y is normalized to [0,1] per lake; convert to meters for depth-wise metrics
                    depth_valid = denormalize_depths_for_lake(depth_values_y[b][sample_mask], lake_id_b)
                    # datetime for each token (numpy datetime64); align with the same valid mask
                    dt_row = None
                    if (self.rank == 0) and (self.BL or (thermo_cfg is not None and thermo_cfg.enabled)):
                        try:
                            # datetime_y is (B, T) numpy datetime64 array from collate_fn padding
                            dt_row = datetime_y[b]
                            # Align with the same mask positions in the original sequence length
                            dt_row = dt_row[sample_mask.detach().cpu().numpy()]
                        except Exception:
                            dt_row = None
                    # Denormalize per-lake (required because scalers are lake-specific)
                    if self.denorm_eval and scalers is not None:
                        y_valid = Normalizer.denormalize_by_var_ids(
                        y_valid, var_ids_valid,
                            scaler_DF=scalers["scaler_DF"],
                            variate_ids_2D=scalers["variate_ids_2D"]
                        )
                        yhat_valid = Normalizer.denormalize_by_var_ids(
                        yhat_valid, var_ids_valid,
                            scaler_DF=scalers["scaler_DF"],
                            variate_ids_2D=scalers["variate_ids_2D"]
                        )
                                    
                    if y_valid.numel() == 0:
                        continue

                    # --- T+N date×horizon matrix accumulation (variate-level; union across depths) ---
                    # IMPORTANT: `time_ids_*` are relative to the *window start* on a regular daily grid.
                    # So the correct origin for T+1..T+pred_len is the *end of the context window*,
                    # not the last observed context token (which may be earlier if observations are sparse).
                    context_len_days = 30
                    origin_time_id = int(context_len_days) - 1
                    if origin_time_id >= 0:
                        if tplusn_target_var_ids is None:
                            var_ids_to_do = [int(v.item()) for v in torch.unique(var_ids_valid)]
                        else:
                            var_ids_to_do = tplusn_target_var_ids
                        for vid in var_ids_to_do:
                            self._tplusn_update_date_horizon_acc(
                                tplusn_acc[int(vid)],
                                dates=dt_valid,
                                time_ids=time_ids_valid,
                                preds=yhat_valid,
                                gts=y_valid,
                                var_ids=var_ids_valid,
                                target_var_id=int(vid),
                                origin_time_id=origin_time_id,
                                max_horizon=max_horizon,
                            )

                    # ---------------- WQL ----------------
                    # Note: we normalize by sum |y| to make WQL scale-invariant (common definition).
                    #       We also guard against non-finite params/targets.
                    # Filter to finite tokens for WQL computation
                    finite_mask = torch.isfinite(y_valid) & torch.isfinite(yhat_valid)
                    if finite_mask.any():
                        y_w = y_valid[finite_mask]
                        var_ids_w = var_ids_valid[finite_mask]
                        # denominator is per-token abs(y)
                        denom = y_w.abs().sum().item()
                        wql_denom_sum += denom
                        lake_wql_denom_sums[lake_id_b] += denom

                        # per-variable denominators
                        for vid in torch.unique(var_ids_w):
                            vid_int = int(vid.item())
                            vm = (var_ids_w == vid)
                            d_v = y_w[vm].abs().sum().item()
                            var_wql_denom_sums[vid_int] += d_v
                            lake_var_wql_denom_sums[lake_id_b][vid_int] += d_v

                    def _std_student_t_ppf(df_tensor: torch.Tensor, q: float):
                        """
                        Return the *standardized* Student-t quantile t_q for each df in df_tensor.
                        Priority:
                          1) torch.special.stdtrit (fast, on-device)
                          2) scipy.stats.t.ppf (robust, CPU)  <-- preferred fallback
                          3) torch.distributions.StudentT.icdf (may be unavailable/unsupported)
                        Returns None only if all methods fail.
                        """
                        qf = float(q)
                        q_tensor = torch.full_like(df_tensor, qf)

                        stdtrit = getattr(getattr(torch, "special", None), "stdtrit", None)
                        if stdtrit is not None:
                            try:
                                return stdtrit(df_tensor, q_tensor)
                            except Exception:
                                pass

                        # Robust fallback: SciPy PPF on CPU
                        try:
                            from scipy import stats as _scipy_stats  # local import to avoid hard dependency at import-time

                            df_np = df_tensor.detach().to("cpu").numpy()
                            # scipy supports array df with scalar q
                            tq_np = _scipy_stats.t.ppf(qf, df_np)
                            tq = torch.as_tensor(tq_np, device=df_tensor.device, dtype=df_tensor.dtype)
                            if torch.isfinite(tq).all():
                                return tq
                            # If SciPy returned inf/nan for some df, fall through to torch icdf / None.
                        except Exception:
                            pass

                        # Analytic/numerical fallback: use StudentT.icdf if available in this torch build.
                        try:
                            d = torch.distributions.StudentT(
                                df=df_tensor,
                                loc=torch.zeros_like(df_tensor),
                                scale=torch.ones_like(df_tensor),
                            )
                            return d.icdf(q_tensor)
                        except Exception:
                            return None

                    # Build quantile predictions
                    if (dist_obj is not None) and isinstance(pred, dict) and all(k in pred for k in ("loc", "scale", "df")):
                        loc_w = pred["loc"][b][sample_mask][finite_mask]
                        scale_w = pred["scale"][b][sample_mask][finite_mask]
                        df_w = pred["df"][b][sample_mask][finite_mask]

                        # denormalize loc/scale to match y_valid space
                        if self.denorm_eval and scalers is not None:
                            loc_w = Normalizer.denormalize_by_var_ids(
                                loc_w, var_ids_w,
                                scaler_DF=scalers["scaler_DF"],
                                variate_ids_2D=scalers["variate_ids_2D"],
                            )
                            scale_w = Normalizer.denormalize_scale_by_var_ids(
                                scale_w, var_ids_w,
                                scaler_DF=scalers["scaler_DF"],
                                variate_ids_2D=scalers["variate_ids_2D"],
                            )

                        # clamp to valid parameter domain
                        df_w = torch.clamp(df_w, min=1e-3)
                        scale_w = torch.clamp(scale_w, min=1e-12)
                        param_finite = torch.isfinite(loc_w) & torch.isfinite(scale_w) & torch.isfinite(df_w)
                        if param_finite.any():
                            y_wp = y_w[param_finite]
                            loc_wp = loc_w[param_finite]
                            scale_wp = scale_w[param_finite]
                            df_wp = df_w[param_finite]
                            var_ids_wp = var_ids_w[param_finite]

                            for qi, q in enumerate(wql_quantiles):
                                t_q = _std_student_t_ppf(df_wp, q)
                                if t_q is None:
                                    # Last resort: approximate quantile via MC sampling (quantile of samples)
                                    if not warned_wql_ppf_fallback and self.rank == 0:
                                        print(
                                            "Warning: Student-t PPF unavailable (torch.special.stdtrit missing/failed; SciPy PPF failed; StudentT.icdf failed). "
                                            f"Falling back to MC quantiles for WQL with {wql_mc_samples} samples (approximation)."
                                        )
                                        warned_wql_ppf_fallback = True
                                    try:
                                        dist_tokens = torch.distributions.StudentT(df=df_wp, loc=loc_wp, scale=scale_wp)
                                        s = dist_tokens.rsample((wql_mc_samples,))
                                    except Exception:
                                        dist_tokens = torch.distributions.StudentT(df=df_wp, loc=loc_wp, scale=scale_wp)
                                        s = dist_tokens.sample((wql_mc_samples,))
                                    yhat_q = torch.quantile(s, float(q), dim=0)
                                else:
                                    yhat_q = loc_wp + scale_wp * t_q

                                diff = y_wp - yhat_q
                                pin = torch.maximum(float(q) * diff, (float(q) - 1.0) * diff)
                                pin_sum = pin.sum().item()
                                wql_pinball_sums[qi] += pin_sum
                                lake_wql_pinball_sums[lake_id_b][qi] += pin_sum
                                # per-variable pinball sums
                                for vid in torch.unique(var_ids_wp):
                                    vid_int = int(vid.item())
                                    vm = (var_ids_wp == vid)
                                    ps_v = pin[vm].sum().item()
                                    var_wql_pinball_sums[vid_int][qi] += ps_v
                                    lake_var_wql_pinball_sums[lake_id_b][vid_int][qi] += ps_v
                        # else: skip WQL for this sample if params are all non-finite
                    else:
                        # Deterministic: degenerate distribution at yhat
                        yhat_w = yhat_valid[finite_mask]
                        for qi, q in enumerate(wql_quantiles):
                            diff = y_w - yhat_w
                            pin = torch.maximum(float(q) * diff, (float(q) - 1.0) * diff)
                            pin_sum = pin.sum().item()
                            wql_pinball_sums[qi] += pin_sum
                            lake_wql_pinball_sums[lake_id_b][qi] += pin_sum
                            # per-variable pinball sums
                            for vid in torch.unique(var_ids_w):
                                vid_int = int(vid.item())
                                vm = (var_ids_w == vid)
                                ps_v = pin[vm].sum().item()
                                var_wql_pinball_sums[vid_int][qi] += ps_v
                                lake_var_wql_pinball_sums[lake_id_b][vid_int][qi] += ps_v

                # Overall metrics (token-wise sums)
                err = yhat_valid - y_valid
                mse_sum += (err * err).sum().item()
                mae_sum += err.abs().sum().item()
                metric_count += y_valid.numel()
                lake_token_counts[lake_id_b] += y_valid.numel()
                                
                # CRPS (token-wise sums)
                crps_tokens = None
                if samples is not None:
                    sample_samples = samples[:, b, sample_mask]  # (K, N_valid)
                    if self.denorm_eval and scalers is not None:
                        for k in range(sample_samples.shape[0]):
                            sample_samples[k] = Normalizer.denormalize_by_var_ids(sample_samples[k], var_ids_valid,
                                scaler_DF=scalers["scaler_DF"],
                                variate_ids_2D=scalers["variate_ids_2D"]
                            )
                    
                    if sample_samples.numel() > 0:
                        yv = y_valid.unsqueeze(0)
                        term1 = (sample_samples - yv).abs().mean(dim=0)  # (N,)
                        if sample_samples.shape[0] >= 2:
                            s1 = sample_samples[:K // 2]
                            s2 = sample_samples[K // 2:K]
                            pair = (s1.unsqueeze(1) - s2.unsqueeze(0)).abs().mean(dim=(0, 1))  # (N,)
                        else:
                            pair = torch.zeros_like(term1)

                        crps_tokens = (term1 - 0.5 * pair)  # (N,)
                        crps_sum += crps_tokens.sum().item()
                        crps_count += int(crps_tokens.numel())
                        lake_crps_sums[lake_id_b] += crps_tokens.sum().item()
                        lake_crps_counts[lake_id_b] += int(crps_tokens.numel())

                    # BL update (rank 0 only) – keep it separate from core metric computation.
                    if self.BL and bl_cfg.enabled and dt_row is not None:
                        try:
                            self._bl_update(
                                bl_cfg,
                                bl_state,
                                lake_id=lake_id_b,
                                lake_name=lake_names[b],
                                dt_row=np.array(dt_row, dtype="datetime64[ns]"),
                                depth_m=depth_valid.detach().cpu().numpy(),
                                var_ids=var_ids_valid.detach().cpu().numpy(),
                                pred=yhat_valid.detach().cpu().numpy(),
                            )
                        except Exception:
                            pass

                    # Thermocline update (rank 0 only) – uses full eval stream, filtered by start/end date.
                    if thermo_cfg is not None and thermo_state is not None and dt_row is not None:
                        try:
                            self._thermo_update(
                                thermo_cfg,
                                thermo_state,
                                lake_id=lake_id_b,
                                lake_name=lake_names[b],
                                dt_row=np.array(dt_row, dtype="datetime64[ns]"),
                                depth_m=depth_valid.detach().cpu().numpy(),
                                var_ids=var_ids_valid.detach().cpu().numpy(),
                                gt=y_valid.detach().cpu().numpy(),
                                pred=yhat_valid.detach().cpu().numpy(),
                            )
                        except Exception:
                            pass

                    # Variate-wise and depth-wise metrics on flattened tokens
                    unique_vars = torch.unique(var_ids_valid)

                    # Depth-wise metrics aggregated across all variates (within this lake)
                    # Group by depth/bin on the full token set for this sample.
                    if depth_valid.numel() > 0:
                        if self.depth_bin_size_m is not None and self.depth_bin_size_m > 0:
                            depth_grp_all = torch.floor(depth_valid / self.depth_bin_size_m) * self.depth_bin_size_m
                        else:
                            p_all = float(10 ** int(self.depth_round_decimals))
                            depth_grp_all = torch.round(depth_valid * p_all) / p_all

                        for dg in torch.unique(depth_grp_all):
                            dg_mask = (depth_grp_all == dg)
                            if not dg_mask.any():
                                continue
                            dk = _depth_key(float(dg.item()))
                            err_d = err[dg_mask]
                            mse_d = (err_d * err_d).sum().item()
                            mae_d = err_d.abs().sum().item()
                            c_d = int(dg_mask.sum().item())

                            lake_depth_mse_sums[lake_id_b][dk] += mse_d
                            lake_depth_mae_sums[lake_id_b][dk] += mae_d
                            lake_depth_counts[lake_id_b][dk] += c_d

                            if crps_tokens is not None:
                                lake_depth_crps_sums[lake_id_b][dk] += crps_tokens[dg_mask].sum().item()

                    # Variate-wise and depth-wise metrics on flattened tokens
                    unique_vars = torch.unique(var_ids_valid)
                    for var_id in unique_vars:
                        var_mask = (var_ids_valid == var_id)
                        if not var_mask.any():
                            continue
                        var_id_int = int(var_id)
                        cnt = int(var_mask.sum().item())

                        # token-wise sums for this variate
                        err_v = err[var_mask]
                        mse_val = (err_v * err_v).sum().item()
                        mae_val = err_v.abs().sum().item()

                        var_mse_sums[var_id_int] += mse_val
                        var_mae_sums[var_id_int] += mae_val
                        var_counts[var_id_int] += cnt

                        lake_var_mse_sums[lake_id_b][var_id_int] += mse_val
                        lake_var_mae_sums[lake_id_b][var_id_int] += mae_val
                        lake_var_counts[lake_id_b][var_id_int] += cnt

                        if crps_tokens is not None:
                            crps_val = crps_tokens[var_mask].sum().item()
                            var_crps_sums[var_id_int] += crps_val
                            lake_var_crps_sums[lake_id_b][var_id_int] += crps_val

                        # Group by (variate, depth/bin) within this lake
                        depth_vals_v = depth_valid[var_mask]
                        if depth_vals_v.numel() > 0:
                            if self.depth_bin_size_m is not None and self.depth_bin_size_m > 0:
                                depth_grp = torch.floor(depth_vals_v / self.depth_bin_size_m) * self.depth_bin_size_m
                            else:
                                p = float(10 ** int(self.depth_round_decimals))
                                depth_grp = torch.round(depth_vals_v * p) / p

                            for dg in torch.unique(depth_grp):
                                dg_mask = (depth_grp == dg)
                                if not dg_mask.any():
                                    continue
                                dk = _depth_key(float(dg.item()))
                                err_d = err_v[dg_mask]
                                mse_d = (err_d * err_d).sum().item()
                                mae_d = err_d.abs().sum().item()
                                c_d = int(dg_mask.sum().item())

                                lake_var_depth_mse_sums[lake_id_b][var_id_int][dk] += mse_d
                                lake_var_depth_mae_sums[lake_id_b][var_id_int][dk] += mae_d
                                lake_var_depth_counts[lake_id_b][var_id_int][dk] += c_d

                                if crps_tokens is not None:
                                    # Align depth grouping with the same tokens used above
                                    crps_v = crps_tokens[var_mask]
                                    lake_var_depth_crps_sums[lake_id_b][var_id_int][dk] += crps_v[dg_mask].sum().item()

            if not math.isfinite(loss_value):
                if self.rank==0:
                    print("Loss is {}, stopping validation".format(loss_value))
 
            # (variates/depth-wise metrics handled above inside the unified metrics block)

            # Store full prediction distribution for uncertainty visualization
            if isinstance(pred, dict) and 'distribution' in pred:
                # Store full distribution parameters
                pred_to_store = {
                    'loc': pred['loc'].detach().clone(),
                    'scale': pred['scale'].detach().clone(),
                    'df': pred['df'].detach().clone(),
                    'mean': pred['distribution'].mean.detach().clone()
                }
            else:
                pred_to_store = pred.detach().clone()
            
            # Only collect plotting data for first num_plot_batches (similar to val_one_epoch)
            # If num_plot_batches < 0 (e.g. -1), collect plotting data for the entire test loader.
            plot_all_batches = False
            try:
                plot_all_batches = int(self.cfg.num_plot_batches) < 0
            except Exception:
                plot_all_batches = False
            if (plot_all_batches or iteration < self.cfg.num_plot_batches) and self.rank == 0:
                # Denormalize plotting data if flag is set (only mean and labels)
                labels_denorm = seq_Y.detach().clone()
                if self.denorm_eval:
                    # Handle batched data - denormalize per sample in batch
                    B = seq_Y.shape[0]
                    for b in range(B):
                        lake_id_b = int(lake_ids[b])
                        scalers = get_scalers_for_lake(lake_id_b)
                        if scalers is not None:
                            # Denormalize this sample's predictions
                            if isinstance(pred_to_store, dict):
                                if 'mean' in pred_to_store:
                                    pred_to_store['mean'][b] = Normalizer.denormalize_by_var_ids(
                                        pred_to_store['mean'][b:b+1], var_ids_y[b:b+1],
                                        scaler_DF=scalers["scaler_DF"],
                                        variate_ids_2D=scalers["variate_ids_2D"]
                                    )[0]
                                # Denormalize loc (location parameter) - same as mean
                                if 'loc' in pred_to_store:
                                    pred_to_store['loc'][b] = Normalizer.denormalize_by_var_ids(
                                        pred_to_store['loc'][b:b+1], var_ids_y[b:b+1],
                                        scaler_DF=scalers["scaler_DF"],
                                        variate_ids_2D=scalers["variate_ids_2D"]
                                    )[0]
                                # Denormalize scale (std) - only multiply by std, no mean shift
                                if 'scale' in pred_to_store:
                                    pred_to_store['scale'][b] = Normalizer.denormalize_scale_by_var_ids(
                                        pred_to_store['scale'][b:b+1], var_ids_y[b:b+1],
                                        scaler_DF=scalers["scaler_DF"],
                                        variate_ids_2D=scalers["variate_ids_2D"]
                                    )[0]
                            else:
                                pred_to_store[b] = Normalizer.denormalize_by_var_ids(
                                    pred_to_store[b:b+1], var_ids_y[b:b+1],
                                    scaler_DF=scalers["scaler_DF"],
                                    variate_ids_2D=scalers["variate_ids_2D"]
                                )[0]
                            # Denormalize this sample's labels
                            labels_denorm[b] = Normalizer.denormalize_by_var_ids(
                                labels_denorm[b:b+1], var_ids_y[b:b+1],
                                scaler_DF=scalers["scaler_DF"],
                                variate_ids_2D=scalers["variate_ids_2D"]
                            )[0]
                preds_list_2d.append(pred_to_store)
                labels_list_2d.append(labels_denorm)
                masks_list_2d.append(mask_out.detach())
                var_ids_2d_list.append(var_ids_y.detach())
                # Store depths in meters (de-normalized) when min/max are available.
                depth_y_store = depth_values_y.detach()
                try:
                    if lake_id_to_depth_minmax is not None:
                        depth_y_store = depth_y_store.clone()
                        for bb in range(depth_y_store.shape[0]):
                            try:
                                lid_bb = int(lake_ids[bb])
                            except Exception:
                                lid_bb = lake_ids[bb]
                            depth_y_store[bb] = denormalize_depths_for_lake(depth_y_store[bb], lid_bb)
                except Exception:
                    pass
                depth_val_list.append(depth_y_store)
                time_val_list.append(time_values_y.detach())
                datetime_list.append(datetime_y)
                lake_names_list.append(lake_names)
                lake_id_list.append(lake_ids)

                # Export per-sample predictions JSON (minimal format) for the same collected windows.
                if sample_pred_fh is not None and sample_gt_fh is not None:
                    try:
                        B = int(seq_Y.shape[0])
                        for b in range(B):
                            # Sample id (prefer dataset idx)
                            try:
                                sid = idx[b]
                                if isinstance(sid, (torch.Tensor, np.ndarray)) and np.ndim(sid) == 0:
                                    sid = sid.item()
                                sid = int(sid)
                            except Exception:
                                sid = int(b)

                            # Pull token-aligned arrays for this sample
                            dt_row = np.asarray(datetime_y[b], dtype=object)
                            vids = var_ids_y[b].detach().cpu().numpy().astype(int)
                            # Depths are already denormalized to meters in `depth_y_store` above (when min/max are available).
                            depths_m = depth_y_store[b].detach().cpu().numpy().astype(float)
                            eval_mask = mask_out[b].detach().cpu().numpy().astype(bool)
                            
                            if isinstance(pred_to_store, dict):
                                mean_row = pred_to_store["mean"][b].detach().cpu().numpy().astype(float)
                                std_row = pred_to_store.get("scale", None)
                                std_row = std_row[b].detach().cpu().numpy().astype(float) if std_row is not None else None
                            else:
                                mean_row = pred_to_store[b].detach().cpu().numpy().astype(float)
                                std_row = None
                            # Ground-truth values (already denormalized if denorm_eval=True above)
                            gt_row_vals = labels_denorm[b].detach().cpu().numpy().astype(float)

                            # Choose token indices: valid prediction tokens (mask_out) with non-padding datetimes
                            # IMPORTANT: For JSON export, we want ALL predictions (even where GT is NaN)
                            # So we use padding_mask only, not eval_mask (which filters NaN GTs)
                            # This ensures consistent depth/date grids for baseline comparison
                            token_idx = []
                            padding_mask_b = padding_mask_y[b].detach().cpu().numpy().astype(bool)
                            for i in range(len(dt_row)):
                                if not bool(padding_mask_b[i]):  # Only filter padding, not NaN
                                    continue
                                dk = _day_key(dt_row[i])
                                if not dk:
                                    continue
                                token_idx.append(i)

                            # Dates are constant across vars/depths; store once at sample level.
                            dates = sorted({ _day_key(dt_row[i]) for i in token_idx if _day_key(dt_row[i]) })
                            date_to_j = {d: j for j, d in enumerate(dates)}
                            T = len(dates)
                            
                            # DEBUG: Check dates extraction
                            # if sid < 3:
                            #     print(f"  token_idx length: {len(token_idx)}")
                            #     print(f"  Extracted dates (T): {T}")
                            #     print(f"  dates: {dates[:5] if len(dates) > 5 else dates}")
                            #     if T == len(set(dt_row)):
                            #         print(f"  SUCCESS: All {T} dates exported!")
                            #     elif T < len(set(dt_row)):
                            #         print(f"  WARNING: Only {T} of {len(set(dt_row))} dates extracted!")
                            #         # Show which dates are missing
                            #         all_dates = sorted(set(_day_key(dt_row[i]) for i in range(len(dt_row)) if _day_key(dt_row[i])))
                            #         exported_dates = set(dates)
                            #         missing_dates = [d for d in all_dates if d not in exported_dates]
                            #         if missing_dates:
                            #             print(f"  Missing dates: {missing_dates[:5]}")
                            if T == 0:
                                continue

                            # Aggregate values by (var, depth, date) (defensive against duplicates)
                            acc = {}  # (vid, depth_key, date) -> [sum_mean, cnt_mean, sum_std, cnt_std]
                            for i in token_idx:
                                d = _day_key(dt_row[i])
                                if d not in date_to_j:
                                    continue
                                vid = int(vids[i])
                                dep = float(np.round(float(depths_m[i]), 6))
                                k = (vid, dep, d)
                                rec = acc.get(k)
                                if rec is None:
                                    rec = [0.0, 0, 0.0, 0]
                                    acc[k] = rec
                                mv = float(mean_row[i]) if np.isfinite(mean_row[i]) else None
                                if mv is not None:
                                    rec[0] += mv
                                    rec[1] += 1
                                if std_row is not None:
                                    sv = float(std_row[i]) if np.isfinite(std_row[i]) else None
                                    if sv is not None:
                                        rec[2] += sv
                                        rec[3] += 1

                            # Build nested output: variate -> mean/std -> depth -> series
                            variates_out = {}
                            # Determine unique (vid, dep) pairs
                            pairs = sorted({ (vid, dep) for (vid, dep, _d) in acc.keys() })
                            for vid, dep in pairs:
                                # Use variable name (preferred) instead of raw id
                                vname = None
                                try:
                                    id_to_var = getattr(self, "id_to_var", None) or getattr(getattr(self, "data", None), "id_to_var", None)
                                    if id_to_var is not None:
                                        vname = id_to_var.get(int(vid), None)
                                        if vname is None:
                                            vname = id_to_var.get(str(int(vid)), None)
                                except Exception:
                                    vname = None
                                vkey = str(vname) if vname is not None else f"var_{int(vid)}"
                                dkey = f"depth_{float(dep):.6f}"
                                vrec = variates_out.setdefault(vkey, {"mean": {}, "std": {}})
                                mean_series = [None] * T
                                std_series = [None] * T
                                for d in dates:
                                    rec = acc.get((vid, dep, d))
                                    if rec is None:
                                        continue
                                    if rec[1] > 0:
                                        mean_series[date_to_j[d]] = float(rec[0] / float(rec[1]))
                                    if std_row is not None and rec[3] > 0:
                                        std_series[date_to_j[d]] = float(rec[2] / float(rec[3]))
                                vrec["mean"][dkey] = mean_series
                                vrec["std"][dkey] = std_series

                            sample_obj = {"dates": dates, "variates": variates_out}
                            skey = f"sample_{sid}"
                            sample_pred_fh.write(json.dumps({skey: sample_obj}, ensure_ascii=False) + "\n")
                            sample_pred_written += 1

                            # Build ground-truth object with the same schema (std = nulls)
                            gt_variates_out = {}
                            # Aggregate GT by (var, depth, date) (same mechanics as preds)
                            gt_acc = {}  # (vid, depth_key, date) -> [sum, cnt]
                            for i in token_idx:
                                d = _day_key(dt_row[i])
                                if d not in date_to_j:
                                    continue
                                vid = int(vids[i])
                                dep = float(np.round(float(depths_m[i]), 6))
                                k = (vid, dep, d)
                                rec = gt_acc.get(k)
                                if rec is None:
                                    rec = [0.0, 0]
                                    gt_acc[k] = rec
                                gv = float(gt_row_vals[i]) if np.isfinite(gt_row_vals[i]) else None
                                if gv is not None:
                                    rec[0] += gv
                                    rec[1] += 1

                            gt_pairs = sorted({ (vid, dep) for (vid, dep, _d) in gt_acc.keys() })
                            for vid, dep in gt_pairs:
                                vname = None
                                try:
                                    id_to_var = getattr(self, "id_to_var", None) or getattr(getattr(self, "data", None), "id_to_var", None)
                                    if id_to_var is not None:
                                        vname = id_to_var.get(int(vid), None)
                                        if vname is None:
                                            vname = id_to_var.get(str(int(vid)), None)
                                except Exception:
                                    vname = None
                                vkey = str(vname) if vname is not None else f"var_{int(vid)}"
                                dkey = f"depth_{float(dep):.6f}"
                                vrec = gt_variates_out.setdefault(vkey, {"mean": {}, "std": {}})
                                mean_series = [None] * T
                                std_series = [None] * T  # keep schema identical; GT has no std
                                for d in dates:
                                    rec = gt_acc.get((vid, dep, d))
                                    if rec is None:
                                        continue
                                    if rec[1] > 0:
                                        mean_series[date_to_j[d]] = float(rec[0] / float(rec[1]))
                                vrec["mean"][dkey] = mean_series
                                vrec["std"][dkey] = std_series

                            gt_obj = {"dates": dates, "variates": gt_variates_out}
                            sample_gt_fh.write(json.dumps({skey: gt_obj}, ensure_ascii=False) + "\n")
                            sample_gt_written += 1
                    except Exception as e:
                        print(f"[export] Warning: failed to write per-sample predictions JSON: {e}")
            
                context_denorm = seq_X.detach()
                var_ids_x_cpu = var_ids_x.detach()
                mask_X_cpu = mask_X.detach()
                datetime_x = sample["datetime_strs_x"]
                
                if self.denorm_eval:
                    for b in range(B):
                        lake_id_b = int(lake_ids[b])
                        scalers = get_scalers_for_lake(lake_id_b)
                        if scalers is None:
                            continue
                        context_denorm[b] = Normalizer.denormalize_by_var_ids(
                            context_denorm[b:b+1],
                            var_ids_x_cpu[b:b+1],
                            scaler_DF=scalers.get("scaler_DF", None),
                            variate_ids_2D=scalers.get("variate_ids_2D", None),
                            scaler_DR=scalers.get("scaler_DR", None),
                            variate_ids_1D=scalers.get("variate_ids_1D", None),
                        )[0]
                        # Keep masked tokens as 0.0 (they were NaN->0 upstream); this makes JSON easier to interpret.
                        context_denorm[b][mask_X_cpu[b] <= 0] = 0.0
                
                context_seq_list.append(context_denorm)
                context_var_ids_list.append(var_ids_x_cpu)
                # Store context depths in meters as well (for consistent plot/export).
                depth_x_store = depth_values_x.detach()
                try:
                    if lake_id_to_depth_minmax is not None:
                        depth_x_store = depth_x_store.clone()
                        for bb in range(depth_x_store.shape[0]):
                            try:
                                lid_bb = int(lake_ids[bb])
                            except Exception:
                                lid_bb = lake_ids[bb]
                            depth_x_store[bb] = denormalize_depths_for_lake(depth_x_store[bb], lid_bb)
                except Exception:
                    pass
                context_depth_vals_list.append(depth_x_store)
                context_time_vals_list.append(time_values_x.detach())
                context_datetime_list.append(datetime_x)
                context_mask_list.append(mask_X_cpu)

            if self.rank==0:
                pbar.update(1)

            # collect per-lake preds & labels to compute lake-wise metrics
            # flatten per-sample masked values and append into per-lake lists
            for i, lake_id in enumerate(lake_ids):
                sel = mask_out[i].bool()
                # Handle both dict and tensor predictions
                if isinstance(pred, dict) and 'distribution' in pred:
                    pred_mean = pred['distribution'].mean
                else:
                    pred_mean = pred
                
                pred_vals = pred_mean[i][sel].detach().cpu()
                label_vals = seq_Y[i][sel].detach().cpu()
                var_ids_sel = var_ids_y[i][sel].detach().cpu()
                lake_id_int = _to_int_lake_id(lake_id)
                scalers = get_scalers_for_lake(lake_id_int)
                
                # Denormalize if flag is set (for lake-wise metrics)
                if self.denorm_eval and scalers is not None:
                    pred_vals = Normalizer.denormalize_by_var_ids(
                        pred_vals, var_ids_sel,
                        scaler_DF=scalers["scaler_DF"],
                        variate_ids_2D=scalers["variate_ids_2D"]
                    )
                    label_vals = Normalizer.denormalize_by_var_ids(
                        label_vals, var_ids_sel,
                        scaler_DF=scalers["scaler_DF"],
                        variate_ids_2D=scalers["variate_ids_2D"]
                    )
                
                # Use int lake_id keys consistently across all lake-wise metrics / JSON outputs
                lake_preds[int(lake_id_int)].append(pred_vals)
                lake_labels[int(lake_id_int)].append(label_vals)

            # build data map for collection (flattened data for irregular grids) - only if saving
            if self.save_preds:
                # Store full prediction distribution if available
                if isinstance(pred, dict) and 'distribution' in pred:
                    pred_np = {
                        'loc': pred['loc'].detach().cpu().numpy(),
                        'scale': pred['scale'].detach().cpu().numpy(),
                        'df': pred['df'].detach().cpu().numpy(),
                        'mean': pred['distribution'].mean.detach().cpu().numpy()
                    }
                else:
                    pred_np = pred.detach().cpu().numpy()
                
                # Handle datetime strings
                datetime_arr = np.array(datetime_y, dtype='S')
                
                # Get sample indices from the batch
                B = seq_Y.shape[0]
                sample_indices = np.array(idx, dtype='int64') if isinstance(idx, (list, np.ndarray, torch.Tensor)) else np.arange(B, dtype='int64')
                if isinstance(sample_indices, torch.Tensor):
                    sample_indices = sample_indices.cpu().numpy()
                if len(sample_indices) != B:
                    sample_indices = np.arange(B, dtype='int64')
                
                labels_np = seq_Y.detach().cpu().numpy()
                masks_np = mask_out.detach().cpu().numpy()
                
                # Compute per-sample MSE and MAE if requested
                per_sample_mse = None
                per_sample_mae = None
                if self.save_per_sample_metrics:
                    per_sample_mse = []
                    per_sample_mae = []
                    for b in range(B):
                        valid_mask = masks_np[b].astype(bool)
                        if valid_mask.sum() > 0:
                            # Get predictions (use mean if dict)
                            if isinstance(pred_np, dict):
                                pred_mean_b = pred_np['mean'][b]
                            else:
                                pred_mean_b = pred_np[b]
                            
                            # Compute MSE and MAE only on valid predictions
                            pred_valid = pred_mean_b[valid_mask]
                            label_valid = labels_np[b][valid_mask]
                            
                            mse = np.mean((pred_valid - label_valid) ** 2)
                            mae = np.mean(np.abs(pred_valid - label_valid))
                            per_sample_mse.append(mse)
                            per_sample_mae.append(mae)
                        else:
                            per_sample_mse.append(np.nan)
                            per_sample_mae.append(np.nan)
                    per_sample_mse = np.array(per_sample_mse, dtype='float32')
                    per_sample_mae = np.array(per_sample_mae, dtype='float32')
                
                data_map = {
                    'preds': pred_np,
                    'labels': labels_np,
                    'masks': masks_np,
                    'var_ids': var_ids_y.detach().cpu().numpy(),
                    'depth_vals': depth_values_y.detach().cpu().numpy(),
                    'time_vals': time_values_y.detach().cpu().numpy(),
                    'lake_names': np.array([str(n).encode('utf-8') if not isinstance(n, bytes) else n for n in lake_names], dtype='S'),
                    'lake_ids': np.array(lake_ids, dtype='int64'),
                    'datetime_strs': datetime_arr,
                    'sample_indices': sample_indices
                }
                if per_sample_mse is not None:
                    data_map['per_sample_mse'] = per_sample_mse
                    data_map['per_sample_mae'] = per_sample_mae
                
                # Group data by lake_id for separate saving
                B = seq_Y.shape[0]
                for b in range(B):
                    lake_id_b = int(lake_ids[b])
                    # Store lake name mapping
                    if b < len(lake_names):
                        lake_name_b = lake_names[b]
                        if isinstance(lake_name_b, bytes):
                            lake_name_str = lake_name_b.decode('utf-8')
                        else:
                            lake_name_str = str(lake_name_b)
                        if lake_name_str and lake_name_str.strip():
                            if lake_id_b not in lake_id_to_name_all or not lake_id_to_name_all[lake_id_b]:
                                lake_id_to_name_all[lake_id_b] = lake_name_str
                    
                    # Extract data for this sample
                    sample_data_map = {}
                    for key, val in data_map.items():
                        if isinstance(val, np.ndarray) and len(val.shape) > 0:
                            if key == 'preds' and isinstance(val, dict):
                                # Handle dict predictions
                                sample_data_map[key] = {k: v[b:b+1] for k, v in val.items()}
                            else:
                                sample_data_map[key] = val[b:b+1] if len(val.shape) > 0 else val
                        else:
                            sample_data_map[key] = val
                    collected_data_by_lake[lake_id_b].append(sample_data_map)
                
                # Also keep original format for backward compatibility (metrics computation)
                collected_data.append(data_map)

                
            if self.rank == 0:
                pbar.update(1)
        
        # After collecting all data, pad and save to HDF5 - organized by lake
        if self.save_preds and collected_data_by_lake:
            # Helper function to sanitize directory names
            def sanitize_dir_name(name):
                """Sanitize a string to be filesystem-safe"""
                import re
                # Replace invalid filesystem characters with underscores
                sanitized = re.sub(r'[<>:"/\\|?*]', '_', str(name))
                sanitized = sanitized.strip(' .')
                sanitized = re.sub(r'[_\s]+', '_', sanitized)
                if len(sanitized) > 200:
                    sanitized = sanitized[:200]
                return sanitized
            
            # Save predictions separately for each lake
            for lake_id, lake_data_list in collected_data_by_lake.items():
                # Get lake name from collected mapping (already populated during batch loop)
                lake_name = lake_id_to_name_all.get(lake_id, f"Lake_{lake_id}")
                sanitized_lake_name = sanitize_dir_name(lake_name) or f"Lake_{lake_id}"
                lake_dir_name = f"{sanitized_lake_name}"
                
                # Create lake-specific directory
                lake_dir = os.path.join(save_dir, lake_dir_name)
                os.makedirs(lake_dir, exist_ok=True)
                
                # Create HDF5 file for this lake
                h5_filename = f"predictions_predlen_{self.pred_len}.h5"
                h5_path = os.path.join(lake_dir, h5_filename)
                h5_file = h5py.File(h5_path, 'w')
                save_preds_dict = {}
                
                # Find the maximum sequence length for this lake
                max_seq_len = 0
                for data_map in lake_data_list:
                    preds = data_map['preds']
                    if isinstance(preds, dict):
                        # For dict predictions, check all keys
                        for key, arr in preds.items():
                            if isinstance(arr, np.ndarray) and len(arr.shape) > 1:
                                max_seq_len = max(max_seq_len, arr.shape[1])
                    else:
                        if isinstance(preds, np.ndarray) and len(preds.shape) > 1:
                            max_seq_len = max(max_seq_len, preds.shape[1])
                    
                    # Check other arrays too
                    for name, arr in data_map.items():
                        if name != 'preds' and isinstance(arr, np.ndarray) and len(arr.shape) > 1:
                            max_seq_len = max(max_seq_len, arr.shape[1])
                
                # Pad and save all data for this lake
                for data_map in lake_data_list:
                    padded_data_map = {}
                
                # Handle predictions (can be dict or array)
                preds = data_map['preds']
                if isinstance(preds, dict):
                    padded_preds = {}
                    for key, arr in preds.items():
                        if isinstance(arr, np.ndarray) and len(arr.shape) > 1:
                            if arr.shape[1] < max_seq_len:
                                pad_width = [(0, 0)] * len(arr.shape)
                                pad_width[1] = (0, max_seq_len - arr.shape[1])
                                pad_value = 0.0
                                padded_arr = np.pad(arr, pad_width, mode='constant', constant_values=pad_value)
                            else:
                                padded_arr = arr
                        else:
                            padded_arr = arr
                        padded_preds[key] = padded_arr
                    padded_data_map['preds'] = padded_preds
                else:
                    # Tensor predictions
                    if isinstance(preds, np.ndarray) and len(preds.shape) > 1:
                        if preds.shape[1] < max_seq_len:
                            pad_width = [(0, 0)] * len(preds.shape)
                            pad_width[1] = (0, max_seq_len - preds.shape[1])
                            padded_data_map['preds'] = np.pad(preds, pad_width, mode='constant', constant_values=0.0)
                        else:
                            padded_data_map['preds'] = preds
                    else:
                        padded_data_map['preds'] = preds
                
                # Pad other arrays
                for name, arr in data_map.items():
                    if name == 'preds':
                        continue  # Already handled
                    
                    if isinstance(arr, np.ndarray) and len(arr.shape) > 1:
                        if arr.shape[1] < max_seq_len:
                            pad_width = [(0, 0)] * len(arr.shape)
                            pad_width[1] = (0, max_seq_len - arr.shape[1])
                            if name in ['masks']:
                                pad_value = 0
                            elif name in ['var_ids']:
                                pad_value = self.pad_value_id
                            elif name in ['datetime_strs']:
                                pad_value = b''
                            else:
                                pad_value = 0
                            padded_arr = np.pad(arr, pad_width, mode='constant', constant_values=pad_value)
                        else:
                            padded_arr = arr
                    else:
                        padded_arr = arr
                    padded_data_map[name] = padded_arr
                
                # Stream each array into HDF5
                for name, arr in padded_data_map.items():
                    if name == 'preds':
                        # Handle dictionary predictions
                        if isinstance(arr, dict):
                            if 'preds' not in save_preds_dict:
                                save_preds_dict['preds'] = {}
                            for key, val_arr in arr.items():
                                pred_key = f'preds_{key}'
                                if pred_key not in save_preds_dict['preds']:
                                    shape = (0,) + val_arr.shape[1:]
                                    maxshape = (None,) + val_arr.shape[1:]
                                    save_preds_dict['preds'][pred_key] = h5_file.create_dataset(
                                        pred_key, shape=shape, maxshape=maxshape, dtype=val_arr.dtype)
                                ds = save_preds_dict['preds'][pred_key]
                                old_size = ds.shape[0]
                                ds.resize(old_size + val_arr.shape[0], axis=0)
                                ds[old_size:old_size + val_arr.shape[0], ...] = val_arr
                        else:
                            # Tensor predictions
                            if 'preds' not in save_preds_dict:
                                shape = (0,) + arr.shape[1:]
                                maxshape = (None,) + arr.shape[1:]
                                save_preds_dict['preds'] = h5_file.create_dataset(
                                    'preds', shape=shape, maxshape=maxshape, dtype=arr.dtype)
                            ds = save_preds_dict['preds']
                            old_size = ds.shape[0]
                            ds.resize(old_size + arr.shape[0], axis=0)
                            ds[old_size:old_size + arr.shape[0], ...] = arr
                    else:
                        # Regular arrays
                        if name not in save_preds_dict:
                            if isinstance(arr, np.ndarray) and len(arr.shape) > 0:
                                shape = (0,) + arr.shape[1:] if len(arr.shape) > 1 else (0,)
                                maxshape = (None,) + arr.shape[1:] if len(arr.shape) > 1 else (None,)
                                if name in ['lake_names', 'datetime_strs']:
                                    # String arrays need special handling
                                    if arr.dtype.kind == 'S':
                                        save_preds_dict[name] = h5_file.create_dataset(
                                            name, shape=shape, maxshape=maxshape, dtype=arr.dtype)
                                    else:
                                        save_preds_dict[name] = h5_file.create_dataset(
                                            name, shape=shape, maxshape=maxshape, dtype=string_dtype(encoding='utf-8'))
                                else:
                                    save_preds_dict[name] = h5_file.create_dataset(
                                        name, shape=shape, maxshape=maxshape, dtype=arr.dtype)
                            else:
                                # Scalar or 0-d array
                                save_preds_dict[name] = h5_file.create_dataset(
                                    name, shape=(0,), maxshape=(None,), dtype=arr.dtype if hasattr(arr, 'dtype') else type(arr))
                        
                        ds = save_preds_dict[name]
                        if isinstance(arr, np.ndarray) and len(arr.shape) > 0:
                            old_size = ds.shape[0]
                            ds.resize(old_size + arr.shape[0], axis=0)
                            if len(arr.shape) > 1:
                                ds[old_size:old_size + arr.shape[0], ...] = arr
                            else:
                                ds[old_size:old_size + arr.shape[0]] = arr
                
                # Close HDF5 file for this lake
                h5_file.close()
                if self.rank == 0:
                    print(f"Saved predictions for {lake_name} (ID: {lake_id}) to {h5_path}")
        
        if self.rank == 0 and collected_data_by_lake:
            print(f"Saved predictions for {len(collected_data_by_lake)} lake(s)")

        # Finalize per-sample predictions JSON export
        if self.rank == 0 and sample_pred_fh is not None:
            try:
                sample_pred_fh.close()
            except Exception:
                pass
        if self.rank == 0 and sample_gt_fh is not None:
            try:
                sample_gt_fh.close()
            except Exception:
                pass
        
        # finalize global nll loss
        avg_batch_loss = batch_loss/len(dataloader)
        avg_batch_loss_tensor = torch.tensor(avg_batch_loss, device=self.device)
        avg_batch_loss_global = reduce_mean(avg_batch_loss_tensor, dist.get_world_size()).item()

        # Aggregate inference profiling metrics across ranks.
        infer_samples_t = torch.tensor(float(infer_samples), device=self.device)
        infer_tokens_t = torch.tensor(float(infer_valid_tokens), device=self.device)
        infer_time_t = torch.tensor(float(infer_step_time_s), device=self.device)
        infer_peak_vram_t = torch.tensor(float(infer_forward_peak_vram_gb), device=self.device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(infer_samples_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(infer_tokens_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(infer_time_t, op=dist.ReduceOp.MAX)
            dist.all_reduce(infer_peak_vram_t, op=dist.ReduceOp.MAX)
        infer_total_samples = float(infer_samples_t.item())
        infer_total_tokens = float(infer_tokens_t.item())
        infer_time_max_s = max(float(infer_time_t.item()), 1e-8)
        infer_throughput_samples_per_s = infer_total_samples / infer_time_max_s
        infer_throughput_tokens_per_s = infer_total_tokens / infer_time_max_s
        infer_peak_vram_gb = float(infer_peak_vram_t.item())

        # compute per-lake masked MSE via F.mse_loss 
        mse_by_lake = {
            lake_id: F.mse_loss(
                    torch.cat(lake_preds[lake_id]),
                    torch.cat(lake_labels[lake_id])
                ).item()
            for lake_id in lake_preds
        }
        
        # compute per-lake masked MAE via F.l1_loss
        mae_by_lake = {
            lake_id: F.l1_loss(
                    torch.cat(lake_preds[lake_id]),
                    torch.cat(lake_labels[lake_id])
                ).item()
            for lake_id in lake_preds
        }
        
        # compute per-lake CRPS (aligned with variate-wise CRPS: Monte Carlo estimate over valid target tokens)
        if saw_distribution and len(lake_crps_counts) > 0:
            crps_by_lake = {}
            for lake_id in lake_preds:
                lid = int(lake_id)
                cnt = int(lake_crps_counts.get(lid, 0))
                if cnt > 0:
                    crps_by_lake[lid] = float(lake_crps_sums[lid] / cnt)
                else:
                        crps_by_lake[lid] = float('nan')
        else:
            # Deterministic forecasts: CRPS reduces to MAE
            crps_by_lake = mae_by_lake.copy()

        if self.rank==0:
            # Handle predictions (can be dict or tensor) - similar to val_one_epoch
            if len(preds_list_2d) > 0:
                if isinstance(preds_list_2d[0], dict):
                    # Use trainer's helper function for each prediction component
                    preds_list_2d = {
                        'loc': self.trainer._pad_and_concat_tensors([p['loc'] for p in preds_list_2d]),
                        'scale': self.trainer._pad_and_concat_tensors([p['scale'] for p in preds_list_2d]),
                        'df': self.trainer._pad_and_concat_tensors([p['df'] for p in preds_list_2d]),
                        'mean': self.trainer._pad_and_concat_tensors([p['mean'] for p in preds_list_2d])
                    }
                else:
                    preds_list_2d = self.trainer._pad_and_concat_tensors(preds_list_2d)
            else:
                preds_list_2d = torch.tensor([])
            
            labels_list_2d = self.trainer._pad_and_concat_tensors(labels_list_2d)
            masks_list_2d = self.trainer._pad_and_concat_tensors(masks_list_2d)
            
            var_ids_2d_list = self.trainer._pad_and_concat_tensors(var_ids_2d_list)
            depth_val_list = self.trainer._pad_and_concat_tensors(depth_val_list)
            time_val_list = self.trainer._pad_and_concat_tensors(time_val_list)
            
            # Flatten/concatenate per-batch meta (loader returns tensor/array per batch; batches can differ in size when drop_last=False)
            if len(lake_id_list) > 0:
                if isinstance(lake_id_list[0], torch.Tensor):
                    lake_id_list = torch.cat(lake_id_list, dim=0).cpu().numpy()
                else:
                    lake_id_list = np.concatenate([np.asarray(x).ravel() for x in lake_id_list])
            else:
                lake_id_list = np.array([])
            if len(lake_names_list) > 0:
                if isinstance(lake_names_list[0], np.ndarray):
                    lake_names_list = np.concatenate(lake_names_list)
                else:
                    lake_names_list = np.concatenate([np.asarray(x).ravel() for x in lake_names_list])
            else:
                lake_names_list = np.array([])
        
       
        # Compute final metrics (lake-wise = overall metrics)
        final_mse = mse_sum / metric_count if metric_count > 0 else float('inf')
        final_mae = mae_sum / metric_count if metric_count > 0 else float('inf')
        if saw_distribution:
            final_crps = crps_sum / crps_count if crps_count > 0 else float('inf')
        else:
            # Deterministic forecasts: CRPS reduces to MAE
            final_crps = final_mae

        # Final WQL (overall + per-lake). If denom is zero, return NaN for WQL.
        wql_by_quantile = {}
        if wql_denom_sum > 0 and len(wql_quantiles) > 0:
            for qi, q in enumerate(wql_quantiles):
                wql_by_quantile[str(q)] = float(2.0 * wql_pinball_sums[qi] / (wql_denom_sum + wql_eps))
            final_wql = float(np.mean(list(wql_by_quantile.values()))) if wql_by_quantile else float('nan')
        else:
            final_wql = float('nan')
            for q in wql_quantiles:
                wql_by_quantile[str(q)] = float('nan')

        wql_by_lake = {}
        for lake_id in lake_preds:
            lid = int(lake_id)
            denom_l = float(lake_wql_denom_sums.get(lid, 0.0))
            if denom_l > 0 and len(wql_quantiles) > 0:
                vals = []
                for qi, q in enumerate(wql_quantiles):
                    vals.append(2.0 * lake_wql_pinball_sums[lid][qi] / (denom_l + wql_eps))
                wql_by_lake[lid] = float(np.mean(vals)) if vals else float('nan')
            else:
                wql_by_lake[lid] = float('nan')

        # Variable-wise WQL (overall) and per-lake per-variable
        wql_by_variate = {}
        for vid_int, denom_v in var_wql_denom_sums.items():
            denom_v = float(denom_v)
            if denom_v > 0 and len(wql_quantiles) > 0:
                vals = []
                for qi in range(len(wql_quantiles)):
                    vals.append(2.0 * var_wql_pinball_sums[vid_int][qi] / (denom_v + wql_eps))
                wql_by_variate[f"var_{int(vid_int)}"] = float(np.mean(vals)) if vals else float('nan')
            else:
                wql_by_variate[f"var_{int(vid_int)}"] = float('nan')

        wql_by_variate_by_lake = {}
        for lake_id, var_map in lake_var_wql_denom_sums.items():
            lid_int = int(lake_id)
            inner = {}
            for vid_int, denom_v in var_map.items():
                denom_v = float(denom_v)
                if denom_v > 0 and len(wql_quantiles) > 0:
                    vals = []
                    for qi in range(len(wql_quantiles)):
                        vals.append(2.0 * lake_var_wql_pinball_sums[lid_int][vid_int][qi] / (denom_v + wql_eps))
                    inner[f"var_{int(vid_int)}"] = float(np.mean(vals)) if vals else float('nan')
                else:
                    inner[f"var_{int(vid_int)}"] = float('nan')
            if inner:
                wql_by_variate_by_lake[lid_int] = inner
        
        # Compute variate-wise metrics
        mse_by_variate = {}
        mae_by_variate = {}
        crps_by_variate = {}
        # Per-lake per-variate metrics
        mse_by_variate_by_lake = {}
        mae_by_variate_by_lake = {}
        crps_by_variate_by_lake = {}
        
        # All variates (no 2D/1D distinction for irregular grids)
        for var_id in var_mse_sums:
            if var_counts[var_id] > 0:
                mse_by_variate[f"var_{var_id}"] = var_mse_sums[var_id] / var_counts[var_id]
                mae_by_variate[f"var_{var_id}"] = var_mae_sums[var_id] / var_counts[var_id]
            if saw_distribution:
                    crps_by_variate[f"var_{var_id}"] = var_crps_sums[var_id] / var_counts[var_id]
            else:
                crps_by_variate[f"var_{var_id}"] = mae_by_variate[f"var_{var_id}"]

        # Build per-lake per-variate
        for lake_id, var_sums in lake_var_mse_sums.items():
            inner_mse = {}
            inner_mae = {}
            inner_crps = {}
            for var_id, mse_sum_l in var_sums.items():
                cnt = lake_var_counts[lake_id][var_id]
                if cnt > 0:
                    inner_mse[f"var_{var_id}"] = mse_sum_l / cnt
                    inner_mae[f"var_{var_id}"] = lake_var_mae_sums[lake_id][var_id] / cnt
                if saw_distribution:
                    crps_sum_l = lake_var_crps_sums[lake_id].get(var_id, 0.0)
                    inner_crps[f"var_{var_id}"] = crps_sum_l / cnt
                else:
                    inner_crps[f"var_{var_id}"] = inner_mae[f"var_{var_id}"]
            if inner_mse:
                mse_by_variate_by_lake[int(lake_id)] = inner_mse
            if inner_mae:
                mae_by_variate_by_lake[int(lake_id)] = inner_mae
            if inner_crps:
                crps_by_variate_by_lake[int(lake_id)] = inner_crps

        # Depth-wise (per lake, per variate, per depth/bin)
        mse_by_variate_by_depth_by_lake = {}
        mae_by_variate_by_depth_by_lake = {}
        crps_by_variate_by_depth_by_lake = {}
        n_obs_by_variate_by_depth_by_lake = {}
        for lake_id, var_map in lake_var_depth_counts.items():
            lid_int = int(lake_id)
            inner_mse = {}
            inner_mae = {}
            inner_crps = {}
            inner_cnt = {}
            for var_id, depth_map in var_map.items():
                var_key = f"var_{int(var_id)}"
                inner_mse[var_key] = {}
                inner_mae[var_key] = {}
                inner_crps[var_key] = {}
                inner_cnt[var_key] = {}
                for depth_key, cnt in depth_map.items():
                    cnt_int = int(cnt)
                    if cnt_int <= 0:
                        continue
                    mse_avg = lake_var_depth_mse_sums[lake_id][var_id][depth_key] / cnt_int
                    mae_avg = lake_var_depth_mae_sums[lake_id][var_id][depth_key] / cnt_int
                    if saw_distribution:
                        crps_avg = lake_var_depth_crps_sums[lake_id][var_id].get(depth_key, 0.0) / cnt_int
                    else:
                        crps_avg = mae_avg
                    inner_mse[var_key][depth_key] = float(mse_avg)
                    inner_mae[var_key][depth_key] = float(mae_avg)
                    inner_crps[var_key][depth_key] = float(crps_avg)
                    inner_cnt[var_key][depth_key] = cnt_int

            # Drop empty leaves to keep JSON compact
            inner_mse = {vk: dv for vk, dv in inner_mse.items() if dv}
            inner_mae = {vk: dv for vk, dv in inner_mae.items() if dv}
            inner_crps = {vk: dv for vk, dv in inner_crps.items() if dv}
            inner_cnt = {vk: dv for vk, dv in inner_cnt.items() if dv}
            if inner_mse:
                mse_by_variate_by_depth_by_lake[lid_int] = inner_mse
            if inner_mae:
                mae_by_variate_by_depth_by_lake[lid_int] = inner_mae
            if inner_crps:
                crps_by_variate_by_depth_by_lake[lid_int] = inner_crps
            if inner_cnt:
                n_obs_by_variate_by_depth_by_lake[lid_int] = inner_cnt
        
        if self.rank == 0:
            print(f"Final Metrics - MSE: {final_mse:.6f}, MAE: {final_mae:.6f}, CRPS: {final_crps:.6f}, WQL: {final_wql:.6f}")
            print(f"Total valid predictions: {metric_count}")
            if mse_by_variate:
                print(f"Variate-wise metrics computed for {len(mse_by_variate)} variates")

        # t+n x Date matrix
        tplusn_by_var = {}
        if max_horizon > 0 and len(tplusn_acc) > 0:
            tplusn_cols = [f"t+{i}" for i in range(1, int(max_horizon) + 1)]
            # Optional: export per-date averages across horizons + plots
            export_tplusn_avg_csv = bool(getattr(self.cfg.evaluator, "export_tplusn_avg_csv", True))
            plot_tplusn_avg_predictions = bool(getattr(self.cfg.evaluator, "plot_tplusn_avg_predictions", True))
            for vid_int, by_date in tplusn_acc.items():
                # stable date ordering
                dates_sorted = sorted(by_date.keys())
                pred_mat = []
                gt_mat = []
                pred_cnt_mat = []
                gt_cnt_mat = []
                for d in dates_sorted:
                    cell = by_date[d]
                    ps = np.asarray(cell["pred_sum"], dtype=float)
                    pc = np.asarray(cell["pred_cnt"], dtype=int)
                    gt_sum = float(cell.get("gt_sum", 0.0))
                    gt_cnt = int(cell.get("gt_cnt", 0))

                    pr = np.full((max_horizon,), np.nan, dtype=float)
                    gt_vec = np.full((max_horizon,), np.nan, dtype=float)
                    mask_p = pc > 0
                    pr[mask_p] = ps[mask_p] / pc[mask_p]
                    if gt_cnt > 0:
                        gt_val = gt_sum / float(gt_cnt)
                        gt_vec[:] = float(gt_val)

                    pred_mat.append(pr.tolist())
                    gt_mat.append(gt_vec.tolist())
                    pred_cnt_mat.append(pc.tolist())
                    gt_cnt_mat.append([int(gt_cnt) for _ in range(int(max_horizon))])

                tplusn_by_var[f"var_{int(vid_int)}"] = {
                    "dates": dates_sorted,
                    "pred": pred_mat,
                    "gt": gt_mat,
                    "pred_cnt": pred_cnt_mat,
                    "gt_cnt": gt_cnt_mat,
                    "max_horizon": int(max_horizon),
                }

                # Also export to CSV (one file per variate) under the current trial save_dir.
                # Rows = dates, cols = t+1..t+max_horizon
                if save_dir is not None:
                    out_dir = os.path.join(save_dir, "tplusn_matrices")
                    os.makedirs(out_dir, exist_ok=True)

                    var_key = f"var_{int(vid_int)}"
                    df_pred = pd.DataFrame(pred_mat, index=dates_sorted, columns=tplusn_cols)
                    df_gt = pd.DataFrame(gt_mat, index=dates_sorted, columns=tplusn_cols)
                    df_pred_cnt = pd.DataFrame(pred_cnt_mat, index=dates_sorted, columns=tplusn_cols)
                    df_gt_cnt = pd.DataFrame(gt_cnt_mat, index=dates_sorted, columns=tplusn_cols)

                    df_pred.to_csv(os.path.join(out_dir, f"{var_key}_pred.csv"))
                    df_gt.to_csv(os.path.join(out_dir, f"{var_key}_gt.csv"))
                    df_pred_cnt.to_csv(os.path.join(out_dir, f"{var_key}_pred_cnt.csv"))
                    df_gt_cnt.to_csv(os.path.join(out_dir, f"{var_key}_gt_cnt.csv"))

                    # Also export a per-date AVG prediction across all available horizons (T+1..T+N) + plot vs GT.
                    # This matches the "avg across T+1..T+N" visualization request.
                    if export_tplusn_avg_csv or plot_tplusn_avg_predictions:
                        try:
                            # AVG across horizons for each date (row-wise mean, ignoring NaNs)
                            pred_avg = df_pred.mean(axis=1, skipna=True)
                            # GT is horizon-invariant (we fill gt_vec[:] per date), so take mean across horizons for robustness
                            gt_by_date = df_gt.mean(axis=1, skipna=True)

                            if export_tplusn_avg_csv:
                                df_avg = pd.DataFrame(
                                    {
                                        "pred_avg_across_tplus": pred_avg.astype(float),
                                        "gt": gt_by_date.astype(float),
                                    },
                                    index=df_pred.index,
                                )
                                df_avg.to_csv(os.path.join(out_dir, f"{var_key}_pred_avg_across_tplus.csv"))

                            if plot_tplusn_avg_predictions:
                                plot_dir = os.path.join(save_dir, "tplusn_plots")
                                os.makedirs(plot_dir, exist_ok=True)
                                # Backward-compat: if the index is epoch seconds, parse with unit="s".
                                idx_str = df_pred.index.astype(str)
                                idx_num = pd.to_numeric(idx_str, errors="coerce")
                                if np.isfinite(idx_num).all() and idx_num.size > 0 and float(np.nanmedian(idx_num)) > 1e8:
                                    x = pd.to_datetime(idx_num.astype("int64"), unit="s", utc=True, errors="coerce").tz_convert("UTC").tz_localize(None)
                                else:
                                    x = pd.to_datetime(idx_str, errors="coerce", utc=True).tz_convert("UTC").tz_localize(None)

                                fig, ax = plt.subplots(figsize=(14, 4))
                                ax.plot(
                                    pred_avg,
                                    color="tab:orange",
                                    linestyle="--",
                                    marker="o",
                                    linewidth=2,
                                    markersize=5,
                                    label="AVG Pred (across T+1..T+N)",
                                )
                                ax.plot(
                                    gt_by_date,
                                    color="tab:blue",
                                    linestyle="-",
                                    marker="o",
                                    linewidth=2,
                                    markersize=5,
                                    alpha=0.9,
                                    label="GT",
                                )
                                ax.set_title(f"{var_key}: per-date AVG across T+1..T+N", fontsize=12)
                                ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
                                ax.tick_params(axis="x", rotation=90, labelsize=9)
                                ax.tick_params(axis="y", labelsize=9)
                                ax.legend(loc="best", fontsize=9, framealpha=0.9)
                                plt.tight_layout()
                                plt.savefig(os.path.join(plot_dir, f"{var_key}_pred_avg_across_tplus_vs_gt.png"), dpi=180, bbox_inches="tight")
                                plt.close(fig)
                        except Exception as e:
                            if self.rank == 0:
                                print(f"[tplusn] Warning: failed to export/plot avg-across-horizons for {var_key}: {e}")

        # --- T+N metrics (overall, by horizon) computed from the date×horizon matrices ---
        # We compute these across *all variates* included in tplusn_acc.
        tplusn_metrics = {}
        tplusn_metrics_by_variate = {}
        if max_horizon > 0 and len(tplusn_acc) > 0:
            def _compute_tplusn_metrics_from_by_date_dicts(by_date_dicts):
                """
                Compute T+N metrics (overall, by horizon) from one or more `by_date` dicts where:
                  cell = {pred_sum, pred_cnt, gt_sum, gt_cnt}, each length=max_horizon (except gt_sum/gt_cnt scalars).
                Returns: dict compatible with `tplusn_metrics`.
                """
                mse_sum_h = np.zeros((max_horizon,), dtype=float)
                mae_sum_h = np.zeros((max_horizon,), dtype=float)
                cnt_h = np.zeros((max_horizon,), dtype=float)
                wql_pin_sums_h = np.zeros((len(wql_quantiles), max_horizon), dtype=float)
                wql_denom_h = np.zeros((max_horizon,), dtype=float)

                for by_date in by_date_dicts:
                    for _d, cell in by_date.items():
                        ps = np.asarray(cell["pred_sum"], dtype=float)
                        pc = np.asarray(cell["pred_cnt"], dtype=float)
                        gt_sum = float(cell.get("gt_sum", 0.0))
                        gt_cnt = float(cell.get("gt_cnt", 0.0))

                        w = np.minimum(pc, gt_cnt)
                        valid = w > 0
                        if not np.any(valid):
                            continue

                        pr = np.full((max_horizon,), np.nan, dtype=float)
                        gt_vec = np.full((max_horizon,), np.nan, dtype=float)
                        pr[pc > 0] = ps[pc > 0] / pc[pc > 0]
                        if gt_cnt > 0:
                            gt_vec[:] = gt_sum / gt_cnt

                        err = pr - gt_vec
                        ok = valid & np.isfinite(err) & np.isfinite(pr) & np.isfinite(gt_vec)
                        if not np.any(ok):
                            continue

                        mse_sum_h[ok] += w[ok] * (err[ok] ** 2)
                        mae_sum_h[ok] += w[ok] * np.abs(err[ok])
                        cnt_h[ok] += w[ok]

                        # WQL denom uses |y| (weighted)
                        wql_denom_h[ok] += w[ok] * np.abs(gt_vec[ok])
                        diff = gt_vec - pr  # y - yhat
                        for qi, q in enumerate(wql_quantiles):
                            qf = float(q)
                            pin = np.maximum(qf * diff, (qf - 1.0) * diff)
                            wql_pin_sums_h[qi, ok] += w[ok] * pin[ok]

                mse_by_h = {}
                mae_by_h = {}
                wql_by_h = {}
                for h in range(1, max_horizon + 1):
                    i = h - 1
                    if cnt_h[i] > 0:
                        mse_by_h[f"t+{h}"] = float(mse_sum_h[i] / cnt_h[i])
                        mae_by_h[f"t+{h}"] = float(mae_sum_h[i] / cnt_h[i])
                    else:
                        mse_by_h[f"t+{h}"] = float("nan")
                        mae_by_h[f"t+{h}"] = float("nan")

                    denom = float(wql_denom_h[i])
                    if denom > 0 and len(wql_quantiles) > 0:
                        vals = []
                        for qi in range(len(wql_quantiles)):
                            vals.append(float(2.0 * wql_pin_sums_h[qi, i] / (denom + wql_eps)))
                        wql_by_h[f"t+{h}"] = float(np.mean(vals)) if vals else float("nan")
                    else:
                        wql_by_h[f"t+{h}"] = float("nan")

                total_cnt = float(np.sum(cnt_h))
                total_mse = float(np.sum(mse_sum_h) / total_cnt) if total_cnt > 0 else float("nan")
                total_mae = float(np.sum(mae_sum_h) / total_cnt) if total_cnt > 0 else float("nan")

                denom_all = float(np.sum(wql_denom_h))
                if denom_all > 0 and len(wql_quantiles) > 0:
                    vals = []
                    for qi in range(len(wql_quantiles)):
                        vals.append(float(2.0 * np.sum(wql_pin_sums_h[qi, :]) / (denom_all + wql_eps)))
                    total_wql = float(np.mean(vals)) if vals else float("nan")
                else:
                    total_wql = float("nan")

                return {
                    "max_horizon": int(max_horizon),
                    "mse_by_horizon": mse_by_h,
                    "mae_by_horizon": mae_by_h,
                    "wql_by_horizon": wql_by_h,
                    "mse": total_mse,
                    "mae": total_mae,
                    "wql": total_wql,
                    "cell_count_weighted": int(total_cnt),
                }

            # Overall across all variates
            tplusn_metrics = _compute_tplusn_metrics_from_by_date_dicts(list(tplusn_acc.values()))

            # Per-variate metrics-by-horizon (requested)
            export_tplusn_metrics_by_variate = bool(getattr(self.cfg.evaluator, "export_tplusn_metrics_by_variate", True))
            if export_tplusn_metrics_by_variate:
                for vid_int, by_date in tplusn_acc.items():
                    tplusn_metrics_by_variate[f"var_{int(vid_int)}"] = _compute_tplusn_metrics_from_by_date_dicts([by_date])
        
        # Match val_one_epoch return format
        ret = {
            'loss': avg_batch_loss_global,  # NLL loss (not reported but kept for compatibility)
            'mse_by_lake': mse_by_lake,     # Lake-wise MSE
            'mae_by_lake': mae_by_lake,     # Lake-wise MAE
            'crps_by_lake': crps_by_lake,   # Lake-wise CRPS
            'wql_by_lake': wql_by_lake,     # Lake-wise WQL
            'mse': final_mse,           # Overall MSE
            'mae': final_mae,           # Overall MAE  
            'crps': final_crps,         # Overall CRPS
            'wql': final_wql,           # Overall WQL (mean over quantiles)
            'wql_quantiles': [float(q) for q in wql_quantiles],
            'wql_by_quantile': wql_by_quantile,  # string(q) -> wql(q)
            'tplusn_by_variate_date_horizon': tplusn_by_var,
            'tplusn_metrics': tplusn_metrics,
            'tplusn_metrics_by_variate': tplusn_metrics_by_variate,
            # Token counts for weighted/unweighted aggregation
            'n_obs': int(metric_count),
            'n_obs_definition': 'Number of valid target tokens (after masks) used to compute metrics.',
            'n_obs_by_lake': {int(lid): int(cnt) for lid, cnt in lake_token_counts.items()},
            'infer_compute_time_s': infer_time_max_s,
            'infer_peak_vram_gb': infer_peak_vram_gb,
            'infer_throughput_samples_per_s': infer_throughput_samples_per_s,
            'infer_throughput_tokens_per_s': infer_throughput_tokens_per_s,
        }
        
        # Variate-wise metrics (always)
        ret['mse_by_variate'] = mse_by_variate
        ret['mae_by_variate'] = mae_by_variate
        ret['crps_by_variate'] = crps_by_variate
        ret['wql_by_variate'] = wql_by_variate
        
        # Per-lake per-variate (always)
        ret['mse_by_variate_by_lake'] = mse_by_variate_by_lake
        ret['mae_by_variate_by_lake'] = mae_by_variate_by_lake
        ret['crps_by_variate_by_lake'] = crps_by_variate_by_lake
        ret['wql_by_variate_by_lake'] = wql_by_variate_by_lake
        ret['n_obs_by_variate'] = {f"var_{int(var_id)}": int(cnt) for var_id, cnt in var_counts.items()}
        ret['n_obs_by_variate_by_lake'] = {
            int(lid): {f"var_{int(vid)}": int(cnt) for vid, cnt in vmap.items()}
            for lid, vmap in lake_var_counts.items()
        }

        # Depth-wise metrics (always)
        ret['mse_by_variate_by_depth_by_lake'] = mse_by_variate_by_depth_by_lake
        ret['mae_by_variate_by_depth_by_lake'] = mae_by_variate_by_depth_by_lake
        ret['crps_by_variate_by_depth_by_lake'] = crps_by_variate_by_depth_by_lake
        ret['n_obs_by_variate_by_depth_by_lake'] = n_obs_by_variate_by_depth_by_lake
        # Depth-wise metrics aggregated across variates (per lake, per depth/bin)
        mse_by_depth_by_lake = {}
        mae_by_depth_by_lake = {}
        crps_by_depth_by_lake = {}
        n_obs_by_depth_by_lake = {}
        for lake_id, depth_map in lake_depth_counts.items():
            lid_int = int(lake_id)
            mse_by_depth_by_lake[lid_int] = {}
            mae_by_depth_by_lake[lid_int] = {}
            crps_by_depth_by_lake[lid_int] = {}
            n_obs_by_depth_by_lake[lid_int] = {}
            for depth_key, cnt in depth_map.items():
                cnt_int = int(cnt)
                if cnt_int <= 0:
                    continue
                mse_avg = lake_depth_mse_sums[lake_id][depth_key] / cnt_int
                mae_avg = lake_depth_mae_sums[lake_id][depth_key] / cnt_int
                if saw_distribution:
                    crps_avg = lake_depth_crps_sums[lake_id].get(depth_key, 0.0) / cnt_int
                else:
                    crps_avg = mae_avg
                mse_by_depth_by_lake[lid_int][depth_key] = float(mse_avg)
                mae_by_depth_by_lake[lid_int][depth_key] = float(mae_avg)
                crps_by_depth_by_lake[lid_int][depth_key] = float(crps_avg)
                n_obs_by_depth_by_lake[lid_int][depth_key] = cnt_int
            # Drop empties for compactness
            mse_by_depth_by_lake[lid_int] = {k: v for k, v in mse_by_depth_by_lake[lid_int].items() if v is not None}
            mae_by_depth_by_lake[lid_int] = {k: v for k, v in mae_by_depth_by_lake[lid_int].items() if v is not None}
            crps_by_depth_by_lake[lid_int] = {k: v for k, v in crps_by_depth_by_lake[lid_int].items() if v is not None}
            n_obs_by_depth_by_lake[lid_int] = {k: v for k, v in n_obs_by_depth_by_lake[lid_int].items() if v is not None}

        ret['mse_by_depth_by_lake'] = mse_by_depth_by_lake
        ret['mae_by_depth_by_lake'] = mae_by_depth_by_lake
        ret['crps_by_depth_by_lake'] = crps_by_depth_by_lake
        ret['n_obs_by_depth_by_lake'] = n_obs_by_depth_by_lake
        ret['depth_bin_size_m'] = self.depth_bin_size_m
        ret['depth_round_decimals'] = int(self.depth_round_decimals)
        ret['depth_key_units'] = "m"
        if lake_id_to_depth_minmax is not None:
            # JSON-friendly: keys as strings and values as floats
            ret['depth_minmax_by_lake'] = {
                str(int(lid)): {"min_depth": float(mm["min_depth"]), "max_depth": float(mm["max_depth"])}
                for lid, mm in lake_id_to_depth_minmax.items()
                if mm is not None and "min_depth" in mm and "max_depth" in mm
            }
        
        # Add plotting data (similar to val_one_epoch)
        if len(preds_list_2d) > 0:
            # preds_list_2d is already concatenated above, just assign it
            ret['preds2d'] = preds_list_2d
            
            ret['labels2d'] = labels_list_2d
            ret['masks2d'] = masks_list_2d
            ret['var_ids_2d'] = var_ids_2d_list
            ret['depth_vals'] = depth_val_list
            ret['time_vals'] = time_val_list
            ret['datetime_strs'] = self.trainer._pad_and_concat_numpy_arrays(datetime_list) if len(datetime_list) > 0 else np.array([])
            ret['lake_names'] = lake_names_list
            ret['lake_ids'] = lake_id_list
            
            # Add context (input X) data for baseline comparison
            if len(context_seq_list) > 0:
                ret['context_seq'] = self.trainer._pad_and_concat_tensors(context_seq_list)
                ret['context_var_ids'] = self.trainer._pad_and_concat_tensors(context_var_ids_list)
                ret['context_depth_vals'] = self.trainer._pad_and_concat_tensors(context_depth_vals_list)
                ret['context_time_vals'] = self.trainer._pad_and_concat_tensors(context_time_vals_list)
                ret['context_datetime_strs'] = self.trainer._pad_and_concat_numpy_arrays(context_datetime_list) if len(context_datetime_list) > 0 else np.array([])
                ret['context_mask'] = self.trainer._pad_and_concat_tensors(context_mask_list)
        
        # Add lake_id_to_name mapping from all batches (not just plotting batches)
        ret['lake_id_to_name'] = lake_id_to_name_all
        
        # Add lake_to_variates mapping (convert sets to sorted lists for JSON serialization)
        ret['lake_to_variates'] = {str(lid): sorted(list(vars_set)) for lid, vars_set in lake_to_variates.items()}
        
        # Add lake embedding trajectory data if enabled
        if self.lake_embed_traj and len(lake_embed_traj_list) > 0:
            ret['lake_embed_traj'] = torch.stack(lake_embed_traj_list)  # (N, d_temporal)
            ret['lake_embed_traj_dates'] = lake_embed_traj_dates  # List of datetime strings
            ret['lake_embed_traj_lake_ids'] = lake_embed_traj_lake_ids  # List of lake IDs
            ret['lake_embed_traj_lake_names'] = lake_embed_traj_lake_names  # List of lake names

        # Thermocline finalize (rank 0 only): write dedicated JSON + plots for the requested window
        try:
            if thermo_cfg is not None and thermo_state is not None:
                self._thermo_finalize(thermo_cfg, thermo_state)
        except Exception as e:
            if self.rank == 0:
                print(f"[thermocline] Warning: failed to finalize thermocline metrics/plots: {e}")

        # Finalize BL experiment (plots + metrics) outside the main loop
        if self.BL and bl_cfg.enabled:
            self._bl_finalize(bl_cfg, bl_state, ret)

        return ret

    def plot_predictions(self, elements, flag, it, save_dir=None):    
        pretty_print(f"Prediction Visualization :: {flag}")
        preds=elements['preds2d']
        gt=elements['labels2d']
        var_ids=elements["var_ids_2d"]
        depth_vals=elements['depth_vals'] # TODO: update depth vals to get raw/original depth values
        time_vals=elements['time_vals']
        datetime_raw_vals = elements['datetime_strs']
        gt_mask=elements['masks2d']
        
        # Extract context (input X) data if available (for baseline comparison)
        context_seq = elements.get('context_seq', None)
        context_var_ids = elements.get('context_var_ids', None)
        context_depth_vals = elements.get('context_depth_vals', None)
        context_time_vals = elements.get('context_time_vals', None)
        context_datetime_strs = elements.get('context_datetime_strs', None)
        context_mask = elements.get('context_mask', None)

        # Best-effort lake name for plot titles
        lake_name_for_title = None
        if 'lake_names' in elements and elements['lake_names'] is not None and len(elements['lake_names']) > 0:
            lake_name_for_title = elements['lake_names'][0]
            if isinstance(lake_name_for_title, (list, np.ndarray)) and len(lake_name_for_title) > 0:
                lake_name_for_title = lake_name_for_title[0]
            if isinstance(lake_name_for_title, bytes):
                lake_name_for_title = lake_name_for_title.decode('utf-8')
            lake_name_for_title = str(lake_name_for_title)
        
        # Handle total_samples for both dict and tensor predictions
        if isinstance(preds, dict):
            total_samples = preds['mean'].shape[0]
        else:
            total_samples = preds.shape[0] if preds is not None else 0

        id_to_var = self.data.id_to_var

        # num_plot_batches controls how many windows we collected for plotting.
        # If -1, we collected the full test stream, so treat "num_windows" as total_samples.
        num_windows = self.cfg.num_plot_batches
        try:
            if int(num_windows) < 0:
                num_windows = total_samples
        except Exception:
            pass
        num_samples = self.cfg.plot_num_samples

        # Check if we have data to plot
        if preds is not None and gt is not None:
            start = 0
            idx = np.arange(start, start+total_samples, 1)
            plt_idx = np.floor(np.linspace(0, 1, 1))
            
            # Set up plots directory for saving all plots
            if save_dir is not None:
                eval_dir = save_dir
                plots_dir = os.path.join(os.path.dirname(eval_dir), "PLOTS", os.path.basename(eval_dir))
                os.makedirs(plots_dir, exist_ok=True)
            else:
                plots_dir = None
            
            try:
                
                # breakpoint()
                # Check if we should use heatmap plotting (regular grid forecasting)
                # if self.regular_grid_forecasting and self.enable_heatmap_plotting:
                    # Use heatmap plotting for regular grid data
                print("\n\nPlotting regular grid heatmaps\n\n")
                regular_grid_save_path = os.path.join(plots_dir, f"regular_grid_heatmaps_{flag}.png") if plots_dir is not None else None
                if self.thermocline:
                    # Water Temp only, 1x2 (GT vs Pred)
                    therm_save_path = os.path.join(plots_dir, f"thermocline_watertemp_heatmap_{flag}.png") if plots_dir is not None else regular_grid_save_path
                    wt_id = self._resolve_var_id(var_name=self.thermocline_var_name_substr, id_to_var=id_to_var)
                    self._plot_regular_grid_heatmaps(
                        gt=gt, 
                        preds=preds, 
                        time_vals=time_vals, 
                        datetime_raw_vals=datetime_raw_vals, 
                        depth_vals=depth_vals,
                        var_ids=var_ids, 
                        gt_mask=gt_mask, 
                        id_to_var=id_to_var,
                        sample_idx=idx, 
                        epoch=it, 
                        train_or_val=flag,
                        save_path=therm_save_path,
                        plot_var_ids=[wt_id] if wt_id is not None else None,
                        max_depths_per_var=int(self.thermocline_max_depths),
                    )
                    if self.thermocline_skip_irregular_plots:
                        return
                    
                # else:
                print("\n\nPlotting irregular grid forecasts\n\n")
                # Use irregular plotting for irregular grid data
                irregular_grid_save_path = os.path.join(plots_dir, f"forecasts_{self.pred_len}_{lake_name_for_title}.pdf") if plots_dir is not None else None
                # self.irregular_plotter.plot_forecast_irregular_grid(
                #     gt_row=gt,
                #     preds_row=preds,
                #     time_vals_row=time_vals,
                #     datetime_raw_vals=datetime_raw_vals,
                #     depth_vals_row=depth_vals,
                #     var_ids_row=var_ids,
                #     mask_row=gt_mask,
                #     feature_dict=id_to_var,
                #     sample_idx=idx,
                #     plt_idx=plt_idx,
                #     epoch=it,
                #     train_or_val=flag,
                #     plot_type=self.cfg.forecast_plot_type,
                #     max_features=14,
                #     max_depths_per_feature=14,
                #     filter_first_pred=True,
                #     depth_units="norm",
                #     plot_interval=self.cfg.plot_interval,
                #     save_path=irregular_grid_save_path
                # )

                self.irregular_plotter.plot_forecast_irregular_grid_single_depth_var(
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
                    depth_units="m",
                    var_names_subset=getattr(getattr(self.cfg, "plotter", None), "var_names_subset", ["WaterTemp_C"]),
                    depth_index=getattr(getattr(self.cfg, "plotter", None), "depth_index", 1),
                    pred_len=self.pred_len,
                    lake_name=lake_name_for_title,
                    plot_interval=self.cfg.plot_interval,
                    save_path=irregular_grid_save_path,
                    depth_name=getattr(getattr(self.cfg, "plotter", None), "depth_name", 1.5),
                    pred_offset=getattr(getattr(self.cfg, "plotter", None), "pred_offset", 0.0),
                    plot_full_timeseries=bool(getattr(getattr(self.cfg, "plotter", None), "plot_full_timeseries", True)),
                    year=getattr(getattr(self.cfg, "plotter", None), "year", None),
                    plot_single_sample=bool(getattr(getattr(self.cfg, "plotter", None), "plot_single_sample", False)),
                    select_single_sample_by_max_context=bool(getattr(getattr(self.cfg, "plotter", None), "select_single_sample_by_max_context", False)),
                    select_single_sample_by_pred_variance=bool(getattr(getattr(self.cfg, "plotter", None), "select_single_sample_by_pred_variance", False)),
                    # Always show axis labels/ticks for interpretability (override config defaults).
                    show_xticks=True,
                    show_xlabel=True,
                    # Relax y-axis constraints: allow per-variable auto scaling.
                    ymin=None,
                    ymax=None,
                    axis_label_fontsize=int(getattr(getattr(self.cfg, "plotter", None), "axis_label_fontsize", 20)),
                    tick_labelsize=int(getattr(getattr(self.cfg, "plotter", None), "tick_labelsize", 20)),
                    ytick_step=None,
                    show_title=bool(getattr(getattr(self.cfg, "plotter", None), "show_title", True)),
                    show_ylabel=True,
                    # Context (input X) data for baseline comparison
                    context_seq_row=context_seq,
                    context_var_ids_row=context_var_ids,
                    context_depth_vals_row=context_depth_vals,
                    context_time_vals_row=context_time_vals,
                    context_datetime_strs=context_datetime_strs,
                    context_mask_row=context_mask
                )

                # Optional: plot stitched T+n average (e.g., avg of T+2..T+14) for the same var/depth selection.
                if bool(getattr(getattr(self.cfg, "plotter", None), "plot_tplusn_avg", False)):
                    self.irregular_plotter.plot_forecast_irregular_grid_single_depth_var_tplusn_average(
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
                        title_prefix=getattr(getattr(self.cfg, "plotter", None), "tplusn_avg_title_prefix", "T+n Avg Forecast"),
                        plot_type=self.cfg.forecast_plot_type,
                        max_features=14,
                        max_depths_per_feature=14,
                        depth_units="m",
                        var_names_subset=getattr(getattr(self.cfg, "plotter", None), "var_names_subset", ["WaterTemp_C"]),
                        depth_index=getattr(getattr(self.cfg, "plotter", None), "depth_index", 1),
                        pred_len=self.pred_len,
                        lake_name=lake_name_for_title,
                        save_path=irregular_grid_save_path,
                        depth_name=getattr(getattr(self.cfg, "plotter", None), "depth_name", 1.5),
                        pred_offset=getattr(getattr(self.cfg, "plotter", None), "pred_offset", 0.0),
                        horizon_start=int(getattr(getattr(self.cfg, "plotter", None), "tplusn_avg_horizon_start", 1)),
                        horizon_end=int(getattr(getattr(self.cfg, "plotter", None), "tplusn_avg_horizon_end", 14)),
                        require_all_horizons=bool(getattr(getattr(self.cfg, "plotter", None), "tplusn_avg_require_all_horizons", False)),
                        use_common_intersection=bool(getattr(getattr(self.cfg, "plotter", None), "tplusn_avg_use_common_intersection", False)),
                        intersection_mode=str(getattr(getattr(self.cfg, "plotter", None), "tplusn_avg_intersection_mode", "range")),
                        filename_prefix=str(getattr(getattr(self.cfg, "plotter", None), "tplusn_avg_filename_prefix", "tplusn_avg")),
                        # Context (input X) data for baseline comparison (passed through for compatibility)
                        context_seq_row=context_seq,
                        context_var_ids_row=context_var_ids,
                        context_depth_vals_row=context_depth_vals,
                        context_time_vals_row=context_time_vals,
                        context_datetime_strs=context_datetime_strs,
                        context_mask_row=context_mask
                    )

                # Also save multi-horizon stitched forecasts (T+1/T+7/T+14/T+21) as a 2x2 figure.
                # Uses the same base save_path but the plotter will prefix the filename with "t+N_preds_".
                # Only plot if T+N metrics are enabled
                if self.do_t_plus_n_metrics:
                    self.irregular_plotter.plot_forecast_irregular_grid_single_depth_var_multi_horizon(
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
                        # IMPORTANT: keep all horizon points so we can stitch T+7/T+14/T+21 correctly
                        filter_first_pred=False,
                        depth_units="norm",
                        var_names_subset=getattr(getattr(self.cfg, "plotter", None), "var_names_subset", ["WaterTemp_C"]),
                        depth_index=getattr(getattr(self.cfg, "plotter", None), "depth_index", 1),
                        pred_len=self.pred_len,
                        lake_name=lake_name_for_title,
                        plot_interval=self.cfg.plot_interval,
                        save_path=irregular_grid_save_path,
                        depth_name=getattr(getattr(self.cfg, "plotter", None), "depth_name", 1.5),
                        pred_offset=getattr(getattr(self.cfg, "plotter", None), "pred_offset", 0.0),
                        horizons=(1, 7, 14, 21),
                        use_common_intersection=True,
                        intersection_mode="range",
                        filename_prefix="t+N_preds",
                        # Context (input X) data for baseline comparison (accepted for compatibility)
                        context_seq_row=context_seq,
                        context_var_ids_row=context_var_ids,
                        context_depth_vals_row=context_depth_vals,
                        context_time_vals_row=context_time_vals,
                        context_datetime_strs=context_datetime_strs,
                        context_mask_row=context_mask
                    )
                
                # Plot prediction error heatmaps (if enabled)
            #     if self.enable_prediction_error_heatmaps:
            #         print("\n\nPlotting prediction error heatmaps\n\n")
            #         prediction_error_save_path = os.path.join(plots_dir, f"prediction_error_heatmaps_{flag}_epoch_{it}.png") if plots_dir is not None else None
                    
            #         # self.irregular_plotter.plot_prediction_error_heatmaps(
            #         #     gt_row=gt,
            #         #     preds_row=preds,
            #         #     time_vals_row=time_vals,
            #         #     datetime_raw_vals=datetime_raw_vals,
            #         #     depth_vals_row=depth_vals,
            #         #     var_ids_row=var_ids,
            #         #     mask_row=gt_mask,
            #         #     feature_dict=id_to_var,
            #         #     sample_idx=idx,
            #         #     epoch=it,
            #         #     train_or_val=flag,
            #         #     depth_min=self.depth_min,
            #         #     depth_max=self.depth_max,
            #         #     save_path=prediction_error_save_path,
            #         #     pred_len=self.pred_len
            #         # )
            #         print("\n\nPlotting T+n prediction errors for date range\n\n")
            #         # Get lake name if available - use the lake from the selected sample (most dense)
            #         lake_name = None
            #         if 'lake_names' in elements and 'lake_ids' in elements:
            #             # Find sample with most observations (same logic as plot_Tn_predictions_for_day)
            #             if isinstance(gt, dict):
            #                 gt_for_count = gt['mean'] if 'mean' in gt else list(gt.values())[0]
            #             else:
            #                 gt_for_count = gt
                        
            #             if hasattr(gt_mask, 'detach'):
            #                 mask_np = gt_mask.detach().cpu().numpy()
            #             else:
            #                 mask_np = np.asarray(gt_mask)
                        
            #             valid_mask = mask_np.astype(bool)
            #             sample_observation_counts = [valid_mask[b].sum() for b in range(valid_mask.shape[0])]
                        
            #             if max(sample_observation_counts) > 0:
            #                 selected_sample_idx = np.argmax(sample_observation_counts)
            #                 # Get lake_id for the selected sample
            #                 lake_ids = elements['lake_ids']
            #                 if len(lake_ids) > selected_sample_idx:
            #                     selected_lake_id = int(lake_ids[selected_sample_idx])
            #                     # Try to get lake name from lake_id_to_name mapping first
            #                     if 'lake_id_to_name' in elements and selected_lake_id in elements['lake_id_to_name']:
            #                         lake_name = elements['lake_id_to_name'][selected_lake_id]
            #                     # Fallback to lake_names array
            #                     elif len(elements['lake_names']) > selected_sample_idx:
            #                         lake_name = elements['lake_names'][selected_sample_idx]
            #                         if isinstance(lake_name, bytes):
            #                             lake_name = lake_name.decode('utf-8')
            #         elif 'lake_names' in elements and len(elements['lake_names']) > 0:
            #             # Fallback: Get the first lake name
            #             lake_name = elements['lake_names'][0]
            #             if isinstance(lake_name, bytes):
            #                 lake_name = lake_name.decode('utf-8')
                    
            #         # Plot for each variable separately
            #         for var_id, var_name in id_to_var.items():
            #             tn_predictions_save_path = os.path.join(plots_dir, f"tn_predictions_{var_name}_{flag}_epoch_{it}.png") if plots_dir is not None else None
            #             self.irregular_plotter.plot_Tn_predictions_for_day(
            #                 preds_row=preds,
            #                 gt_row=gt,
            #                 datetime_raw_vals=datetime_raw_vals,
            #                 depth_vals_row=depth_vals,
            #                 var_ids_row=var_ids,
            #                 mask_row=gt_mask,
            #                 feature_dict=id_to_var,
            #                 epoch=it,
            #                 train_or_val=flag,
            #                 var_name=var_name,
            #                 lake_name=lake_name,
            #                 start_date=None,  # Auto-select (prefers May-June)
            #                 end_date=None,   # Auto-select (30 days after start)
            #                 horizons=[1, 7, 14, 30],  # T+1, T+7, T+14, T+30
            #                 max_depths=20,
            #                 depth_min=self.depth_min,
            #                 depth_max=self.depth_max,
            #                 save_path=tn_predictions_save_path
            #             )
                    
            #         # Plot single sample forecast for a specific depth
            #         print("\n\nPlotting single sample forecast with context window\n\n")
            #         # Find sample with most observations
            #         if isinstance(gt, dict):
            #             gt_for_count = gt['mean'] if 'mean' in gt else list(gt.values())[0]
            #         else:
            #             gt_for_count = gt
                    
            #         if hasattr(gt_for_count, 'detach'):
            #             gt_np = gt_for_count.detach().cpu().numpy()
            #         else:
            #             gt_np = np.asarray(gt_for_count)
                    
            #         if hasattr(gt_mask, 'detach'):
            #             mask_np = gt_mask.detach().cpu().numpy()
            #         else:
            #             mask_np = np.asarray(gt_mask)
                    
            #         valid_mask = mask_np.astype(bool)
            #         sample_observation_counts = [valid_mask[b].sum() for b in range(valid_mask.shape[0])]
            #         selected_sample_idx = np.argmax(sample_observation_counts)
                    
            #         # Get lake name for the selected sample
            #         single_sample_lake_name = lake_name
            #         if 'lake_ids' in elements and 'lake_id_to_name' in elements:
            #             lake_ids = elements['lake_ids']
            #             if len(lake_ids) > selected_sample_idx:
            #                 selected_lake_id = int(lake_ids[selected_sample_idx])
            #                 if selected_lake_id in elements['lake_id_to_name']:
            #                     single_sample_lake_name = elements['lake_id_to_name'][selected_lake_id]
                    
            #         # Use a simple depth value
            #         target_depth = 2.0
                    
            #         # Plot for first variable
            #         first_var_name = list(id_to_var.values())[0]
            #         single_sample_save_path = os.path.join(plots_dir, f"single_sample_forecast_{first_var_name}_depth_{target_depth:.1f}_{flag}_epoch_{it}.png") if plots_dir is not None else None
            #         self.irregular_plotter.plot_single_sample_forecast(
            #             inputs_row=gt,
            #             gt_row=gt,
            #             preds_row=preds,
            #             time_vals_row=time_vals,
            #             datetime_raw_vals=datetime_raw_vals,
            #             depth_vals_row=depth_vals,
            #             var_ids_row=var_ids,
            #             mask_row=gt_mask,
            #             feature_dict=id_to_var,
            #             sample_idx=selected_sample_idx,
            #             var_name=first_var_name,
            #             target_depth=target_depth,
            #             context_len=None,
            #             horizons=[1, 7, 14, 30],
            #             confidence_level=0.95,
            #             epoch=it,
            #             train_or_val=flag,
            #             lake_name=single_sample_lake_name,
            #             save_path=single_sample_save_path
            #         )
            except Exception as e:
                if self.rank == 0:
                    print(f"Skipping plot due to error: {e}")
        else:
            print("No data available for plotting")

    def _plot_regular_grid_heatmaps(self, gt, preds, time_vals, datetime_raw_vals, 
                                   depth_vals, var_ids, gt_mask, id_to_var, 
                                   sample_idx, epoch, train_or_val, save_path=None,
                                   plot_var_ids: Optional[list] = None,
                                   max_depths_per_var: int = 20):
        """
        Plot regular grid heatmaps for spatio-temporal forecasting
        
        Args:
            gt: Ground truth data
            preds: Predictions (can be dict with uncertainty or tensor)
            time_vals: Time values
            datetime_raw_vals: Datetime values
            depth_vals: Depth values
            var_ids: Variable IDs
            gt_mask: Ground truth mask
            id_to_var: Variable ID to name mapping
            sample_idx: Sample indices
            epoch: Current epoch
            train_or_val: 'train' or 'val'
            save_path: Optional path to save the figure
        """
        # Convert to numpy for plotting
        if hasattr(gt, 'cpu'):
            gt_np = gt.cpu().numpy()
        else:
            gt_np = gt
            
        if hasattr(preds, 'cpu'):
            if isinstance(preds, dict):
                preds_np = {k: v.cpu().numpy() if hasattr(v, 'cpu') else v for k, v in preds.items()}
            else:
                preds_np = preds.cpu().numpy()
        else:
            preds_np = preds
            
        if hasattr(var_ids, 'cpu'):
            var_ids_np = var_ids.cpu().numpy()
        else:
            var_ids_np = var_ids
            
        if hasattr(gt_mask, 'cpu'):
            gt_mask_np = gt_mask.cpu().numpy()
        else:
            gt_mask_np = gt_mask
        
        # Convert depth_vals to numpy
        if hasattr(depth_vals, 'cpu'):
            depth_vals_np = depth_vals.cpu().numpy()
        else:
            depth_vals_np = depth_vals
        
        # Convert time_vals to numpy
        if hasattr(time_vals, 'cpu'):
            time_vals_np = time_vals.cpu().numpy()
        else:
            time_vals_np = time_vals
        
        # Convert sample_idx to numpy
        if hasattr(sample_idx, 'cpu'):
            sample_idx_np = sample_idx.cpu().numpy()
        else:
            sample_idx_np = sample_idx

        
        # Get dimensions from prediction shape (should be regular grid)
        if isinstance(preds_np, dict):
            pred_mean_np = preds_np['mean']
        else:
            pred_mean_np = preds_np
        
        B = pred_mean_np.shape[0]
        valid_mask = gt_mask_np.astype(bool)
        
        # Prepare first datetime per sample for x-axis
        dt_array = np.asarray(datetime_raw_vals)
        first_datetimes = []
        for b in range(B):
            sel = valid_mask[b]
            if sel.any():
                try:
                    dts = pd.to_datetime(dt_array[b][sel])
                    first_datetimes.append(dts.min())
                except Exception:
                    first_datetimes.append(None)
            else:
                first_datetimes.append(None)
        
        # Unique variables across valid tokens
        unique_vars = np.unique(var_ids_np[valid_mask])
        if plot_var_ids is not None:
            # Filter to requested var ids (drop Nones)
            wanted = [int(v) for v in plot_var_ids if v is not None]
            plot_vars = np.array([v for v in wanted if v in set(unique_vars.tolist())], dtype=unique_vars.dtype)
        else:
            plot_vars = unique_vars[:6]
        if len(plot_vars) == 0:
            print("No valid variables to plot in heatmap")
            return
        
        fig, axes = plt.subplots(2, len(plot_vars), figsize=(18,6), squeeze=False)
        
        for vi, var_id in enumerate(plot_vars):
            depths_for_var = np.unique(depth_vals_np[(var_ids_np == var_id) & valid_mask])
            depths_for_var = np.sort(depths_for_var)[: max(2, int(max_depths_per_var))]
            Dn = len(depths_for_var)
            gt_mat = np.full((Dn, B), np.nan)
            pred_mat = np.full((Dn, B), np.nan)
            
            # Optional inverse scaling params per variable (removed - denormalization handled earlier if denorm_eval=True)
            mu, sigma = None, None
            
            for b in range(B):
                sel = valid_mask[b]
                if not sel.any():
                    continue
                for di, depth_val in enumerate(depths_for_var):
                    idxs = np.where(sel & (var_ids_np[b] == var_id) & (depth_vals_np[b] == depth_val))[0]
                    if idxs.size == 0:
                        continue
                    try:
                        dts = pd.to_datetime(dt_array[b][idxs])
                        earliest_idx = idxs[np.argmin(dts.values)]
                    except Exception:
                        earliest_idx = idxs[0]
                    gt_val = gt_np[b, earliest_idx]
                    pred_val = pred_mean_np[b, earliest_idx]
                    # Inverse normalize if stats available
                    if mu is not None:
                        gt_val = gt_val * sigma + mu
                        pred_val = pred_val * sigma + mu
                    gt_mat[di, b] = gt_val
                    pred_mat[di, b] = pred_val
            
            # Compute shared colorbar limits from both GT and predictions
            gt_masked = np.ma.masked_invalid(gt_mat)
            pred_masked = np.ma.masked_invalid(pred_mat)
            vmin = min(np.nanmin(gt_masked), np.nanmin(pred_masked))
            vmax = max(np.nanmax(gt_masked), np.nanmax(pred_masked))
            
            # Plot GT heatmap
            ax_gt = axes[vi, 0]
            cmap_gt = plt.cm.get_cmap('coolwarm').copy()
            try:
                cmap_gt.set_bad(color='#ffcccc')
            except Exception:
                pass
            im1 = ax_gt.imshow(gt_masked, aspect='auto', cmap=cmap_gt, origin='lower', vmin=vmin, vmax=vmax)

            var_name = id_to_var.get(int(var_id), f"Variable {int(var_id)}")
            ax_gt.set_title(f"Ground-Truth - {var_name}")
            ax_gt.set_ylabel("Depth (m)")
            ax_gt.set_xticks(np.arange(B)[:: max(1, B // 10)])
            # Depth tick labels from original values
            ax_gt.set_yticks(np.arange(Dn))
            try:
                if self.depth_min is not None and self.depth_max is not None:
                    depth_labels = [f"{(float(d) * (self.depth_max - self.depth_min) + self.depth_min):.2f}" for d in depths_for_var]
                else:
                    depth_labels = [f"{float(d):.2f}" for d in depths_for_var]
            except Exception:
                depth_labels = [str(d) for d in depths_for_var]
            ax_gt.set_yticklabels(depth_labels)
            try:
                date_labels = pd.to_datetime(first_datetimes).strftime('%Y-%m-%d').tolist()
            except Exception:
                date_labels = [str(x) for x in first_datetimes]
            xt = ax_gt.get_xticks().astype(int)
            ax_gt.set_xticklabels([date_labels[i] if 0 <= i < len(date_labels) else '' for i in xt], rotation=90)
            fig.colorbar(im1, ax=ax_gt, fraction=0.046, pad=0.04)
            
            # Plot Pred heatmap
            ax_pr = axes[vi, 1]
            cmap_pr = plt.cm.get_cmap('coolwarm').copy()
            try:
                cmap_pr.set_bad(color='#ffcccc')
            except Exception:
                pass
            im2 = ax_pr.imshow(pred_masked, aspect='auto', cmap=cmap_pr, origin='lower', vmin=vmin, vmax=vmax)
            ax_pr.set_title(f"Predictions - {var_name}")
            ax_pr.set_ylabel("Depth (m)")
            ax_pr.set_yticks(np.arange(Dn))
            ax_pr.set_yticklabels(depth_labels)
            ax_pr.set_xticks(np.arange(B)[:: max(1, B // 10)])
            xt2 = ax_pr.get_xticks().astype(int)
            ax_pr.set_xticklabels([date_labels[i] if 0 <= i < len(date_labels) else '' for i in xt2], rotation=90)
            fig.colorbar(im2, ax=ax_pr, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        # Save to disk if save_path is provided
        if save_path is not None:
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            plt.savefig(save_path, dpi=180, bbox_inches='tight')
            print(f"Saved regular grid heatmaps to {save_path}")
        
        wandb.log({f"Regular Grid Heatmaps ({train_or_val})": wandb.Image(plt)})
        plt.close()
        return

    def run(self, datasets, flag, plot_datasets, scaling=None):
        """
        Runs evaluation over 'num_trials' trials, computes mean and std of RMSE,
        saves predictions incrementally per trial, and logs summary to W&B.
        """
        # Experimental branch: only extract embeddings and exit
        if getattr(self.cfg.evaluator, "experiment", False):
            if self.rank==0:
                pretty_print("Experiment mode: extracting representations (no metrics computation).")
            loader = build_dataloader(datasets, 
                                    self.cfg.dataloader,
                                    self.pad_value_id, 
                                    self.pad_value_default)
            # Choose output directory
            out_dir = os.path.join(self.cfg.evaluator.output_dir, f"{self.cfg.run_name}_embeddings")
            os.makedirs(out_dir, exist_ok=True)
            extract_embeddings_from_loader(cfg=self.cfg,
                                           model=self.model,
                                           loader=loader,
                                           output_dir=out_dir,
                                           device=self.device)
            return {"embeddings_output_dir": out_dir}

        if self.rank==0:
            config = init_wandb(self.trainer.trainer, self.cfg.task_name)

        loader = build_dataloader(datasets, 
                                self.cfg.dataloader,
                                self.pad_value_id, 
                                self.pad_value_default)
        trial_mses    = []
        trial_maes    = []
        trial_crpss   = []
        trial_wqls    = []
        trial_infer_throughput_samples = []
        trial_infer_throughput_tokens = []
        trial_infer_peak_vram_gb = []
        trial_infer_compute_time_s = []
        per_trial_lake_mse = []
        per_trial_lake_mae = []
        per_trial_lake_crps = []
        per_trial_lake_wql = []
        per_trial_variate_mse = []
        per_trial_variate_mae = []
        per_trial_variate_crps = []
        per_trial_variate_wql = []
        # per-lake per-variate (per trial)
        per_trial_lake_variate_mse = []
        per_trial_lake_variate_mae = []
        per_trial_lake_variate_crps = []
        per_trial_lake_variate_wql = []
        all_preds2d     = []
        # all_preds1d     = []
        all_depths      = []
        lake_id_to_name    = {}
        lake_mse_trials    = defaultdict(list)
        lake_mae_trials    = defaultdict(list)
        lake_crps_trials   = defaultdict(list)
        lake_wql_trials    = defaultdict(list)
        variate_mse_trials = defaultdict(list)
        variate_mae_trials = defaultdict(list)
        variate_crps_trials = defaultdict(list)
        variate_wql_trials = defaultdict(list)
        # aggregated per-lake per-variate across trials
        lake_var_mse_trials = defaultdict(lambda: defaultdict(list))
        lake_var_mae_trials = defaultdict(lambda: defaultdict(list))
        lake_var_crps_trials = defaultdict(lambda: defaultdict(list))
        lake_var_wql_trials = defaultdict(lambda: defaultdict(list))
        
        # Store first trial's lake_to_variates for JSON output
        first_trial_lake_to_variates = None
        # Store first trial's token counts / depth-wise metrics (counts are data-dependent and should be identical across trials)
        first_trial_n_obs = None
        first_trial_n_obs_definition = None
        first_trial_n_obs_by_lake = None
        first_trial_n_obs_by_variate = None
        first_trial_n_obs_by_variate_by_lake = None
        first_trial_depthwise = None
        first_trial_depthwise_n_obs = None
        first_trial_depthwise_n_obs_by_depth = None
        first_trial_depth_bin_size_m = None
        first_trial_depth_round_decimals = None
        first_trial_depth_key_units = None
        first_trial_depth_minmax_by_lake = None
        first_trial_lake_id_to_name = None
        first_trial_depthwise_variate_id_to_name = None
        # Store first trial's WQL metadata (quantiles + per-quantile WQL)
        first_trial_wql_quantiles = None
        first_trial_wql_by_quantile = None
        # Store first trial's T+N metrics computed from date×horizon matrices
        first_trial_tplusn_metrics = None

        # loop over trials
        num_trials = getattr(self.cfg.evaluator, 'num_trials', 1)
        raw_id_to_var = OmegaConf.to_container(self.cfg.data.id_to_var, resolve=True)
        id_to_var      = { int(k): v for k, v in raw_id_to_var.items() }
        self.id_to_var = id_to_var
        
        # Load depth min/max from dataset for denormalization of tick labels
        try:
            ds0 = datasets[0] if isinstance(datasets, list) else datasets
        except Exception:
            ds0 = None
        if ds0 is not None and hasattr(ds0, 'min_depth') and hasattr(ds0, 'max_depth'):
            self.depth_min = getattr(ds0, 'min_depth', None)
            self.depth_max = getattr(ds0, 'max_depth', None)
        
        # Build mapping from lake_id to scalers for denormalization (variables)
        lake_id_to_scalers = {}
        # Build mapping from lake_id -> {min_depth, max_depth} for depth de-normalization (meters)
        lake_id_to_depth_minmax = {}
        datasets_list = datasets if isinstance(datasets, list) else [datasets]
        for ds in datasets_list:
            if hasattr(ds, 'lake_id') and hasattr(ds, 'norm') and ds.norm is not None:
                # Ensure stable int lake_id keys so denormalization behaves consistently.
                try:
                    lake_id = int(ds.lake_id)
                except Exception:
                    lake_id = ds.lake_id
                lake_id_to_scalers[lake_id] = {
                    'scaler_DR': ds.norm.scaler_DR,
                    'scaler_DF': ds.norm.scaler_DF,
                    'variate_ids_1D': ds.variate_ids_1D,
                    'variate_ids_2D': ds.variate_ids_2D
                }
            if hasattr(ds, 'lake_id') and hasattr(ds, 'min_depth') and hasattr(ds, 'max_depth'):
                try:
                    lid = int(ds.lake_id)
                    lake_id_to_depth_minmax[lid] = {
                        "min_depth": float(getattr(ds, "min_depth")),
                        "max_depth": float(getattr(ds, "max_depth")),
                    }
                except Exception:
                    pass
        
        # No global normalization loading here; denormalization uses per-lake scalers from dataset when enabled
        # try:
        #     ds0 = datasets[0] if isinstance(datasets, list) else datasets
        # except Exception:
        #     ds0 = None
        
        # if ds0 is not None and hasattr(ds0, 'norm') and hasattr(ds0, 'normalization_stats_path'):
        #     try:
        #         breakpoint()
        #         self.var_norm_params = Normalizer.get_variable_normalization_params(
        #             global_stats_path=ds0.normalization_stats_path)
        #     except Exception as e:
        #         if self.rank == 0:
        #             print(f"Warning: failed to load normalization from dataset.norm: {e}")
        
        # if self.var_norm_params is None:
        #     # from utils.util import Normalizer
        #     # normalization_stats_path = f"{self.cfg.server_prefix}/lakefm/dev/norm_stats"
        #     raise RuntimeError(f"Failed to load global normalization stats, and dataset.norm does not supply them. Expected stats at: {ds0.normalization_stats_path}")

        # prepare output directory if saving
        save_dir = None
        if self.cfg.evaluator.output_dir is None:
            save_dir = os.path.join(self.cfg.evaluator.ckpt_dir, f"{self.cfg.run_name}_evaluation")
        else:
            save_dir = os.path.join(self.cfg.evaluator.output_dir, f"{self.cfg.run_name}_evaluation")
        
        # Store evaluation-level save_dir for plot saving
        eval_save_dir = save_dir

        for trial_idx in range(num_trials):
            # optional reseed for reproducibility
            seed = getattr(self.cfg, 'seed', None)
            if seed is not None:
                torch.manual_seed(seed + trial_idx)
                np.random.seed(seed + trial_idx)
            # run one trial
            trial_dir = os.path.join(save_dir, f"trial_{trial_idx}")
            eval_dict = self.test_once(
                loader,
                trial_idx=trial_idx,
                save_dir=trial_dir,
                scaling=scaling,
                lake_id_to_scalers=lake_id_to_scalers,
                lake_id_to_depth_minmax=lake_id_to_depth_minmax,
            )
            if self.plot:
                self.plot_predictions(eval_dict, flag=flag, it=trial_idx, save_dir=eval_save_dir)

            if trial_idx == 0:
                first_trial_n_obs = eval_dict.get("n_obs", None)
                first_trial_n_obs_definition = eval_dict.get("n_obs_definition", None)
                first_trial_n_obs_by_lake = eval_dict.get("n_obs_by_lake", None)
                first_trial_n_obs_by_variate = eval_dict.get("n_obs_by_variate", None)
                first_trial_n_obs_by_variate_by_lake = eval_dict.get("n_obs_by_variate_by_lake", None)
                first_trial_wql_quantiles = eval_dict.get("wql_quantiles", None)
                first_trial_wql_by_quantile = eval_dict.get("wql_by_quantile", None)
                first_trial_tplusn_metrics = eval_dict.get("tplusn_metrics", None)
                first_trial_depthwise = {
                    "mse_by_variate_by_depth_by_lake": eval_dict.get("mse_by_variate_by_depth_by_lake", None),
                    "mae_by_variate_by_depth_by_lake": eval_dict.get("mae_by_variate_by_depth_by_lake", None),
                    "crps_by_variate_by_depth_by_lake": eval_dict.get("crps_by_variate_by_depth_by_lake", None),
                    # Aggregated across variates
                    "mse_by_depth_by_lake": eval_dict.get("mse_by_depth_by_lake", None),
                    "mae_by_depth_by_lake": eval_dict.get("mae_by_depth_by_lake", None),
                    "crps_by_depth_by_lake": eval_dict.get("crps_by_depth_by_lake", None),
                }
                first_trial_depthwise_n_obs = eval_dict.get("n_obs_by_variate_by_depth_by_lake", None)
                first_trial_depthwise_n_obs_by_depth = eval_dict.get("n_obs_by_depth_by_lake", None)
                first_trial_depth_bin_size_m = eval_dict.get("depth_bin_size_m", None)
                first_trial_depth_round_decimals = eval_dict.get("depth_round_decimals", None)
                first_trial_depth_key_units = eval_dict.get("depth_key_units", None)
                first_trial_depth_minmax_by_lake = eval_dict.get("depth_minmax_by_lake", None)
                first_trial_lake_id_to_name = eval_dict.get("lake_id_to_name", None)
                # Prefer cfg-derived mapping for stability across datasets
                first_trial_depthwise_variate_id_to_name = {f"var_{int(k)}": str(v) for k, v in id_to_var.items()}

            # record global metrics
            trial_mses.append(eval_dict['mse'])
            trial_maes.append(eval_dict['mae'])
            trial_crpss.append(eval_dict['crps'])
            trial_wqls.append(eval_dict.get('wql', float('nan')))
            trial_infer_throughput_samples.append(
                float(eval_dict.get('infer_throughput_samples_per_s', float('nan')))
            )
            trial_infer_throughput_tokens.append(
                float(eval_dict.get('infer_throughput_tokens_per_s', float('nan')))
            )
            trial_infer_peak_vram_gb.append(
                float(eval_dict.get('infer_peak_vram_gb', float('nan')))
            )
            trial_infer_compute_time_s.append(
                float(eval_dict.get('infer_compute_time_s', float('nan')))
            )
            
            # build lake_id->name mapping from all trials and all batches
            # Priority: use mapping from test_once (all batches) if available, else fallback to plotting batches
            if 'lake_id_to_name' in eval_dict:
                # This comes from all batches in test_once
                for lid, lname in eval_dict['lake_id_to_name'].items():
                    lid_int = int(lid)
                    # Only update if we don't have a name yet or if current name is empty
                    if lid_int not in lake_id_to_name or not lake_id_to_name[lid_int]:
                        if lname and lname.strip():
                            lake_id_to_name[lid_int] = lname
            elif 'lake_ids' in eval_dict and 'lake_names' in eval_dict:
                # Fallback to plotting batches (for backward compatibility)
                for lake_id, lake_name in zip(eval_dict['lake_ids'], eval_dict['lake_names']):
                    lid_int = int(lake_id)
                    if lid_int not in lake_id_to_name or not lake_id_to_name[lid_int]:
                        lname = lake_name.decode('utf-8') if isinstance(lake_name, bytes) else str(lake_name)
                        if lname and lname.strip():
                            lake_id_to_name[lid_int] = lname
            
            # collect lake-wise metrics
            per_trial_lake_mse.append(eval_dict.get('mse_by_lake', {}))
            per_trial_lake_mae.append(eval_dict.get('mae_by_lake', {}))
            per_trial_lake_crps.append(eval_dict.get('crps_by_lake', {}))
            per_trial_lake_wql.append(eval_dict.get('wql_by_lake', {}))
            
            for lake_id, v in eval_dict.get('mse_by_lake', {}).items():
                lake_mse_trials[lake_id].append(v)
            for lake_id, v in eval_dict.get('mae_by_lake', {}).items():
                lake_mae_trials[lake_id].append(v)
            for lake_id, v in eval_dict.get('crps_by_lake', {}).items():
                lake_crps_trials[lake_id].append(v)
            for lake_id, v in eval_dict.get('wql_by_lake', {}).items():
                lake_wql_trials[lake_id].append(v)

            # collect variate-wise metrics (always)
            per_trial_variate_mse.append(eval_dict.get('mse_by_variate', {}))
            per_trial_variate_mae.append(eval_dict.get('mae_by_variate', {}))
            per_trial_variate_crps.append(eval_dict.get('crps_by_variate', {}))
            per_trial_variate_wql.append(eval_dict.get('wql_by_variate', {}))

            for var_id, v in eval_dict.get('mse_by_variate', {}).items():
                variate_mse_trials[var_id].append(v)
            for var_id, v in eval_dict.get('mae_by_variate', {}).items():
                variate_mae_trials[var_id].append(v)
            for var_id, v in eval_dict.get('crps_by_variate', {}).items():
                variate_crps_trials[var_id].append(v)
            for var_id, v in eval_dict.get('wql_by_variate', {}).items():
                variate_wql_trials[var_id].append(v)

            # collect per-lake per-variate metrics (always)
            lb_mse = eval_dict.get('mse_by_variate_by_lake', {}) or {}
            lb_mae = eval_dict.get('mae_by_variate_by_lake', {}) or {}
            lb_crps = eval_dict.get('crps_by_variate_by_lake', {}) or {}
            lb_wql = eval_dict.get('wql_by_variate_by_lake', {}) or {}
            per_trial_lake_variate_mse.append(lb_mse)
            per_trial_lake_variate_mae.append(lb_mae)
            per_trial_lake_variate_crps.append(lb_crps)
            per_trial_lake_variate_wql.append(lb_wql)
            # aggregate across trials
            for lake_id, d in lb_mse.items():
                for var_k, val in d.items():
                    lake_var_mse_trials[int(lake_id)][var_k].append(val)
            for lake_id, d in lb_mae.items():
                for var_k, val in d.items():
                    lake_var_mae_trials[int(lake_id)][var_k].append(val)
            for lake_id, d in lb_crps.items():
                for var_k, val in d.items():
                    lake_var_crps_trials[int(lake_id)][var_k].append(val)
            for lake_id, d in lb_wql.items():
                for var_k, val in d.items():
                    lake_var_wql_trials[int(lake_id)][var_k].append(val)
            # Store and print lake_to_variates mapping from first trial only (for debugging)
            if trial_idx == 0 and 'lake_to_variates' in eval_dict:
                first_trial_lake_to_variates = eval_dict['lake_to_variates']
                if self.rank == 0:
                    pretty_print("=" * 70)
                    pretty_print("Variables per Lake (from evaluation dataset - Trial 0):")
                    pretty_print("=" * 70)
                    for lake_id_str in sorted(first_trial_lake_to_variates.keys(), key=lambda x: int(x)):
                        lake_id_int = int(lake_id_str)
                        lake_name = lake_id_to_name.get(lake_id_int, f"Lake_{lake_id_int}")
                        variates = sorted(first_trial_lake_to_variates[lake_id_str])
                        variate_names = [self.id_to_var.get(vid, f"var_{vid}") for vid in variates]
                        pretty_print(f"  {lake_name} (ID: {lake_id_int}): {len(variates)} variates")
                        pretty_print(f"    Variate IDs: {variates}")
                        pretty_print(f"    Variate Names: {variate_names}")
                    
                    # Print summary
                    all_unique_variates = set()
                    for variates_list in first_trial_lake_to_variates.values():
                        all_unique_variates.update(variates_list)
                    pretty_print(f"\nTotal unique variates across all lakes: {len(all_unique_variates)}")
                    pretty_print(f"Variate IDs: {sorted(all_unique_variates)}")
                    variate_names_all = [self.id_to_var.get(vid, f"var_{vid}") for vid in sorted(all_unique_variates)]
                    pretty_print(f"Variate Names: {variate_names_all}")
                    pretty_print("=" * 70)
            
            if self.save_preds:
                all_preds2d.append(eval_dict['preds2d'])
                all_depths.append(eval_dict['depth_vals'].cpu().numpy())
            
            mse = eval_dict['mse']
            pretty_print(f"Trial {trial_idx} MSE: {mse:.4f}")

        # summary metrics
        mean_mse = float(np.mean(trial_mses))
        std_mse = float(np.std(trial_mses))
        mean_mae = float(np.mean(trial_maes))
        std_mae = float(np.std(trial_maes))
        mean_crps = float(np.mean(trial_crpss))
        std_crps = float(np.std(trial_crpss))
        mean_wql = float(np.nanmean(trial_wqls)) if trial_wqls else float('nan')
        std_wql = float(np.nanstd(trial_wqls)) if trial_wqls else float('nan')
        mean_infer_thr_samples = float(np.nanmean(trial_infer_throughput_samples)) if trial_infer_throughput_samples else float('nan')
        std_infer_thr_samples = float(np.nanstd(trial_infer_throughput_samples)) if trial_infer_throughput_samples else float('nan')
        mean_infer_thr_tokens = float(np.nanmean(trial_infer_throughput_tokens)) if trial_infer_throughput_tokens else float('nan')
        std_infer_thr_tokens = float(np.nanstd(trial_infer_throughput_tokens)) if trial_infer_throughput_tokens else float('nan')
        mean_infer_peak_vram = float(np.nanmean(trial_infer_peak_vram_gb)) if trial_infer_peak_vram_gb else float('nan')
        std_infer_peak_vram = float(np.nanstd(trial_infer_peak_vram_gb)) if trial_infer_peak_vram_gb else float('nan')
        mean_infer_compute_time = float(np.nanmean(trial_infer_compute_time_s)) if trial_infer_compute_time_s else float('nan')
        std_infer_compute_time = float(np.nanstd(trial_infer_compute_time_s)) if trial_infer_compute_time_s else float('nan')
        
        # per-lake mean/std for all metrics, and embed per-variable stats per lake
        lake_stats = {}
        for lake_id in lake_mse_trials:
            stats = {
                'mse_mean': float(np.mean(lake_mse_trials[lake_id])),
                'mse_std':  float(np.std(lake_mse_trials[lake_id])),
                'mae_mean': float(np.mean(lake_mae_trials[lake_id])),
                'mae_std':  float(np.std(lake_mae_trials[lake_id])),
                'crps_mean': float(np.mean(lake_crps_trials[lake_id])),
                'crps_std':  float(np.std(lake_crps_trials[lake_id])),
                'wql_mean': float(np.nanmean(lake_wql_trials[lake_id])) if lake_wql_trials.get(lake_id) else float('nan'),
                'wql_std':  float(np.nanstd(lake_wql_trials[lake_id]))  if lake_wql_trials.get(lake_id) else float('nan'),
                'name': lake_id_to_name.get(lake_id, '')
            }
            # Build nested variate_stats for this lake if available
            if lake_var_mse_trials.get(lake_id):
                inner_vs = {}
                for var_k in lake_var_mse_trials[lake_id].keys():
                    mse_vals = lake_var_mse_trials[lake_id].get(var_k, [])
                    mae_vals = lake_var_mae_trials[lake_id].get(var_k, [])
                    crps_vals = lake_var_crps_trials[lake_id].get(var_k, [])
                    wql_vals = lake_var_wql_trials[lake_id].get(var_k, [])
                    if mse_vals or mae_vals or crps_vals:
                        # extract numeric id if var_k is like 'var_1'
                        try:
                            vid_int = int(str(var_k).split('_')[-1])
                        except Exception:
                            vid_int = None
                        inner_vs[var_k] = {
                            'mse_mean': float(np.mean(mse_vals)) if mse_vals else float('nan'),
                            'mse_std':  float(np.std(mse_vals))  if mse_vals else float('nan'),
                            'mae_mean': float(np.mean(mae_vals)) if mae_vals else float('nan'),
                            'mae_std':  float(np.std(mae_vals))  if mae_vals else float('nan'),
                            'crps_mean': float(np.mean(crps_vals)) if crps_vals else float('nan'),
                            'crps_std':  float(np.std(crps_vals))  if crps_vals else float('nan'),
                            'wql_mean': float(np.nanmean(wql_vals)) if wql_vals else float('nan'),
                            'wql_std':  float(np.nanstd(wql_vals))  if wql_vals else float('nan'),
                            'name': self.id_to_var.get(vid_int, str(var_k)) if vid_int is not None else str(var_k)
                        }
                if inner_vs:
                    stats['variate_stats'] = inner_vs
            lake_stats[lake_id] = stats
        
        # per-variate mean/std for all metrics
        variate_stats = {}
        for var_id in variate_mse_trials:
            # Extract integer ID from string like 'var_1' -> 1
            try:
                vid_int = int(str(var_id).split('_')[-1])
            except Exception:
                vid_int = None
            variate_stats[var_id] = {
                'mse_mean': float(np.mean(variate_mse_trials[var_id])),
                'mse_std':  float(np.std(variate_mse_trials[var_id])),
                'mae_mean': float(np.mean(variate_mae_trials[var_id])),
                'mae_std':  float(np.std(variate_mae_trials[var_id])),
                'crps_mean': float(np.mean(variate_crps_trials[var_id])),
                'crps_std':  float(np.std(variate_crps_trials[var_id])),
                'wql_mean': float(np.nanmean(variate_wql_trials[var_id])) if variate_wql_trials.get(var_id) else float('nan'),
                'wql_std':  float(np.nanstd(variate_wql_trials[var_id]))  if variate_wql_trials.get(var_id) else float('nan'),
                'name': self.id_to_var.get(vid_int, f"var_{var_id}") if vid_int is not None else f"var_{var_id}"
            }

        # write JSON summary
        json_dict = {
            "trial_mse": trial_mses,
            "trial_mae": trial_maes,
            "trial_crps": trial_crpss,
            "trial_wql": trial_wqls,
            "mean_mse":    mean_mse,
            "std_mse":     std_mse,
            "mean_mae":    mean_mae,
            "std_mae":     std_mae,
            "mean_crps":   mean_crps,
            "std_crps":    std_crps,
            "mean_wql":    mean_wql,
            "std_wql":     std_wql,
            "model_param_count_total": int(self.total_params),
            "model_param_count_trainable": int(self.trainable_params),
            "trial_infer_throughput_samples_per_s": trial_infer_throughput_samples,
            "trial_infer_throughput_tokens_per_s": trial_infer_throughput_tokens,
            "trial_infer_peak_vram_gb": trial_infer_peak_vram_gb,
            "trial_infer_compute_time_s": trial_infer_compute_time_s,
            "mean_infer_throughput_samples_per_s": mean_infer_thr_samples,
            "std_infer_throughput_samples_per_s": std_infer_thr_samples,
            "mean_infer_throughput_tokens_per_s": mean_infer_thr_tokens,
            "std_infer_throughput_tokens_per_s": std_infer_thr_tokens,
            "mean_infer_peak_vram_gb": mean_infer_peak_vram,
            "std_infer_peak_vram_gb": std_infer_peak_vram,
            "mean_infer_compute_time_s": mean_infer_compute_time,
            "std_infer_compute_time_s": std_infer_compute_time,
            "lake_stats":     lake_stats,
            "eval_dataset": self.cfg.evaluator.eval_dataset,
            "pretrain_dataset": list(self.data.pretrain_dataset),
            "model_epoch":  self.model_epoch,
        }
        # NOTE: thermocline metrics are written to a dedicated JSON per-trial under trial_*/THERMOCLINE/.
        # Token counts for weighted/unweighted aggregation
        if first_trial_n_obs_definition is not None:
            json_dict["n_obs_definition"] = first_trial_n_obs_definition
        if first_trial_n_obs is not None:
            json_dict["n_obs"] = first_trial_n_obs
        if first_trial_n_obs_by_lake is not None:
            json_dict["n_obs_by_lake"] = first_trial_n_obs_by_lake
        # WQL metadata
        if first_trial_wql_quantiles is not None:
            json_dict["wql_quantiles"] = first_trial_wql_quantiles
        if first_trial_wql_by_quantile is not None:
            json_dict["wql_by_quantile"] = first_trial_wql_by_quantile
        # Variate-wise summary metrics (always)
        json_dict["variate_stats"] = variate_stats
        if first_trial_n_obs_by_variate is not None:
            json_dict["n_obs_by_variate"] = first_trial_n_obs_by_variate
        if first_trial_n_obs_by_variate_by_lake is not None:
            json_dict["n_obs_by_variate_by_lake"] = first_trial_n_obs_by_variate_by_lake

        # Depth-wise metrics payload (always)
        if first_trial_depthwise is not None:
            json_dict["depthwise_metrics"] = first_trial_depthwise
        if first_trial_depthwise_n_obs is not None:
            json_dict["depthwise_n_obs"] = first_trial_depthwise_n_obs
        try:
            if first_trial_depthwise_n_obs_by_depth is not None:
                json_dict["depthwise_n_obs_by_depth"] = first_trial_depthwise_n_obs_by_depth
        except Exception:
            pass
        json_dict["depth_bin_size_m"] = first_trial_depth_bin_size_m
        json_dict["depth_round_decimals"] = first_trial_depth_round_decimals
        if first_trial_depth_key_units is not None:
            json_dict["depth_key_units"] = first_trial_depth_key_units
        if first_trial_depth_minmax_by_lake is not None:
            json_dict["depth_minmax_by_lake"] = first_trial_depth_minmax_by_lake
        if first_trial_lake_id_to_name is not None:
            json_dict["depthwise_lake_id_to_name"] = first_trial_lake_id_to_name
        if first_trial_depthwise_variate_id_to_name is not None:
            json_dict["depthwise_variate_id_to_name"] = first_trial_depthwise_variate_id_to_name
        if first_trial_tplusn_metrics:
            json_dict["tplusn_metrics"] = first_trial_tplusn_metrics

        # Add lake_to_variates mapping from first trial to JSON
        if first_trial_lake_to_variates is not None:
            json_dict["lake_to_variates"] = first_trial_lake_to_variates

        json_path = os.path.join(save_dir, "evaluation_summary.json")
        with open(json_path, "w") as jf:
            json.dump(json_dict, jf, indent=2, ensure_ascii=False)

        pretty_print(f"Saved JSON summary (losses + preds) to {json_path}")
        
        # Save lake embedding trajectories if enabled
        if self.lake_embed_traj and 'lake_embed_traj' in eval_dict:
            # Group embeddings by lake_id (keep lakes separate, don't concatenate)
            embeddings_np = eval_dict['lake_embed_traj'].cpu().numpy()  # (N, d_temporal)
            dates_np = np.array(eval_dict['lake_embed_traj_dates'])
            lake_ids_np = np.array(eval_dict['lake_embed_traj_lake_ids'])
            lake_names_np = np.array(eval_dict['lake_embed_traj_lake_names'])
            
            # Group by lake_id
            unique_lake_ids = np.unique(lake_ids_np)
            save_dict = {}
            
            for lake_id in unique_lake_ids:
                mask = lake_ids_np == lake_id
                lake_name = lake_names_np[mask][0]  # Get lake name (same for all samples of this lake)
                
                # Store per-lake data with lake_id as key prefix
                save_dict[f'lake_{lake_id}_embeddings'] = embeddings_np[mask]  # (n_samples, d_temporal)
                save_dict[f'lake_{lake_id}_dates'] = dates_np[mask]  # (n_samples,)
                save_dict[f'lake_{lake_id}_name'] = lake_name
            
            # Save all lakes in one file, but separated by lake_id
            lake_embed_traj_path = os.path.join(save_dir, "lake_embedding_trajectories.npz")
            np.savez(lake_embed_traj_path, **save_dict)
            
            pretty_print(f"Saved lake embedding trajectories to {lake_embed_traj_path}")
            pretty_print(f"  - Total samples: {len(dates_np)}")
            pretty_print(f"  - Number of lakes: {len(unique_lake_ids)}")
            for lake_id in unique_lake_ids:
                mask = lake_ids_np == lake_id
                lake_name = lake_names_np[mask][0]
                n_samples = mask.sum()
                pretty_print(f"    • Lake {lake_id} ({lake_name}): {n_samples} samples, shape {embeddings_np[mask].shape}")

        # Handle predictions (can be dict or tensor) - similar to val_one_epoch
        if len(all_preds2d) > 0 and isinstance(all_preds2d[0], dict):
            # For dict predictions, stack each component separately
            preds2d_stack = {
                'loc': np.stack([p['loc'].cpu().numpy() for p in all_preds2d], axis=0),
                'scale': np.stack([p['scale'].cpu().numpy() for p in all_preds2d], axis=0),
                'df': np.stack([p['df'].cpu().numpy() for p in all_preds2d], axis=0),
                'mean': np.stack([p['mean'].cpu().numpy() for p in all_preds2d], axis=0)
            }
            preds2d_mean = {
                'loc': preds2d_stack['loc'].mean(axis=0),
                'scale': preds2d_stack['scale'].mean(axis=0),
                'df': preds2d_stack['df'].mean(axis=0),
                'mean': preds2d_stack['mean'].mean(axis=0)
            }
            preds2d_std = {
                'loc': preds2d_stack['loc'].std(axis=0),
                'scale': preds2d_stack['scale'].std(axis=0),
                'df': preds2d_stack['df'].std(axis=0),
                'mean': preds2d_stack['mean'].std(axis=0)
            }
        else:
            # For tensor predictions, use numpy stacking
            preds2d_stack = np.stack([p.cpu().numpy() if hasattr(p, 'cpu') else p for p in all_preds2d], axis=0)
            preds2d_mean = preds2d_stack.mean(axis=0)
            preds2d_std = preds2d_stack.std(axis=0)

        # save the summaries to HDF5
        summary_h5 = os.path.join(save_dir, "preds_summary.h5")
        with h5py.File(summary_h5, 'w') as hf:
            for i in range(num_trials):
                grp = hf.create_group(f"trial_{i}")
                # this will store in a list form
                # lake-wise MSE for this trial
                # lake_dict = per_trial_lake_mse[i]
                # lake_ids_i = np.array(list(lake_dict.keys()), dtype=np.int64)
                # lake_vals_i = np.array(list(lake_dict.values()), dtype=np.float32)
                # grp.create_dataset('mse_by_lake_ids', data=lake_ids_i)
                # grp.create_dataset('mse_by_lake',     data=lake_vals_i)
                # if self.do_variate_mse:
                #     var_dict = per_trial_variate_mse[i]
                #     var_ids_i = np.array(list(var_dict.keys()), dtype=np.int64)
                #     var_vals_i = np.array(list(var_dict.values()), dtype=np.float32)
                #     grp.create_dataset('mse_by_var_ids',   data=var_ids_i)
                #     grp.create_dataset('mse_by_variate',   data=var_vals_i)
                #                 # lake-wise: one dataset per lake for all metrics
                lake_mse_dict = per_trial_lake_mse[i] or {}
                lake_mae_dict = per_trial_lake_mae[i] or {}
                lake_crps_dict = per_trial_lake_crps[i] or {}
                lake_wql_dict = per_trial_lake_wql[i] or {}
                for lid in lake_mse_dict:
                    if lid in lake_mse_dict:
                        grp.create_dataset(f"lake_{lid}_mse", data=np.array(lake_mse_dict[lid], dtype=np.float32))
                    if lid in lake_mae_dict:
                        grp.create_dataset(f"lake_{lid}_mae", data=np.array(lake_mae_dict[lid], dtype=np.float32))
                    if lid in lake_crps_dict:
                        grp.create_dataset(f"lake_{lid}_crps", data=np.array(lake_crps_dict[lid], dtype=np.float32))
                    if lid in lake_wql_dict:
                        grp.create_dataset(f"lake_{lid}_wql", data=np.array(lake_wql_dict[lid], dtype=np.float32))
                # per-lake per-variate for this trial
                if per_trial_lake_variate_mse:
                    lb_mse = per_trial_lake_variate_mse[i] or {}
                    lb_mae = per_trial_lake_variate_mae[i] or {}
                    lb_crps = per_trial_lake_variate_crps[i] or {}
                    lb_wql = per_trial_lake_variate_wql[i] or {}
                    for lid, var_dict in lb_mse.items():
                        for var_k, val in var_dict.items():
                            grp.create_dataset(f"lake_{lid}_{var_k}_mse", data=np.array(val, dtype=np.float32))
                    for lid, var_dict in lb_mae.items():
                        for var_k, val in var_dict.items():
                            grp.create_dataset(f"lake_{lid}_{var_k}_mae", data=np.array(val, dtype=np.float32))
                    for lid, var_dict in lb_crps.items():
                        for var_k, val in var_dict.items():
                            grp.create_dataset(f"lake_{lid}_{var_k}_crps", data=np.array(val, dtype=np.float32))
                    for lid, var_dict in lb_wql.items():
                        for var_k, val in var_dict.items():
                            grp.create_dataset(f"lake_{lid}_{var_k}_wql", data=np.array(val, dtype=np.float32))
                
                # variate-wise: one dataset per variate for all metrics
                if per_trial_variate_mse:
                    var_mse_dict = per_trial_variate_mse[i] or {}
                    var_mae_dict = per_trial_variate_mae[i] or {}
                    var_crps_dict = per_trial_variate_crps[i] or {}
                    var_wql_dict = per_trial_variate_wql[i] or {}
                    for vid in var_mse_dict:
                        if vid in var_mse_dict:
                            grp.create_dataset(f"variate_{vid}_mse", data=np.array(var_mse_dict[vid], dtype=np.float32))
                        if vid in var_mae_dict:
                            grp.create_dataset(f"variate_{vid}_mae", data=np.array(var_mae_dict[vid], dtype=np.float32))
                        if vid in var_crps_dict:
                            grp.create_dataset(f"variate_{vid}_crps", data=np.array(var_crps_dict[vid], dtype=np.float32))
                        if vid in var_wql_dict:
                            grp.create_dataset(f"variate_{vid}_wql", data=np.array(var_wql_dict[vid], dtype=np.float32))
                if self.save_preds:
                    # raw preds for this trial - handle dict vs tensor
                    if isinstance(preds2d_stack, dict):
                        pred_grp = grp.create_group("preds2d")
                        for key in preds2d_stack:
                            pred_grp.create_dataset(key, data=preds2d_stack[key][i])
                    else:
                        grp.create_dataset("preds2d", data=preds2d_stack[i])
                    grp.create_dataset("depth_vals", data=all_depths[i])
                # scalar metrics
                grp.create_dataset("mse",   data=np.array(trial_mses[i], dtype=np.float32))
                grp.create_dataset("mae",   data=np.array(trial_maes[i], dtype=np.float32))
                grp.create_dataset("crps",  data=np.array(trial_crpss[i], dtype=np.float32))
                grp.create_dataset("wql",   data=np.array(trial_wqls[i], dtype=np.float32))
            if self.save_preds:
                # Handle dict vs tensor for mean/std
                if isinstance(preds2d_mean, dict):
                    mean_grp = hf.create_group("preds2d_mean")
                    std_grp = hf.create_group("preds2d_std")
                    for key in preds2d_mean:
                        mean_grp.create_dataset(key, data=preds2d_mean[key])
                        std_grp.create_dataset(key, data=preds2d_std[key])
                else:
                    hf.create_dataset("preds2d_mean", data=preds2d_mean)
                    hf.create_dataset("preds2d_std",  data=preds2d_std)
            hf.attrs["mse_mean"] = mean_mse
            hf.attrs["mse_std"]  = std_mse
            hf.attrs["mae_mean"] = mean_mae
            hf.attrs["mae_std"]  = std_mae
            hf.attrs["crps_mean"] = mean_crps
            hf.attrs["crps_std"]  = std_crps
            hf.attrs["wql_mean"] = mean_wql
            hf.attrs["wql_std"]  = std_wql
            hf.attrs["eval_dataset"] = self.cfg.evaluator.eval_dataset

            if self.cfg.data.pretrain_dataset:
                ds = list(self.cfg.data.pretrain_dataset)   # cast to Python list
                hf.attrs["pretrain_dataset"] = json.dumps(ds)
            # lake statistics
            ls_ids   = np.array(list(lake_stats.keys()), dtype=np.int64)
            ls_mse_mean  = np.array([lake_stats[l]['mse_mean'] for l in lake_stats], dtype=np.float32)
            ls_mse_std   = np.array([lake_stats[l]['mse_std']  for l in lake_stats], dtype=np.float32)
            ls_mae_mean  = np.array([lake_stats[l]['mae_mean'] for l in lake_stats], dtype=np.float32)
            ls_mae_std   = np.array([lake_stats[l]['mae_std']  for l in lake_stats], dtype=np.float32)
            ls_crps_mean = np.array([lake_stats[l]['crps_mean'] for l in lake_stats], dtype=np.float32)
            ls_crps_std  = np.array([lake_stats[l]['crps_std']  for l in lake_stats], dtype=np.float32)
            ls_wql_mean  = np.array([lake_stats[l].get('wql_mean', np.nan) for l in lake_stats], dtype=np.float32)
            ls_wql_std   = np.array([lake_stats[l].get('wql_std', np.nan)  for l in lake_stats], dtype=np.float32)
            ls_names = np.array([lake_stats[l]['name'].encode('utf-8') for l in lake_stats],
                                dtype=string_dtype(encoding='utf-8'))
            grp_ls   = hf.create_group('lake_stats')
            grp_ls.create_dataset('ids',   data=ls_ids)
            grp_ls.create_dataset('mse_mean',  data=ls_mse_mean)
            grp_ls.create_dataset('mse_std',   data=ls_mse_std)
            grp_ls.create_dataset('mae_mean',  data=ls_mae_mean)
            grp_ls.create_dataset('mae_std',   data=ls_mae_std)
            grp_ls.create_dataset('crps_mean', data=ls_crps_mean)
            grp_ls.create_dataset('crps_std',  data=ls_crps_std)
            grp_ls.create_dataset('wql_mean',  data=ls_wql_mean)
            grp_ls.create_dataset('wql_std',   data=ls_wql_std)
            grp_ls.create_dataset('names', data=ls_names, dtype=string_dtype('utf-8'))

            # variate statistics
            if variate_stats:
                # Extract numeric IDs from string keys like 'var_1', 'var_2', etc.
                vs_ids = np.array([int(k.split('_')[1]) for k in variate_stats.keys()], dtype=np.int64)
                vs_mse_mean  = np.array([variate_stats[v]['mse_mean'] for v in variate_stats], dtype=np.float32)
                vs_mse_std   = np.array([variate_stats[v]['mse_std']  for v in variate_stats], dtype=np.float32)
                vs_mae_mean  = np.array([variate_stats[v]['mae_mean'] for v in variate_stats], dtype=np.float32)
                vs_mae_std   = np.array([variate_stats[v]['mae_std']  for v in variate_stats], dtype=np.float32)
                vs_crps_mean = np.array([variate_stats[v]['crps_mean'] for v in variate_stats], dtype=np.float32)
                vs_crps_std  = np.array([variate_stats[v]['crps_std']  for v in variate_stats], dtype=np.float32)
                vs_wql_mean  = np.array([variate_stats[v].get('wql_mean', np.nan) for v in variate_stats], dtype=np.float32)
                vs_wql_std   = np.array([variate_stats[v].get('wql_std', np.nan)  for v in variate_stats], dtype=np.float32)
                vs_names = np.array([variate_stats[v]['name'].encode('utf-8') for v in variate_stats],
                                    dtype=string_dtype('utf-8'))
                grp_vs   = hf.create_group('variate_stats')
                grp_vs.create_dataset('ids',   data=vs_ids)
                grp_vs.create_dataset('mse_mean',  data=vs_mse_mean)
                grp_vs.create_dataset('mse_std',   data=vs_mse_std)
                grp_vs.create_dataset('mae_mean',  data=vs_mae_mean)
                grp_vs.create_dataset('mae_std',   data=vs_mae_std)
                grp_vs.create_dataset('crps_mean', data=vs_crps_mean)
                grp_vs.create_dataset('crps_std',  data=vs_crps_std)
                grp_vs.create_dataset('wql_mean',  data=vs_wql_mean)
                grp_vs.create_dataset('wql_std',   data=vs_wql_std)
                grp_vs.create_dataset('names', data=vs_names, dtype=string_dtype('utf-8'))

            # lake-wise embedded variate statistics (aggregated across trials)
            if lake_stats:
                grp_lvs = hf.create_group('lake_variate_stats')
                for lake_id, lstats in lake_stats.items():
                    if 'variate_stats' not in lstats:
                        continue
                    subgrp = grp_lvs.create_group(f"lake_{lake_id}")
                    var_keys = list(lstats['variate_stats'].keys())
                    # store ids as integers when possible
                    var_ids = []
                    var_names = []
                    mse_mean = []
                    mse_std = []
                    mae_mean = []
                    mae_std = []
                    crps_mean = []
                    crps_std = []
                    for vk in var_keys:
                        try:
                            vid = int(str(vk).split('_')[-1])
                        except Exception:
                            vid = -1
                        vs = lstats['variate_stats'][vk]
                        var_ids.append(vid)
                        var_names.append(vs.get('name',''))
                        mse_mean.append(vs.get('mse_mean', np.nan))
                        mse_std.append(vs.get('mse_std', np.nan))
                        mae_mean.append(vs.get('mae_mean', np.nan))
                        mae_std.append(vs.get('mae_std', np.nan))
                        crps_mean.append(vs.get('crps_mean', np.nan))
                        crps_std.append(vs.get('crps_std', np.nan))
                    subgrp.create_dataset('ids', data=np.array(var_ids, dtype=np.int64))
                    subgrp.create_dataset('names', data=np.array([n.encode('utf-8') for n in var_names], dtype=string_dtype('utf-8')))
                    subgrp.create_dataset('mse_mean', data=np.array(mse_mean, dtype=np.float32))
                    subgrp.create_dataset('mse_std', data=np.array(mse_std, dtype=np.float32))
                    subgrp.create_dataset('mae_mean', data=np.array(mae_mean, dtype=np.float32))
                    subgrp.create_dataset('mae_std', data=np.array(mae_std, dtype=np.float32))
                    subgrp.create_dataset('crps_mean', data=np.array(crps_mean, dtype=np.float32))
                    subgrp.create_dataset('crps_std', data=np.array(crps_std, dtype=np.float32))


        pretty_print(f"Saved preds mean/std to {summary_h5}")

        # log summary to W&B under evaluator namespace
        if self.rank == 0:
            wandb.init(
                project=self.cfg.trainer.wandb_project,
                name=self.cfg.trainer.wandb_name,
                config=OmegaConf.to_container(self.cfg, resolve=True),
                save_code=self.cfg.evaluator.save_code
            )

            # Log final metrics to wandb.summary
            if self.rank == 0:
                # Global metrics
                wandb.summary.update({
                    "eval/mse_mean": mean_mse,
                    "eval/mse_std": std_mse,
                    "eval/mae_mean": mean_mae,
                    "eval/mae_std": std_mae,
                    "eval/crps_mean": mean_crps,
                    "eval/crps_std": std_crps,
                    "eval/wql_mean": mean_wql,
                    "eval/wql_std": std_wql,
                    "eval/model_param_count_total": int(self.total_params),
                    "eval/model_param_count_trainable": int(self.trainable_params),
                    "eval/infer_throughput_samples_per_s_mean": mean_infer_thr_samples,
                    "eval/infer_throughput_samples_per_s_std": std_infer_thr_samples,
                    "eval/infer_throughput_tokens_per_s_mean": mean_infer_thr_tokens,
                    "eval/infer_throughput_tokens_per_s_std": std_infer_thr_tokens,
                    "eval/infer_peak_vram_gb_mean": mean_infer_peak_vram,
                    "eval/infer_peak_vram_gb_std": std_infer_peak_vram,
                    "eval/infer_compute_time_s_mean": mean_infer_compute_time,
                    "eval/infer_compute_time_s_std": std_infer_compute_time,
                })
                
                # Lake-wise metrics
                for lake_id, stats in lake_stats.items():
                    wandb.summary.update({
                        f"eval/lake_{lake_id}_mse_mean": stats['mse_mean'],
                        f"eval/lake_{lake_id}_mse_std": stats['mse_std'],
                        f"eval/lake_{lake_id}_mae_mean": stats['mae_mean'],
                        f"eval/lake_{lake_id}_mae_std": stats['mae_std'],
                        f"eval/lake_{lake_id}_crps_mean": stats['crps_mean'],
                        f"eval/lake_{lake_id}_crps_std": stats['crps_std'],
                        f"eval/lake_{lake_id}_wql_mean": stats.get('wql_mean', float('nan')),
                        f"eval/lake_{lake_id}_wql_std": stats.get('wql_std', float('nan')),
                    })
                
                # Variate-wise metrics
                    for vid, stats in variate_stats.items():
                        wandb.summary.update({
                            f"eval/var_{vid}_mse_mean": stats['mse_mean'],
                            f"eval/var_{vid}_mse_std": stats['mse_std'],
                            f"eval/var_{vid}_mae_mean": stats['mae_mean'],
                            f"eval/var_{vid}_mae_std": stats['mae_std'],
                            f"eval/var_{vid}_crps_mean": stats['crps_mean'],
                            f"eval/var_{vid}_crps_std": stats['crps_std'],
                            f"eval/var_{vid}_wql_mean": stats.get('wql_mean', float('nan')),
                            f"eval/var_{vid}_wql_std": stats.get('wql_std', float('nan')),
                        })
                
                wandb.finish()

        return {
            "eval/mse_mean": mean_mse,
            "eval/mse_std":  std_mse,
            "eval/mae_mean": mean_mae,
            "eval/mae_std":  std_mae,
            "eval/crps_mean": mean_crps,
            "eval/crps_std":  std_crps,
            "eval/wql_mean": mean_wql,
            "eval/wql_std":  std_wql,
            "eval/model_param_count_total": int(self.total_params),
            "eval/model_param_count_trainable": int(self.trainable_params),
            "eval/infer_throughput_samples_per_s_mean": mean_infer_thr_samples,
            "eval/infer_throughput_samples_per_s_std": std_infer_thr_samples,
            "eval/infer_throughput_tokens_per_s_mean": mean_infer_thr_tokens,
            "eval/infer_throughput_tokens_per_s_std": std_infer_thr_tokens,
            "eval/infer_peak_vram_gb_mean": mean_infer_peak_vram,
            "eval/infer_peak_vram_gb_std": std_infer_peak_vram,
            "eval/infer_compute_time_s_mean": mean_infer_compute_time,
            "eval/infer_compute_time_s_std": std_infer_compute_time,
            "lake_stats":     lake_stats,
            "variate_stats": variate_stats
        }

