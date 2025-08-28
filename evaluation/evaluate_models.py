
# Description:
#   Model evaluation script — compare different training strategies with multiple IQA metrics.
#   Supported metrics (via helper module): PSNR, ERGAS, SAM, Q index, SSIM (and aggregates).
#
# Environment variables (override as needed):
#   PROJECT_ROOT   (default: parent directory of this file)
#   DATA_DIR       (default: <PROJECT_ROOT>/dataset)
#   INPUT_DIR      (default: <DATA_DIR>/input_restacked_16)
#   LABEL_DIR      (default: <DATA_DIR>/label)
#   CHECKPOINT_DIR (default: <PROJECT_ROOT>/checkpoints)
#   OUTPUT_DIR     (default: <PROJECT_ROOT>/outputs)
#   BATCH_SIZE     (default: 8)
#   MAX_BATCHES    (default: 50)   # limit batches per band for speed
#   NUM_WORKERS    (default: 2)
#   IN_CHANNELS    (default: 16)   # expected by single-band models
#   IMAGE_SIZE     (default: 128)
#   DEVICE         (default: auto; "cuda" if available else "cpu")
#   ALLOW_RESTACK_5_TO_16 (default: "1")  If inputs have 5 channels, restack to 16 by a standard pattern.
#
# Notes:
#   - This script tries to import `denormalize_prediction` from train.denormalize_utils; if missing,
#     a no-op fallback is used (assumes inputs are already in display/range scale).

from __future__ import annotations
import os
import sys
import json
import glob
import warnings
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# Ensure sibling modules (e.g., image_quality_metrics) are importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Safe import: denormalization utility
try:
    from train.denormalize_utils import denormalize_prediction  # type: ignore
except Exception:
    def denormalize_prediction(x: torch.Tensor, target_band: int) -> torch.Tensor:
        # Fallback: no-op (assumes x already in the correct scale)
        return x

# Metrics helpers (expected to exist alongside this script)
from image_quality_metrics import (
    calculate_all_metrics,        # not used directly but kept for API parity
    batch_calculate_metrics,
    aggregate_metrics,
)

# Model
from monai.networks.nets import SwinUNETR


# -----------------------------
# Soft-coded path resolution
# -----------------------------
def resolve_paths() -> Dict[str, str]:
    """
    Resolve data/checkpoint/output directories using environment variables with safe defaults.
    """
    this_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.environ.get("PROJECT_ROOT", os.path.dirname(this_dir))
    data_dir = os.environ.get("DATA_DIR", os.path.join(project_root, "dataset"))
    input_dir = os.environ.get("INPUT_DIR", os.path.join(data_dir, "input_restacked_16"))
    label_dir = os.environ.get("LABEL_DIR", os.path.join(data_dir, "label"))
    checkpoint_dir = os.environ.get("CHECKPOINT_DIR", os.path.join(project_root, "checkpoints"))
    output_dir = os.environ.get("OUTPUT_DIR", os.path.join(project_root, "outputs"))
    return {
        "PROJECT_ROOT": project_root,
        "DATA_DIR": data_dir,
        "INPUT_DIR": input_dir,
        "LABEL_DIR": label_dir,
        "CHECKPOINT_DIR": checkpoint_dir,
        "OUTPUT_DIR": output_dir,
    }


# -----------------------------
# Model and dataset
# -----------------------------
class SpecSwin_SingleBand(nn.Module):
    def __init__(self, in_channels: int = 16, img_size: Tuple[int, int] = (128, 128), spatial_dims: int = 2):
        super().__init__()
        self.model = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=1,
            feature_size=48,
            spatial_dims=spatial_dims,
            use_checkpoint=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def _maybe_to_chw(t: torch.Tensor) -> torch.Tensor:
    """Convert (H, W, C) to (C, H, W) if needed."""
    if t.ndim == 3 and t.shape[0] not in (5, 16) and t.shape[-1] in (5, 16):
        return t.permute(2, 0, 1).contiguous()
    return t


def _restack_5_to_16(x5: torch.Tensor) -> torch.Tensor:
    """
    Restack a 5-channel input to 16 channels using a common pattern:
      indices = [0,1,2,3,4, 1,3,0,2,4, 0,4,3,2,1, 0]
    """
    indices = [0, 1, 2, 3, 4, 1, 3, 0, 2, 4, 0, 4, 3, 2, 1, 0]
    return x5[indices]


class SingleBandDataset(Dataset):
    """
    Dataset for single-band reconstruction.
    Expects paired *.pt files in INPUT_DIR and LABEL_DIR.

    Input (.pt) may be:
      - dict with key 'input' -> tensor (C,H,W) or (H,W,C)
      - tensor (C,H,W) or (H,W,C)
    Label (.pt) may be:
      - dict with key 'label' -> tensor (B,H,W) or (H,W,B), optionally with 'band_indices' (order mapping)
      - tensor (B,H,W) or (H,W,B)

    target_band_id is the real spectral band id. If 'band_indices' mapping exists in label dict,
    we use its index; otherwise we assume target_band_id equals the zero-based band index.
    """
    def __init__(self, input_dir: str, label_dir: str, target_band_id: int, in_channels: int = 16):
        self.input_dir = input_dir
        self.label_dir = label_dir
        self.target_band_id = int(target_band_id)
        self.in_channels = int(in_channels)

        self.input_files = sorted(glob.glob(os.path.join(input_dir, "*.pt")))
        self.label_files = sorted(glob.glob(os.path.join(label_dir, "*.pt")))
        print(f"Found {len(self.input_files)} input files, {len(self.label_files)} label files")

    def __len__(self) -> int:
        return min(len(self.input_files), len(self.label_files))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load input
        in_raw = torch.load(self.input_files[idx], map_location="cpu")
        if isinstance(in_raw, dict) and "input" in in_raw:
            in_t = in_raw["input"]
        else:
            in_t = in_raw
        in_t = _maybe_to_chw(in_t).float()  # (C,H,W)

        if in_t.ndim != 3:
            raise ValueError(f"Unexpected input tensor shape for {self.input_files[idx]}: {tuple(in_t.shape)}")

        if in_t.shape[0] not in (5, 16):
            raise ValueError(f"Input channels must be 5 or 16, got {in_t.shape[0]} in {self.input_files[idx]}")

        # Optionally restack 5->16
        allow_restack = os.environ.get("ALLOW_RESTACK_5_TO_16", "1").strip() not in ("0", "false", "False")
        if self.in_channels == 16 and in_t.shape[0] == 5:
            if allow_restack:
                in_t = _restack_5_to_16(in_t)
            else:
                raise RuntimeError("Input has 5 channels but model expects 16. Set ALLOW_RESTACK_5_TO_16=1 to enable.")

        # Load label
        lb_raw = torch.load(self.label_files[idx], map_location="cpu")
        if isinstance(lb_raw, dict) and "label" in lb_raw:
            lb = lb_raw["label"]
            band_indices = None
            for key in ("band_indices", "bands", "band_ids", "band_order"):
                if key in lb_raw:
                    band_indices = list(lb_raw[key])
                    break
        else:
            lb = lb_raw
            band_indices = None

        # Normalize label array shape to (B,H,W)
        if isinstance(lb, torch.Tensor):
            t = lb
        else:
            t = torch.tensor(lb)

        if t.ndim != 3:
            raise ValueError(f"Unexpected label tensor shape for {self.label_files[idx]}: {tuple(t.shape)}")

        if t.shape[0] < t.shape[-1]:  # likely (B,H,W)
            BHW = t
        else:  # likely (H,W,B)
            BHW = t.permute(2, 0, 1).contiguous()

        # Map band id -> index
        if band_indices is not None and self.target_band_id in band_indices:
            bidx = int(band_indices.index(self.target_band_id))
        else:
            # Fall back: assume band indices are 0..B-1
            bidx = int(self.target_band_id)
            if bidx >= BHW.shape[0]:
                raise IndexError(
                    f"Target band index {bidx} is out of range for label tensor with {BHW.shape[0]} bands. "
                    "Provide a correct 'band_indices' mapping in label files."
                )

        target_tensor = BHW[bidx].unsqueeze(0).float()  # (1,H,W)
        return in_t, target_tensor


# -----------------------------
# Evaluation helpers
# -----------------------------
def evaluate_model(model: nn.Module, data_loader: DataLoader, device: torch.device,
                   model_name: str, target_band: int, max_batches: int) -> Optional[Dict[str, float]]:
    """Evaluate one model on a target band and aggregate metrics."""
    model.eval()
    all_metrics: List[Dict[str, float]] = []

    print(f"Evaluating model: {model_name}, target band: {target_band}")
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(data_loader, desc=f"Evaluating {model_name}")):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward
            predictions = model(inputs)

            # Denormalize (if utility provided; otherwise no-op)
            predictions = denormalize_prediction(predictions, target_band)
            targets = denormalize_prediction(targets, target_band)

            # Batch-wise metrics
            batch_metrics = batch_calculate_metrics(predictions, targets, data_range=1.0)
            all_metrics.extend(batch_metrics)

            if batch_idx + 1 >= max_batches:
                break

    if not all_metrics:
        return None

    aggregated = aggregate_metrics(all_metrics)
    aggregated["model"] = model_name
    aggregated["target_band"] = target_band
    return aggregated


def load_model(model_path: str, device: torch.device, in_channels: int, img_size: int) -> Optional[nn.Module]:
    """Load a trained single-band model checkpoint safely."""
    model = SpecSwin_SingleBand(in_channels=in_channels, img_size=(img_size, img_size), spatial_dims=2)
    try:
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint

        # Skip 3D weights (5D patch embed) if any
        pe = state_dict.get("model.swinViT.patch_embed.proj.weight", None)
        if pe is not None and hasattr(pe, "shape") and len(pe.shape) == 5:
            print(f"Skipping 3D model weights: {model_path}")
            return None

        model.load_state_dict(state_dict)
        print(f"Loaded model: {model_path}")
    except Exception as e:
        print(f"Failed to load model {model_path}: {e}")
        return None

    model.to(device)
    return model


def find_available_models(checkpoint_dir: str) -> List[Dict]:
    """
    Discover all available models organized by strategy (supports cascade + fine-tuning schemes).
    Strategy directory layout:
      <CHECKPOINT_DIR>/<strategy>/
        models/*.pth
        cascade_levels.txt          # optional (maps Band_xxx -> Level_y)
        fine_tuning_plan.json       # optional (lists fine_tune_bands or bands)
    Returns a list of strategy dicts with metadata and per-band model paths.
    """
    strategies: List[Dict] = []

    if not os.path.isdir(checkpoint_dir):
        print(f"Checkpoint directory not found: {checkpoint_dir}")
        return strategies

    for strategy_dir in sorted(os.listdir(checkpoint_dir)):
        strategy_path = os.path.join(checkpoint_dir, strategy_dir)
        if not os.path.isdir(strategy_path):
            continue

        models_path = os.path.join(strategy_path, "models")
        cascade_file = os.path.join(strategy_path, "cascade_levels.txt")
        fine_tune_file = os.path.join(strategy_path, "fine_tuning_plan.json")

        if not os.path.isdir(models_path):
            continue

        model_files = glob.glob(os.path.join(models_path, "*_best_model.pth"))

        # Parse cascade levels
        cascade_bands: Dict[int, int] = {}
        if os.path.exists(cascade_file):
            try:
                with open(cascade_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if "Band_" in line and "->" in line and "Level_" in line:
                            # Example line: "Band_005 -> Level_0 (Strategy: importance)"
                            parts = line.strip().split(" -> ")
                            if len(parts) != 2:
                                continue
                            band_str = parts[0].replace("Band_", "")
                            level_str = parts[1].split(" ")[0].replace("Level_", "")
                            try:
                                band_num = int(band_str)
                                level_num = int(level_str)
                                cascade_bands[band_num] = level_num
                            except ValueError:
                                continue
            except Exception as e:
                print(f"Warning: could not read cascade file {cascade_file}: {e}")

        # Parse fine-tuning bands
        fine_tune_bands: set[int] = set()
        if os.path.exists(fine_tune_file):
            try:
                with open(fine_tune_file, "r", encoding="utf-8") as f:
                    fine_tune_data = json.load(f)
                    if "fine_tune_bands" in fine_tune_data:
                        fine_tune_bands = set(int(b) for b in fine_tune_data["fine_tune_bands"])
                    elif "bands" in fine_tune_data:
                        fine_tune_bands = set(int(b) for b in fine_tune_data["bands"])
            except Exception as e:
                print(f"Warning: could not read fine-tuning file {fine_tune_file}: {e}")

        # Analyze per-band models
        band_models: Dict[int, Dict] = {}
        cascade_count = 0
        fine_tune_count = 0

        for model_file in model_files:
            filename = os.path.basename(model_file)
            if "band_" not in filename:
                continue
            try:
                band_str = filename.split("band_")[1].split("_")[0]
                band_num = int(band_str)
            except Exception:
                continue

            # Decide phase/level
            if band_num in cascade_bands:
                phase = "cascade"
                level = cascade_bands[band_num]
                cascade_count += 1
            elif band_num in fine_tune_bands or len(cascade_bands) > 0:
                phase = "fine_tune"
                level = -1
                fine_tune_count += 1
            else:
                phase = "unknown"
                level = -1

            band_models[band_num] = {
                "path": model_file,
                "level": level,
                "training_phase": phase,
            }

        if band_models:
            strategies.append({
                "name": strategy_dir,
                "path": strategy_path,
                "band_models": band_models,
                "cascade_bands": cascade_bands,
                "fine_tune_bands": fine_tune_bands,
                "cascade_count": cascade_count,
                "fine_tune_count": fine_tune_count,
                "total_models": len(band_models),
            })

    return strategies


# -----------------------------
# Visualization & reporting
# -----------------------------
def create_visualization(results_df: pd.DataFrame, output_dir: str) -> None:
    """Create charts for evaluation results (bar charts, heatmaps, phase comparisons)."""
    print("\nCreating visualizations...")

    os.makedirs(output_dir, exist_ok=True)

    # Combined figure — strategy comparisons
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Model Performance Comparison Across Strategies", fontsize=16, fontweight="bold")

    metrics = ["psnr_mean", "ergas_mean", "sam_mean", "q_index_mean", "ssim_mean"]
    metric_names = ["PSNR", "ERGAS", "SAM", "Q Index", "SSIM"]

    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        row, col = divmod(i, 3)
        ax = axes[row, col]

        # Mean per strategy
        asc = (name in ("ERGAS", "SAM"))  # lower-better metrics sorted ascending
        strategy_means = results_df.groupby("model")[metric].mean().sort_values(ascending=asc)

        bars = ax.bar(range(len(strategy_means)), strategy_means.values)
        ax.set_title(f"{name} Comparison", fontweight="bold")
        ax.set_xticks(range(len(strategy_means)))
        ax.set_xticklabels(strategy_means.index, rotation=45, ha="right")
        ax.grid(True, alpha=0.3)

        for bar, value in zip(bars, strategy_means.values):
            ax.text(bar.get_x() + bar.get_width()/2.0, bar.get_height(), f"{value:.3f}",
                    ha="center", va="bottom", fontsize=9)

    # Remove the empty subplot if 5 metrics only
    fig.delaxes(axes[1, 2])
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "model_evaluation_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {plot_path}")

    # Heatmaps — strategy x band performance
    plt.figure(figsize=(16, 12))
    if "training_phase" in results_df.columns:
        # Split cascade vs fine-tune
        cascade_data = results_df[results_df["training_phase"] == "cascade"]
        finetune_data = results_df[results_df["training_phase"] == "fine_tune"]

        if not cascade_data.empty:
            plt.subplot(2, 2, 1)
            psnr_cascade = cascade_data.pivot_table(index="model", columns="target_band",
                                                    values="psnr_mean", aggfunc="mean")
            sns.heatmap(psnr_cascade, annot=True, fmt=".2f", cmap="viridis",
                        cbar_kws={"label": "PSNR (dB)"})
            plt.title("PSNR — Cascade Bands", fontweight="bold")
            plt.xlabel("Target Band")
            plt.ylabel("Strategy")

            plt.subplot(2, 2, 2)
            ssim_cascade = cascade_data.pivot_table(index="model", columns="target_band",
                                                    values="ssim_mean", aggfunc="mean")
            sns.heatmap(ssim_cascade, annot=True, fmt=".3f", cmap="viridis",
                        cbar_kws={"label": "SSIM"})
            plt.title("SSIM — Cascade Bands", fontweight="bold")
            plt.xlabel("Target Band")
            plt.ylabel("Strategy")

        if not finetune_data.empty:
            plt.subplot(2, 2, 3)
            psnr_finetune = finetune_data.pivot_table(index="model", columns="target_band",
                                                      values="psnr_mean", aggfunc="mean")
            sns.heatmap(psnr_finetune, annot=True, fmt=".2f", cmap="plasma",
                        cbar_kws={"label": "PSNR (dB)"})
            plt.title("PSNR — Fine-tuning Bands", fontweight="bold")
            plt.xlabel("Target Band")
            plt.ylabel("Strategy")

            plt.subplot(2, 2, 4)
            ssim_finetune = finetune_data.pivot_table(index="model", columns="target_band",
                                                      values="ssim_mean", aggfunc="mean")
            sns.heatmap(ssim_finetune, annot=True, fmt=".3f", cmap="plasma",
                        cbar_kws={"label": "SSIM"})
            plt.title("SSIM — Fine-tuning Bands", fontweight="bold")
            plt.xlabel("Target Band")
            plt.ylabel("Strategy")
    else:
        # Legacy case (no phase column)
        psnr_pivot = results_df.pivot_table(index="model", columns="target_band",
                                            values="psnr_mean", aggfunc="mean")

        plt.subplot(2, 1, 1)
        sns.heatmap(psnr_pivot, annot=True, fmt=".2f", cmap="viridis",
                    cbar_kws={"label": "PSNR (dB)"})
        plt.title("PSNR Heatmap (Strategy × Band)", fontweight="bold")
        plt.xlabel("Target Band")
        plt.ylabel("Strategy")

        ssim_pivot = results_df.pivot_table(index="model", columns="target_band",
                                            values="ssim_mean", aggfunc="mean")

        plt.subplot(2, 1, 2)
        sns.heatmap(ssim_pivot, annot=True, fmt=".3f", cmap="viridis",
                    cbar_kws={"label": "SSIM"})
        plt.title("SSIM Heatmap (Strategy × Band)", fontweight="bold")
        plt.xlabel("Target Band")
        plt.ylabel("Strategy")

    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, "model_evaluation_heatmap.png")
    plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {heatmap_path}")

    # Training phase comparisons
    if "training_phase" in results_df.columns:
        plt.figure(figsize=(15, 10))
        phase_perf = results_df.groupby(["model", "training_phase"]).agg({
            "psnr_mean": "mean",
            "ssim_mean": "mean",
            "ergas_mean": "mean",
            "sam_mean": "mean",
        }).reset_index()

        # PSNR: cascade vs finetune
        plt.subplot(2, 2, 1)
        for strategy in phase_perf["model"].unique():
            sd = phase_perf[phase_perf["model"] == strategy]
            c_psnr = sd[sd["training_phase"] == "cascade"]["psnr_mean"].values
            f_psnr = sd[sd["training_phase"] == "fine_tune"]["psnr_mean"].values
            x = [0, 1]
            y = [c_psnr[0] if len(c_psnr) else 0, f_psnr[0] if len(f_psnr) else 0]
            plt.plot(x, y, "o-", label=strategy, linewidth=2, markersize=8)
        plt.xticks([0, 1], ["Cascade", "Fine-tuning"])
        plt.ylabel("PSNR (dB)")
        plt.title("Cascade vs Fine-tuning — PSNR")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # SSIM: cascade vs finetune
        plt.subplot(2, 2, 2)
        for strategy in phase_perf["model"].unique():
            sd = phase_perf[phase_perf["model"] == strategy]
            c_ssim = sd[sd["training_phase"] == "cascade"]["ssim_mean"].values
            f_ssim = sd[sd["training_phase"] == "fine_tune"]["ssim_mean"].values
            x = [0, 1]
            y = [c_ssim[0] if len(c_ssim) else 0, f_ssim[0] if len(f_ssim) else 0]
            plt.plot(x, y, "s-", label=strategy, linewidth=2, markersize=8)
        plt.xticks([0, 1], ["Cascade", "Fine-tuning"])
        plt.ylabel("SSIM")
        plt.title("Cascade vs Fine-tuning — SSIM")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Cascade levels — PSNR
        cascade_data = results_df[results_df["training_phase"] == "cascade"]
        if not cascade_data.empty and "level" in cascade_data.columns:
            plt.subplot(2, 2, 3)
            lv_perf = cascade_data.groupby(["model", "level"]).agg({"psnr_mean": "mean"}).reset_index()
            for strategy in lv_perf["model"].unique():
                sd = lv_perf[lv_perf["model"] == strategy]
                plt.plot(sd["level"], sd["psnr_mean"], "o-", label=strategy, linewidth=2, markersize=6)
            plt.xlabel("Cascade Level")
            plt.ylabel("PSNR (dB)")
            plt.title("PSNR by Cascade Level")
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.subplot(2, 2, 4)
            lv_perf_ssim = cascade_data.groupby(["model", "level"]).agg({"ssim_mean": "mean"}).reset_index()
            for strategy in lv_perf_ssim["model"].unique():
                sd = lv_perf_ssim[lv_perf_ssim["model"] == strategy]
                plt.plot(sd["level"], sd["ssim_mean"], "s-", label=strategy, linewidth=2, markersize=6)
            plt.xlabel("Cascade Level")
            plt.ylabel("SSIM")
            plt.title("SSIM by Cascade Level")
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        phase_path = os.path.join(output_dir, "cascade_vs_finetune_performance.png")
        plt.savefig(phase_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {phase_path}")


def print_summary(results_df: pd.DataFrame) -> None:
    """Print a textual summary of evaluation results."""
    print("\n" + "=" * 80)
    print("Model Evaluation Summary — Cascade Training + Fine-tuning")
    print("=" * 80)

    # 1) Per-strategy stats
    strategy_summary = results_df.groupby("model").agg({
        "psnr_mean": ["mean", "std"],
        "ergas_mean": ["mean", "std"],
        "sam_mean": ["mean", "std"],
        "q_index_mean": ["mean", "std"],
        "ssim_mean": ["mean", "std"],
        "target_band": "count",
    }).round(4)
    print("\n1. Average performance by strategy:")
    print(strategy_summary)

    # 2) By training phase
    if "training_phase" in results_df.columns:
        print("\n2. By training phase:")
        phase_summary = results_df.groupby(["model", "training_phase"]).agg({
            "psnr_mean": "mean",
            "ssim_mean": "mean",
            "ergas_mean": "mean",
            "sam_mean": "mean",
            "target_band": "count",
        }).round(4)
        print(phase_summary)

        print("\n3. Overall comparison: Cascade vs Fine-tuning:")
        overall_phase = results_df.groupby("training_phase").agg({
            "psnr_mean": ["mean", "std"],
            "ssim_mean": ["mean", "std"],
            "ergas_mean": ["mean", "std"],
            "sam_mean": ["mean", "std"],
        }).round(4)
        print(overall_phase)

    # 3) Best strategies per metric
    print("\n4. Best strategies (by metric):")
    best_psnr = results_df.groupby("model")["psnr_mean"].mean().sort_values(ascending=False)
    best_ssim = results_df.groupby("model")["ssim_mean"].mean().sort_values(ascending=False)
    best_ergas = results_df.groupby("model")["ergas_mean"].mean().sort_values(ascending=True)
    best_sam = results_df.groupby("model")["sam_mean"].mean().sort_values(ascending=True)
    best_q = results_df.groupby("model")["q_index_mean"].mean().sort_values(ascending=False)

    if len(best_psnr) > 0:
        print(f"  Highest PSNR   : {best_psnr.index[0]} ({best_psnr.iloc[0]:.3f} dB)")
    if len(best_ssim) > 0:
        print(f"  Highest SSIM   : {best_ssim.index[0]} ({best_ssim.iloc[0]:.3f})")
    if len(best_ergas) > 0:
        print(f"  Lowest ERGAS   : {best_ergas.index[0]} ({best_ergas.iloc[0]:.3f})")
    if len(best_sam) > 0:
        print(f"  Lowest SAM     : {best_sam.index[0]} ({best_sam.iloc[0]:.3f}°)")
    if len(best_q) > 0:
        print(f"  Highest Q Index: {best_q.index[0]} ({best_q.iloc[0]:.3f})")

    # 4) Cascade-level analysis
    if "level" in results_df.columns and "training_phase" in results_df.columns:
        cascade_data = results_df[results_df["training_phase"] == "cascade"]
        if not cascade_data.empty:
            print("\n5. Cascade levels summary:")
            level_summary = cascade_data.groupby("level").agg({
                "psnr_mean": "mean",
                "ssim_mean": "mean",
                "target_band": "count",
            }).round(3)
            print(level_summary)

    # 5) Composite ranking (normalized scores)
    print("\n6. Composite ranking (normalized score average):")
    normalized = results_df.groupby("model").agg({
        "psnr_mean": "mean",
        "ergas_mean": "mean",
        "sam_mean": "mean",
        "q_index_mean": "mean",
        "ssim_mean": "mean",
    })

    if normalized.empty:
        print("No data to rank.")
        return

    # Higher-better: PSNR, Q, SSIM; Lower-better: ERGAS, SAM
    def _minmax(x: pd.Series, invert: bool = False) -> pd.Series:
        denom = (x.max() - x.min()) or 1.0
        s = (x - x.min()) / denom
        return 1.0 - s if invert else s

    comp = pd.DataFrame(index=normalized.index)
    comp["PSNR"] = _minmax(normalized["psnr_mean"])
    comp["Q"] = _minmax(normalized["q_index_mean"])
    comp["SSIM"] = _minmax(normalized["ssim_mean"])
    comp["ERGAS"] = _minmax(normalized["ergas_mean"], invert=True)
    comp["SAM"] = _minmax(normalized["sam_mean"], invert=True)
    comp["Composite"] = comp.mean(axis=1)

    ranking = comp["Composite"].sort_values(ascending=False)
    for i, (name, score) in enumerate(ranking.items(), start=1):
        print(f"  {i}. {name}: {score:.3f}")


# -----------------------------
# Main
# -----------------------------
def main():
    paths = resolve_paths()
    INPUT_DIR = paths["INPUT_DIR"]
    LABEL_DIR = paths["LABEL_DIR"]
    CHECKPOINT_DIR = paths["CHECKPOINT_DIR"]
    OUTPUT_DIR = paths["OUTPUT_DIR"]

    DEVICE = torch.device(os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))
    MAX_BATCHES = int(os.environ.get("MAX_BATCHES", "50"))
    NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "2"))
    IN_CHANNELS = int(os.environ.get("IN_CHANNELS", "16"))
    IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", "128"))

    print(f"Using device: {DEVICE}")
    print("Resolved paths:")
    print(f"  INPUT_DIR      = {INPUT_DIR}")
    print(f"  LABEL_DIR      = {LABEL_DIR}")
    print(f"  CHECKPOINT_DIR = {CHECKPOINT_DIR}")
    print(f"  OUTPUT_DIR     = {OUTPUT_DIR}")

    # Discover strategies and models
    strategies = find_available_models(CHECKPOINT_DIR)
    print(f"Found {len(strategies)} training strategies:")
    for s in strategies:
        print(f"\nStrategy: {s['name']}")
        print(f"  Total models        : {s['total_models']}")
        print(f"  Cascade band count  : {s['cascade_count']}")
        print(f"  Fine-tune band count: {s['fine_tune_count']}")

        if s["cascade_bands"]:
            levels: Dict[int, List[int]] = {}
            for band, level in s["cascade_bands"].items():
                levels.setdefault(level, []).append(band)
            print("  Cascade level distribution:")
            for lv in sorted(levels.keys()):
                sample = levels[lv][:5]
                tail = f", ...+{len(levels[lv]) - 5}" if len(levels[lv]) > 5 else ""
                print(f"    Level_{lv}: {len(levels[lv])} bands {sample}{tail}")

        if s["fine_tune_bands"]:
            ft_bands = sorted(s["fine_tune_bands"])
            print(f"  Fine-tune band range: {ft_bands[0]} - {ft_bands[-1]} (total {len(ft_bands)})")

    # Choose test bands (a few from cascade + a stride from finetune)
    test_bands: List[int] = []
    for s in strategies:
        if s["cascade_bands"]:
            cascade_bands = sorted(list(s["cascade_bands"].keys()))
            test_bands.extend(cascade_bands[:5])  # first 5 cascade bands as representatives
            break  # one strategy's cascade bands are sufficient as representatives

    if strategies and strategies[0]["fine_tune_bands"]:
        ft = sorted(list(strategies[0]["fine_tune_bands"]))
        if ft:
            stride = max(1, len(ft) // 5)  # pick ~5 samples across the range
            test_bands.extend(ft[::stride])

    # Unique & limit
    test_bands = sorted(list(set(test_bands)))[:10]
    print(f"\nSelected test bands: {test_bands}")

    all_results: List[Dict[str, float]] = []

    # Evaluate each strategy on the selected bands
    for s in strategies:
        print(f"\nEvaluating strategy: {s['name']}")
        for band_num in sorted(s["band_models"].keys()):
            if band_num not in test_bands:
                continue

            model_info = s["band_models"][band_num]
            model_path = model_info["path"]
            level = model_info["level"]
            phase = model_info["training_phase"]
            phase_label = "Cascade" if phase == "cascade" else "Fine-tune" if phase == "fine_tune" else "Unknown"
            level_label = f"L{level}" if level >= 0 else "N/A"

            print(f"  Band {band_num} ({phase_label} {level_label})")

            # Data loader
            dataset = SingleBandDataset(INPUT_DIR, LABEL_DIR, band_num, in_channels=IN_CHANNELS)
            loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

            # Model
            model = load_model(model_path, DEVICE, in_channels=IN_CHANNELS, img_size=IMAGE_SIZE)
            if model is None:
                continue

            # Evaluate
            try:
                metrics = evaluate_model(model, loader, DEVICE, s["name"], band_num, max_batches=MAX_BATCHES)
                if metrics:
                    metrics["level"] = level
                    metrics["training_phase"] = phase
                    all_results.append(metrics)
                    # Print a brief line
                    if "psnr_mean" in metrics and "ssim_mean" in metrics:
                        print(f"    PSNR: {metrics['psnr_mean']:.3f}±{metrics.get('psnr_std', np.nan):.3f}, "
                              f"SSIM: {metrics['ssim_mean']:.3f}±{metrics.get('ssim_std', np.nan):.3f}")
            except Exception as e:
                print(f"    Evaluation failed: {e}")
                continue

    # Save & visualize
    if all_results:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        results_df = pd.DataFrame(all_results)
        results_path = os.path.join(OUTPUT_DIR, "model_evaluation_results.csv")
        results_df.to_csv(results_path, index=False)
        print(f"\nResults saved to: {results_path}")

        create_visualization(results_df, OUTPUT_DIR)
        print_summary(results_df)
    else:
        print("No models were successfully evaluated.")


if __name__ == "__main__":
    main()
