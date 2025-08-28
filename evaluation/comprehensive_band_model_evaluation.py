
# Description:
#   Compare reconstruction performance across many spectral bands and multiple model strategies.
#   For a selected set of bands, this script:
#     - finds per-band checkpoints under each strategy,
#     - evaluates 6 metrics (PSNR, ERGAS, SAM, Q-Index, SSIM, RMSE),
#     - saves a CSV with results,
#     - creates line plots and heatmaps for comparison,
#     - prints summary statistics and a composite ranking.
#
# Environment variables (override as needed):
#   PROJECT_ROOT   (default: parent directory of this file)
#   DATA_DIR       (default: <PROJECT_ROOT>/dataset)
#   INPUT_DIR      (default: <DATA_DIR>/input_restacked_16)
#   LABEL_DIR      (default: <DATA_DIR>/label)
#   CHECKPOINT_DIR (default: <PROJECT_ROOT>/checkpoints)
#   OUTPUT_DIR     (default: <PROJECT_ROOT>/outputs)
#   STRATEGIES     (default: "physical_repeated_full_219_bands,physical_full_219_bands")
#                  Comma-separated list of strategy folders under CHECKPOINT_DIR
#   MAX_SAMPLES    (default: 100)  number of .pt samples per band to evaluate (upper bound)
#   BATCH_SIZE     (default: 8)
#   MAX_BATCHES    (default: 10)   limit number of batches per band for speed
#   NUM_TEST_BANDS (default: 20)   number of bands to evaluate (evenly spaced subset)
#   IN_CHANNELS    (default: 16)   input channels expected by the per-band model
#   IMAGE_SIZE     (default: 128)  spatial size expected by the model
#   DEVICE         (default: auto; "cuda" if available else "cpu")

from __future__ import annotations
import os
import sys
import glob
import json
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

# If helper modules live next to this file, keep local import path.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Image quality metrics (expected to be provided by a sibling module)
from image_quality_metrics import (
    calculate_psnr, calculate_ergas, calculate_sam,
    calculate_q_index, calculate_ssim, calculate_rmse
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


def parse_strategies(checkpoint_dir: str) -> List[str]:
    """
    Parse strategy directory names from STRATEGIES env var (comma-separated).
    Fallback to two common strategy folders if not provided.
    """
    env_val = os.environ.get(
        "STRATEGIES",
        "physical_repeated_full_219_bands,physical_full_219_bands"
    )
    strategies = [s.strip() for s in env_val.split(",") if s.strip()]
    # Keep only those that actually exist under CHECKPOINT_DIR
    existing = []
    for s in strategies:
        if os.path.isdir(os.path.join(checkpoint_dir, s, "models")):
            existing.append(s)
    if not existing:
        print("No strategy directories found from STRATEGIES; "
              "make sure they exist under CHECKPOINT_DIR/<strategy>/models")
    return existing


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


class SingleBandDataset(Dataset):
    """
    Dataset that loads paired input (C,H,W) and a single target band (1,H,W) for a given band index.
    Expects *.pt files in INPUT_DIR and LABEL_DIR. Each .pt should be either:
      - a dict with keys: 'input' (C,H,W) and 'label' (B,H,W) or similar, or
      - a tensor of shape (H,W,C) or (C,H,W) (less strict; we try to handle both).
    """
    def __init__(self, input_dir: str, label_dir: str, target_band_idx: int, max_samples: int = 50):
        self.input_dir = input_dir
        self.label_dir = label_dir
        self.target_band_idx = target_band_idx

        all_input_files = sorted(glob.glob(os.path.join(input_dir, "*.pt")))
        all_label_files = sorted(glob.glob(os.path.join(label_dir, "*.pt")))
        num_files = min(max_samples, len(all_input_files), len(all_label_files))
        self.input_files = all_input_files[:num_files]
        self.label_files = all_label_files[:num_files]

        print(f"Band {target_band_idx}: evaluating with {len(self.input_files)} samples")

    def __len__(self) -> int:
        return len(self.input_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input_data = torch.load(self.input_files[idx], map_location="cpu")
        if isinstance(input_data, dict) and "input" in input_data:
            input_tensor = input_data["input"].float()  # (C, H, W)
            if input_tensor.ndim == 3 and input_tensor.shape[0] in (5, 16):
                pass
            elif input_tensor.ndim == 3 and input_tensor.shape[-1] in (5, 16):
                input_tensor = input_tensor.permute(2, 0, 1).float()
            else:
                raise ValueError("Unsupported input tensor shape in dict['input']")
        else:
            # tensor-only path
            t = input_data if isinstance(input_data, torch.Tensor) else torch.tensor(input_data)
            if t.ndim == 3 and t.shape[0] in (5, 16):
                input_tensor = t.float()
            elif t.ndim == 3 and t.shape[-1] in (5, 16):
                input_tensor = t.permute(2, 0, 1).float()
            else:
                raise ValueError("Unsupported standalone input tensor shape")

        label_data = torch.load(self.label_files[idx], map_location="cpu")
        if isinstance(label_data, dict) and "label" in label_data:
            label_full = label_data["label"]
            if label_full.ndim == 3:  # (B, H, W)
                if self.target_band_idx >= label_full.shape[0]:
                    raise IndexError("target_band_idx out of range for label tensor")
                label_tensor = label_full[self.target_band_idx].unsqueeze(0).float()
            else:
                raise ValueError("Unsupported label tensor shape in dict['label']")
        else:
            t = label_data if isinstance(label_data, torch.Tensor) else torch.tensor(label_data)
            if t.ndim == 3 and self.target_band_idx < t.shape[-1]:  # (H, W, B)
                label_tensor = t[:, :, self.target_band_idx].unsqueeze(0).float()
            else:
                raise ValueError("Unsupported standalone label tensor shape")

        return input_tensor, label_tensor


# -----------------------------
# Metrics
# -----------------------------
def calculate_all_metrics_safe(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """
    Compute all 6 metrics safely on [0,1]-clipped arrays:
      PSNR, ERGAS, SAM, Q-Index, SSIM, RMSE
    """
    pred_np = predictions.detach().cpu().numpy() if isinstance(predictions, torch.Tensor) else predictions
    tgt_np = targets.detach().cpu().numpy() if isinstance(targets, torch.Tensor) else targets

    pred_np = np.clip(pred_np, 0, 1)
    tgt_np = np.clip(tgt_np, 0, 1)

    metrics: Dict[str, float] = {}

    try:
        v = calculate_psnr(pred_np, tgt_np, max_val=1.0)
        metrics["PSNR"] = 100.0 if not np.isfinite(v) else v
    except Exception:
        metrics["PSNR"] = np.nan

    try:
        v = calculate_ergas(pred_np, tgt_np)
        metrics["ERGAS"] = 0.0 if not np.isfinite(v) else v
    except Exception:
        metrics["ERGAS"] = np.nan

    try:
        v = calculate_sam(pred_np, tgt_np)
        metrics["SAM"] = 0.0 if not np.isfinite(v) else v
    except Exception:
        metrics["SAM"] = np.nan

    try:
        v = calculate_q_index(pred_np, tgt_np)
        metrics["Q_Index"] = 1.0 if not np.isfinite(v) else v
    except Exception:
        metrics["Q_Index"] = np.nan

    try:
        v = calculate_ssim(pred_np, tgt_np)
        metrics["SSIM"] = 1.0 if not np.isfinite(v) else v
    except Exception:
        metrics["SSIM"] = np.nan

    try:
        v = calculate_rmse(pred_np, tgt_np)
        metrics["RMSE"] = 0.0 if not np.isfinite(v) else v
    except Exception:
        metrics["RMSE"] = np.nan

    return metrics


# -----------------------------
# Model discovery / loading
# -----------------------------
def find_all_models_and_bands(checkpoint_dir: str, strategies: List[str]) -> Tuple[Dict[str, Dict[int, str]], List[int]]:
    """
    Discover per-band checkpoints for each strategy under CHECKPOINT_DIR/<strategy>/models.
    Priority per band: fine_tune/finetune > best_model > cascade > any band_XXX_*.pth
    Returns:
        models_info: {strategy_name: {band_num: checkpoint_path}}
        all_bands:   sorted list of all discovered band indices
    """
    print("Scanning available models...")
    models_info: Dict[str, Dict[int, str]] = {}
    all_bands: set[int] = set()

    for strategy in strategies:
        strategy_path = os.path.join(checkpoint_dir, strategy)
        models_path = os.path.join(strategy_path, "models")
        if not os.path.isdir(models_path):
            continue

        band_models: Dict[int, str] = {}
        model_files = glob.glob(os.path.join(models_path, "*.pth"))
        for fp in model_files:
            fn = os.path.basename(fp)
            if "band_" not in fn:
                continue
            try:
                band_part = fn.split("band_")[1]
                band_num = int(band_part.split("_")[0])
            except Exception:
                continue

            # Select best file per band by priority
            choose = False
            if band_num not in band_models:
                choose = True
            else:
                # Upgrade if better type appears
                current = band_models[band_num]
                if "fine_tune" in fn or "finetune" in fn:
                    choose = True
                elif "best_model" in fn and "cascade" in os.path.basename(current):
                    choose = True

            if choose:
                band_models[band_num] = fp
                all_bands.add(band_num)

        if band_models:
            models_info[strategy] = band_models
            print(f"Strategy {strategy}: found {len(band_models)} band models")

    return models_info, sorted(list(all_bands))


def load_model_safe(model_path: str, device: torch.device, in_channels: int, img_size: int) -> Optional[nn.Module]:
    """
    Safely load a SpecSwin-based single-band model (2D). Returns None on failure.
    """
    model = SpecSwin_SingleBand(in_channels=in_channels, img_size=(img_size, img_size))

    try:
        checkpoint = torch.load(model_path, map_location=device)

        state_dict = checkpoint
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        # Skip 3D models (5D patch_embed weights)
        pw = state_dict.get("model.swinViT.patch_embed.proj.weight", None)
        if pw is not None and hasattr(pw, "shape") and len(pw.shape) == 5:
            print(f"Skipping 3D weights: {model_path}")
            return None

        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"Failed to load model {model_path}: {e}")
        return None


# -----------------------------
# Evaluation
# -----------------------------
def evaluate_model_on_band(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    max_batches: int
) -> Optional[Dict[str, float]]:
    """
    Evaluate one model on one band over a limited number of batches.
    Returns average metrics dict or None.
    """
    metrics_list: List[Dict[str, float]] = []
    with torch.no_grad():
        for bidx, (inputs, targets) in enumerate(data_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            predictions = model(inputs)

            # per-sample metrics within the batch
            for i in range(predictions.shape[0]):
                pred_sample = predictions[i].unsqueeze(0)
                tgt_sample = targets[i].unsqueeze(0)
                m = calculate_all_metrics_safe(pred_sample, tgt_sample)
                metrics_list.append(m)

            if bidx + 1 >= max_batches:
                break

    if not metrics_list:
        return None

    avg: Dict[str, float] = {}
    for name in ["PSNR", "ERGAS", "SAM", "Q_Index", "SSIM", "RMSE"]:
        vals = [m[name] for m in metrics_list if not np.isnan(m[name])]
        avg[name] = float(np.mean(vals)) if vals else np.nan
    return avg


# -----------------------------
# Plotting
# -----------------------------
def create_line_plots(results_df: pd.DataFrame, save_dir: str) -> None:
    """
    Create 6 subplots (PSNR, ERGAS, SAM, Q-Index, SSIM, RMSE) vs Band, colored by Strategy.
    """
    print("\nCreating line plots...")
    sns.set_style("whitegrid")

    metrics_info = [
        ("PSNR", "PSNR (dB)", True),   # higher is better
        ("ERGAS", "ERGAS", False),     # lower is better
        ("SAM", "SAM (°)", False),     # lower is better
        ("Q_Index", "Q-Index", True),  # higher is better
        ("SSIM", "SSIM", True),        # higher is better
        ("RMSE", "RMSE", False),       # lower is better
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Reconstruction Performance Across Bands by Strategy", fontsize=16, fontweight="bold")

    strategies = results_df["Strategy"].unique()
    colors = plt.cm.Set1(np.linspace(0, 1, len(strategies)))
    color_map = dict(zip(strategies, colors))

    for idx, (metric, ylabel, higher_better) in enumerate(metrics_info):
        r, c = idx // 3, idx % 3
        ax = axes[r, c]

        for strategy in strategies:
            sdf = results_df[results_df["Strategy"] == strategy].sort_values("Band").dropna(subset=[metric])
            if not sdf.empty:
                ax.plot(sdf["Band"], sdf[metric], marker="o", linewidth=2, markersize=4,
                        label=strategy, color=color_map[strategy], alpha=0.85)

        ax.set_xlabel("Band Index")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{metric} Comparison", fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        if metric == "PSNR":
            ax.set_ylim(bottom=0)
        elif metric in ["SSIM", "Q_Index"]:
            ax.set_ylim(0, 1)
        elif metric in ["ERGAS", "SAM", "RMSE"]:
            ax.set_ylim(bottom=0)

    plt.tight_layout()
    out_path = os.path.join(save_dir, "comprehensive_band_model_line_plots.png")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Line plots saved to: {out_path}")

    # Individual detailed plots per metric
    for metric, ylabel, _ in metrics_info:
        plt.figure(figsize=(12, 8))
        for strategy in strategies:
            sdf = results_df[results_df["Strategy"] == strategy].sort_values("Band").dropna(subset=[metric])
            if not sdf.empty:
                plt.plot(sdf["Band"], sdf[metric], marker="o", linewidth=3, markersize=6, label=strategy, alpha=0.85)
        plt.xlabel("Band Index", fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(f"{metric} — Band-wise Comparison by Strategy", fontsize=14, fontweight="bold")
        plt.grid(True, alpha=0.3)
        plt.legend()
        if metric == "PSNR":
            plt.ylim(bottom=0)
        elif metric in ["SSIM", "Q_Index"]:
            plt.ylim(0, 1)
        elif metric in ["ERGAS", "SAM", "RMSE"]:
            plt.ylim(bottom=0)

        path = os.path.join(save_dir, f"{metric}_band_comparison.png")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
    print(f"Individual metric plots saved under: {save_dir}")

    create_heatmaps(results_df, save_dir)


def create_heatmaps(results_df: pd.DataFrame, save_dir: str) -> None:
    """
    Create heatmaps (Strategy x Band) for each metric.
    """
    print("Creating heatmaps...")
    metrics = ["PSNR", "ERGAS", "SAM", "Q_Index", "SSIM", "RMSE"]

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("Performance Heatmaps by Strategy and Band", fontsize=16, fontweight="bold")

    for idx, metric in enumerate(metrics):
        r, c = idx // 3, idx % 3
        ax = axes[r, c]

        pivot = results_df.pivot_table(index="Strategy", columns="Band", values=metric, aggfunc="mean")
        cmap = "viridis" if metric in ["PSNR", "Q_Index", "SSIM"] else "viridis_r"
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap=cmap, ax=ax, cbar_kws={"label": metric})
        ax.set_title(f"{metric} Heatmap", fontweight="bold")
        ax.set_xlabel("Band Index")
        ax.set_ylabel("Strategy")

    plt.tight_layout()
    out_path = os.path.join(save_dir, "comprehensive_band_model_heatmaps.png")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Heatmaps saved to: {out_path}")


# -----------------------------
# Reporting
# -----------------------------
def print_summary_statistics(results_df: pd.DataFrame) -> None:
    """
    Print summary statistics and a composite ranking across metrics.
    """
    print("\n" + "=" * 80)
    print("Summary of Band-Model Evaluation Results")
    print("=" * 80)

    # 1) Per-strategy averages
    print("\n1. Average performance by strategy:")
    strategy_stats = results_df.groupby("Strategy").agg({
        "PSNR": ["mean", "std", "count"],
        "ERGAS": ["mean", "std"],
        "SAM": ["mean", "std"],
        "Q_Index": ["mean", "std"],
        "SSIM": ["mean", "std"],
        "RMSE": ["mean", "std"],
    }).round(3)
    print(strategy_stats)

    # 2) Best strategy per metric
    print("\n2. Best strategy per metric:")
    for metric in ["PSNR", "ERGAS", "SAM", "Q_Index", "SSIM", "RMSE"]:
        means = results_df.groupby("Strategy")[metric].mean()
        if means.empty:
            continue
        if metric in ["PSNR", "Q_Index", "SSIM"]:
            best_strategy, best_value = means.idxmax(), means.max()
            direction = "highest"
        else:
            best_strategy, best_value = means.idxmin(), means.min()
            direction = "lowest"
        print(f"  {metric} ({direction}): {best_strategy} ({best_value:.3f})")

    # 3) Band performance overview
    print("\n3. Band-wise performance (mean PSNR & SSIM):")
    band_stats = results_df.groupby("Band").agg({"PSNR": "mean", "SSIM": "mean"}).round(3)
    print(band_stats)

    # Best / worst band by PSNR
    if not band_stats.empty:
        best_band_psnr = band_stats["PSNR"].idxmax()
        worst_band_psnr = band_stats["PSNR"].idxmin()
        print(f"\nBand with highest PSNR: {best_band_psnr} ({band_stats.loc[best_band_psnr, 'PSNR']:.3f} dB)")
        print(f"Band with lowest PSNR : {worst_band_psnr} ({band_stats.loc[worst_band_psnr, 'PSNR']:.3f} dB)")

    # 4) Composite ranking across normalized metrics
    print("\n4. Composite strategy ranking (normalized scores):")
    strategy_means = results_df.groupby("Strategy").agg({
        "PSNR": "mean", "ERGAS": "mean", "SAM": "mean",
        "Q_Index": "mean", "SSIM": "mean", "RMSE": "mean"
    })

    if strategy_means.empty:
        print("No data for composite ranking.")
        return

    norm = pd.DataFrame(index=strategy_means.index)
    # Higher is better
    for metric in ["PSNR", "Q_Index", "SSIM"]:
        col = strategy_means[metric]
        denom = (col.max() - col.min()) or 1.0
        norm[metric] = (col - col.min()) / denom
    # Lower is better
    for metric in ["ERGAS", "SAM", "RMSE"]:
        col = strategy_means[metric]
        denom = (col.max() - col.min()) or 1.0
        norm[metric] = 1.0 - (col - col.min()) / denom

    composite = norm.mean(axis=1).sort_values(ascending=False)
    print("Ranking (composite score):")
    for i, (name, score) in enumerate(composite.items(), start=1):
        print(f"  {i}. {name}: {score:.3f}")


# -----------------------------
# Main
# -----------------------------
def main():
    # Resolve config from env
    paths = resolve_paths()
    INPUT_DIR = paths["INPUT_DIR"]
    LABEL_DIR = paths["LABEL_DIR"]
    CHECKPOINT_DIR = paths["CHECKPOINT_DIR"]
    OUTPUT_DIR = paths["OUTPUT_DIR"]

    # Runtime controls
    DEVICE = torch.device(os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    IN_CHANNELS = int(os.environ.get("IN_CHANNELS", "16"))
    IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", "128"))
    MAX_SAMPLES = int(os.environ.get("MAX_SAMPLES", "100"))
    BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))
    MAX_BATCHES = int(os.environ.get("MAX_BATCHES", "10"))
    NUM_TEST_BANDS = int(os.environ.get("NUM_TEST_BANDS", "20"))

    print(f"Using device: {DEVICE}")
    print("Resolved paths:")
    print(f"  INPUT_DIR      = {INPUT_DIR}")
    print(f"  LABEL_DIR      = {LABEL_DIR}")
    print(f"  CHECKPOINT_DIR = {CHECKPOINT_DIR}")
    print(f"  OUTPUT_DIR     = {OUTPUT_DIR}")

    strategies = parse_strategies(CHECKPOINT_DIR)
    if not strategies:
        print("No strategies to evaluate. Set STRATEGIES env var appropriately.")
        return
    print(f"Strategies: {strategies}")

    # Discover models and bands
    models_info, all_bands = find_all_models_and_bands(CHECKPOINT_DIR, strategies)
    if not all_bands:
        print("No band models were found.")
        return

    print(f"\nDiscovered {len(models_info)} strategy(ies), covering {len(all_bands)} band(s)")
    print(f"Band range: {min(all_bands)} - {max(all_bands)}")

    # Select a subset of bands evenly
    num_test = min(NUM_TEST_BANDS, len(all_bands))
    step = max(1, len(all_bands) // num_test)
    test_bands = all_bands[::step][:num_test]
    print(f"Selected test bands: {test_bands}")

    # Results collection
    all_results: List[Dict[str, float]] = []

    # Evaluate each strategy on selected bands
    for strategy_name, band_models in models_info.items():
        print(f"\nEvaluating strategy: {strategy_name}")
        for band_idx in tqdm(test_bands, desc=f"Evaluating {strategy_name}"):
            if band_idx not in band_models:
                print(f"  Skipping band {band_idx} (no model)")
                continue

            model_path = band_models[band_idx]
            model = load_model_safe(model_path, DEVICE, in_channels=IN_CHANNELS, img_size=IMAGE_SIZE)
            if model is None:
                print(f"  Skipping band {band_idx} (failed to load model)")
                continue

            try:
                dataset = SingleBandDataset(INPUT_DIR, LABEL_DIR, band_idx, max_samples=MAX_SAMPLES)
                loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

                metrics = evaluate_model_on_band(model, loader, DEVICE, max_batches=MAX_BATCHES)
                if metrics:
                    record = {"Strategy": strategy_name, "Band": band_idx, **metrics}
                    all_results.append(record)
                    print(f"  Band {band_idx}: PSNR={metrics['PSNR']:.2f} dB, SSIM={metrics['SSIM']:.3f}")
                else:
                    print(f"  Band {band_idx}: evaluation returned no metrics")
            except Exception as e:
                print(f"  Error evaluating band {band_idx}: {e}")
                continue

    if not all_results:
        print("No successful evaluations.")
        return

    # Save results CSV
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results_df = pd.DataFrame(all_results)
    results_csv = os.path.join(OUTPUT_DIR, "comprehensive_band_model_results.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"\nResults saved to: {results_csv}")

    # Plots
    create_line_plots(results_df, OUTPUT_DIR)

    # Summary
    print_summary_statistics(results_df)


if __name__ == "__main__":
    main()
