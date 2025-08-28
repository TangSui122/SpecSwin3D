from __future__ import annotations
import os
import sys
import glob
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import random
from typing import List, Optional, Tuple, Dict

# ----------------------------
# Path resolution (soft-coded)
# ----------------------------
def resolve_paths() -> Dict[str, str]:
    """
    Resolve directories using environment variables with safe defaults:
      - PROJECT_ROOT: default = parent directory of this file
      - DATA_DIR: default = <PROJECT_ROOT>/dataset
      - CHECKPOINT_DIR: default = <PROJECT_ROOT>/checkpoints
      - VIS_DIR: default = <PROJECT_ROOT>/visualize/visualization_results
      - MODEL_DIRS: optional comma-separated list of directories containing band_XXX_*.pth
    """
    this_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.environ.get("PROJECT_ROOT", os.path.dirname(this_dir))
    data_dir = os.environ.get("DATA_DIR", os.path.join(project_root, "dataset"))
    checkpoint_dir = os.environ.get("CHECKPOINT_DIR", os.path.join(project_root, "checkpoints"))
    vis_dir = os.environ.get(
        "VIS_DIR",
        os.path.join(project_root, "visualize", "visualization_results")
    )
    model_dirs_env = os.environ.get("MODEL_DIRS", "").strip()

    return {
        "PROJECT_ROOT": project_root,
        "DATA_DIR": data_dir,
        "CHECKPOINT_DIR": checkpoint_dir,
        "VIS_DIR": vis_dir,
        "MODEL_DIRS_ENV": model_dirs_env
    }

def get_model_dirs(checkpoint_dir: str, model_dirs_env: str) -> Tuple[List[str], List[str]]:
    """
    Determine model directories to search for per-band checkpoints.

    If MODEL_DIRS env var is provided (comma-separated), use those.
    Otherwise, try common strategy subfolders under CHECKPOINT_DIR (relative, not absolute).
    Returns (dirs, names)
    """
    if model_dirs_env:
        dirs = [d.strip() for d in model_dirs_env.split(",") if d.strip()]
        names = [os.path.basename(os.path.normpath(d)) for d in dirs]
        return dirs, names

    # Fallback to relative strategy folders under CHECKPOINT_DIR (no hardcoded absolute paths)
    # You can rename or extend this list without exposing local paths.
    candidates = [
        "variance_importance_full_219_bands",
        "uniform_full_219_bands",
        "spectral_physics_importance_full_219_bands",
        "physical_full_219_bands",
        "mutual_info_importance_full_219_bands",
        "correlation_importance_full_219_bands",
    ]
    dirs = []
    names = []
    for c in candidates:
        model_dir = os.path.join(checkpoint_dir, c, "models")
        if os.path.isdir(model_dir):
            dirs.append(model_dir)
            names.append(c)
    return dirs, names


# ----------------------------
# Model and dataset
# ----------------------------
from monai.networks.nets import SwinUNETR

class SpecSwin_16Band(torch.nn.Module):
    def __init__(self, in_channels=16, out_channels=219, img_size=(128, 128)):
        super().__init__()
        self.model = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=48,
            spatial_dims=2,
            use_checkpoint=False
        )
    def forward(self, x):
        return self.model(x)

class RestackedSpectralDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper that loads .pt files from:
      <DATA_DIR>/input_restacked_16_with_indices and <DATA_DIR>/label (by default)
    """
    def __init__(self, input_dir: str, label_dir: str):
        self.input_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".pt")])
        self.label_files = sorted([f for f in os.listdir(label_dir) if f.endswith(".pt")])
        self.input_dir = input_dir
        self.label_dir = label_dir
        assert len(self.input_files) == len(self.label_files), "Input and label file count mismatch"

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.input_files[idx])
        label_path = os.path.join(self.label_dir, self.label_files[idx])
        input_data = torch.load(input_path, map_location="cpu")
        label_data = torch.load(label_path, map_location="cpu")
        input_tensor = input_data["input"] if isinstance(input_data, dict) else input_data
        label_tensor = label_data["label"] if isinstance(label_data, dict) else label_data
        return input_tensor.float(), label_tensor.float()


# ----------------------------
# Utilities
# ----------------------------
def load_model(model_path: str, device: torch.device) -> SpecSwin_16Band:
    """Load a trained model checkpoint and return a ready-to-eval model."""
    print(f"Loading model: {model_path}")
    model = SpecSwin_16Band(in_channels=16, out_channels=1, img_size=(128, 128)).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    epoch = checkpoint.get("epoch", "Unknown")
    val_loss = checkpoint.get("val_loss", None)
    if isinstance(val_loss, (int, float)):
        print(f"Model loaded. Epoch={epoch}, ValLoss={val_loss:.6f}")
    else:
        print(f"Model loaded. Epoch={epoch}, ValLoss={val_loss}")
    return model

def find_band_checkpoint(model_dir: str, band_idx: int) -> Optional[str]:
    """
    Find a checkpoint for the given band in model_dir.
    Preference: band_xxx_best_model.pth, otherwise any band_xxx_*.pth
    """
    best = os.path.join(model_dir, f"band_{band_idx:03d}_best_model.pth")
    if os.path.exists(best):
        return best
    pattern = os.path.join(model_dir, f"band_{band_idx:03d}_*.pth")
    matches = sorted(glob.glob(pattern))
    if matches:
        return matches[0]
    return None


# ----------------------------
# Visualization helpers
# ----------------------------
def visualize_input_16bands(input_tensor: torch.Tensor, title: str = "16-Band Input (Restacked)"):
    """Visualize 16-band input using a global dynamic range."""
    fig, axes = plt.subplots(4, 4, figsize=(12, 10))
    axes = axes.flatten()

    # Display labels for the 16 restacked channels (edit as needed)
    band_names = ["30", "20", "9", "40", "52", "20", "40", "30", "9", "52", "30", "20", "9", "40", "52", "30"]

    global_min = input_tensor.min().item()
    global_max = input_tensor.max().item()
    print(f"16-band visualization range: [{global_min:.6f}, {global_max:.6f}]")

    for i in range(16):
        band_data = input_tensor[i].cpu().numpy()
        band_min = band_data.min()
        band_max = band_data.max()
        im = axes[i].imshow(band_data, cmap="viridis", vmin=global_min, vmax=global_max)
        axes[i].set_title(f"Band {i}: {band_names[i]}\n[{band_min:.3f}, {band_max:.3f}]", fontsize=7)
        axes[i].axis("off")
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig

def visualize_original_5bands(original_dir: str, sample_idx: int):
    """
    Visualize the first 5 channels of a restacked sample as 'original 5 bands'.
    original_dir defaults to <DATA_DIR>/input_restacked_16.
    """
    if not os.path.isdir(original_dir):
        print(f"Original input directory does not exist: {original_dir}")
        return None

    original_files = sorted([f for f in os.listdir(original_dir) if f.endswith(".pt")])
    if sample_idx >= len(original_files):
        print("Sample index out of range for original inputs.")
        return None

    original_file = os.path.join(original_dir, original_files[sample_idx])
    original_data = torch.load(original_file, map_location="cpu")
    original_tensor = original_data["input"] if isinstance(original_data, dict) else original_data

    global_min = original_tensor.min().item()
    global_max = original_tensor.max().item()
    print(f"Original 5-band visualization range: [{global_min:.6f}, {global_max:.6f}]")

    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    band_names = ["30nm", "20nm", "9nm", "40nm", "52nm"]
    for i in range(5):
        band_data = original_tensor[i].cpu().numpy()
        band_min = band_data.min()
        band_max = band_data.max()
        im = axes[i].imshow(band_data, cmap="viridis", vmin=global_min, vmax=global_max)
        axes[i].set_title(f"Original Band {i}: {band_names[i]}\n[{band_min:.3f}, {band_max:.3f}]", fontsize=9)
        axes[i].axis("off")
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    plt.suptitle("Original 5-Band Input", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig

def visualize_hyperspectral_output(
    output_tensor: torch.Tensor,
    target_tensor: Optional[torch.Tensor] = None,
    title: str = "219-Band Hyperspectral Output"
):
    """Visualize predicted (and optionally ground-truth) bands and difference maps."""
    bands_to_show = [0, 10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 218]  # 13 bands
    num_bands = len(bands_to_show)
    rows = 3 if target_tensor is not None else 2
    fig, axes = plt.subplots(rows, num_bands, figsize=(20, 6 * rows))

    output_tensor = output_tensor.cpu()
    if target_tensor is not None:
        target_tensor = target_tensor.cpu()

    if target_tensor is not None:
        combined_tensor = torch.cat([output_tensor, target_tensor], dim=0)
        global_min = combined_tensor.min().item()
        global_max = combined_tensor.max().item()
    else:
        global_min = output_tensor.min().item()
        global_max = output_tensor.max().item()

    print(f"Hyperspectral visualization range: [{global_min:.6f}, {global_max:.6f}]")

    for i, band_idx in enumerate(bands_to_show):
        pred_data = output_tensor[band_idx].cpu().numpy()
        im1 = axes[0, i].imshow(pred_data, cmap="viridis", vmin=global_min, vmax=global_max)
        axes[0, i].set_title(f"Predicted Band {band_idx}\n[{pred_data.min():.3f}, {pred_data.max():.3f}]", fontsize=7)
        axes[0, i].axis("off")
        plt.colorbar(im1, ax=axes[0, i], fraction=0.046, pad=0.04)

        if target_tensor is not None:
            true_data = target_tensor[band_idx].cpu().numpy()
            im2 = axes[1, i].imshow(true_data, cmap="viridis", vmin=global_min, vmax=global_max)
            axes[1, i].set_title(f"Ground Truth Band {band_idx}\n[{true_data.min():.3f}, {true_data.max():.3f}]", fontsize=7)
            axes[1, i].axis("off")
            plt.colorbar(im2, ax=axes[1, i], fraction=0.046, pad=0.04)

            diff = np.abs(pred_data - true_data)
            diff_max = diff.max() if diff.size > 0 else 1.0
            im3 = axes[2, i].imshow(diff, cmap="hot", vmin=0, vmax=diff_max)
            axes[2, i].set_title(f"Difference Band {band_idx}\n[0.000, {diff_max:.3f}]", fontsize=7)
            axes[2, i].axis("off")
            plt.colorbar(im3, ax=axes[2, i], fraction=0.046, pad=0.04)

    plt.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()
    return fig

def analyze_spectral_curves(
    output_tensor: torch.Tensor,
    target_tensor: Optional[torch.Tensor] = None,
    num_pixels: int = 5
):
    """Plot spectral curves for several random pixel locations."""
    H, W = output_tensor.shape[1], output_tensor.shape[2]
    output_tensor = output_tensor.cpu()
    if target_tensor is not None:
        target_tensor = target_tensor.cpu()

    pixel_coords = [(random.randint(0, H - 1), random.randint(0, W - 1)) for _ in range(num_pixels)]
    fig, axes = plt.subplots(1, num_pixels, figsize=(20, 4))
    if num_pixels == 1:
        axes = [axes]

    wavelengths = np.arange(219)  # assuming 219 output bands

    if target_tensor is not None:
        combined_min = min(output_tensor.min().item(), target_tensor.min().item())
        combined_max = max(output_tensor.max().item(), target_tensor.max().item())
    else:
        combined_min = output_tensor.min().item()
        combined_max = output_tensor.max().item()
    print(f"Spectral curves Y-axis range: [{combined_min:.6f}, {combined_max:.6f}]")

    for i, (h, w) in enumerate(pixel_coords):
        pred_spectrum = output_tensor[:, h, w].cpu().numpy()
        axes[i].plot(wavelengths, pred_spectrum, "b-", label="Predicted", alpha=0.8, linewidth=2)

        if target_tensor is not None:
            true_spectrum = target_tensor[:, h, w].cpu().numpy()
            axes[i].plot(wavelengths, true_spectrum, "r-", label="Ground Truth", alpha=0.8, linewidth=2)

        axes[i].set_title(f"Pixel ({h}, {w}) Spectral Curve", fontsize=10)
        axes[i].set_xlabel("Band Index")
        axes[i].set_ylabel("Reflectance")
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
        axes[i].set_ylim(combined_min * 0.95, combined_max * 1.05)

    plt.suptitle("Random Pixel Spectral Curves Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


# ----------------------------
# Main
# ----------------------------
def main():
    paths = resolve_paths()
    PROJECT_ROOT = paths["PROJECT_ROOT"]
    DATA_DIR = paths["DATA_DIR"]
    CHECKPOINT_DIR = paths["CHECKPOINT_DIR"]
    VIS_DIR = paths["VIS_DIR"]
    MODEL_DIRS_ENV = paths["MODEL_DIRS_ENV"]

    print("Resolved paths:")
    print(f"  PROJECT_ROOT : {PROJECT_ROOT}")
    print(f"  DATA_DIR     : {DATA_DIR}")
    print(f"  CHECKPOINT_DIR: {CHECKPOINT_DIR}")
    print(f"  VIS_DIR      : {VIS_DIR}")
    if MODEL_DIRS_ENV:
        print(f"  MODEL_DIRS   : {MODEL_DIRS_ENV}")

    # Prepare directories
    input_dir = os.path.join(DATA_DIR, "input_restacked_16_with_indices")
    label_dir = os.path.join(DATA_DIR, "label")
    original_dir = os.path.join(DATA_DIR, "input_restacked_16")
    os.makedirs(VIS_DIR, exist_ok=True)

    # Pick device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model directories and labels
    model_dirs, model_names = get_model_dirs(CHECKPOINT_DIR, MODEL_DIRS_ENV)
    if not model_dirs:
        print("No model directories found. You can set MODEL_DIRS env var (comma-separated).")
        return

    # Load dataset (one sample for demonstration; you can loop as needed)
    dataset = RestackedSpectralDataset(input_dir, label_dir)
    print(f"Dataset size: {len(dataset)} samples")
    sample_idx = 0
    input_tensor, target_tensor = dataset[sample_idx]

    # Load "original 5-band" from the restacked 16-channel sample
    original_files = sorted([f for f in os.listdir(original_dir) if f.endswith(".pt")])
    if not original_files:
        print(f"No files found in original_dir: {original_dir}")
        return
    original_file = os.path.join(original_dir, original_files[sample_idx])
    original_data = torch.load(original_file, map_location="cpu")
    original_tensor = original_data["input"] if isinstance(original_data, dict) else original_data
    input_bands5 = original_tensor[:5]  # [5, H, W]

    # Visualize selected bands across models
    band_indices = list(range(0, 219, 10))
    if 218 not in band_indices:
        band_indices.append(218)

    # Keep a representative model for later single-sample inference
    representative_model: Optional[SpecSwin_16Band] = None

    for band_idx in band_indices:
        print(f"\nVisualizing band {band_idx} ...")
        model_results = []

        for m, model_dir in enumerate(model_dirs):
            ckpt = find_band_checkpoint(model_dir, band_idx)
            if not ckpt:
                print(f"Model not found for band {band_idx:03d} in {model_dir}")
                model_results.append(None)
                continue

            model = load_model(ckpt, device)
            if representative_model is None:
                representative_model = model  # store first successfully loaded model

            with torch.no_grad():
                input_batch = input_tensor.unsqueeze(0).to(device)  # [1, 16, 128, 128]
                output_batch = model(input_batch)

                # If out_channels==1 -> single band reconstruction; else multi-band
                if output_batch.shape[1] == 1:
                    pred_band = output_batch[0, 0].cpu()
                else:
                    pred_band = output_batch[0, band_idx].cpu()
                model_results.append(pred_band)

        # Ground truth for this band
        gt_band = target_tensor[band_idx].cpu()

        # Plot: 5 input bands + N model predictions + ground truth
        cols = 5 + len(model_dirs) + 1
        fig, axes = plt.subplots(1, cols, figsize=(4.0 * cols, 3))
        # First 5 input bands
        for i in range(5):
            axes[i].imshow(input_bands5[i].numpy(), cmap="viridis")
            axes[i].set_title(f"Input Band {i+1}", fontsize=9)
            axes[i].axis("off")
        # Model predictions
        offset = 5
        for i in range(len(model_dirs)):
            if model_results[i] is not None:
                axes[offset + i].imshow(model_results[i].numpy(), cmap="viridis")
                axes[offset + i].set_title(f"{model_names[i]}", fontsize=9)
            else:
                axes[offset + i].set_title(f"{model_names[i]} (No Model)", fontsize=9)
            axes[offset + i].axis("off")
        # Ground truth
        axes[-1].imshow(gt_band.numpy(), cmap="viridis")
        axes[-1].set_title("Ground Truth", fontsize=9)
        axes[-1].axis("off")

        plt.tight_layout()
        out_path = os.path.join(VIS_DIR, f"{band_idx:03d}_band.png")
        plt.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"Saved: {out_path}")

    # Randomly evaluate one sample with the representative model if available
    sample_idx = random.randint(0, len(dataset) - 1)
    print(f"\nRandomly selected sample #{sample_idx}")
    input_tensor, target_tensor = dataset[sample_idx]

    print("\nData range debugging:")
    print(f"  Input tensor shape:  {tuple(input_tensor.shape)}")
    print(f"  Input tensor range:  [{input_tensor.min().item():.6f}, {input_tensor.max().item():.6f}]")
    print(f"  Input tensor mean:   {input_tensor.mean().item():.6f}")
    print(f"  Input unique values: {len(torch.unique(input_tensor))}")
    print(f"  Target tensor shape: {tuple(target_tensor.shape)}")
    print(f"  Target tensor range: [{target_tensor.min().item():.6f}, {target_tensor.max().item():.6f}]")
    print(f"  Target tensor mean:  {target_tensor.mean().item():.6f}")
    print(f"  Target unique values:{len(torch.unique(target_tensor))}")

    if representative_model is not None:
        representative_model.eval()
        with torch.no_grad():
            input_batch = input_tensor.unsqueeze(0).to(device)  # [1, 16, 128, 128]
            output_batch = representative_model(input_batch)    # [1, C, 128, 128]
            output_tensor = output_batch.squeeze(0).cpu()       # [C, 128, 128]

        # If single-band model, cannot directly compute metrics vs 219-band target
        if output_tensor.shape[0] == 1:
            print("\nRepresentative model outputs a single band; skipping full 219-band metrics.")
        else:
            mse = nn.MSELoss()(output_tensor, target_tensor).item()
            mae = nn.L1Loss()(output_tensor, target_tensor).item()
            rmse = np.sqrt(mse)
            print(f"\nSample #{sample_idx} reconstruction quality:")
            print(f"  MSE : {mse:.6f}")
            print(f"  MAE : {mae:.6f}")
            print(f"  RMSE: {rmse:.6f}")

        print(f"  Input shape : {tuple(input_tensor.shape)}")
        print(f"  Output shape: {tuple(output_tensor.shape)}")
        print(f"  Target shape: {tuple(target_tensor.shape)}")

        # Ensure save directory exists
        os.makedirs(VIS_DIR, exist_ok=True)

        # Visualize original 5 bands
        print("\nGenerating visualizations...")
        fig1 = visualize_original_5bands(original_dir, sample_idx)
        if fig1:
            fig1.savefig(os.path.join(VIS_DIR, f"original_5bands_sample_{sample_idx}.png"),
                         dpi=300, bbox_inches="tight")
            plt.close(fig1)

        # Visualize 16-band input
        fig2 = visualize_input_16bands(input_tensor, f"16-Band Input (Sample #{sample_idx})")
        fig2.savefig(os.path.join(VIS_DIR, f"input_16bands_sample_{sample_idx}.png"),
                     dpi=300, bbox_inches="tight")
        plt.close(fig2)

        # Visualize 219-band output (if available)
        if output_tensor.shape[0] > 1:
            fig3 = visualize_hyperspectral_output(
                output_tensor, target_tensor,
                f"219-Band Hyperspectral Reconstruction (Sample #{sample_idx})"
            )
            fig3.savefig(os.path.join(VIS_DIR, f"hyperspectral_output_sample_{sample_idx}.png"),
                         dpi=300, bbox_inches="tight")
            plt.close(fig3)

        # Spectral curves
        if output_tensor.shape[0] > 1:
            fig4 = analyze_spectral_curves(output_tensor, target_tensor, num_pixels=5)
            fig4.savefig(os.path.join(VIS_DIR, f"spectral_curves_sample_{sample_idx}.png"),
                         dpi=300, bbox_inches="tight")
            plt.close(fig4)
    else:
        print("\nNo representative model was loaded; skipping single-sample inference and plots.")

    print(f"\nVisualization completed. Results saved to: {VIS_DIR}")
    print("Generated files include (depending on availability):")
    print("  - {band}_band.png for selected bands")
    print("  - original_5bands_sample_*.png")
    print("  - input_16bands_sample_*.png")
    print("  - hyperspectral_output_sample_*.png")
    print("  - spectral_curves_sample_*.png")

if __name__ == "__main__":
    main()
