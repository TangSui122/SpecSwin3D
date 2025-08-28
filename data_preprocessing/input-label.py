import os
import glob
import tifffile
import torch
import numpy as np
from typing import Tuple, List, Dict


# -----------------------------
# Path resolution (soft-coded)
# -----------------------------
def resolve_paths() -> Dict[str, str]:
    """
    Resolve directories using environment variables with safe defaults:
      - PROJECT_ROOT   (default: parent directory of this file)
      - DATA_DIR       (default: <PROJECT_ROOT>/dataset)
      - TIF_DIR        (default: <DATA_DIR>/tif)
      - OUT_INPUT_DIR  (default: <DATA_DIR>/input)
      - OUT_LABEL_DIR  (default: <DATA_DIR>/label)
    """
    this_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.environ.get("PROJECT_ROOT", os.path.dirname(this_dir))
    data_dir = os.environ.get("DATA_DIR", os.path.join(project_root, "dataset"))
    tif_dir = os.environ.get("TIF_DIR", os.path.join(data_dir, "tif"))
    out_input_dir = os.environ.get("OUT_INPUT_DIR", os.path.join(data_dir, "input"))
    out_label_dir = os.environ.get("OUT_LABEL_DIR", os.path.join(data_dir, "label"))
    return {
        "PROJECT_ROOT": project_root,
        "DATA_DIR": data_dir,
        "TIF_DIR": tif_dir,
        "OUT_INPUT_DIR": out_input_dir,
        "OUT_LABEL_DIR": out_label_dir,
    }


# -----------------------------
# Normalization
# -----------------------------
def normalize_data(data: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Simple robust normalization to [0, 1] using 1st and 99th percentiles.

    Returns:
        normalized (float32), p1, p99
    """
    p1 = float(np.percentile(data, 1))
    p99 = float(np.percentile(data, 99))
    denom = (p99 - p1) if (p99 - p1) != 0 else 1e-8
    normalized = np.clip((data - p1) / (denom + 1e-12), 0, 1).astype(np.float32)
    return normalized, p1, p99


# -----------------------------
# Main preprocessing
# -----------------------------
def main():
    paths = resolve_paths()
    TIF_DIR = paths["TIF_DIR"]
    OUT_INPUT_DIR = paths["OUT_INPUT_DIR"]
    OUT_LABEL_DIR = paths["OUT_LABEL_DIR"]

    os.makedirs(OUT_INPUT_DIR, exist_ok=True)
    os.makedirs(OUT_LABEL_DIR, exist_ok=True)

    # Selected input bands (kept from original logic)
    input_bands: List[int] = [30, 20, 9, 40, 52]
    # Label bands are all others in [0, 223] except input bands (total 224 channels)
    label_band_indices: List[int] = [i for i in range(224) if i not in input_bands]

    # Collect all .tif files (customize the pattern if needed)
    tif_files = sorted(glob.glob(os.path.join(TIF_DIR, "patch_*.tif")))
    print("Found {} .tif files in: {}".format(len(tif_files), TIF_DIR))

    for i, tif_path in enumerate(tif_files):
        # full_img shape is either (224, H, W) or (H, W, 224)
        full_img = tifffile.imread(tif_path)
        shape = full_img.shape

        # If shape is (H, W, 224), transpose to (224, H, W)
        if shape[0] != 224:
            if shape[-1] == 224:
                full_img = np.transpose(full_img, (2, 0, 1))  # [H, W, C] -> [C, H, W]
            else:
                print("Skip {}: unsupported shape {}".format(tif_path, shape))
                continue

        # Crop to 128x128 (top-left corner to match original behavior)
        full_img = full_img[:, :128, :128]

        # Build input and label tensors
        input_tensor = full_img[input_bands, :, :]
        label_tensor = full_img[label_band_indices, :, :]

        # Normalize input
        normalized_input = np.zeros_like(input_tensor, dtype=np.float32)
        input_norm_params = []
        for idx in range(len(input_bands)):
            normalized_input[idx], p1, p99 = normalize_data(input_tensor[idx])
            input_norm_params.append({"p1": p1, "p99": p99})

        # Normalize label
        normalized_label = np.zeros_like(label_tensor, dtype=np.float32)
        label_norm_params = []
        for idx in range(len(label_band_indices)):
            normalized_label[idx], p1, p99 = normalize_data(label_tensor[idx])
            label_norm_params.append({"p1": p1, "p99": p99})

        base = os.path.splitext(os.path.basename(tif_path))[0]

        # Save input as dict with normalization params
        torch.save(
            {
                "input": torch.tensor(normalized_input),
                "band_indices": input_bands,
                "norm_params": input_norm_params,
            },
            os.path.join(OUT_INPUT_DIR, base + ".pt"),
        )

        # Save label as dict with normalization params
        torch.save(
            {
                "label": torch.tensor(normalized_label),
                "band_indices": label_band_indices,
                "norm_params": label_norm_params,
            },
            os.path.join(OUT_LABEL_DIR, base + ".pt"),
        )

        if (i + 1) % 20 == 0:
            print("[{}/{}] normalized and saved".format(i + 1, len(tif_files)))

    print("\nAll samples have been generated and normalized to [0, 1].")

    # Verification on the first sample
    if len(tif_files) > 0:
        first_base = os.path.splitext(os.path.basename(tif_files[0]))[0]
        sample_input = torch.load(os.path.join(OUT_INPUT_DIR, first_base + ".pt"))
        sample_label = torch.load(os.path.join(OUT_LABEL_DIR, first_base + ".pt"))

        in_min = float(sample_input["input"].min())
        in_max = float(sample_input["input"].max())
        lb_min = float(sample_label["label"].min())
        lb_max = float(sample_label["label"].max())

        print("\nNormalization check:")
        print("  Input range : [{:.3f}, {:.3f}]".format(in_min, in_max))
        print("  Label range : [{:.3f}, {:.3f}]".format(lb_min, lb_max))


if __name__ == "__main__":
    main()
