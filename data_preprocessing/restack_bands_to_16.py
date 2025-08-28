# Environment variables (optional):
#   PROJECT_ROOT            (default: parent directory of this file)
#   DATA_DIR                (default: <PROJECT_ROOT>/dataset)
#   INPUT_DIR               (default: <DATA_DIR>/input)
#   OUTPUT_RESTACKED_16_DIR (default: <DATA_DIR>/input_restacked_16)

import os
import glob
import sys
from typing import Optional, Dict, List

import torch
from tqdm import tqdm


# -----------------------------
# Path resolution (soft-coded)
# -----------------------------
def resolve_paths() -> Dict[str, str]:
    """
    Resolve directories using environment variables with safe defaults:
      - PROJECT_ROOT:            defaults to the parent directory of this file
      - DATA_DIR:                defaults to <PROJECT_ROOT>/dataset
      - INPUT_DIR:               defaults to <DATA_DIR>/input
      - OUTPUT_RESTACKED_16_DIR: defaults to <DATA_DIR>/input_restacked_16
    """
    this_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.environ.get("PROJECT_ROOT", os.path.dirname(this_dir))
    data_dir = os.environ.get("DATA_DIR", os.path.join(project_root, "dataset"))
    input_dir = os.environ.get("INPUT_DIR", os.path.join(data_dir, "input"))
    output_dir = os.environ.get("OUTPUT_RESTACKED_16_DIR", os.path.join(data_dir, "input_restacked_16"))
    return {
        "PROJECT_ROOT": project_root,
        "DATA_DIR": data_dir,
        "INPUT_DIR": input_dir,
        "OUTPUT_DIR": output_dir,
    }


# -----------------------------
# Core logic
# -----------------------------
def restack_input_bands(
    input_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> None:
    """
    Restack 5-band input tensors into 16-band tensors.

    Original 5 bands (indices in the original dataset): [30, 20, 9, 40, 52]
    Target 16-band order follows pattern '1234524135154321':
      [30, 20, 9, 40, 52, 20, 40, 30, 9, 52, 30, 52, 40, 9, 20, 30]
    Corresponding tensor channel indices (from the 5-channel tensor): [0, 1, 2, 3, 4, 1, 3, 0, 2, 4, 0, 4, 3, 2, 1, 0]

    The script expects .pt files in INPUT_DIR, each containing a dict with key 'input'
    shaped as [5, H, W]. It writes restacked tensors as dicts to OUTPUT_DIR with keys:
      - 'input'                   -> FloatTensor [16, H, W]
      - 'original_band_indices'   -> [30, 20, 9, 40, 52]
      - 'restacked_band_indices'  -> list of 16 band indices per the pattern above
      - 'restack_mapping'         -> list of source channel indices
      - 'restack_pattern'         -> '1234524135154321'
    """
    # Resolve paths
    paths = resolve_paths()
    input_dir = input_dir or paths["INPUT_DIR"]
    output_dir = output_dir or paths["OUTPUT_DIR"]

    os.makedirs(output_dir, exist_ok=True)

    # Indices in the 5-channel tensor to create the 16-channel tensor
    restack_indices: List[int] = [0, 1, 2, 3, 4, 1, 3, 0, 2, 4, 0, 4, 3, 2, 1, 0]

    # Scan .pt files
    pt_files = sorted(glob.glob(os.path.join(input_dir, "*.pt")))
    if not pt_files:
        print("No .pt files found in input directory.")
        print(f"Path checked: {input_dir}")
        return

    print(f"Starting restack for {len(pt_files)} files")
    print(f"Input directory : {input_dir}")
    print(f"Output directory: {output_dir}")
    print("Band reordering: 5 bands -> 16 bands")
    print("Pattern       : 1234524135154321")
    print(f"Index mapping : {restack_indices}\n")

    success_count = 0
    error_count = 0

    for pt_file in tqdm(pt_files, desc="Processing files"):
        try:
            data = torch.load(pt_file, map_location="cpu")

            # Expect a dict with 'input' tensor of shape [5, H, W]
            if not isinstance(data, dict) or "input" not in data:
                print(f"Skip {os.path.basename(pt_file)}: data must be a dict with key 'input'")
                error_count += 1
                continue

            original_tensor = data["input"]

            # Validate shape
            if not (isinstance(original_tensor, torch.Tensor) and original_tensor.ndim == 3 and original_tensor.shape[0] == 5):
                print(
                    f"Skip {os.path.basename(pt_file)}: tensor must be [5, H, W], got {tuple(original_tensor.shape)}"
                )
                error_count += 1
                continue

            # Restack to 16 channels
            restacked_tensor = original_tensor[restack_indices]  # [16, H, W]

            # Prepare metadata
            new_data = {
                "input": restacked_tensor,
                "original_band_indices": [30, 20, 9, 40, 52],
                "restacked_band_indices": [30, 20, 9, 40, 52, 20, 40, 30, 9, 52, 30, 52, 40, 9, 20, 30],
                "restack_mapping": restack_indices,
                "restack_pattern": "1234524135154321",
            }

            # Save
            output_file = os.path.join(output_dir, os.path.basename(pt_file))
            torch.save(new_data, output_file)
            success_count += 1

        except Exception as e:
            print(f"Error processing {os.path.basename(pt_file)}: {e}")
            error_count += 1

    print("\nRestack completed.")
    print(f"  Succeeded: {success_count}")
    print(f"  Failed   : {error_count}")
    print(f"  Output   : {output_dir}")

    if success_count > 0:
        verify_first_file(output_dir)


# -----------------------------
# Verification
# -----------------------------
def verify_first_file(output_dir: str) -> None:
    """Load and print a brief summary of the first output file in output_dir."""
    pt_files = sorted(glob.glob(os.path.join(output_dir, "*.pt")))
    if not pt_files:
        return

    try:
        sample_path = pt_files[0]
        data = torch.load(sample_path, map_location="cpu")
        print(f"\nVerifying first output file: {os.path.basename(sample_path)}")
        print(f"  Type        : {type(data)}")
        print(f"  Keys        : {list(data.keys())}")

        if "input" in data:
            tensor = data["input"]
            tmin = tensor.min().item() if tensor.numel() > 0 else float("nan")
            tmax = tensor.max().item() if tensor.numel() > 0 else float("nan")
            print(f"  Tensor shape: {tuple(tensor.shape)}")
            print(f"  Dtype       : {tensor.dtype}")
            print(f"  Value range : [{tmin:.4f}, {tmax:.4f}]")

        if "restacked_band_indices" in data:
            print(f"  Restacked band indices: {data['restacked_band_indices']}")
        if "restack_pattern" in data:
            print(f"  Restack pattern       : {data['restack_pattern']}")
        if "restack_mapping" in data:
            print(f"  Tensor index mapping  : {data['restack_mapping']}")

    except Exception as e:
        print(f"Error verifying output file: {e}")


# -----------------------------
# Entrypoint
# -----------------------------
if __name__ == "__main__":
    restack_input_bands()
