# Environment variables (optional):
#   PROJECT_ROOT                  (default: parent directory of this file)
#   DATA_DIR                      (default: <PROJECT_ROOT>/dataset)
#   INPUT_DIR                     (default: <DATA_DIR>/input)
#   OUTPUT_RESTACKED_16_REPEATED  (default: <DATA_DIR>/input_restacked_16_repeated)

import os
import glob
import torch
from tqdm import tqdm
from typing import Dict, Optional, List


# -----------------------------
# Path resolution (soft-coded)
# -----------------------------
def resolve_paths() -> Dict[str, str]:
    """
    Resolve directories using environment variables with safe defaults.
    """
    this_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.environ.get("PROJECT_ROOT", os.path.dirname(this_dir))
    data_dir = os.environ.get("DATA_DIR", os.path.join(project_root, "dataset"))
    input_dir = os.environ.get("INPUT_DIR", os.path.join(data_dir, "input"))
    output_dir = os.environ.get(
        "OUTPUT_RESTACKED_16_REPEATED",
        os.path.join(data_dir, "input_restacked_16_repeated"),
    )
    return {
        "PROJECT_ROOT": project_root,
        "DATA_DIR": data_dir,
        "INPUT_DIR": input_dir,
        "OUTPUT_DIR": output_dir,
    }


# -----------------------------
# Core logic
# -----------------------------
def restack_input_bands_repeated(
    input_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> None:
    """
    Restack 5-band input tensors to 16 bands using a repeated pattern.

    Original 5 bands (by index in the original dataset): [30, 20, 9, 40, 52]
    Repeated pattern over channel indices (from the 5-channel tensor):
      [0,0,0,0, 1,1,1, 2,2,2, 3,3,3, 4,4,4]
    Which corresponds to band indices:
      [30,30,30,30, 20,20,20, 9,9,9, 40,40,40, 52,52,52]

    Input format:
      Each .pt file in INPUT_DIR must be a dict with key 'input' of shape [5, H, W]
    Output format:
      A dict saved to OUTPUT_DIR with keys:
        - 'input'                 -> FloatTensor [16, H, W]
        - 'original_band_indices' -> [30, 20, 9, 40, 52]
        - 'restacked_band_info'   -> metadata of the repeated sequence
        - 'restack_indices'       -> list of source channel indices
        - 'restack_pattern'       -> string representation of the indices
    """
    # Resolve paths
    paths = resolve_paths()
    input_dir = input_dir or paths["INPUT_DIR"]
    output_dir = output_dir or paths["OUTPUT_DIR"]

    os.makedirs(output_dir, exist_ok=True)

    # New repeated indices: [0,0,0,0,1,1,1,2,2,2,3,3,3,4,4,4]
    repeated_indices: List[int] = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]

    # Collect .pt files
    pt_files = sorted(glob.glob(os.path.join(input_dir, "*.pt")))
    if not pt_files:
        print("No .pt files found in the input directory.")
        print(f"Path checked: {input_dir}")
        return

    print(f"Starting restack for {len(pt_files)} files")
    print(f"Input directory : {input_dir}")
    print(f"Output directory: {output_dir}")
    print("Band reordering: 5 bands -> 16 bands")
    print("Repeated indices: [0,0,0,0, 1,1,1, 2,2,2, 3,3,3, 4,4,4]")
    print("Corresponding bands: [30,30,30,30, 20,20,20, 9,9,9, 40,40,40, 52,52,52]")
    print(f"Index mapping    : {repeated_indices}\n")

    success_count = 0
    error_count = 0

    # Process each file
    for pt_file in tqdm(pt_files, desc="Processing files"):
        try:
            # Load original data
            data = torch.load(pt_file, map_location="cpu")

            # Validate structure
            if not isinstance(data, dict) or "input" not in data:
                print(f"Skip {os.path.basename(pt_file)}: data must be a dict with key 'input'")
                error_count += 1
                continue

            # Fetch original tensor
            original_tensor = data["input"]  # shape: [5, H, W]

            # Validate tensor shape
            if not (
                isinstance(original_tensor, torch.Tensor)
                and original_tensor.ndim == 3
                and original_tensor.shape[0] == 5
            ):
                print(
                    f"Skip {os.path.basename(pt_file)}: tensor must be [5, H, W], got {tuple(original_tensor.shape)}"
                )
                error_count += 1
                continue

            # Restack by repeated pattern
            restacked_tensor = original_tensor[repeated_indices]  # shape: [16, H, W]

            # Build output dict
            new_data = {
                "input": restacked_tensor,
                "original_band_indices": [30, 20, 9, 40, 52],
                "restacked_band_info": {
                    "pattern": "repeated_4x_each_first_band_then_3x_next",
                    "channels_0_3": [30, 30, 30, 30],
                    "channels_4_6": [20, 20, 20],
                    "channels_7_9": [9, 9, 9],
                    "channels_10_12": [40, 40, 40],
                    "channels_13_15": [52, 52, 52],
                    "full_sequence": [30, 30, 30, 30, 20, 20, 20, 9, 9, 9, 40, 40, 40, 52, 52, 52],
                },
                "wavelengths": {
                    "Band_30": "647nm (Red)",
                    "Band_20": "550nm (Green)",
                    "Band_9": "443nm (Blue)",
                    "Band_40": "723nm (Red Edge)",
                    "Band_52": "840nm (NIR)",
                },
                "restack_indices": repeated_indices,
                "restack_pattern": "[0,0,0,0,1,1,1,2,2,2,3,3,3,4,4,4]",
                "design_principle": "Each original band is repeated multiple times to enhance feature learning.",
            }

            # Save file
            output_file = os.path.join(output_dir, os.path.basename(pt_file))
            torch.save(new_data, output_file)
            success_count += 1

        except Exception as e:
            print(f"Error processing {os.path.basename(pt_file)}: {e}")
            error_count += 1

    print("\nProcessing finished.")
    print(f"  Succeeded: {success_count}")
    print(f"  Failed   : {error_count}")
    print(f"  Output   : {output_dir}")

    if success_count > 0:
        verify_first_file(output_dir)


# -----------------------------
# Verification helpers
# -----------------------------
def verify_first_file(output_dir: str) -> None:
    """Verify the first output file format and value ranges."""
    pt_files = sorted(glob.glob(os.path.join(output_dir, "*.pt")))
    if not pt_files:
        return

    try:
        data = torch.load(pt_files[0], map_location="cpu")
        print(f"\nVerifying first output file: {os.path.basename(pt_files[0])}")
        print(f"  Data type : {type(data)}")
        print(f"  Dict keys : {list(data.keys())}")

        if "input" in data:
            tensor = data["input"]
            print(f"  Tensor shape: {tuple(tensor.shape)}")
            print(f"  Dtype       : {tensor.dtype}")
            vmin = tensor.min().item() if tensor.numel() > 0 else float("nan")
            vmax = tensor.max().item() if tensor.numel() > 0 else float("nan")
            print(f"  Value range : [{vmin:.4f}, {vmax:.4f}]")

            # Validate repeat pattern
            if tensor.shape[0] == 16:
                print("\nPer-channel value range (repeat pattern validation):")

                # Band 30: channels 0–3
                print("  Channels 0–3 (Band 30, 647nm Red): repeated 4x")
                for i in range(0, 4):
                    ch_min = tensor[i].min().item()
                    ch_max = tensor[i].max().item()
                    print(f"    Channel {i}: [{ch_min:.4f}, {ch_max:.4f}]")
                    if i > 0:
                        identical = torch.allclose(tensor[0], tensor[i])
                        print(f"    Identical to channel 0: {identical}")

                # Band 20: channels 4–6
                print("  Channels 4–6 (Band 20, 550nm Green): repeated 3x")
                for i in range(4, 7):
                    ch_min = tensor[i].min().item()
                    ch_max = tensor[i].max().item()
                    print(f"    Channel {i}: [{ch_min:.4f}, {ch_max:.4f}]")
                    if i > 4:
                        identical = torch.allclose(tensor[4], tensor[i])
                        print(f"    Identical to channel 4: {identical}")

                # Band 9: channels 7–9
                print("  Channels 7–9 (Band 9, 443nm Blue): repeated 3x")
                for i in range(7, 10):
                    ch_min = tensor[i].min().item()
                    ch_max = tensor[i].max().item()
                    print(f"    Channel {i}: [{ch_min:.4f}, {ch_max:.4f}]")
                    if i > 7:
                        identical = torch.allclose(tensor[7], tensor[i])
                        print(f"    Identical to channel 7: {identical}")

                # Band 40: channels 10–12
                print("  Channels 10–12 (Band 40, 723nm Red Edge): repeated 3x")
                for i in range(10, 13):
                    ch_min = tensor[i].min().item()
                    ch_max = tensor[i].max().item()
                    print(f"    Channel {i}: [{ch_min:.4f}, {ch_max:.4f}]")
                    if i > 10:
                        identical = torch.allclose(tensor[10], tensor[i])
                        print(f"    Identical to channel 10: {identical}")

                # Band 52: channels 13–15
                print("  Channels 13–15 (Band 52, 840nm NIR): repeated 3x")
                for i in range(13, 16):
                    ch_min = tensor[i].min().item()
                    ch_max = tensor[i].max().item()
                    print(f"    Channel {i}: [{ch_min:.4f}, {ch_max:.4f}]")
                    if i > 13:
                        identical = torch.allclose(tensor[13], tensor[i])
                        print(f"    Identical to channel 13: {identical}")

        if "restacked_band_info" in data:
            print("\nRestack metadata:")
            band_info = data["restacked_band_info"]
            if "full_sequence" in band_info:
                print(f"  Full sequence: {band_info['full_sequence']}")
            if "pattern" in band_info:
                print(f"  Pattern      : {band_info['pattern']}")

    except Exception as e:
        print(f"Error verifying output file: {e}")


def compare_with_original(
    input_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> None:
    """Compare the first original input with the first restacked output."""
    paths = resolve_paths()
    input_dir = input_dir or paths["INPUT_DIR"]
    output_dir = output_dir or paths["OUTPUT_DIR"]

    input_files = sorted(glob.glob(os.path.join(input_dir, "*.pt")))
    output_files = sorted(glob.glob(os.path.join(output_dir, "*.pt")))

    if not input_files or not output_files:
        print("Comparison files not found.")
        return

    try:
        original_data = torch.load(input_files[0], map_location="cpu")
        restacked_data = torch.load(output_files[0], map_location="cpu")

        original_tensor = original_data["input"]  # [5, H, W]
        restacked_tensor = restacked_data["input"]  # [16, H, W]

        print("\nData comparison:")
        print(f"  Original shape: {tuple(original_tensor.shape)}")
        print(f"  Restacked shape: {tuple(restacked_tensor.shape)}")

        repeated_indices = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
        print("\nRestack validation:")
        all_correct = True
        for i, orig_idx in enumerate(repeated_indices):
            identical = torch.allclose(original_tensor[orig_idx], restacked_tensor[i])
            status = "OK" if identical else "Mismatch"
            print(f"  Channel {i:2d} <- Original band {orig_idx}: {status}")
            if not identical:
                all_correct = False

        if all_correct:
            print("\nValidation passed: all channels correctly correspond to original bands.")
        else:
            print("\nValidation failed: please check the code or inputs.")

    except Exception as e:
        print(f"Error during comparison: {e}")


# -----------------------------
# Entrypoint
# -----------------------------
if __name__ == "__main__":
    print("Creating repeated-pattern 16-channel inputs")
    print("Pattern indices: [0,0,0,0,1,1,1,2,2,2,3,3,3,4,4,4]")
    print("Design: each original band is repeated several times to enhance feature learning.\n")

    # Restack
    restack_input_bands_repeated()

    # Compare for quick validation
    print("\n" + "=" * 60)
    compare_with_original()
