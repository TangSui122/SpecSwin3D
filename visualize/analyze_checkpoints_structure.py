# create: analyze_checkpoints_structure.py
import os
import json
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional

# ------------------------------
# Path resolution (soft-coded)
# ------------------------------
def resolve_default_checkpoint_dir() -> str:
    """
    Resolve the default checkpoints directory using environment variables with safe fallbacks:
      - PROJECT_ROOT: defaults to the parent of this file's directory
      - CHECKPOINT_DIR: defaults to <PROJECT_ROOT>/checkpoints
    """
    this_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.environ.get("PROJECT_ROOT", os.path.dirname(this_dir))
    checkpoint_dir = os.environ.get("CHECKPOINT_DIR", os.path.join(project_root, "checkpoints"))
    return checkpoint_dir


def analyze_checkpoints_structure(checkpoint_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Analyze the structure of a checkpoints directory.

    For each strategy subfolder, this function inspects:
      - presence of training_summary.json
      - presence of models/ and tensorboard_logs/
      - counts *.pth model files, extracts band indices (band_XXX_best_model.pth)
      - counts TensorBoard event files and total size
      - reads training_summary.json to compute average RMSE/MAE if present

    Args:
        checkpoint_dir: Path to the checkpoints directory. If None, it will be
                        resolved from environment variables (soft-coded).

    Returns:
        A list of dictionaries, each describing one strategy.
    """
    if checkpoint_dir is None:
        checkpoint_dir = resolve_default_checkpoint_dir()

    if not os.path.exists(checkpoint_dir):
        print(f"Directory does not exist: {checkpoint_dir}")
        return []

    print(f"Analyzing checkpoints directory: {checkpoint_dir}")
    print("=" * 80)

    strategies: List[Dict[str, Any]] = []
    total_size_bytes = 0

    for item in os.listdir(checkpoint_dir):
        item_path = os.path.join(checkpoint_dir, item)
        if not os.path.isdir(item_path):
            continue

        print(f"\nStrategy directory: {item}")

        # Key paths
        summary_file = os.path.join(item_path, "training_summary.json")
        models_dir = os.path.join(item_path, "models")
        tensorboard_dir = os.path.join(item_path, "tensorboard_logs")

        strategy_info: Dict[str, Any] = {
            "name": item,
            "has_summary": os.path.exists(summary_file),
            "has_models": os.path.exists(models_dir),
            "has_tensorboard": os.path.exists(tensorboard_dir),
            "model_count": 0,
            "tensorboard_files": 0,
        }

        # Analyze model files
        if os.path.exists(models_dir):
            model_files = glob.glob(os.path.join(models_dir, "*.pth"))
            strategy_info["model_count"] = len(model_files)

            if model_files:
                # Extract band indices from filenames like: band_012_best_model.pth
                band_numbers: List[int] = []
                for model_file in model_files:
                    filename = os.path.basename(model_file)
                    if filename.startswith("band_") and filename.endswith("_best_model.pth"):
                        try:
                            band_num = int(filename.split("_")[1])
                            band_numbers.append(band_num)
                        except ValueError:
                            pass

                if band_numbers:
                    band_numbers.sort()
                    strategy_info["band_range"] = f"{min(band_numbers)}-{max(band_numbers)}"
                    strategy_info["bands"] = band_numbers

            print(f"   Model files: {strategy_info['model_count']}")
            if "band_range" in strategy_info:
                print(f"   Band range: {strategy_info['band_range']}")

        # Analyze TensorBoard event files
        if os.path.exists(tensorboard_dir):
            tb_files: List[str] = []
            for root, _, files in os.walk(tensorboard_dir):
                for f in files:
                    if f.startswith("events.out.tfevents"):
                        tb_files.append(os.path.join(root, f))

            strategy_info["tensorboard_files"] = len(tb_files)
            print(f"   TensorBoard files: {len(tb_files)}")

            tb_size = sum(os.path.getsize(f) for f in tb_files) if tb_files else 0
            tb_size_gb = tb_size / (1024 ** 3)
            print(f"   TensorBoard size: {tb_size_gb:.2f} GB")
            total_size_bytes += tb_size

        # Read training summary and compute averages
        if os.path.exists(summary_file):
            try:
                with open(summary_file, "r", encoding="utf-8") as f:
                    summary = json.load(f)

                strategy_info["summary"] = summary
                completed_bands = len(summary.get("training_results", {}))
                total_bands = summary.get("total_bands", 0)
                print(f"   Training summary: {completed_bands}/{total_bands} bands completed")

                if summary.get("training_results"):
                    results = summary["training_results"]
                    rmse_values = [r.get("rmse", 0) for r in results.values()]
                    mae_values = [r.get("mae", 0) for r in results.values()]

                    if rmse_values:
                        avg_rmse = sum(rmse_values) / len(rmse_values)
                        avg_mae = sum(mae_values) / len(mae_values)
                        print(f"   Average performance: RMSE={avg_rmse:.6f}, MAE={avg_mae:.6f}")
                        strategy_info["avg_rmse"] = avg_rmse
                        strategy_info["avg_mae"] = avg_mae

            except Exception as e:
                print(f"   Failed to read training summary: {e}")

        strategies.append(strategy_info)

    # Summary
    print("\n" + "=" * 80)
    print("Overall statistics:")
    print(f"   Total strategies: {len(strategies)}")
    print(f"   Total models: {sum(s.get('model_count', 0) for s in strategies)}")
    print(f"   Total TensorBoard size: {total_size_bytes / (1024 ** 3):.2f} GB")

    # Ranking by performance if available (lower RMSE is better)
    strategies_with_perf = [s for s in strategies if "avg_rmse" in s]
    if strategies_with_perf:
        strategies_with_perf.sort(key=lambda x: x["avg_rmse"])
        print("\nStrategy performance ranking (by RMSE):")
        for i, s in enumerate(strategies_with_perf, 1):
            print(f"   {i}. {s['name']:<25}: RMSE={s['avg_rmse']:.6f}, completed {s.get('model_count', 0)} bands")

    print("\nTensorBoard launch command:")
    print(f"   tensorboard --logdir=\"{checkpoint_dir}\" --port=6006")

    return strategies


if __name__ == "__main__":
    # If you want to pass a custom path, set CHECKPOINT_DIR env var, or pass an argument here.
    analyze_checkpoints_structure()
