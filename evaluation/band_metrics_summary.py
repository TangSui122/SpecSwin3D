from __future__ import annotations
import os
import csv
import argparse
from typing import List, Dict
import pandas as pd


# -----------------------------
# Defaults (soft-coded paths)
# -----------------------------
def default_paths() -> Dict[str, str]:
    """
    Resolve default input/output paths using environment variables with safe fallbacks:
      - PROJECT_ROOT (default: parent directory of this file)
      - RESULTS_CSV  (default: <PROJECT_ROOT>/outputs/comprehensive_evaluation_results.csv)
      - OUTPUT_CSV   (default: <PROJECT_ROOT>/outputs/custom_band_metrics_summary.csv)
    """
    this_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.environ.get("PROJECT_ROOT", os.path.dirname(this_dir))
    results_csv = os.environ.get(
        "RESULTS_CSV",
        os.path.join(project_root, "outputs", "comprehensive_evaluation_results.csv"),
    )
    output_csv = os.environ.get(
        "OUTPUT_CSV",
        os.path.join(project_root, "outputs", "custom_band_metrics_summary.csv"),
    )
    return {"PROJECT_ROOT": project_root, "RESULTS_CSV": results_csv, "OUTPUT_CSV": output_csv}


def parse_bands(bands_str: str | None) -> List[int]:
    """
    Parse a comma/space separated bands string. If None/empty, return a sensible default.
    Example env/arg: "10,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210"
    """
    if not bands_str:
        return [10, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210]
    parts = [p.strip() for p in bands_str.replace(" ", ",").split(",") if p.strip()]
    out: List[int] = []
    for p in parts:
        try:
            out.append(int(p))
        except ValueError:
            raise ValueError(f"Invalid band value: {p!r}")
    return out


# -----------------------------
# Core
# -----------------------------
def summarize_metrics(
    results_path: str,
    output_path: str,
    custom_bands: List[int],
    metrics: List[str] | None = None,
    metric_names: List[str] | None = None,
) -> None:
    """
    Build a summary CSV for selected bands and strategies.

    Args:
        results_path: Path to the comprehensive evaluation CSV
        output_path:  Where to write the summary CSV
        custom_bands: Bands to include (exact match on 'band' column)
        metrics:      Column names in results CSV to extract (defaults below)
        metric_names: Friendly names for the metrics in the output header
    """
    # Defaults to mirror original script, preserving column -> label mapping
    if metrics is None:
        metrics = ['PSNR_mean', 'ERGAS_mean', 'SAM_mean', 'Q_Index_mean', 'SSIM_mean', 'RMSE_mean']
    if metric_names is None:
        metric_names = ['PSNR', 'ERGAS', 'SAM', 'Q-Index', 'SSIM', 'RMSE']

    # Read results
    if not os.path.isfile(results_path):
        raise FileNotFoundError(f"Results CSV not found: {results_path}")
    results_df = pd.read_csv(results_path)

    # Basic column validation
    required_cols = {'band', 'strategy'}
    missing_req = required_cols - set(results_df.columns)
    if missing_req:
        raise ValueError(f"Missing required column(s) in results CSV: {sorted(missing_req)}")

    missing_metrics = [m for m in metrics if m not in results_df.columns]
    if missing_metrics:
        raise ValueError(f"Missing metric column(s) in results CSV: {missing_metrics}")

    rows: List[Dict[str, object]] = []
    # Iterate requested bands
    for band in custom_bands:
        band_df = results_df[results_df['band'] == band]
        if band_df.empty:
            # Skip silently if a band isn't present
            continue

        # Iterate strategies present for this band
        for strategy in band_df['strategy'].dropna().unique():
            strat_df = band_df[band_df['strategy'] == strategy]
            if strat_df.empty:
                continue

            # Take the first row for this (band, strategy); align with original behavior
            row_src = strat_df.iloc[0]
            row: Dict[str, object] = {'band': int(band), 'strategy': str(strategy)}
            for m_name, m_col in zip(metric_names, metrics):
                row[m_name] = row_src[m_col]
            rows.append(row)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Write CSV (preserve desired column order)
    header = ['band', 'strategy'] + metric_names
    with open(output_path, "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"CSV saved to: {output_path}")
    print(f"Rows written: {len(rows)}")


# -----------------------------
# CLI
# -----------------------------
def main():
    paths = default_paths()
    ap = argparse.ArgumentParser(
        description="Summarize per-band metrics for selected bands and strategies (soft-coded paths)."
    )
    ap.add_argument("--results-csv", type=str, default=paths["RESULTS_CSV"],
                    help="Path to the comprehensive evaluation CSV (default from RESULTS_CSV env or project default).")
    ap.add_argument("--output-csv", type=str, default=paths["OUTPUT_CSV"],
                    help="Path to write the summary CSV (default from OUTPUT_CSV env or project default).")
    ap.add_argument("--bands", type=str, default=os.environ.get("CUSTOM_BANDS", ""),
                    help='Comma/space separated bands to include (e.g. "10,50,60,..."). '
                         "Defaults to a standard list if not provided.")
    args = ap.parse_args()

    custom_bands = parse_bands(args.bands)
    summarize_metrics(
        results_path=args.results_csv,
        output_path=args.output_csv,
        custom_bands=custom_bands,
    )


if __name__ == "__main__":
    main()
