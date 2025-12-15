"""Formatting utilities for evaluation results."""

import pandas as pd
from tabulate import tabulate


def print_summary(results: dict):
    """Print summary table of results."""
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    
    has_pesq = any("pesq" in r for r in results.values())
    has_stoi = any("stoi" in r for r in results.values())
    
    headers = ["Variant", "SI-SDR (dB)"]
    if has_pesq:
        headers.append("PESQ")
    if has_stoi:
        headers.append("STOI")
    headers.append("Samples")
    
    table_data = []
    for variant, metrics in sorted(results.items()):
        row = [variant, f"{metrics['si_sdr']:.2f}"]
        if has_pesq:
            row.append(f"{metrics['pesq']:.2f}" if "pesq" in metrics else "N/A")
        if has_stoi:
            row.append(f"{metrics['stoi']:.3f}" if "stoi" in metrics else "N/A")
        row.append(metrics["num_samples"])
        table_data.append(row)
    
    if len(results) > 1:
        avg_row = [
            "AVERAGE",
            f"{sum(r['si_sdr'] for r in results.values()) / len(results):.2f}",
        ]
        if has_pesq:
            pesq_values = [r["pesq"] for r in results.values() if "pesq" in r]
            avg_row.append(
                f"{sum(pesq_values) / len(pesq_values):.2f}" if pesq_values else "N/A"
            )
        if has_stoi:
            stoi_values = [r["stoi"] for r in results.values() if "stoi" in r]
            avg_row.append(
                f"{sum(stoi_values) / len(stoi_values):.3f}" if stoi_values else "N/A"
            )
        avg_row.append("")
        table_data.append(avg_row)
    
    print(tabulate(table_data, headers=headers, tablefmt="simple"))
    print("=" * 80)


def save_results_csv(results: dict, output_path: str):
    """Save results to CSV file."""
    rows = []
    for variant, metrics in results.items():
        row = {
            "variant": variant,
            "si_sdr_db": metrics["si_sdr"],
            "num_samples": metrics["num_samples"],
        }
        if "pesq" in metrics:
            row["pesq"] = metrics["pesq"]
        if "stoi" in metrics:
            row["stoi"] = metrics["stoi"]
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
