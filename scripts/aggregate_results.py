#!/usr/bin/env python3
import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from parse_stats import parse_stats_file

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    results_cap = os.path.join(project_root, "results", "capacity_sweep")
    results_inf = os.path.join(project_root, "results", "inference_sweep")
    results_arch = os.path.join(project_root, "results", "architecture_sweep")
    
    all_data = []

    # 0. Parse Architecture Sweep (Prefill)
    if os.path.exists(results_arch):
        for root, dirs, files in os.walk(results_arch):
            if "stats.txt" in files:
                stats_path = os.path.join(root, "stats.txt")
                parts = stats_path.split(os.sep)
                try:
                    matrix_n_str = [p for p in parts if p.startswith("matrix_")][0]
                    matrix_size = int(matrix_n_str.split("_")[1])
                    mode = parts[-2]
                    
                    w_frac = 0.5 if mode == "hybrid" else (1.0 if mode == "pure_hbm" else 0.0)
                    
                    stats = parse_stats_file(stats_path, mode, measured_iters=10)
                    if stats and stats['simTicks'] > 0:
                        all_data.append({
                            'Phase': 'Prefill (Arch Sweep)',
                            'MatrixSize_N': matrix_size,
                            'Weight_in_HBM_Frac': w_frac,
                            'Precision_Bytes': 4,
                            'Tensor_Shards': 1,
                            'KV_in_HBM': None,
                            'Latency_ms': (stats['simTicks'] / 1e12) * 1000.0,
                            'Energy_Per_Token_nJ': stats['energy_per_token_nj'],
                            'EDP': stats['edp'],
                            'LPDDR5_Traffic_MB': stats['lpddr5_read_bytes'] / 1e6,
                            'HBM_Traffic_MB': stats['hbm_read_bytes'] / 1e6
                        })
                except Exception as e:
                    pass

    # 1. Parse Capacity Sweep (Prefill)
    # These were run with measured_iters = 5
    if os.path.exists(results_cap):
        for root, dirs, files in os.walk(results_cap):
            if "stats.txt" in files:
                stats_path = os.path.join(root, "stats.txt")
                
                # Path format: results/capacity_sweep/matrix_N/cap_0.5/w_0.x/stats.txt
                parts = stats_path.split(os.sep)
                try:
                    matrix_n_str = [p for p in parts if p.startswith("matrix_")][0]
                    w_frac_str = [p for p in parts if p.startswith("w_")][0]
                    
                    matrix_size = int(matrix_n_str.split("_")[1])
                    w_frac = float(w_frac_str.split("_")[1])
                    
                    stats = parse_stats_file(stats_path, "hybrid", measured_iters=5)
                    if stats and stats['simTicks'] > 0:
                        all_data.append({
                            'Phase': 'Prefill',
                            'MatrixSize_N': matrix_size,
                            'Weight_in_HBM_Frac': w_frac,
                            'Precision_Bytes': 4, # Baseline gemm is FP32
                            'Tensor_Shards': 1,
                            'KV_in_HBM': None,
                            'Latency_ms': (stats['simTicks'] / 1e12) * 1000.0,
                            'Energy_Per_Token_nJ': stats['energy_per_token_nj'],
                            'EDP': stats['edp'],
                            'LPDDR5_Traffic_MB': stats['lpddr5_read_bytes'] / 1e6,
                            'HBM_Traffic_MB': stats['hbm_read_bytes'] / 1e6
                        })
                except Exception as e:
                    print(f"Skipping {stats_path}: {e}")

    # 2. Parse Inference Sweep (Decode)
    # These were run with measured_iters = 1 (after the manual optimization!)
    if os.path.exists(results_inf):
        for root, dirs, files in os.walk(results_inf):
            if "stats.txt" in files:
                stats_path = os.path.join(root, "stats.txt")
                
                # Path format: results/inference_sweep/prec_X/shard_Y/kv_frac_Z/stats.txt
                parts = stats_path.split(os.sep)
                try:
                    prec_str = [p for p in parts if p.startswith("prec_")][0]
                    shard_str = [p for p in parts if p.startswith("shard_")][0]
                    kv_str = [p for p in parts if p.startswith("kv_frac_")][0]
                    
                    prec = int(prec_str.split("_")[1])
                    shard = int(shard_str.split("_")[1])
                    kv_frac = float(kv_str.split("_")[2])
                    
                    # VERY IMPORTANT: measured_iters=1 to get accurate numbers!
                    stats = parse_stats_file(stats_path, "hybrid", measured_iters=1)
                    if stats and stats['simTicks'] > 0:
                        all_data.append({
                            'Phase': 'Decode',
                            'MatrixSize_N': 8192, # Base sequence length
                            'Weight_in_HBM_Frac': None,
                            'Precision_Bytes': prec,
                            'Tensor_Shards': shard,
                            'KV_in_HBM': (kv_frac == 1.0),
                            'Latency_ms': (stats['simTicks'] / 1e12) * 1000.0,
                            'Energy_Per_Token_nJ': stats['energy_per_token_nj'],
                            'EDP': stats['edp'],
                            'LPDDR5_Traffic_MB': stats['lpddr5_read_bytes'] / 1e6,
                            'HBM_Traffic_MB': stats['hbm_read_bytes'] / 1e6
                        })
                except Exception as e:
                    pass

    # Save to CSV
    df = pd.DataFrame(all_data)
    if not df.empty:
        # Sort for readability
        df = df.sort_values(by=['Phase', 'MatrixSize_N', 'Precision_Bytes', 'Tensor_Shards', 'Weight_in_HBM_Frac', 'KV_in_HBM'])
        out_csv = os.path.join(project_root, "results", "All_Simulations_Master_Results.csv")
        df.to_csv(out_csv, index=False)
        print(f"Successfully aggregated {len(df)} simulation results into: {out_csv}")
    else:
        print("No valid stats files found to aggregate.")

if __name__ == "__main__":
    main()
