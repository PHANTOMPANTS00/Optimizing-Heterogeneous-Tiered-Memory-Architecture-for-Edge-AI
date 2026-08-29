#!/usr/bin/env python3
import os
import subprocess
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from parse_stats import parse_stats_file
from ieee_style import set_ieee_style, get_markers

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    gem5_root = os.path.dirname(project_root)
    gem5_opt = os.path.join(gem5_root, "build", "ALL", "gem5.opt")
    config_script = os.path.join(project_root, "configs", "tiered_memory_sys.py")
    results_base_dir = os.path.join(project_root, "results", "architecture_sweep")
    binary = os.path.join(project_root, "src_benchmarks", "gemm_bench")
    
    modes = ["pure_lpddr5", "pure_hbm", "hybrid"]
    sizes = [256, 512, 1024, 2048]
    
    processes = []
    
    os.makedirs(os.path.join(project_root, "plots"), exist_ok=True)
    
    for N in sizes:
        for mode in modes:
            outdir = os.path.join(results_base_dir, f"matrix_{N}", mode)
            os.makedirs(outdir, exist_ok=True)
            stats_file = os.path.join(outdir, "stats.txt")
            
            if os.path.exists(stats_file):
                print(f"Skipping [ N={N} | mode={mode} ]")
                continue
            
            # gemm_bench args: <N> <W_addr> <X_addr> <Y_addr> <batch>
            options_str = f"{N} 0x1000000000 0x1100000000 0x1200000000 8"
            
            cmd = [
                gem5_opt,
                "--outdir", outdir,
                config_script,
                "--mem-mode", mode,
                "--binary", binary,
                "--options", options_str
            ]
            
            if mode == "hybrid":
                # For hybrid, use the optimal load balancing: 0.5 capacity, 0.5 weight fraction (Hot weights in HBM)
                cmd.extend([
                    "--hbm-capacity-fraction", "0.5",
                    "--hbm-weight-fraction", "0.5"
                ])
                
            print(f"Launching [ N={N} | mode={mode} ]...")
            log_file = open(os.path.join(outdir, "sim_log.txt"), "w")
            p = subprocess.Popen(cmd, stdout=log_file, stderr=log_file)
            processes.append((p, log_file))
            
            if len(processes) >= 4:
                for p_obj, l_file in processes:
                    p_obj.wait()
                    l_file.close()
                processes = []

    if processes:
        for p_obj, l_file in processes:
            p_obj.wait()
            l_file.close()
            
    # Parse stats
    results = []
    for N in sizes:
        for mode in modes:
            stats_path = os.path.join(results_base_dir, f"matrix_{N}", mode, "stats.txt")
            stats = parse_stats_file(stats_path, mode, measured_iters=5)
            if stats and stats['simTicks'] > 0:
                results.append({
                    'MatrixSize': N,
                    'Architecture': 'Pure LPDDR5' if mode == 'pure_lpddr5' else 'Pure HBM' if mode == 'pure_hbm' else 'Hybrid (Load Balanced)',
                    'Latency_ms': (stats['simTicks'] / 1e12) * 1000.0,
                    'EDP': stats['edp'],
                    'LPDDR5_Traffic_MB': stats.get('lpddr5_read_mb', 0.0),
                    'HBM_Traffic_MB': stats.get('hbm_read_mb', 0.0)
                })
                
    df = pd.DataFrame(results)
    if not df.empty:
        # Save CSV
        df.to_csv(os.path.join(results_base_dir, "architecture_sweep_results.csv"), index=False)
        
        # Calculate speedup relative to Pure LPDDR5
        baseline = df[df['Architecture'] == 'Pure LPDDR5'].set_index('MatrixSize')['Latency_ms']
        df['Speedup'] = df.apply(lambda row: baseline[row['MatrixSize']] / row['Latency_ms'], axis=1)

        # Plot 1: Architecture Comparison (Speedup)
        figsize = set_ieee_style("single")
        plt.figure(figsize=figsize)
        
        ax = sns.lineplot(data=df, x="MatrixSize", y="Speedup", hue="Architecture", 
                          style="Architecture", markers=True, dashes=False, 
                          linewidth=1.5, markersize=5)

        plt.title("Architecture Speedup (Prefill GEMM)", fontsize=9)
        plt.ylabel("Speedup over LPDDR5", fontsize=9)
        plt.xlabel("Matrix Dimension (N)", fontsize=9)
        plt.axhline(1.0, color='gray', linestyle='--', linewidth=1) # Baseline
        plt.xticks(sizes, fontsize=8)
        plt.yticks(fontsize=8)
        plt.legend(title='', fontsize=7)
        
        plt.tight_layout()
        plot_path = os.path.join(project_root, "plots", "fig0_architecture_comparison.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved {plot_path}")
        
if __name__ == "__main__":
    main()
