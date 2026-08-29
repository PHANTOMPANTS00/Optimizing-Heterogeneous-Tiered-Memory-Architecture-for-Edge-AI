#!/usr/bin/env python3
import os
import subprocess
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add scripts directory to path to import parse_stats
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from parse_stats import parse_stats_file
from ieee_style import set_ieee_style

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    gem5_root = os.path.dirname(project_root)
    gem5_opt = os.path.join(gem5_root, "build", "ALL", "gem5.opt")
    config_script = os.path.join(project_root, "configs", "tiered_memory_sys.py")
    results_base_dir = os.path.join(project_root, "results", "capacity_sweep")
    binary = os.path.join(project_root, "src_benchmarks", "gemm_bench")
    
    cap_fractions = [0.125, 0.25, 0.5]
    weight_fractions = [0.0, 0.25, 0.5, 0.75, 1.0]
    sizes = [256, 512, 1024]
    
    processes = []
    
    # ensure plots dir exists
    os.makedirs(os.path.join(project_root, "plots"), exist_ok=True)
    
    for N in sizes:
        for cap_frac in cap_fractions:
            for w_frac in weight_fractions:
                outdir = os.path.join(results_base_dir, f"matrix_{N}", f"cap_{cap_frac}", f"w_{w_frac}")
                os.makedirs(outdir, exist_ok=True)
                stats_file = os.path.join(outdir, "stats.txt")
                
                if os.path.exists(stats_file):
                    print(f"Skipping [ N={N} | cap={cap_frac} | w={w_frac} ]")
                    continue
                
                # gemm_bench args: <N> <W_addr> <X_addr> <Y_addr> <batch>
                options_str = f"{N} 0x1000000000 0x1100000000 0x1200000000 8"
                cmd = [
                    gem5_opt,
                    "--outdir", outdir,
                    config_script,
                    "--mem-mode", "hybrid",
                    "--hbm-capacity-fraction", str(cap_frac),
                    "--hbm-weight-fraction", str(w_frac),
                    "--binary", binary,
                    "--options", options_str
                ]
                
                print(f"Launching [ N={N} | cap={cap_frac} | w={w_frac} ]...")
                log_file = open(os.path.join(outdir, "sim_log.txt"), "w")
                p = subprocess.Popen(cmd, stdout=log_file, stderr=log_file)
                processes.append((p, N, cap_frac, w_frac, log_file))
                
                # 12 parallel simulations max
                if len(processes) >= 12:
                    for p, _, _, _, log_file in processes:
                        p.wait()
                        log_file.close()
                    processes = []

    if processes:
        for p, _, _, _, log_file in processes:
            p.wait()
            log_file.close()
            
    # Now parse stats and plot
    results = []
    for N in sizes:
        for cap_frac in cap_fractions:
            for w_frac in weight_fractions:
                stats_path = os.path.join(results_base_dir, f"matrix_{N}", f"cap_{cap_frac}", f"w_{w_frac}", "stats.txt")
                stats = parse_stats_file(stats_path, "hybrid", measured_iters=5)
                if stats and stats['simTicks'] > 0:
                    results.append({
                        'MatrixSize': N,
                        'HBM_Capacity': cap_frac,
                        'HBM_Weight': w_frac,
                        'Energy_Per_Token_nJ': stats['energy_per_token_nj'],
                        'EDP': stats['edp']
                    })
                    
    df = pd.DataFrame(results)
    if not df.empty:
        df.to_csv(os.path.join(results_base_dir, "capacity_sweep_results.csv"), index=False)
        
        # Convert capacity to string for categorical coloring and styles
        df["HBM_Capacity"] = df["HBM_Capacity"].astype(str)
        
        # Plot Energy per token
        figsize_double = set_ieee_style("double")
        
        g = sns.relplot(
            data=df,
            x="HBM_Weight", y="Energy_Per_Token_nJ",
            hue="HBM_Capacity", style="HBM_Capacity",
            col="MatrixSize",
            kind="line", markers=True, dashes=True, alpha=0.9, linewidth=1.5,
            height=figsize_double[1], aspect=(figsize_double[0]/3)/figsize_double[1], facet_kws={'sharey': False}
        )
        g.set_axis_labels("HBM Weight Fraction", "Energy/Token (nJ)", fontsize=9)
        g.set_titles("N = {col_name}", size=9)
        g.fig.subplots_adjust(top=0.82, wspace=0.3)
        g.fig.suptitle("Energy Efficiency of Hybrid Memory Tiering (Prefill GEMM)", fontsize=10)
        
        # Adjust tick fonts
        for ax in g.axes.flat:
            ax.tick_params(axis='both', which='major', labelsize=8)
            
        plot_path = os.path.join(project_root, "plots", "fig5_energy_per_token.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved {plot_path}")
        
        # Plot EDP
        g2 = sns.relplot(
            data=df,
            x="HBM_Weight", y="EDP",
            hue="HBM_Capacity", style="HBM_Capacity",
            col="MatrixSize",
            kind="line", markers=True, dashes=True, alpha=0.9, linewidth=1.5,
            height=figsize_double[1], aspect=(figsize_double[0]/3)/figsize_double[1], facet_kws={'sharey': False}
        )
        g2.set_axis_labels("HBM Weight Fraction", "EDP", fontsize=9)
        g2.set_titles("N = {col_name}", size=9)
        g2.fig.subplots_adjust(top=0.82, wspace=0.3)
        g2.fig.suptitle("Energy-Delay Product of Hybrid Memory Tiering", fontsize=10)
        
        # Adjust tick fonts
        for ax in g2.axes.flat:
            ax.tick_params(axis='both', which='major', labelsize=8)
            
        plot_path2 = os.path.join(project_root, "plots", "fig6_edp.png")
        plt.savefig(plot_path2, dpi=300, bbox_inches='tight')
        print(f"Saved {plot_path2}")

if __name__ == "__main__":
    main()
