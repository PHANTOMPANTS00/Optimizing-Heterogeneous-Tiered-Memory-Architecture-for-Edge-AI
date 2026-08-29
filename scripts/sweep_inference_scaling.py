#!/usr/bin/env python3
import os
import subprocess
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from parse_stats import parse_stats_file
from ieee_style import set_ieee_style, get_hatch_patterns

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    gem5_root = os.path.dirname(project_root)
    gem5_opt = os.path.join(gem5_root, "build", "ALL", "gem5.opt")
    config_script = os.path.join(project_root, "configs", "tiered_memory_sys.py")
    results_base_dir = os.path.join(project_root, "results", "inference_sweep")
    binary = os.path.join(project_root, "src_benchmarks", "kv_cache_bench")
    
    # 4=FP32, 2=FP16, 1=INT8
    precisions = [4, 2, 1]
    # Tensor Parallelism shards
    shards = [1, 2, 4, 8]
    # KV Cache mapped entirely to HBM (1.0) vs LPDDR5 (0.0)
    kv_placements = [1.0, 0.0]
    
    seq_len = 8192
    d = 4096
    
    processes = []
    
    os.makedirs(os.path.join(project_root, "plots"), exist_ok=True)
    
    for p_bytes in precisions:
        for shard in shards:
            for kv_frac in kv_placements:
                outdir = os.path.join(results_base_dir, f"prec_{p_bytes}", f"shard_{shard}", f"kv_frac_{kv_frac}")
                os.makedirs(outdir, exist_ok=True)
                stats_file = os.path.join(outdir, "stats.txt")
                
                if os.path.exists(stats_file):
                    print(f"Skipping [ prec={p_bytes} | shard={shard} | kv_frac={kv_frac} ]")
                    continue
                
                # kv_cache_bench args: <seq_len> <d> <elem_bytes> <shard_rows> <K_addr> <Q_addr> <out_addr>
                options_str = f"{seq_len} {d} {p_bytes} {shard} 0x1000000000 0x1100000000 0x1200000000"
                cmd = [
                    gem5_opt,
                    "--outdir", outdir,
                    config_script,
                    "--mem-mode", "hybrid",
                    "--hbm-capacity-fraction", "0.5",
                    "--hbm-weight-fraction", str(kv_frac),
                    "--binary", binary,
                    "--options", options_str
                ]
                
                print(f"Launching [ prec={p_bytes} | shard={shard} | kv_frac={kv_frac} ]...")
                log_file = open(os.path.join(outdir, "sim_log.txt"), "w")
                p = subprocess.Popen(cmd, stdout=log_file, stderr=log_file)
                processes.append((p, p_bytes, shard, kv_frac, log_file))
                
                # 4 parallel simulations max to prevent OOM
                if len(processes) >= 4:
                    for p_obj, _, _, _, l_file in processes:
                        p_obj.wait()
                        l_file.close()
                    processes = []

    if processes:
        for p_obj, _, _, _, l_file in processes:
            p_obj.wait()
            l_file.close()
            
    # Parse stats
    results = []
    for p_bytes in precisions:
        for shard in shards:
            for kv_frac in kv_placements:
                stats_path = os.path.join(results_base_dir, f"prec_{p_bytes}", f"shard_{shard}", f"kv_frac_{kv_frac}", "stats.txt")
                stats = parse_stats_file(stats_path, "hybrid", measured_iters=1)
                if stats and stats['simTicks'] > 0:
                    results.append({
                        'Precision_Bytes': p_bytes,
                        'Shards': shard,
                        'Memory_Tier': 'KV in HBM' if kv_frac == 1.0 else 'KV in LPDDR5',
                        'Energy_Per_Token_nJ': stats['energy_per_token_nj'],
                        'EDP': stats['edp'],
                        'Latency_ms': (stats['simTicks'] / 1e12) * 1000.0
                    })
                    
    df = pd.DataFrame(results)
    if not df.empty:
        df.to_csv(os.path.join(results_base_dir, "inference_sweep_results.csv"), index=False)
        
        # Plot 1: KV Scaling (Latency vs Shards)
        figsize = set_ieee_style("single")
        plt.figure(figsize=figsize)
        # Filter for FP16 (typical inference)
        df_fp16 = df[df['Precision_Bytes'] == 2]
        if not df_fp16.empty:
            sns.lineplot(data=df_fp16, x="Shards", y="Latency_ms", hue="Memory_Tier", 
                         style="Memory_Tier", markers=True, dashes=False, linewidth=1.5, markersize=5)
            plt.title("Tensor Parallelism Scaling (FP16)", fontsize=9)
            plt.ylabel("Decode Latency (ms)", fontsize=9)
            plt.xlabel("Number of Shards", fontsize=9)
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)
            plt.legend(title='', fontsize=7)
            
            plt.tight_layout()
            plot_path = os.path.join(project_root, "plots", "fig7_kv_scaling.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Saved {plot_path}")
        
        # Plot 2: Precision Sweep (EDP vs Precision)
        plt.figure(figsize=figsize)
        # Filter for 4 shards
        df_4shards = df[df['Shards'] == 4].copy()
        if not df_4shards.empty:
            df_4shards['Precision'] = df_4shards['Precision_Bytes'].map({4: 'FP32', 2: 'FP16', 1: 'INT8'})
            ax = sns.barplot(data=df_4shards, x="Precision", y="EDP", hue="Memory_Tier", 
                             edgecolor='black', linewidth=1)
            
            plt.yscale("log")
            plt.title("EDP vs Precision (4 Shards)", fontsize=9)
            plt.ylabel("EDP", fontsize=9)
            plt.xlabel("Precision", fontsize=9)
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)
            plt.legend(title='', fontsize=7)
            
            plt.tight_layout()
            plot_path2 = os.path.join(project_root, "plots", "fig8_precision_sweep.png")
            plt.savefig(plot_path2, dpi=300, bbox_inches='tight')
            print(f"Saved {plot_path2}")

if __name__ == "__main__":
    main()
