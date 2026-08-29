#!/usr/bin/env python3
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from parse_stats import parse_stats_file
from ieee_style import set_ieee_style, get_hatch_patterns

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    results_cap = os.path.join(project_root, "results", "capacity_sweep")
    results_inf = os.path.join(project_root, "results", "inference_sweep")
    plot_dir = os.path.join(project_root, "plots")

    os.makedirs(plot_dir, exist_ok=True)


    # 2. Parse Inference Sweep for Speedup & Bandwidth (fig3, fig4)
    # We will use Shard=4, Prec=2 (FP16) as our representative configuration
    inf_results = []
    
    # Baseline: kv_frac = 0.0
    base_path_inf = os.path.join(results_inf, "prec_2", "shard_4", "kv_frac_0.0", "stats.txt")
    base_stats_inf = parse_stats_file(base_path_inf, "hybrid", measured_iters=1)
    
    hbm_path_inf = os.path.join(results_inf, "prec_2", "shard_4", "kv_frac_1.0", "stats.txt")
    hbm_stats_inf = parse_stats_file(hbm_path_inf, "hybrid", measured_iters=1)
    
    if base_stats_inf and hbm_stats_inf and base_stats_inf['simTicks'] > 0:
        speedup = base_stats_inf['simTicks'] / hbm_stats_inf['simTicks']
        
        # Fig 3: Overall System Speedup (Decode)
        figsize = set_ieee_style("single")
        plt.figure(figsize=figsize)
        hatches = get_hatch_patterns()
        
        bars = plt.bar(['KV in LPDDR5', 'KV in HBM'], [1.0, speedup], 
                       color=['tab:blue', 'tab:red'], edgecolor='black', linewidth=1)
                       
        plt.title('Decode Speedup', fontsize=9)
        plt.ylabel('Speedup (x)', fontsize=9)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, f'{yval:.2f}x', 
                     ha='center', va='bottom', fontsize=8, fontweight='bold')
        plt.ylim(0, max(1.0, speedup) * 1.2)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "fig3_dual_speedup_bw.png"), dpi=300, bbox_inches='tight')
        
        # Fig 4: Bandwidth Utilization
        def avg_bw(bw_list):
            return sum(bw_list)/len(bw_list) if bw_list else 0.0
            
        lpddr5_bw_base = avg_bw(base_stats_inf['lpddr5_bw_util'])
        hbm_bw_base = avg_bw(base_stats_inf['hbm_bw_util'])
        
        lpddr5_bw_hbm = avg_bw(hbm_stats_inf['lpddr5_bw_util'])
        hbm_bw_hbm = avg_bw(hbm_stats_inf['hbm_bw_util'])
        
        figsize4 = set_ieee_style("single")
        plt.figure(figsize=figsize4)
        
        df_fig4 = pd.DataFrame({
            'Placement': ['KV in LPDDR5', 'KV in LPDDR5', 'KV in HBM', 'KV in HBM'],
            'Memory Tier': ['DDR5', 'HBM', 'DDR5', 'HBM'],
            'Utilization (%)': [lpddr5_bw_base, hbm_bw_base, lpddr5_bw_hbm, hbm_bw_hbm]
        })
        
        sns.barplot(data=df_fig4, x='Placement', y='Utilization (%)', hue='Memory Tier', 
                    edgecolor='black', palette=['tab:blue', 'tab:orange'], linewidth=1)
        
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.ylabel('Utilization (%)', fontsize=9)
        plt.xlabel('')
        plt.title('Bandwidth Utilization (Decode)', fontsize=9)
        plt.legend(title='Memory Tier', fontsize=7, title_fontsize=7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "fig4_optimal_tiering.png"), dpi=300, bbox_inches='tight')
        
    print("Performance metric plots generated successfully!")

if __name__ == "__main__":
    main()
