#!/usr/bin/env python3
import os
import re
import csv
import sys

def parse_stats_file(filepath, mem_mode, measured_iters=10):
    stats = {
        'simSeconds': 0.0,
        'simTicks': 0,
        'lpddr5_read_bytes': 0,
        'hbm_read_bytes': 0,
        'lpddr5_bw_util': [],
        'hbm_bw_util': [],
        'l3_hits': 0,
        'l3_misses': 0,
        'lpddr5_energy_pJ': 0.0,
        'hbm_energy_pJ': 0.0,
    }
    
    if not os.path.exists(filepath):
        return None

    dump_block_count = 0

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('---'):
                dump_block_count += 1
                if dump_block_count >= 2:
                    break
                continue
                
            parts = re.split(r'\s+', line)
            if len(parts) < 2:
                continue
                
            key = parts[0]
            try:
                val_str = parts[1]
                if '.' in val_str:
                    val = float(val_str)
                else:
                    val = int(val_str)
            except ValueError:
                continue

            # Core Execution Metrics
            if key == 'simTicks':
                stats['simTicks'] = int(val)
            elif key == 'simSeconds':
                stats['simSeconds'] = float(val)

            # L3 Cache Stats
            elif 'l3cache' in key.lower():
                if 'overallHits::total' in key:
                    stats['l3_hits'] += int(val)
                elif 'overallMisses::total' in key:
                    stats['l3_misses'] += int(val)

            # Memory Read Traffic
            elif 'bytesRead::total' in key:
                if mem_mode == 'pure_lpddr5':
                    if 'mem_ctrl' in key.lower() or 'dram' in key.lower():
                        stats['lpddr5_read_bytes'] += int(val)
                elif mem_mode == 'pure_hbm':
                    if 'mem_ctrl' in key.lower() or 'dram' in key.lower():
                        stats['hbm_read_bytes'] += int(val)
                elif mem_mode == 'hybrid':
                    if 'lpddr5' in key.lower():
                        stats['lpddr5_read_bytes'] += int(val)
                    elif 'hbm' in key.lower():
                        stats['hbm_read_bytes'] += int(val)

            # Memory Bandwidth Utilization
            elif 'busUtil' in key and 'busUtilRead' not in key and 'busUtilWrite' not in key:
                if mem_mode == 'pure_lpddr5':
                    if 'mem_ctrl' in key.lower() or 'dram' in key.lower():
                        stats['lpddr5_bw_util'].append(float(val))
                elif mem_mode == 'pure_hbm':
                    if 'mem_ctrl' in key.lower() or 'dram' in key.lower():
                        stats['hbm_bw_util'].append(float(val))
                elif mem_mode == 'hybrid':
                    if 'lpddr5' in key.lower():
                        stats['lpddr5_bw_util'].append(float(val))
                    elif 'hbm' in key.lower():
                        stats['hbm_bw_util'].append(float(val))

            # Energy Metrics (totalEnergy per rank is in pJ)
            elif 'totalEnergy' in key:
                if mem_mode == 'pure_lpddr5':
                    if 'mem_ctrl' in key.lower() or 'dram' in key.lower():
                        stats['lpddr5_energy_pJ'] += float(val)
                elif mem_mode == 'pure_hbm':
                    if 'mem_ctrl' in key.lower() or 'dram' in key.lower():
                        stats['hbm_energy_pJ'] += float(val)
                elif mem_mode == 'hybrid':
                    if 'lpddr5' in key.lower():
                        stats['lpddr5_energy_pJ'] += float(val)
                    elif 'hbm' in key.lower():
                        stats['hbm_energy_pJ'] += float(val)

    # Safely compute averages
    stats['lpddr5_bw_util_avg'] = sum(stats['lpddr5_bw_util']) / len(stats['lpddr5_bw_util']) if stats['lpddr5_bw_util'] else 0.0
    stats['hbm_bw_util_avg'] = sum(stats['hbm_bw_util']) / len(stats['hbm_bw_util']) if stats['hbm_bw_util'] else 0.0
    
    # Convert bytes to MB
    stats['lpddr5_read_mb'] = stats['lpddr5_read_bytes'] / (1024 * 1024)
    stats['hbm_read_mb'] = stats['hbm_read_bytes'] / (1024 * 1024)

    # Convert energy pJ to nJ
    stats['lpddr5_energy_nj'] = stats['lpddr5_energy_pJ'] / 1000.0
    stats['hbm_energy_nj'] = stats['hbm_energy_pJ'] / 1000.0
    stats['total_energy_nj'] = stats['lpddr5_energy_nj'] + stats['hbm_energy_nj']

    # Energy per token and EDP
    stats['energy_per_token_nj'] = stats['total_energy_nj'] / measured_iters
    stats['edp'] = stats['total_energy_nj'] * stats['simSeconds']

    # L3 hit rate
    l3_total = stats['l3_hits'] + stats['l3_misses']
    stats['l3_hit_rate'] = stats['l3_hits'] / l3_total if l3_total > 0 else 0.0
    
    return stats