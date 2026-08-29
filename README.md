# 🧠 Optimizing Heterogeneous Tiered Memory Architecture for Edge AI

![C++](https://img.shields.io/badge/C++-Microbenchmarks-blue.svg)
![Python](https://img.shields.io/badge/Python-Automation_Scripts-green.svg)
![gem5](https://img.shields.io/badge/gem5-Simulator-orange.svg)
![Arm Architecture](https://img.shields.io/badge/Arm_Architecture-AArch64-blue)

A simulation-based exploration of modern memory architecture configurations for Edge AI and Large Language Model (LLM) Inference, designed and evaluated using the gem5 simulator on an Arm v8-A CPU architecture.

## Table of Contents
- [About](#about)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage & Simulation](#usage--simulation)
- [Project Structure](#project-structure)
- [Results & Analysis](#results--analysis)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## About
This project addresses the critical "memory-wall" bottleneck in modern Edge AI and Large Language Model (LLM) inference. By evaluating **Pure LPDDR5**, **Pure High Bandwidth Memory (HBM)**, and novel **Hybrid Tiered Memory** architectures, the project identifies optimal memory subsystems for edge devices. It utilizes the gem5 simulator and an out-of-order ARM CPU to measure latency, energy-delay product (EDP), and traffic distribution during prefill and decoding inference phases.

## Key Features
- **Armv8-A Optimization**: Leverages AArch64 for all C++ microbenchmarks to accurately model modern mobile and edge System-on-Chips (SoCs).
- **Hybrid Memory Controller**: Implements a custom `Edge_HBM_Inference` tiered memory subsystem that maps "hot" parameters (GEMM blocks, KV caches) to HBM and bulk parameters to LPDDR5, achieving near-HBM speedups with reduced capacity footprints.
- **Inference Workload Modeling**: Simulates compute-bound prefill phases (`gemm_bench`) and memory-bound decode phases (`kv_cache_bench`) with Tensor Parallelism sharding.
- **Automated Experimentation**: Provides Python automation scripts for running parallel gem5 simulations and generating IEEE-standard performance visualizations.

## System Architecture

### CPU Model
- **Core Architecture**: Arm v8-A Out-of-Order CPU (`O3CPU`)
- **Number of Cores**: 4 Cores (designed for 4-way Tensor Parallelism shards)
- **Simulation Mode**: Syscall Emulation (SE) Mode

### Memory Subsystem Configurations
In all configurations, the total memory capacity is constrained to **4GiB** to ensure a fair comparison for edge device limits.
1. **Pure LPDDR5 (4GiB)**: The industry-standard baseline for mobile/edge compute. **Advantage**: Highly cost-effective and low-power, providing sufficient bulk capacity for large model weights.
2. **Pure HBM (4GiB)**: Represents a premium edge SoC utilizing on-package memory. **Advantage**: Offers exceptionally high bandwidth, significantly reducing latency for memory-bound tasks like the KV cache decode phase.
3. **Hybrid Tiered Memory (2GiB HBM + 2GiB LPDDR5)**: A load-balanced system with static weight partitioning. **Advantage**: Achieves the optimal tradeoff by placing latency-sensitive "hot" parameters in HBM and bulk "cold" weights in LPDDR5, delivering near-HBM performance at a reduced area and cost footprint.

## Prerequisites
Ensure the following software and dependencies are installed before running the simulations:
- **gem5 Simulator**: (Built and accessible, usually expected at `../../build/ALL/gem5.opt`)
- **Cross-Compiler**: `aarch64-linux-gnu-g++` (for compiling the AArch64 targets)
- **Python 3.x**: Required packages: `pandas`, `matplotlib`, `seaborn`, `numpy`

## Installation
Clone the repository and build the C++ microbenchmarks targeting the Arm architecture.

```bash
# Clone the repository
git clone https://github.com/your-username/optimizing_heterogeneous_tiered_memory.git
cd optimizing_heterogeneous_tiered_memory

# Navigate to the benchmarks directory and compile
cd src_benchmarks
make all
```

## Usage & Simulation
Run the automated Python scripts to launch the parallel gem5 simulation sweeps.

```bash
# Navigate to the scripts directory
cd ../scripts

# 1. Run the Architecture Comparison Sweep
python3 sweep_architecture_comparison.py

# 2. (Optional) Run other configuration sweeps
python3 sweep_capacity_ratio.py
python3 sweep_inference_scaling.py

# 3. Aggregate all simulation stats into a master CSV
python3 aggregate_results.py
```

## Project Structure
```text
📦 optimizing_heterogeneous_tiered_memory
 ┣ 📂 configs/            # Custom gem5 configuration scripts (e.g., tiered_memory_sys.py)
 ┣ 📂 src_benchmarks/     # C++ microbenchmarks and Makefile (AArch64 targets)
 ┣ 📂 scripts/            # Python automation for parallel simulations & analysis
 ┣ 📂 plots/              # Auto-generated, publication-ready visualizations
 ┗ 📂 results/            # Raw gem5 simulation outputs and aggregated CSVs
```

## Results & Analysis
The analysis scripts automatically generate publication-ready (IEEE-formatted) visualizations saved in the `plots/` directory. These plots highlight performance speedups, energy efficiency (EDP), and tiered memory traffic balancing against the LPDDR5 baseline. 

## Contributing
We welcome contributions to expand the simulated topologies or optimize the memory controller. Please review the project guidelines and submit a Pull Request with detailed descriptions of your configuration changes.

## License
Distributed under the MIT License. See `LICENSE` for more information.

## Contact
For questions regarding the simulation setup or project findings, please open an issue in the repository or contact the project maintainers.

---
*Developed as part of the Optimizing Heterogeneous Tiered Memory Architecture for Edge AI course project and optimized for Arm Ecosystem best practices.*
