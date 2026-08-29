import argparse
from typing import List, Sequence, Tuple
from m5.objects import AddrRange, Port, MemCtrl, AbstractMemory
from m5.util.convert import toMemorySize

from gem5.components.boards.simple_board import SimpleBoard
from gem5.components.processors.simple_processor import SimpleProcessor
from gem5.components.processors.cpu_types import CPUTypes
from gem5.isas import ISA
from gem5.components.memory.abstract_memory_system import AbstractMemorySystem
from gem5.components.memory.memory import ChanneledMemory
from gem5.components.memory.hbm import HBM2Stack
from gem5.components.memory.dram_interfaces.lpddr5 import LPDDR5_6400_1x16_BG_BL32
from gem5.resources.resource import BinaryResource
from gem5.utils.override import overrides
from gem5.simulate.simulator import Simulator

from three_level_cache_hierarchy import PrivateL1PrivateL2SharedL3CacheHierarchy

PAGE_SIZE = 4096


def align_up(value: int, alignment: int = PAGE_SIZE) -> int:
    return (value + alignment - 1) & ~(alignment - 1)

from gem5.components.memory.hbm import HighBandwidthMemory
from gem5.components.memory.dram_interfaces.hbm import HBM_2000_4H_1x64

class Edge_HBM_Inference(HBM_2000_4H_1x64):
    page_policy = "open"
    read_buffer_size = 128
    write_buffer_size = 128


class TieredMemory(AbstractMemorySystem):
    """
    A custom hybrid memory system combining LPDDR5 and HBM into contiguous regions.
    With default SimpleBoard (base=0x0, 4GiB total):
      HBM    occupies [0x00000000, 0x80000000)  (lower 2 GiB)
      LPDDR5 occupies [0x80000000, 0x100000000) (upper 2 GiB)
    """
    def __init__(self, lpddr5_size: str, hbm_size: str):
        super().__init__()
        # LPDDR5_6400 with 2 channels -- realistic for budget edge SoCs
        self.lpddr5 = ChanneledMemory(LPDDR5_6400_1x16_BG_BL32, 2, 64, size=lpddr5_size)
        self.hbm = HighBandwidthMemory(Edge_HBM_Inference, 4, 256, size=hbm_size)

        self._lpddr5_size = toMemorySize(lpddr5_size)
        self._hbm_size = toMemorySize(hbm_size)
        self._size = self._lpddr5_size + self._hbm_size

    @overrides(AbstractMemorySystem)
    def incorporate_memory(self, board) -> None:
        self.lpddr5.incorporate_memory(board)
        self.hbm.incorporate_memory(board)

    @overrides(AbstractMemorySystem)
    def get_mem_ports(self) -> Sequence[Tuple[AddrRange, Port]]:
        return self.lpddr5.get_mem_ports() + self.hbm.get_mem_ports()

    @overrides(AbstractMemorySystem)
    def get_memory_controllers(self) -> List[MemCtrl]:
        return self.lpddr5.get_memory_controllers() + self.hbm.get_memory_controllers()

    @overrides(AbstractMemorySystem)
    def get_mem_interfaces(self) -> List[AbstractMemory]:
        return self.lpddr5.get_mem_interfaces() + self.hbm.get_mem_interfaces()

    @overrides(AbstractMemorySystem)
    def get_size(self) -> int:
        return self._size

    @overrides(AbstractMemorySystem)
    def set_memory_range(self, ranges: List[AddrRange]) -> None:
        if len(ranges) != 1 or ranges[0].size() != self._size:
            raise Exception("TieredMemory requires a single range matching its size.")

        base = ranges[0].start

        # HBM occupies base range (hot activations / KV cache)
        # LPDDR5 occupies upper range (cold weights / large memory fallback)
        hbm_range = AddrRange(start=base, size=self._hbm_size)
        lpddr5_range = AddrRange(start=base + self._hbm_size, size=self._lpddr5_size)

        self.hbm.set_memory_range([hbm_range])
        self.lpddr5.set_memory_range([lpddr5_range])

    @overrides(AbstractMemorySystem)
    def get_uninterleaved_range(self) -> List[AddrRange]:
        return self.lpddr5.get_uninterleaved_range() + self.hbm.get_uninterleaved_range()


from gem5.simulate.exit_event import ExitEvent

def workbegin_handler():
    print("Core 1 hit WORKBEGIN. (Waiting for others...)")
    yield False
    print("Core 2 hit WORKBEGIN. (Waiting for others...)")
    yield False
    print("Core 3 hit WORKBEGIN. (Waiting for others...)")
    yield False
    print("Core 4 hit WORKBEGIN. All cores ready! Resetting stats to start ROI...")
    import m5
    m5.stats.reset()
    while True:
        yield False

def workend_handler():
    print("Core 1 hit WORKEND. (Waiting for others...)")
    yield False
    print("Core 2 hit WORKEND. (Waiting for others...)")
    yield False
    print("Core 3 hit WORKEND. (Waiting for others...)")
    yield False
    print("Core 4 hit WORKEND. All cores finished! Dumping stats...")
    import m5
    m5.stats.dump()
    while True:
        yield False

def dumpstats_handler():
    while True:
        yield False

def inject_power_models(memory_system):
    # Iterate through all memory controllers in the system
    for ctrl in memory_system.get_memory_controllers():
        drams = []
        if hasattr(ctrl, 'dram'):
            drams.append(ctrl.dram)
        if hasattr(ctrl, 'dram_2'):
            drams.append(ctrl.dram_2)

        for dram in drams:
            dram.enable_dram_powerdown = True
            
            # Distinguish HBM from LPDDR5 based on presence of dram_2 
            # (gem5 HBM pseudo-channels use dram and dram_2)
            is_hbm = hasattr(ctrl, 'dram_2')

            if is_hbm:
                # HBM2 Power Model (1.2V core, 2.5V wordline)
                dram.VDD = "1.2V"
                dram.VDD2 = "2.5V"
                dram.IDD0 = "150mA"
                dram.IDD02 = "5mA"
                dram.IDD2N = "35mA"
                dram.IDD2N2 = "5mA"
                dram.IDD3N = "45mA"
                dram.IDD3N2 = "5mA"
                dram.IDD4W = "350mA"
                dram.IDD4W2 = "5mA"
                dram.IDD4R = "350mA"
                dram.IDD4R2 = "5mA"
                dram.IDD5 = "200mA"
                dram.IDD52 = "5mA"
                dram.IDD3P0 = "30mA"
                dram.IDD3P02 = "5mA"
                dram.IDD3P1 = "30mA"
                dram.IDD3P12 = "5mA"
                dram.IDD2P0 = "30mA"
                dram.IDD2P02 = "5mA"
                dram.IDD2P1 = "30mA"
                dram.IDD2P12 = "5mA"
                dram.IDD6 = "15mA"
                dram.IDD62 = "5mA"
            else:
                # LPDDR5 Power Model (1.05V core, 1.05V I/O)
                dram.VDD = "1.05V"
                dram.VDD2 = "1.05V"
                dram.IDD0 = "40mA"
                dram.IDD02 = "40mA"
                dram.IDD2N = "15mA"
                dram.IDD2N2 = "15mA"
                dram.IDD3N = "20mA"
                dram.IDD3N2 = "20mA"
                dram.IDD4W = "80mA"
                dram.IDD4W2 = "80mA"
                dram.IDD4R = "80mA"
                dram.IDD4R2 = "80mA"
                dram.IDD5 = "100mA"
                dram.IDD52 = "100mA"
                dram.IDD3P0 = "10mA"
                dram.IDD3P02 = "10mA"
                dram.IDD3P1 = "10mA"
                dram.IDD3P12 = "10mA"
                dram.IDD2P0 = "10mA"
                dram.IDD2P02 = "10mA"
                dram.IDD2P1 = "10mA"
                dram.IDD2P12 = "10mA"
                dram.IDD6 = "5mA"
                dram.IDD62 = "5mA"

def main():
    parser = argparse.ArgumentParser(description="Tiered Memory SE Mode Config")
    parser.add_argument(
        "--mem-mode",
        type=str,
        choices=["pure_lpddr5", "hybrid", "pure_hbm"],
        default="hybrid",
        help="Memory configuration to use."
    )
    parser.add_argument(
        "--binary",
        type=str,
        required=True,
        help="Path to the compiled AArch64 benchmark binary."
    )
    parser.add_argument(
        "--options",
        type=str,
        default="",
        help="Command line arguments for the benchmark."
    )
    parser.add_argument(
        "--hbm-weight-fraction",
        type=float,
        default=0.5,
        help="Fraction (0.0-1.0) of the weight matrix W placed in the HBM "
             "tier when --mem-mode=hybrid. 0.0 = all weights in LPDDR5, "
             "1.0 = all weights in HBM. Sweep this to find the optimal "
             "tiering point for a given matrix size / bandwidth ratio.",
    )
    parser.add_argument(
        "--hbm-capacity-fraction",
        type=float,
        default=0.5,
        help="Fraction of the total 4GiB memory allocated to HBM in hybrid mode (default 0.5 = 2GiB HBM, 2GiB LPDDR5).",
    )
    parser.add_argument(
        "--use-prefetcher",
        action="store_true",
        help="Enable stride prefetchers on L1d/L2 (off by default, matching "
             "the original baseline).",
    )
    args = parser.parse_args()

    if args.mem_mode == "hybrid":
        if not (0.0 <= args.hbm_weight_fraction <= 1.0):
            raise ValueError("--hbm-weight-fraction must be in [0.0, 1.0]")
        if not (0.0 <= args.hbm_capacity_fraction <= 1.0):
            raise ValueError("--hbm-capacity-fraction must be in [0.0, 1.0]")

    # Create the 3-level Cache Hierarchy (Private L1+L2, Shared L3)
    cache_hierarchy = PrivateL1PrivateL2SharedL3CacheHierarchy(
        l1d_size="32KiB",
        l1i_size="32KiB",
        l2_size="512KiB",
        l3_size="2MiB",
        l3_assoc=16,
        use_prefetcher=args.use_prefetcher,
    )

    # Configure the memory system based on the mode
    # Total memory size is 4GiB for fairness (4096 MiB)
    TOTAL_MIB = 4096
    
    if args.mem_mode == "hybrid":
        hbm_mib = int(TOTAL_MIB * args.hbm_capacity_fraction)
        lpddr5_mib = TOTAL_MIB - hbm_mib
        if hbm_mib == 0 or lpddr5_mib == 0:
            raise ValueError("Hybrid mode requires non-zero capacity for both HBM and LPDDR5.")
    else:
        hbm_mib = TOTAL_MIB if args.mem_mode == "pure_hbm" else 0
        lpddr5_mib = TOTAL_MIB if args.mem_mode == "pure_lpddr5" else 0

    if args.mem_mode == "pure_lpddr5":
        memory = ChanneledMemory(LPDDR5_6400_1x16_BG_BL32, 2, 64, size="4GiB")
    elif args.mem_mode == "pure_hbm":
        memory = HighBandwidthMemory(Edge_HBM_Inference, 4, 256, size="4GiB")
    elif args.mem_mode == "hybrid":
        memory = TieredMemory(lpddr5_size=f"{lpddr5_mib}MiB", hbm_size=f"{hbm_mib}MiB")
    else:
        raise ValueError("Invalid memory mode.")

    # Explicitly inject VDD and IDD parameters to activate DRAMPower
    inject_power_models(memory)

    # Create the Processor
    processor = SimpleProcessor(
        cpu_type=CPUTypes.O3,
        isa=ISA.ARM,
        num_cores=4
    )

    # Create the Board
    # SimpleBoard uses base=0x0 by default, which is correct for SE mode.
    # With 4GiB memory, the physical range is [0x0, 0x100000000).
    # For hybrid: HBM=[0x0, HBM_SIZE), LPDDR5=[HBM_SIZE, 0x100000000).
    board = SimpleBoard(
        clk_freq="3GHz",
        processor=processor,
        memory=memory,
        cache_hierarchy=cache_hierarchy,
    )

    # Attach the binary workload (4 processes, one per core)
    binary_resource = BinaryResource(local_path=args.binary)
    arguments_list = args.options.split() if args.options else []
    board.set_se_multi_binary_workload(
        binaries=[binary_resource] * 4,
        arguments=[arguments_list] * 4
    )

    # Prepare Simulator
    simulator = Simulator(
        board=board,
        on_exit_event={
            ExitEvent.WORKBEGIN: workbegin_handler(),
            ExitEvent.WORKEND: workend_handler()
        }
    )
    print(f"Starting gem5 simulation with {args.mem_mode} memory "
          f"(hbm_weight_fraction={args.hbm_weight_fraction if args.mem_mode == 'hybrid' else 'n/a'})...")

    # Explicitly instantiate the C++ objects first so we can modify process page tables
    simulator._instantiate()

    # ===================================================================
    # Enforce deterministic memory partitioning in SE Mode for ALL modes.
    #
    # Virtual addresses (VA_A/VA_B/VA_C for W/x/y) must be chosen to avoid
    # collisions with the SE-mode process layout:
    #   - Binary text:  0x400000 - 0x600000
    #   - Heap/brk:     ~0x600000+
    #   - mmap region:  grows DOWN from 0x4000000000
    #   - Stack:        around 0x7fffff0000
    #
    # We use 0x10_0000_0000+ (64 GiB+) which is safely in unused VA space.
    #
    # Physical addresses must fall within the board's memory range [0x0, 0x100000000).
    # The SE-mode binary code/data occupy low physical pages (~0x0 - 0x1000000),
    # so matrix physical mappings start at 0x10000000 (256 MiB) to avoid overlap.
    #
    # CHANGE vs. original: the per-core physical stride is now computed
    # from the actual matrix size N instead of a hardcoded 64 MiB. This
    # removes the silent N <= ~4096 ceiling and the risk of cross-core
    # physical-address aliasing for larger matrices -- we now fail loudly
    # via an assertion instead of corrupting data silently.
    # ===================================================================
    is_kv_cache = "kv_cache_bench" in args.binary

    if is_kv_cache:
        # Arguments for kv_cache_bench: <seq_len> <d> <elem_bytes> <shard_rows> <K_addr> <Q_addr> <out_addr>
        seq_len = int(arguments_list[0]) if len(arguments_list) > 0 else 256
        d = int(arguments_list[1]) if len(arguments_list) > 1 else 256
        elem_bytes = int(arguments_list[2]) if len(arguments_list) > 2 else 4
        shard_rows = int(arguments_list[3]) if len(arguments_list) > 3 else 1
        
        seq_len = seq_len // shard_rows
        
        w_size_bytes = align_up(seq_len * d * elem_bytes)  # KV Cache matrix (K)
        v_size_bytes = align_up(d * elem_bytes)            # Query vector (Q)
        out_size_bytes = align_up(seq_len * 4)             # Output vector
        
        # Override VA variables to match kv_cache_bench expectations
        VA_A = int(arguments_list[4], 16) if len(arguments_list) > 4 else 0x1000000000
        VA_B = int(arguments_list[5], 16) if len(arguments_list) > 5 else 0x1100000000
        VA_C = int(arguments_list[6], 16) if len(arguments_list) > 6 else 0x1200000000
    else:
        # Arguments for gemm_bench/gemv_bench: <N> <VA_A> <VA_B> <VA_C> <batch>
        N = int(arguments_list[0]) if len(arguments_list) > 0 else 256
        batch = int(arguments_list[4]) if len(arguments_list) > 4 else 1
        
        w_size_bytes = align_up(N * N * 4)
        v_size_bytes = align_up(N * batch * 4)
        out_size_bytes = v_size_bytes

        VA_A = int(arguments_list[1], 16) if len(arguments_list) > 1 else 0x1000000000
        VA_B = int(arguments_list[2], 16) if len(arguments_list) > 2 else 0x1100000000
        VA_C = int(arguments_list[3], 16) if len(arguments_list) > 3 else 0x1200000000

    if args.mem_mode == "hybrid":
        hbm_w_bytes = align_up(int(w_size_bytes * args.hbm_weight_fraction))
        lpddr5_w_bytes = w_size_bytes - hbm_w_bytes
    else:
        hbm_w_bytes = 0
        lpddr5_w_bytes = 0

    # Per-core physical stride: big enough to hold W + x + y for this core
    # with headroom, rounded up to a page and to a sane minimum so small-N
    # runs still get the original 64 MiB of isolation (avoids false
    # sharing / prefetcher cross-talk between cores' regions).
    MIN_CORE_STRIDE = 64 * 1024 * 1024
    needed_stride = align_up(w_size_bytes + v_size_bytes + out_size_bytes + PAGE_SIZE)
    per_core_stride = max(MIN_CORE_STRIDE, needed_stride)

    num_cores = board.get_processor().get_num_cores()

    # Physical layout base offsets within each tier
    MATRIX_REGION_BASE = 0x10000000  # 256 MiB, clear of SE-mode code/heap
    HBM_BASE = 0x00000000
    LPDDR5_BASE = hbm_mib * 1024 * 1024 if args.mem_mode == "hybrid" else 0

    if args.mem_mode == "hybrid":
        # Guard: each core's slice of both tiers must fit within that
        # tier's capacity budget, and must not run past the tier boundary
        # (which would silently alias into the other tier).
        hbm_budget = hbm_mib * 1024 * 1024
        lpddr5_budget = lpddr5_mib * 1024 * 1024
        
        max_hbm_offset = MATRIX_REGION_BASE + (num_cores - 1) * per_core_stride + hbm_w_bytes + v_size_bytes
        if max_hbm_offset > hbm_budget:
            raise ValueError(f"HBM physical allocation exceeded {hbm_mib} MiB! Increase capacity or decrease matrix size/fraction.")
        
        max_lpddr5_offset = MATRIX_REGION_BASE + (num_cores - 1) * per_core_stride + lpddr5_w_bytes
        if max_lpddr5_offset > lpddr5_budget:
            raise ValueError(f"LPDDR5 physical allocation exceeded {lpddr5_mib} MiB! Increase capacity or decrease matrix size/fraction.")
    else:
        max_offset = MATRIX_REGION_BASE + (num_cores - 1) * per_core_stride + w_size_bytes + v_size_bytes + out_size_bytes
        total_budget = TOTAL_MIB * 1024 * 1024
        assert max_offset <= total_budget, (
            f"Memory overflow: need up to 0x{max_offset:x} bytes but total "
            f"memory is {total_budget} bytes. Reduce N or num_cores."
        )

    for i, core in enumerate(board.get_processor().get_cores()):
        process = core.core.workload[0]
        offset = i * per_core_stride

        if args.mem_mode == "hybrid":
            # HBM=[0x0, 0x80000000), LPDDR5=[0x80000000, 0x100000000)
            # hbm_weight_fraction of W goes to HBM, the rest to LPDDR5.
            if hbm_w_bytes > 0:
                process.map(VA_A, HBM_BASE + MATRIX_REGION_BASE + offset,
                            hbm_w_bytes, cacheable=True)
            if lpddr5_w_bytes > 0:
                process.map(VA_A + hbm_w_bytes,
                            LPDDR5_BASE + MATRIX_REGION_BASE + offset,
                            lpddr5_w_bytes, cacheable=True)

            # Activations (x) and output (y) stay in HBM -- hot, small,
            # reused every decode step.
            process.map(VA_B, HBM_BASE + MATRIX_REGION_BASE + offset + hbm_w_bytes,
                        v_size_bytes, cacheable=True)
            process.map(VA_C, HBM_BASE + MATRIX_REGION_BASE + offset + hbm_w_bytes + v_size_bytes,
                        out_size_bytes, cacheable=True)

        else:
            # pure_lpddr5 and pure_hbm: single contiguous 4GiB at [0x0, 0x100000000)
            process.map(VA_A, MATRIX_REGION_BASE + offset, w_size_bytes, cacheable=True)
            process.map(VA_B, MATRIX_REGION_BASE + offset + w_size_bytes, v_size_bytes, cacheable=True)
            process.map(VA_C, MATRIX_REGION_BASE + offset + w_size_bytes + v_size_bytes, out_size_bytes, cacheable=True)

    # Run the simulation
    simulator.run()
    print("Simulation complete.")


if True:
    main()