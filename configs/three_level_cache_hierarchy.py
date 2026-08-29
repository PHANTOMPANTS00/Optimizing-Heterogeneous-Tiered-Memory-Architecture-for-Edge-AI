"""
Private L1 + Private L2 + Shared L3 Cache Hierarchy

Topology:
    CPU Core 0    CPU Core 1    CPU Core 2    CPU Core 3
      L1i L1d      L1i L1d      L1i L1d      L1i L1d   (Private, 32KiB)
        L2            L2            L2            L2     (Private, 512KiB)
         \\             |             |             /
          ----------- L3 Bus (L2XBar) -----------
                         |
                    Shared L3 Cache                      (Shared, 2MiB)
                         |
                    Memory Bus (SystemXBar)
                         |
               Memory Controllers (LPDDR5/HBM)

This mirrors real ARM Cortex-A78/X-series cluster designs.

CHANGES vs. original:
  - Optional stride prefetcher on L1d/L2 (off by default, matching the
    original's baseline). Real edge SoCs almost universally prefetch;
    leaving this off silently biases achieved bandwidth downward relative
    to real hardware. Enable via use_prefetcher=True to get a more
    realistic baseline, or sweep both settings to quantify the effect.
"""

from typing import Optional

from m5.objects import (
    BadAddr,
    BaseCPU,
    BaseXBar,
    Cache,
    L2XBar,
    StridePrefetcher,
    SystemXBar,
)
from m5.params import Port

from gem5.isas import ISA
from gem5.utils.override import *
from gem5.components.boards.abstract_board import AbstractBoard
from gem5.components.cachehierarchies.abstract_cache_hierarchy import (
    AbstractCacheHierarchy,
)
from gem5.components.cachehierarchies.abstract_two_level_cache_hierarchy import (
    AbstractTwoLevelCacheHierarchy,
)
from gem5.components.cachehierarchies.classic.abstract_classic_cache_hierarchy import (
    AbstractClassicCacheHierarchy,
)
from gem5.components.cachehierarchies.classic.caches.l1dcache import L1DCache
from gem5.components.cachehierarchies.classic.caches.l1icache import L1ICache
from gem5.components.cachehierarchies.classic.caches.l2cache import L2Cache


class PrivateL1PrivateL2SharedL3CacheHierarchy(
    AbstractClassicCacheHierarchy, AbstractTwoLevelCacheHierarchy
):
    """
    Each core has private L1i + L1d + L2 caches.
    All cores share a single L3 cache connected to the memory bus.
    """

    def _get_default_membus(self) -> SystemXBar:
        membus = SystemXBar(width=64)
        membus.badaddr_responder = BadAddr()
        membus.default = membus.badaddr_responder.pio
        return membus

    def __init__(
        self,
        l1d_size: str,
        l1i_size: str,
        l2_size: str,
        l3_size: str = "2MiB",
        l3_assoc: int = 16,
        membus: Optional[BaseXBar] = None,
        use_prefetcher: bool = False,
    ) -> None:
        AbstractClassicCacheHierarchy.__init__(self=self)
        AbstractTwoLevelCacheHierarchy.__init__(
            self,
            l1i_size=l1i_size,
            l1i_assoc=8,
            l1d_size=l1d_size,
            l1d_assoc=8,
            l2_size=l2_size,
            l2_assoc=4,
        )

        self._l3_size = l3_size
        self._l3_assoc = l3_assoc
        self._use_prefetcher = use_prefetcher
        self.membus = membus if membus else self._get_default_membus()

    @overrides(AbstractClassicCacheHierarchy)
    def get_mem_side_port(self) -> Port:
        return self.membus.mem_side_ports

    @overrides(AbstractClassicCacheHierarchy)
    def get_cpu_side_port(self) -> Port:
        return self.membus.cpu_side_ports

    @overrides(AbstractCacheHierarchy)
    def incorporate_cache(self, board: AbstractBoard) -> None:
        # System port for functional access
        board.connect_system_port(self.membus.cpu_side_ports)

        # Connect memory controller ports to the memory bus
        for _, port in board.get_mem_ports():
            self.membus.mem_side_ports = port

        # ---- Shared L3 ----
        # L3 bus collects all L2 miss traffic
        self.l3bus = L2XBar()

        # Shared L3 cache sits between L3 bus and memory bus
        self.l3cache = Cache(
            size=self._l3_size,
            assoc=self._l3_assoc,
            tag_latency=20,
            data_latency=20,
            response_latency=1,
            mshrs=32,
            tgts_per_mshr=16,
            writeback_clean=False,
        )
        # L3 bus -> L3 cache -> memory bus
        self.l3bus.mem_side_ports = self.l3cache.cpu_side
        self.l3cache.mem_side = self.membus.cpu_side_ports

        # ---- Per-core private L1 + L2 ----
        self.l2buses = [
            L2XBar() for _ in range(board.get_processor().get_num_cores())
        ]

        for i, cpu in enumerate(board.get_processor().get_cores()):
            # Create private L2 for this core
            l2_cache = L2Cache(size=self._l2_size)
            l1d_cache = L1DCache(size=self._l1d_size)

            if self._use_prefetcher:
                # Stride prefetcher on L1d and L2 -- realistic for edge
                # ARM cores and important for fair comparison against
                # real hardware achieved-bandwidth numbers.
                l1d_cache.prefetcher = StridePrefetcher()
                l2_cache.prefetcher = StridePrefetcher()

            l2_node = self.add_root_child(f"l2-cache-{i}", l2_cache)
            # Create private L1i and L1d under L2
            l1i_node = l2_node.add_child(
                f"l1i-cache-{i}", L1ICache(size=self._l1i_size)
            )
            l1d_node = l2_node.add_child(f"l1d-cache-{i}", l1d_cache)

            # L1 -> L2 bus -> L2 cache
            l1i_node.cache.mem_side = self.l2buses[i].cpu_side_ports
            l1d_node.cache.mem_side = self.l2buses[i].cpu_side_ports
            self.l2buses[i].mem_side_ports = l2_node.cache.cpu_side

            # L2 cache -> L3 bus (shared)
            l2_node.cache.mem_side = self.l3bus.cpu_side_ports

            # Connect CPU ports
            cpu.connect_icache(l1i_node.cache.cpu_side)
            cpu.connect_dcache(l1d_node.cache.cpu_side)

            # Table walker connects to L2 bus
            cpu.connect_walker_ports(
                self.l2buses[i].cpu_side_ports,
                self.l2buses[i].cpu_side_ports,
            )

            if board.get_processor().get_isa() == ISA.X86:
                int_req_port = self.membus.mem_side_ports
                int_resp_port = self.membus.cpu_side_ports
                cpu.connect_interrupt(int_req_port, int_resp_port)
            else:
                cpu.connect_interrupt()

        if board.has_coherent_io():
            self._setup_io_cache(board)

    def _setup_io_cache(self, board: AbstractBoard) -> None:
        self.iocache = Cache(
            assoc=8,
            tag_latency=50,
            data_latency=50,
            response_latency=50,
            mshrs=20,
            size="1KiB",
            tgts_per_mshr=12,
            addr_ranges=board.mem_ranges,
        )
        self.iocache.mem_side = self.membus.cpu_side_ports
        self.iocache.cpu_side = board.get_mem_side_coherent_io_port()