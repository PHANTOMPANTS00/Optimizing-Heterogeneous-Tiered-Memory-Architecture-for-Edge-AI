CXX = aarch64-linux-gnu-g++
GEM5_DIR = ../..
CXXFLAGS = -O3 -static -march=armv8.4-a+dotprod+fp16 -Wall -Wextra -DGEM5_M5OPS -I$(GEM5_DIR)/include
LDFLAGS = -L$(GEM5_DIR)/util/m5/build/arm64/out -lm5

TARGETS = gemm_bench kv_cache_bench

all: $(TARGETS)

gemm_bench: gemm_bench.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

kv_cache_bench: kv_cache_bench.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGETS)
