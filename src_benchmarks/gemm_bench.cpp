#include <iostream>
#include <cstdlib>
#include <sys/mman.h>
#include <cstdint>
#include <iomanip>
#include <string>
#include <algorithm>
#include <arm_neon.h>

// ============================================================================
// GEMM Benchmark: Y = X * W^T
// 
// This models the prefill phase of LLM inference (processing a prompt of 
// length 'batch'). Here X is [batch x N], W is [N x N], Y is [batch x N].
// 
// By looping over the rows of W (outer loop) and the batch dimension of X (inner loop),
// each row of W is brought into the L1 cache once and reused 'batch' times. 
// This gives a much higher arithmetic intensity than GEMV, shifting the 
// performance bottleneck from memory bandwidth towards compute.
// ============================================================================

#ifdef GEM5_M5OPS
#include <gem5/m5ops.h>
#else
static inline void m5_reset_stats(uint64_t, uint64_t) {}
static inline void m5_dump_stats(uint64_t, uint64_t) {}
static inline void m5_work_begin(uint64_t, uint64_t) {}
static inline void m5_work_end(uint64_t, uint64_t) {}
#endif

float* allocate_matrix(size_t num_elements, uint64_t fixed_addr) {
    if (fixed_addr != 0) {
        return reinterpret_cast<float*>(fixed_addr);
    }
    size_t size = num_elements * sizeof(float);
    void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (ptr == MAP_FAILED) {
        perror("mmap failed");
        exit(EXIT_FAILURE);
    }
    return static_cast<float*>(ptr);
}

void init_matrix(float* mat, size_t count, float base_val) {
    for (size_t i = 0; i < count; ++i) {
        mat[i] = base_val + static_cast<float>(i % 100) / 100.0f;
    }
}

// Y = X * W^T (W is N x N, X is batch x N, Y is batch x N)
void gemm_neon(const float* W, const float* X, float* Y, int N, int batch) {
    // Outer loop: rows of W
    for (int i = 0; i < N; ++i) {
        // Inner loop: batch dimension of X
        for (int b = 0; b < batch; ++b) {
            float32x4_t acc = vdupq_n_f32(0.0f);
            int j = 0;

            // Process 16 elements per iteration
            for (; j <= N - 16; j += 16) {
                float32x4_t w0 = vld1q_f32(&W[i * N + j]);
                float32x4_t w1 = vld1q_f32(&W[i * N + j + 4]);
                float32x4_t w2 = vld1q_f32(&W[i * N + j + 8]);
                float32x4_t w3 = vld1q_f32(&W[i * N + j + 12]);

                float32x4_t x0 = vld1q_f32(&X[b * N + j]);
                float32x4_t x1 = vld1q_f32(&X[b * N + j + 4]);
                float32x4_t x2 = vld1q_f32(&X[b * N + j + 8]);
                float32x4_t x3 = vld1q_f32(&X[b * N + j + 12]);

                acc = vfmaq_f32(acc, w0, x0);
                acc = vfmaq_f32(acc, w1, x1);
                acc = vfmaq_f32(acc, w2, x2);
                acc = vfmaq_f32(acc, w3, x3);
            }

            for (; j <= N - 4; j += 4) {
                float32x4_t w_vec = vld1q_f32(&W[i * N + j]);
                float32x4_t x_vec = vld1q_f32(&X[b * N + j]);
                acc = vfmaq_f32(acc, w_vec, x_vec);
            }

            float sum = vaddvq_f32(acc);

            for (; j < N; ++j) {
                sum += W[i * N + j] * X[b * N + j];
            }

            Y[b * N + i] = sum;
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " N [addrW_hex] [addrX_hex] [addrY_hex] [batch] [warmup_iters] [measured_iters]"
                  << std::endl;
        return EXIT_FAILURE;
    }

    int N = std::stoi(argv[1]);
    uint64_t addrW = (argc > 2) ? std::stoull(argv[2], nullptr, 16) : 0;
    uint64_t addrX = (argc > 3) ? std::stoull(argv[3], nullptr, 16) : 0;
    uint64_t addrY = (argc > 4) ? std::stoull(argv[4], nullptr, 16) : 0;
    int batch          = (argc > 5) ? std::stoi(argv[5]) : 8;
    int warmup_iters   = (argc > 6) ? std::stoi(argv[6]) : 1;
    int measured_iters = (argc > 7) ? std::stoi(argv[7]) : 5;

    std::cout << "GEMM Benchmark (N=" << N << ", batch=" << batch << ")" << std::endl;
    std::cout << "Allocating buffers..." << std::endl;

    size_t w_elements = static_cast<size_t>(N) * N;
    size_t x_elements = static_cast<size_t>(batch) * N;
    size_t y_elements = static_cast<size_t>(batch) * N;

    float* W = allocate_matrix(w_elements, addrW);
    float* X = allocate_matrix(x_elements, addrX);
    float* Y = allocate_matrix(y_elements, addrY);

    if (addrW) std::cout << "W mapped at: 0x" << std::hex << addrW << std::dec << std::endl;
    if (addrX) std::cout << "X mapped at: 0x" << std::hex << addrX << std::dec << std::endl;
    if (addrY) std::cout << "Y mapped at: 0x" << std::hex << addrY << std::dec << std::endl;

    init_matrix(W, w_elements, 1.0f);
    init_matrix(X, x_elements, 0.5f);
    for (size_t i = 0; i < y_elements; ++i) Y[i] = 0.0f;

    std::cout << "Running warmup GEMM iterations..." << std::endl;
    for (int it = 0; it < warmup_iters; ++it) {
        gemm_neon(W, X, Y, N, batch);
    }

    m5_work_begin(0, 0);

    std::cout << "Running measured GEMM iterations..." << std::endl;
    for (int it = 0; it < measured_iters; ++it) {
        gemm_neon(W, X, Y, N, batch);
    }

    m5_work_end(0, 0);

    std::cout << "GEMM completed. Y[0] = " << Y[0] << std::endl;

    if (addrW == 0) munmap(W, w_elements * sizeof(float));
    if (addrX == 0) munmap(X, x_elements * sizeof(float));
    if (addrY == 0) munmap(Y, y_elements * sizeof(float));

    return EXIT_SUCCESS;
}
