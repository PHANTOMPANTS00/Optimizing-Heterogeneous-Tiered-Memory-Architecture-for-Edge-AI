#include <iostream>
#include <cstdlib>
#include <sys/mman.h>
#include <cstdint>
#include <iomanip>
#include <string>
#include <algorithm>
#include <arm_neon.h>
#include <string.h>

#ifdef GEM5_M5OPS
#include <gem5/m5ops.h>
#else
static inline void m5_reset_stats(uint64_t, uint64_t) {}
static inline void m5_dump_stats(uint64_t, uint64_t) {}
static inline void m5_work_begin(uint64_t, uint64_t) {}
static inline void m5_work_end(uint64_t, uint64_t) {}
#endif

// ============================================================================
// KV Cache Benchmark
// Models the decode phase of LLM inference: computing attention over past tokens.
// Q * K^T (where Q is 1xd, K is seq_len x d).
// ============================================================================

void* allocate_matrix(size_t num_bytes, uint64_t fixed_addr) {
    if (fixed_addr != 0) {
        return reinterpret_cast<void*>(fixed_addr);
    }
    void* ptr = mmap(nullptr, num_bytes, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (ptr == MAP_FAILED) {
        perror("mmap failed");
        exit(EXIT_FAILURE);
    }
    return ptr;
}

void init_matrix_fp32(float* mat, size_t count, float base_val) {
    memset(mat, 0, count * sizeof(float));
}

void init_matrix_int8(int8_t* mat, size_t count, int8_t base_val) {
    memset(mat, 0, count * sizeof(int8_t));
}

// Compute Q * K^T for FP32 (Q is 1xd, K is seq_len x d)
void kv_attention_fp32(const float* K, const float* Q, float* out, int seq_len, int d) {
    for (int i = 0; i < seq_len; ++i) {
        float32x4_t acc = vdupq_n_f32(0.0f);
        int j = 0;
        for (; j <= d - 16; j += 16) {
            float32x4_t k0 = vld1q_f32(&K[i * d + j]);
            float32x4_t k1 = vld1q_f32(&K[i * d + j + 4]);
            float32x4_t k2 = vld1q_f32(&K[i * d + j + 8]);
            float32x4_t k3 = vld1q_f32(&K[i * d + j + 12]);

            float32x4_t q0 = vld1q_f32(&Q[j]);
            float32x4_t q1 = vld1q_f32(&Q[j + 4]);
            float32x4_t q2 = vld1q_f32(&Q[j + 8]);
            float32x4_t q3 = vld1q_f32(&Q[j + 12]);

            acc = vmlaq_f32(acc, k0, q0);
            acc = vmlaq_f32(acc, k1, q1);
            acc = vmlaq_f32(acc, k2, q2);
            acc = vmlaq_f32(acc, k3, q3);
        }
        for (; j < d; ++j) {
            out[i] += K[i * d + j] * Q[j];
        }
        out[i] += vgetq_lane_f32(acc, 0) + vgetq_lane_f32(acc, 1) +
                  vgetq_lane_f32(acc, 2) + vgetq_lane_f32(acc, 3);
    }
}

// Compute Q * K^T for INT8 using VDOT (SDOT)
// Note: ARMv8.4-A dot product instruction
void kv_attention_int8(const int8_t* K, const int8_t* Q, int32_t* out, int seq_len, int d) {
    for (int i = 0; i < seq_len; ++i) {
        int32x4_t acc = vdupq_n_s32(0);
        int j = 0;
        for (; j <= d - 16; j += 16) {
            int8x16_t k_vec = vld1q_s8(&K[i * d + j]);
            int8x16_t q_vec = vld1q_s8(&Q[j]);
            acc = vdotq_s32(acc, k_vec, q_vec);
        }
        int32_t sum = 0;
        for (; j < d; ++j) {
            sum += static_cast<int32_t>(K[i * d + j]) * static_cast<int32_t>(Q[j]);
        }
        out[i] = sum + vgetq_lane_s32(acc, 0) + vgetq_lane_s32(acc, 1) +
                       vgetq_lane_s32(acc, 2) + vgetq_lane_s32(acc, 3);
    }
}

void init_matrix_fp16(float16_t* mat, size_t count, float16_t base_val) {
    memset(mat, 0, count * sizeof(float16_t));
}

// Compute Q * K^T for FP16 (Q is 1xd, K is seq_len x d)
void kv_attention_fp16(const float16_t* K, const float16_t* Q, float* out, int seq_len, int d) {
    for (int i = 0; i < seq_len; ++i) {
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        int j = 0;
        for (; j <= d - 16; j += 16) {
            float16x8_t k0 = vld1q_f16(&K[i * d + j]);
            float16x8_t k1 = vld1q_f16(&K[i * d + j + 8]);

            float16x8_t q0 = vld1q_f16(&Q[j]);
            float16x8_t q1 = vld1q_f16(&Q[j + 8]);

            // First 8 elements
            acc0 = vmlaq_f32(acc0, vcvt_f32_f16(vget_low_f16(k0)), vcvt_f32_f16(vget_low_f16(q0)));
            acc1 = vmlaq_f32(acc1, vcvt_high_f32_f16(k0), vcvt_high_f32_f16(q0));
            // Next 8 elements
            acc0 = vmlaq_f32(acc0, vcvt_f32_f16(vget_low_f16(k1)), vcvt_f32_f16(vget_low_f16(q1)));
            acc1 = vmlaq_f32(acc1, vcvt_high_f32_f16(k1), vcvt_high_f32_f16(q1));
        }
        float sum = 0;
        for (; j < d; ++j) {
            sum += static_cast<float>(K[i * d + j]) * static_cast<float>(Q[j]);
        }
        out[i] = sum + vgetq_lane_f32(acc0, 0) + vgetq_lane_f32(acc0, 1) +
                       vgetq_lane_f32(acc0, 2) + vgetq_lane_f32(acc0, 3) +
                       vgetq_lane_f32(acc1, 0) + vgetq_lane_f32(acc1, 1) +
                       vgetq_lane_f32(acc1, 2) + vgetq_lane_f32(acc1, 3);
    }
}

int main(int argc, char* argv[]) {
    // Usage: kv_cache_bench <seq_len> <d> <elem_bytes> <shard_rows> <K_addr> <Q_addr> <out_addr>
    if (argc < 8) {
        std::cerr << "Usage: " << argv[0] << " <seq_len> <d> <elem_bytes> <shard_rows> <K_addr> <Q_addr> <out_addr>\n";
        return EXIT_FAILURE;
    }

    int seq_len = std::stoi(argv[1]);
    int d = std::stoi(argv[2]);
    int elem_bytes = std::stoi(argv[3]);
    int shard_rows = std::stoi(argv[4]);
    
    // Apply sharding (tensor parallelism divides the sequence length / rows among shards)
    seq_len = seq_len / shard_rows;

    uint64_t k_addr = std::stoull(argv[5], nullptr, 16);
    uint64_t q_addr = std::stoull(argv[6], nullptr, 16);
    uint64_t out_addr = std::stoull(argv[7], nullptr, 16);

    void* K = allocate_matrix(seq_len * d * elem_bytes, k_addr);
    void* Q = allocate_matrix(d * elem_bytes, q_addr);
    void* out = allocate_matrix(seq_len * 4, out_addr); // Output is always FP32/INT32 (4 bytes)

    if (elem_bytes == 4) {
        init_matrix_fp32(static_cast<float*>(K), seq_len * d, 1.0f);
        init_matrix_fp32(static_cast<float*>(Q), d, 0.5f);
    } else if (elem_bytes == 2) {
        init_matrix_fp16(static_cast<float16_t*>(K), seq_len * d, static_cast<float16_t>(1.0f));
        init_matrix_fp16(static_cast<float16_t*>(Q), d, static_cast<float16_t>(0.5f));
    } else if (elem_bytes == 1) {
        init_matrix_int8(static_cast<int8_t*>(K), seq_len * d, 1);
        init_matrix_int8(static_cast<int8_t*>(Q), d, 2);
    } else {
        std::cerr << "Unsupported elem_bytes: " << elem_bytes << "\n";
        return EXIT_FAILURE;
    }

    // Warmup
    if (elem_bytes == 4) {
        kv_attention_fp32(static_cast<float*>(K), static_cast<float*>(Q), static_cast<float*>(out), seq_len, d);
    } else if (elem_bytes == 2) {
        kv_attention_fp16(static_cast<float16_t*>(K), static_cast<float16_t*>(Q), static_cast<float*>(out), seq_len, d);
    } else if (elem_bytes == 1) {
        kv_attention_int8(static_cast<int8_t*>(K), static_cast<int8_t*>(Q), static_cast<int32_t*>(out), seq_len, d);
    }

    m5_work_begin(0, 0);

    const int measured_iters = 1;
    for (int it = 0; it < measured_iters; ++it) {
        if (elem_bytes == 4) {
            kv_attention_fp32(static_cast<float*>(K), static_cast<float*>(Q), static_cast<float*>(out), seq_len, d);
        } else if (elem_bytes == 2) {
            kv_attention_fp16(static_cast<float16_t*>(K), static_cast<float16_t*>(Q), static_cast<float*>(out), seq_len, d);
        } else if (elem_bytes == 1) {
            kv_attention_int8(static_cast<int8_t*>(K), static_cast<int8_t*>(Q), static_cast<int32_t*>(out), seq_len, d);
        }
    }

    m5_work_end(0, 0);

    return EXIT_SUCCESS;
}
