#include "runtime/engine.h"
#include "runtime/fully_connected.h"
#include <immintrin.h>
#include <cstring>

namespace raif {

void init_fully_connected();

namespace {
using MatMulFn = void(*)(float*, const float*, const float*, int, int, int);

void matmul_ref(float* C, const float* A, const float* B, int M, int N, int K) {
    for(int i=0;i<M;i++) {
        for(int j=0;j<N;j++) {
            float sum = 0.0f;
            for(int k=0;k<K;k++) {
                sum += A[i*K+k] * B[k*N+j];
            }
            C[i*N+j] = sum;
        }
    }
}

void matmul_avx2(float* C, const float* A, const float* B, int M, int N, int K) {
    // Simple AVX2 implementation with 8-float vectors
    for(int i=0;i<M;i++) {
        for(int j=0;j<N;j++) {
            __m256 sum = _mm256_setzero_ps();
            int k=0;
            for(; k+8<=K; k+=8) {
                __m256 va = _mm256_loadu_ps(A + i*K + k);
                // Broadcast B element since column stride may not be contiguous
                __m256 vbroadcast = _mm256_set1_ps(*(B + k*N + j));
                sum = _mm256_fmadd_ps(va, vbroadcast, sum);
            }
            float scalar_sum = 0.0f;
            for(;k<K;k++) {
                scalar_sum += A[i*K+k] * B[k*N+j];
            }
            alignas(32) float tmp[8];
            _mm256_store_ps(tmp, sum);
            for(int t=0;t<8;t++) scalar_sum += tmp[t];
            C[i*N+j] = scalar_sum;
        }
    }
}

void matmul_avx512(float* C, const float* A, const float* B,
                   int M, int N, int K) {
#ifdef __AVX512F__
    for(int i=0;i<M;i++) {
        for(int j=0;j<N;j++) {
            __m512 sum = _mm512_setzero_ps();
            int k=0;
            for(; k+16<=K; k+=16) {
                __m512 va = _mm512_loadu_ps(A + i*K + k);
                __m512 vb = _mm512_set1_ps(*(B + k*N + j));
                sum = _mm512_fmadd_ps(va, vb, sum);
            }
            alignas(64) float tmp[16];
            _mm512_store_ps(tmp, sum);
            float scalar_sum = 0.f;
            for(int t=0;t<16;t++) scalar_sum += tmp[t];
            for(; k<K; ++k) scalar_sum += A[i*K+k]*B[k*N+j];
            C[i*N+j] = scalar_sum;
        }
    }
#else
    matmul_avx2(C, A, B, M, N, K);
#endif
}

void matmul_amx(float* C, const float* A, const float* B,
                int M, int N, int K) {
    // Placeholder implementation. In a real implementation this would use
    // AMX tile instructions.
    matmul_avx512(C, A, B, M, N, K);
}

MatMulFn matmul_impl = matmul_ref;

} // anonymous namespace

void init() {
#ifdef __x86_64__
    if (__builtin_cpu_supports("avx512f")) {
        matmul_impl = matmul_avx512;
    } else if (__builtin_cpu_supports("avx2")) {
        matmul_impl = matmul_avx2;
    }
#endif
    init_fully_connected();
}

void matmul(float* C, const float* A, const float* B, int M, int N, int K) {
    matmul_impl(C, A, B, M, N, K);
}

} // namespace raif

