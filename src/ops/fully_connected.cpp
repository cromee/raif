#include "runtime/fully_connected.h"
#include <immintrin.h>
#include <cstring>

namespace raif {

namespace {
using FcFn = void(*)(float*, const float*, const float*, const float*, int, int, int);

void fc_ref(float* output, const float* input, const float* weights, const float* bias,
            int batch, int out_features, int in_features) {
    for(int b=0; b<batch; ++b) {
        const float* x = input + b * in_features;
        for(int o=0; o<out_features; ++o) {
            const float* w = weights + o * in_features;
            float sum = bias ? bias[o] : 0.0f;
            for(int i=0; i<in_features; ++i) {
                sum += x[i] * w[i];
            }
            output[b * out_features + o] = sum;
        }
    }
}

void fc_avx2(float* output, const float* input, const float* weights, const float* bias,
             int batch, int out_features, int in_features) {
    for(int b=0; b<batch; ++b) {
        const float* x = input + b * in_features;
        for(int o=0; o<out_features; ++o) {
            const float* w = weights + o * in_features;
            __m256 vec_sum = _mm256_setzero_ps();
            int i=0;
            for(; i+8<=in_features; i+=8) {
                __m256 xv = _mm256_loadu_ps(x + i);
                __m256 wv = _mm256_loadu_ps(w + i);
                vec_sum = _mm256_fmadd_ps(xv, wv, vec_sum);
            }
            alignas(32) float tmp[8];
            _mm256_store_ps(tmp, vec_sum);
            float sum = bias ? bias[o] : 0.0f;
            for(int t=0; t<8; ++t) sum += tmp[t];
            for(; i<in_features; ++i) sum += x[i] * w[i];
            output[b * out_features + o] = sum;
        }
    }
}

FcFn fc_impl = fc_ref;

} // anonymous namespace

void fully_connected(float* output, const float* input, const float* weights, const float* bias,
                     int batch, int out_features, int in_features) {
    fc_impl(output, input, weights, bias, batch, out_features, in_features);
}

void init_fully_connected() {
#ifdef __x86_64__
    if (__builtin_cpu_supports("avx2")) {
        fc_impl = fc_avx2;
    }
#endif
}

} // namespace raif

