#include "runtime/activation.h"
#include <algorithm>
#include <cmath>
#include <immintrin.h>

namespace raif {

void relu_ref(float* dst, const float* src, int len) {
    for(int i=0;i<len;++i) dst[i] = std::max(0.0f, src[i]);
}

void relu_avx2(float* dst, const float* src, int len) {
    int i=0;
    __m256 zero = _mm256_setzero_ps();
    for(; i+8<=len; i+=8) {
        __m256 v = _mm256_loadu_ps(src + i);
        __m256 r = _mm256_max_ps(zero, v);
        _mm256_storeu_ps(dst + i, r);
    }
    for(; i<len; ++i) dst[i] = std::max(0.0f, src[i]);
}

void gelu_ref(float* dst, const float* src, int len) {
    const float c = std::sqrt(2.0f / M_PI);
    for(int i=0;i<len;++i) {
        float x = src[i];
        float t = c * (x + 0.044715f * x * x * x);
        dst[i] = 0.5f * x * (1.0f + std::tanh(t));
    }
}

void gelu_avx2(float* dst, const float* src, int len) {
    gelu_ref(dst, src, len);
}

void sigmoid_ref(float* dst, const float* src, int len) {
    for(int i=0;i<len;++i) {
        dst[i] = 1.0f / (1.0f + std::exp(-src[i]));
    }
}

void sigmoid_avx2(float* dst, const float* src, int len) {
    sigmoid_ref(dst, src, len);
}

void softmax_ref(float* dst, const float* src, int len) {
    float max_v = src[0];
    for(int i=1;i<len;++i) max_v = std::max(max_v, src[i]);
    float sum = 0.0f;
    for(int i=0;i<len;++i) {
        dst[i] = std::exp(src[i] - max_v);
        sum += dst[i];
    }
    for(int i=0;i<len;++i) dst[i] /= sum;
}

void softmax_avx2(float* dst, const float* src, int len) {
    softmax_ref(dst, src, len);
}

} // namespace raif

