#include "runtime/activation.h"
#include <algorithm>
#include <cmath>
#include <immintrin.h>
#include <limits>

namespace raif {

static inline __m256 exp256_ps(__m256 x) {
    const __m256 exp_hi = _mm256_set1_ps(88.3762626647949f);
    const __m256 exp_lo = _mm256_set1_ps(-88.3762626647949f);

    x = _mm256_min_ps(x, exp_hi);
    x = _mm256_max_ps(x, exp_lo);

    const __m256 log2ef = _mm256_set1_ps(1.44269504088896341f);
    const __m256 c1 = _mm256_set1_ps(0.693359375f);
    const __m256 c2 = _mm256_set1_ps(-2.12194440e-4f);

    __m256 fx = _mm256_fmadd_ps(x, log2ef, _mm256_set1_ps(0.5f));
    __m256i emm0 = _mm256_cvttps_epi32(fx);
    fx = _mm256_cvtepi32_ps(emm0);
    fx = _mm256_floor_ps(fx);

    __m256 z = _mm256_fnmadd_ps(fx, c1, x);
    x = _mm256_fnmadd_ps(fx, c2, z);

    __m256 y = _mm256_set1_ps(1.9875691500E-4f);
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.3981999507E-3f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(8.3334519073E-3f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(4.1665795894E-2f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.6666665459E-1f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(5.0000001201E-1f));

    y = _mm256_fmadd_ps(y, _mm256_mul_ps(x, x), x);
    y = _mm256_add_ps(y, _mm256_set1_ps(1.0f));

    emm0 = _mm256_cvttps_epi32(fx);
    emm0 = _mm256_add_epi32(emm0, _mm256_set1_epi32(127));
    emm0 = _mm256_slli_epi32(emm0, 23);
    __m256 pow2n = _mm256_castsi256_ps(emm0);

    return _mm256_mul_ps(y, pow2n);
}

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
    int i = 0;
    const __m256 c = _mm256_set1_ps(std::sqrt(2.0f / M_PI));
    const __m256 k = _mm256_set1_ps(0.044715f);
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 one = _mm256_set1_ps(1.0f);
    for(; i + 8 <= len; i += 8) {
        __m256 x = _mm256_loadu_ps(src + i);
        __m256 x3 = _mm256_mul_ps(_mm256_mul_ps(x, x), x);
        __m256 t = _mm256_mul_ps(c, _mm256_fmadd_ps(k, x3, x));
        __m256 neg_two_t = _mm256_mul_ps(_mm256_set1_ps(-2.0f), t);
        __m256 exp_neg = exp256_ps(neg_two_t);
        __m256 sigmoid = _mm256_div_ps(one, _mm256_add_ps(one, exp_neg));
        __m256 tanh_t = _mm256_fmsub_ps(_mm256_set1_ps(2.0f), sigmoid, one);
        __m256 out = _mm256_mul_ps(half, _mm256_mul_ps(x, _mm256_add_ps(one, tanh_t)));
        _mm256_storeu_ps(dst + i, out);
    }
    for(; i < len; ++i) {
        float x = src[i];
        float t = std::sqrt(2.0f / M_PI) * (x + 0.044715f * x * x * x);
        dst[i] = 0.5f * x * (1.0f + std::tanh(t));
    }
}

void sigmoid_ref(float* dst, const float* src, int len) {
    for(int i=0;i<len;++i) {
        dst[i] = 1.0f / (1.0f + std::exp(-src[i]));
    }
}

void sigmoid_avx2(float* dst, const float* src, int len) {
    int i = 0;
    const __m256 one = _mm256_set1_ps(1.0f);
    for(; i + 8 <= len; i += 8) {
        __m256 x = _mm256_loadu_ps(src + i);
        __m256 neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
        __m256 exp_neg = exp256_ps(neg_x);
        __m256 denom = _mm256_add_ps(one, exp_neg);
        __m256 out = _mm256_div_ps(one, denom);
        _mm256_storeu_ps(dst + i, out);
    }
    for(; i < len; ++i) {
        dst[i] = 1.0f / (1.0f + std::exp(-src[i]));
    }
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
    if(len <= 0) return;

    int i = 0;
    __m256 max_vec = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
    for(; i + 8 <= len; i += 8) {
        __m256 v = _mm256_loadu_ps(src + i);
        max_vec = _mm256_max_ps(max_vec, v);
    }
    float max_tmp[8];
    _mm256_storeu_ps(max_tmp, max_vec);
    float max_v = max_tmp[0];
    for(int j=1; j<8; ++j) max_v = std::max(max_v, max_tmp[j]);
    for(; i < len; ++i) max_v = std::max(max_v, src[i]);

    i = 0;
    __m256 sum_vec = _mm256_setzero_ps();
    __m256 max_broadcast = _mm256_set1_ps(max_v);
    for(; i + 8 <= len; i += 8) {
        __m256 v = _mm256_sub_ps(_mm256_loadu_ps(src + i), max_broadcast);
        __m256 e = exp256_ps(v);
        _mm256_storeu_ps(dst + i, e);
        sum_vec = _mm256_add_ps(sum_vec, e);
    }
    float sum_tmp[8];
    _mm256_storeu_ps(sum_tmp, sum_vec);
    float sum = sum_tmp[0] + sum_tmp[1] + sum_tmp[2] + sum_tmp[3] +
                sum_tmp[4] + sum_tmp[5] + sum_tmp[6] + sum_tmp[7];
    for(; i < len; ++i) {
        float e = std::exp(src[i] - max_v);
        dst[i] = e;
        sum += e;
    }

    __m256 sum_broadcast = _mm256_set1_ps(sum);
    i = 0;
    for(; i + 8 <= len; i += 8) {
        __m256 v = _mm256_loadu_ps(dst + i);
        __m256 r = _mm256_div_ps(v, sum_broadcast);
        _mm256_storeu_ps(dst + i, r);
    }
    for(; i < len; ++i) dst[i] /= sum;
}

} // namespace raif

