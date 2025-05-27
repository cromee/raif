#include "raif/relu.h"
#include <algorithm>
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

} // namespace raif
