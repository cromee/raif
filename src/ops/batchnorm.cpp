#include "runtime/batchnorm.h"
#include <cmath>
#include <immintrin.h>

namespace raif {

void batchnorm_forward(float* output,
                       const float* input,
                       const float* mean,
                       const float* var,
                       const float* weight,
                       const float* bias,
                       float epsilon,
                       int N,
                       int C,
                       int H,
                       int W) {
    int spatial = H * W;
    for(int n=0; n<N; ++n) {
        for(int c=0; c<C; ++c) {
            float m = mean[c];
            float ivar = 1.0f / std::sqrt(var[c] + epsilon);
            float w = weight ? weight[c] : 1.0f;
            float b = bias ? bias[c] : 0.0f;
            const float* src = input + ((n*C + c) * spatial);
            float* dst = output + ((n*C + c) * spatial);
#ifdef __AVX2__
            __m256 vm = _mm256_set1_ps(m);
            __m256 vivar = _mm256_set1_ps(ivar);
            __m256 vw = _mm256_set1_ps(w);
            __m256 vb = _mm256_set1_ps(b);
            int i = 0;
            for(; i + 8 <= spatial; i += 8) {
                __m256 v = _mm256_loadu_ps(src + i);
                __m256 norm = _mm256_mul_ps(_mm256_sub_ps(v, vm), vivar);
                __m256 out = _mm256_fmadd_ps(norm, vw, vb);
                _mm256_storeu_ps(dst + i, out);
            }
            for(; i < spatial; ++i) {
                float norm = (src[i] - m) * ivar;
                dst[i] = norm * w + b;
            }
#else
            for(int i=0; i<spatial; ++i) {
                float norm = (src[i] - m) * ivar;
                dst[i] = norm * w + b;
            }
#endif
        }
    }
}

} // namespace raif

