#ifndef RAIF_POOLING_H
#define RAIF_POOLING_H

namespace raif {

void max_pool2d_ref(float* dst, const float* src,
                    int N, int C, int H, int W,
                    int kernel_h, int kernel_w,
                    int stride_h, int stride_w,
                    int pad_h, int pad_w);

void avg_pool2d_ref(float* dst, const float* src,
                    int N, int C, int H, int W,
                    int kernel_h, int kernel_w,
                    int stride_h, int stride_w,
                    int pad_h, int pad_w);

} // namespace raif

#endif // RAIF_POOLING_H
