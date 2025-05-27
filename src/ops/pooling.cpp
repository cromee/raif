#include "runtime/pooling.h"
#include <algorithm>
#include <limits>

namespace raif {

void max_pool2d_ref(float* dst, const float* src,
                    int N, int C, int H, int W,
                    int kernel_h, int kernel_w,
                    int stride_h, int stride_w,
                    int pad_h, int pad_w) {
    int out_h = (H + 2*pad_h - kernel_h) / stride_h + 1;
    int out_w = (W + 2*pad_w - kernel_w) / stride_w + 1;
    for(int n=0; n<N; ++n) {
        for(int c=0; c<C; ++c) {
            for(int oh=0; oh<out_h; ++oh) {
                for(int ow=0; ow<out_w; ++ow) {
                    float max_val = -std::numeric_limits<float>::infinity();
                    for(int kh=0; kh<kernel_h; ++kh) {
                        int ih = oh*stride_h - pad_h + kh;
                        if(ih < 0 || ih >= H) continue;
                        for(int kw=0; kw<kernel_w; ++kw) {
                            int iw = ow*stride_w - pad_w + kw;
                            if(iw < 0 || iw >= W) continue;
                            float val = src[((n*C + c)*H + ih)*W + iw];
                            max_val = std::max(max_val, val);
                        }
                    }
                    dst[((n*C + c)*out_h + oh)*out_w + ow] = max_val;
                }
            }
        }
    }
}

void avg_pool2d_ref(float* dst, const float* src,
                    int N, int C, int H, int W,
                    int kernel_h, int kernel_w,
                    int stride_h, int stride_w,
                    int pad_h, int pad_w) {
    int out_h = (H + 2*pad_h - kernel_h) / stride_h + 1;
    int out_w = (W + 2*pad_w - kernel_w) / stride_w + 1;
    for(int n=0; n<N; ++n) {
        for(int c=0; c<C; ++c) {
            for(int oh=0; oh<out_h; ++oh) {
                for(int ow=0; ow<out_w; ++ow) {
                    float sum_val = 0.0f;
                    int count = 0;
                    for(int kh=0; kh<kernel_h; ++kh) {
                        int ih = oh*stride_h - pad_h + kh;
                        if(ih < 0 || ih >= H) continue;
                        for(int kw=0; kw<kernel_w; ++kw) {
                            int iw = ow*stride_w - pad_w + kw;
                            if(iw < 0 || iw >= W) continue;
                            sum_val += src[((n*C + c)*H + ih)*W + iw];
                            count++;
                        }
                    }
                    dst[((n*C + c)*out_h + oh)*out_w + ow] =
                        count > 0 ? sum_val / count : 0.0f;
                }
            }
        }
    }
}

} // namespace raif

