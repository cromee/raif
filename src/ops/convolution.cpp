#include "runtime/convolution.h"
#include "kernels/cpu/winograd/winograd_conv.h"
#include <cstring>

namespace raif {

void conv2d_3x3_winograd(const float* input, const float* filter, float* output,
                          int batches, int in_channels, int out_channels,
                          int height, int width) {
    std::memset(output, 0, sizeof(float)*batches*out_channels*height*width);
    winograd_conv2d_3x3(input, filter, output, batches, in_channels, out_channels,
                        height, width);
}

void conv2d_5x5_winograd(const float* input, const float* filter, float* output,
                          int batches, int in_channels, int out_channels,
                          int height, int width) {
    std::memset(output, 0, sizeof(float)*batches*out_channels*height*width);
    winograd_conv2d_5x5(input, filter, output, batches, in_channels, out_channels,
                        height, width);
}

void conv2d_ref(const float* input, const float* filter, float* output,
                int batches, int in_channels, int out_channels,
                int height, int width,
                int kernel_h, int kernel_w,
                int stride,
                PaddingType padding) {
    int pad_h = (padding == PADDING_ZERO) ? (kernel_h / 2) : 0;
    int pad_w = (padding == PADDING_ZERO) ? (kernel_w / 2) : 0;

    int out_h = (height + 2 * pad_h - kernel_h) / stride + 1;
    int out_w = (width + 2 * pad_w - kernel_w) / stride + 1;

    for(int b=0; b<batches; ++b) {
        for(int oc=0; oc<out_channels; ++oc) {
            for(int oh=0; oh<out_h; ++oh) {
                for(int ow=0; ow<out_w; ++ow) {
                    float sum = 0.0f;
                    for(int ic=0; ic<in_channels; ++ic) {
                        for(int kh=0; kh<kernel_h; ++kh) {
                            int ih = oh * stride - pad_h + kh;
                            if(ih < 0 || ih >= height) continue;
                            for(int kw=0; kw<kernel_w; ++kw) {
                                int iw = ow * stride - pad_w + kw;
                                if(iw < 0 || iw >= width) continue;
                                int in_idx = ((b*in_channels + ic)*height + ih)*width + iw;
                                int f_idx = ((oc*in_channels + ic)*kernel_h + kh)*kernel_w + kw;
                                sum += input[in_idx] * filter[f_idx];
                            }
                        }
                    }
                    int out_idx = ((b*out_channels + oc)*out_h + oh)*out_w + ow;
                    output[out_idx] = sum;
                }
            }
        }
    }
}

} // namespace raif

