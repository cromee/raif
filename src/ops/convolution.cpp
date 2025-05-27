#include "runtime/convolution.h"
#include "kernels/cpu/winograd/winograd_conv.h"
#include "runtime/engine.h"
#include <cstring>
#include <vector>

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

namespace {

void im2col(float* col,
            const float* data,
            int channels,
            int height,
            int width,
            int kernel_h,
            int kernel_w,
            int pad_h,
            int pad_w,
            int stride,
            int out_h,
            int out_w) {
    int csize = kernel_h * kernel_w;
    for(int c=0; c<channels; ++c) {
        for(int kh=0; kh<kernel_h; ++kh) {
            for(int kw=0; kw<kernel_w; ++kw) {
                int col_row = c * csize + kh * kernel_w + kw;
                for(int oh=0; oh<out_h; ++oh) {
                    int ih = oh * stride - pad_h + kh;
                    for(int ow=0; ow<out_w; ++ow) {
                        int iw = ow * stride - pad_w + kw;
                        float val = 0.f;
                        if(ih >=0 && ih < height && iw >=0 && iw < width) {
                            int idx = (c*height + ih)*width + iw;
                            val = data[idx];
                        }
                        int col_idx = (col_row * out_h + oh) * out_w + ow;
                        col[col_idx] = val;
                    }
                }
            }
        }
    }
}

} // anonymous namespace

void conv2d_im2col(const float* input, const float* filter, float* output,
                   int batches, int in_channels, int out_channels,
                   int height, int width,
                   int kernel_h, int kernel_w,
                   int stride,
                   PaddingType padding) {
    int pad_h = (padding == PADDING_ZERO) ? (kernel_h / 2) : 0;
    int pad_w = (padding == PADDING_ZERO) ? (kernel_w / 2) : 0;

    int out_h = (height + 2 * pad_h - kernel_h) / stride + 1;
    int out_w = (width + 2 * pad_w - kernel_w) / stride + 1;

    int col_rows = in_channels * kernel_h * kernel_w;
    int col_cols = out_h * out_w;
    std::vector<float> col(col_rows * col_cols);

    for(int b=0; b<batches; ++b) {
        const float* input_ptr = input + b * in_channels * height * width;
        im2col(col.data(), input_ptr,
               in_channels, height, width,
               kernel_h, kernel_w,
               pad_h, pad_w,
               stride, out_h, out_w);

        const float* filter_ptr = filter; // [out_channels, col_rows]
        float* out_ptr = output + b * out_channels * out_h * out_w;
        raif::matmul(out_ptr, filter_ptr, col.data(),
                     out_channels, col_cols, col_rows);
    }
}

} // namespace raif

