#ifndef RAIF_CONVOLUTION_H
#define RAIF_CONVOLUTION_H

namespace raif {

void conv2d_3x3_winograd(const float* input, const float* filter, float* output,
                         int batches, int in_channels, int out_channels,
                         int height, int width);

void conv2d_5x5_winograd(const float* input, const float* filter, float* output,
                         int batches, int in_channels, int out_channels,
                         int height, int width);

enum PaddingType {
    PADDING_ZERO,
    PADDING_VALID
};

void conv2d_ref(const float* input, const float* filter, float* output,
                int batches, int in_channels, int out_channels,
                int height, int width,
                int kernel_h, int kernel_w,
                int stride,
                PaddingType padding);

void conv2d_im2col(const float* input, const float* filter, float* output,
                   int batches, int in_channels, int out_channels,
                   int height, int width,
                   int kernel_h, int kernel_w,
                   int stride,
                   PaddingType padding);

} // namespace raif

#endif // RAIF_CONVOLUTION_H
