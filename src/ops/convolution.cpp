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

} // namespace raif

