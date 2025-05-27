#ifndef RAIF_WINOGRAD_CONV_H
#define RAIF_WINOGRAD_CONV_H

namespace raif {

void winograd_conv2d_3x3(const float* input, const float* filter, float* output,
                         int batches, int in_channels, int out_channels,
                         int height, int width);

void winograd_conv2d_5x5(const float* input, const float* filter, float* output,
                         int batches, int in_channels, int out_channels,
                         int height, int width);

}

#endif // RAIF_WINOGRAD_CONV_H
