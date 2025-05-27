#ifndef RAIF_FLATTEN_H
#define RAIF_FLATTEN_H

namespace raif {

// Flatten the input tensor to 2D [N, C*H*W].
void flatten(float* output,
             const float* input,
             int N,
             int C,
             int H,
             int W);

} // namespace raif

#endif // RAIF_FLATTEN_H
