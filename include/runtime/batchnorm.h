#ifndef RAIF_BATCHNORM_H
#define RAIF_BATCHNORM_H

namespace raif {

// Batch normalization for inference.
// Parameters:
//  input  - [N, C, H, W]
//  mean   - [C]
//  var    - [C]
//  weight - [C] (can be nullptr)
//  bias   - [C] (can be nullptr)
//  epsilon - small constant for numerical stability
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
                       int W);

} // namespace raif

#endif // RAIF_BATCHNORM_H
