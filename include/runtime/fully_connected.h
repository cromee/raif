#ifndef RAIF_FULLY_CONNECTED_H
#define RAIF_FULLY_CONNECTED_H

namespace raif {

// Compute a fully connected layer.
// - output: [batch, out_features]
// - input: [batch, in_features]
// - weights: [out_features, in_features]
// - bias: [out_features] (can be nullptr)
// Parameters:
//   batch        - number of input samples
//   out_features - number of output nodes
//   in_features  - number of input nodes
void fully_connected(float* output,
                     const float* input,
                     const float* weights,
                     const float* bias,
                     int batch,
                     int out_features,
                     int in_features);

} // namespace raif

#endif // RAIF_FULLY_CONNECTED_H
