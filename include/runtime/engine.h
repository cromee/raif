#ifndef RAIF_ENGINE_H
#define RAIF_ENGINE_H

namespace raif {

void init();

void fully_connected(float* output,
                     const float* input,
                     const float* weights,
                     const float* bias,
                     int batch,
                     int out_features,
                     int in_features);

void matmul(float* C, const float* A, const float* B, int M, int N, int K);

} // namespace raif

#endif // RAIF_ENGINE_H
