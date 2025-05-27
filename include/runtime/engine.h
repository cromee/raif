#ifndef RAIF_ENGINE_H
#define RAIF_ENGINE_H

namespace raif {

void init();

void matmul(float* C, const float* A, const float* B, int M, int N, int K);

} // namespace raif

#endif // RAIF_ENGINE_H
