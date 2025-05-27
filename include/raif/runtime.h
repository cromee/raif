#ifndef RAIF_RUNTIME_H
#define RAIF_RUNTIME_H

namespace raif {

void init();

void matmul(float* C, const float* A, const float* B, int M, int N, int K);

} // namespace raif

#endif // RAIF_RUNTIME_H
