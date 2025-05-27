#ifndef RAIF_ACTIVATION_H
#define RAIF_ACTIVATION_H

namespace raif {

void relu_ref(float* dst, const float* src, int len);
void relu_avx2(float* dst, const float* src, int len);

}

#endif // RAIF_ACTIVATION_H
