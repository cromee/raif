#include "runtime/flatten.h"
#include <cstring>

namespace raif {

void flatten(float* output, const float* input,
             int N, int C, int H, int W) {
    std::memcpy(output, input, sizeof(float) * N * C * H * W);
}

} // namespace raif

