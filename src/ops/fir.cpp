#include "runtime/fir.h"
#include "kernels/cpu/signal/fir.h"

namespace raif {

void fir_filter_ref(const float* input, const float* coeffs, float* output,
                    int length, int num_taps) {
    fir_filter(input, coeffs, output, length, num_taps);
}

} // namespace raif
