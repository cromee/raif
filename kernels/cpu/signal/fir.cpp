#include "kernels/cpu/signal/fir.h"

namespace raif {

void fir_filter(const float* input, const float* coeffs, float* output,
                int length, int num_taps) {
    for(int n=0; n<length; ++n) {
        float acc = 0.f;
        for(int k=0; k<num_taps; ++k) {
            if(n-k >= 0) acc += input[n-k] * coeffs[k];
        }
        output[n] = acc;
    }
}

} // namespace raif
