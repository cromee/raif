#ifndef RAIF_FIR_OP_H
#define RAIF_FIR_OP_H

namespace raif {

void fir_filter_ref(const float* input, const float* coeffs, float* output,
                    int length, int num_taps);

}

#endif // RAIF_FIR_OP_H
