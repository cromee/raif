#ifndef RAIF_FIR_H
#define RAIF_FIR_H

namespace raif {

void fir_filter(const float* input, const float* coeffs, float* output,
                int length, int num_taps);

} // namespace raif

#endif // RAIF_FIR_H
