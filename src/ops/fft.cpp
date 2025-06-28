#include "runtime/fft.h"
#include "kernels/cpu/signal/fft.h"

namespace raif {

void fft_ref(std::complex<float>* data, int n) {
    fft_radix2(data, n, false);
}

void ifft_ref(std::complex<float>* data, int n) {
    fft_radix2(data, n, true);
}

} // namespace raif
