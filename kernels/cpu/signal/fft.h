#ifndef RAIF_FFT_RADIX2_H
#define RAIF_FFT_RADIX2_H

#include <complex>

namespace raif {

void fft_radix2(std::complex<float>* data, int n, bool inverse);

} // namespace raif

#endif // RAIF_FFT_RADIX2_H
