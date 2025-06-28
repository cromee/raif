#ifndef RAIF_FFT_H
#define RAIF_FFT_H

#include <complex>

namespace raif {

void fft_ref(std::complex<float>* data, int n);
void ifft_ref(std::complex<float>* data, int n);

}

#endif // RAIF_FFT_H
