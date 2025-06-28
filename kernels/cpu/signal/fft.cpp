#include "kernels/cpu/signal/fft.h"
#include <vector>
#include <cmath>

namespace raif {

static void fft_rec(std::complex<float>* data, int n, bool inverse) {
    if(n <= 1) return;
    int half = n / 2;
    std::vector<std::complex<float>> even(half), odd(half);
    for(int i=0; i<half; ++i) {
        even[i] = data[2*i];
        odd[i] = data[2*i+1];
    }
    fft_rec(even.data(), half, inverse);
    fft_rec(odd.data(), half, inverse);
    float sign = inverse ? 1.0f : -1.0f;
    for(int k=0; k<half; ++k) {
        float angle = 2.0f * static_cast<float>(M_PI) * k / n;
        std::complex<float> w(std::cos(angle), sign * std::sin(angle));
        std::complex<float> t = w * odd[k];
        data[k] = even[k] + t;
        data[k+half] = even[k] - t;
    }
}

void fft_radix2(std::complex<float>* data, int n, bool inverse) {
    fft_rec(data, n, inverse);
    if(inverse) {
        for(int i=0;i<n;++i) data[i] /= static_cast<float>(n);
    }
}

} // namespace raif
