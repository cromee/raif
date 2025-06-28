#include <iostream>
#include "runtime/engine.h"
#include "runtime/activation.h"
#include "runtime/convolution.h"
#include "runtime/fft.h"
#include "runtime/fir.h"

int main() {
    raif::init();
    const int M = 2, N = 2, K = 2;
    float A[M*K] = {1,2,3,4};
    float B[K*N] = {5,6,7,8};
    float C[M*N] = {0};
    raif::matmul(C, A, B, M, N, K);
    for(int i=0;i<M*N;i++) std::cout << C[i] << " ";
    std::cout << std::endl;

    float X[4] = {-1.0f, 0.0f, 1.0f, 2.0f};
    float Y[4];
    raif::relu_ref(Y, X, 4);
    raif::gelu_ref(Y, X, 4);
    raif::sigmoid_ref(Y, X, 4);
    raif::softmax_ref(Y, X, 4);

    // Simple convolution test
    const int BATCH = 1, IC = 1, OC = 1, H = 4, W = 4;
    float input[BATCH*IC*H*W] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9,10,11,12,
        13,14,15,16
    };
    float filter[1*1*3*3] = {
        1,0,-1,
        1,0,-1,
        1,0,-1
    };
    const int KH = 3, KW = 3, STRIDE = 1;
    float out[BATCH*OC*H*W] = {0};
    raif::conv2d_im2col(input, filter, out,
                        BATCH, IC, OC, H, W, KH, KW, STRIDE,
                        raif::PADDING_ZERO);
    for(int i=0;i<H*W;i++) std::cout << out[i] << " ";
    std::cout << std::endl;

    // Simple FFT/IFFT test
    std::complex<float> data[4] = {1.0f, 0.0f, -1.0f, 0.0f};
    raif::fft_ref(data, 4);
    raif::ifft_ref(data, 4);

    // FIR filter test
    float coeffs[3] = {0.25f, 0.5f, 0.25f};
    float fir_out[4];
    raif::fir_filter_ref(X, coeffs, fir_out, 4, 3);

    return 0;
}
