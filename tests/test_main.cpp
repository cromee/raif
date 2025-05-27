#include <iostream>
#include "runtime/engine.h"
#include "runtime/activation.h"

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

    return 0;
}
