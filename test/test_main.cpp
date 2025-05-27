#include <iostream>
#include "raif/runtime.h"

int main() {
    raif::init();
    const int M = 2, N = 2, K = 2;
    float A[M*K] = {1,2,3,4};
    float B[K*N] = {5,6,7,8};
    float C[M*N] = {0};
    raif::matmul(C, A, B, M, N, K);
    for(int i=0;i<M*N;i++) std::cout << C[i] << " ";
    std::cout << std::endl;
    return 0;
}
