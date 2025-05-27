#include "raif/runtime.h"
#include <iostream>

int main() {
    raif::init();
    const int M=1, N=1, K=4;
    float A[K] = {1,2,3,4};
    float B[K] = {5,6,7,8};
    float C[1] = {0};
    raif::matmul(C, A, B, M, N, K);
    std::cout << "Result: " << C[0] << std::endl;
    return 0;
}
