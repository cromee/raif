#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include "runtime/convolution.h"
#include "runtime/batchnorm.h"
#include "runtime/activation.h"
#include "runtime/pooling.h"
#include "runtime/fully_connected.h"
#include "runtime/flatten.h"

using namespace raif;

// Simple helper to initialize weights
static void init_vector(std::vector<float>& v) {
    for(auto& x : v) {
        x = static_cast<float>(std::rand()) / RAND_MAX - 0.5f;
    }
}

int main() {
    const int batch = 1;
    const int in_c = 3;
    const int h = 32;
    const int w = 32;

    std::vector<float> input(batch*in_c*h*w, 1.0f);

    // First conv layer
    std::vector<float> conv_w(64*in_c*3*3);
    init_vector(conv_w);
    std::vector<float> conv_mean(64, 0.0f), conv_var(64, 1.0f);
    std::vector<float> conv_weight(64, 1.0f), conv_bias(64, 0.0f);

    std::vector<float> x(batch*64*h*w);
    conv2d_3x3_winograd(input.data(), conv_w.data(), x.data(), batch, in_c, 64, h, w);
    std::vector<float> tmp(batch*64*h*w);
    batchnorm_forward(tmp.data(), x.data(), conv_mean.data(), conv_var.data(), conv_weight.data(), conv_bias.data(), 1e-5f, batch, 64, h, w);
    relu_avx2(x.data(), tmp.data(), tmp.size());

    // Residual block 1
    std::vector<float> rb1_w1(64*64*3*3), rb1_w2(64*64*3*3);
    init_vector(rb1_w1); init_vector(rb1_w2);
    std::vector<float> rb1_mean1(64,0.0f), rb1_var1(64,1.0f), rb1_wt1(64,1.0f), rb1_bs1(64,0.0f);
    std::vector<float> rb1_mean2(64,0.0f), rb1_var2(64,1.0f), rb1_wt2(64,1.0f), rb1_bs2(64,0.0f);

    std::vector<float> rb1_out(batch*64*h*w);
    conv2d_3x3_winograd(x.data(), rb1_w1.data(), rb1_out.data(), batch, 64, 64, h, w);
    batchnorm_forward(tmp.data(), rb1_out.data(), rb1_mean1.data(), rb1_var1.data(), rb1_wt1.data(), rb1_bs1.data(), 1e-5f, batch, 64, h, w);
    relu_avx2(rb1_out.data(), tmp.data(), tmp.size());
    conv2d_3x3_winograd(rb1_out.data(), rb1_w2.data(), tmp.data(), batch, 64, 64, h, w);
    batchnorm_forward(rb1_out.data(), tmp.data(), rb1_mean2.data(), rb1_var2.data(), rb1_wt2.data(), rb1_bs2.data(), 1e-5f, batch, 64, h, w);
    for(size_t i=0;i<rb1_out.size();++i) rb1_out[i] += x[i];
    relu_avx2(x.data(), rb1_out.data(), rb1_out.size());

    // Residual block 2
    std::vector<float> rb2_w1(64*64*3*3), rb2_w2(64*64*3*3);
    init_vector(rb2_w1); init_vector(rb2_w2);
    std::vector<float> rb2_mean1(64,0.0f), rb2_var1(64,1.0f), rb2_wt1(64,1.0f), rb2_bs1(64,0.0f);
    std::vector<float> rb2_mean2(64,0.0f), rb2_var2(64,1.0f), rb2_wt2(64,1.0f), rb2_bs2(64,0.0f);

    conv2d_3x3_winograd(x.data(), rb2_w1.data(), rb1_out.data(), batch, 64, 64, h, w);
    batchnorm_forward(tmp.data(), rb1_out.data(), rb2_mean1.data(), rb2_var1.data(), rb2_wt1.data(), rb2_bs1.data(), 1e-5f, batch, 64, h, w);
    relu_avx2(rb1_out.data(), tmp.data(), tmp.size());
    conv2d_3x3_winograd(rb1_out.data(), rb2_w2.data(), tmp.data(), batch, 64, 64, h, w);
    batchnorm_forward(rb1_out.data(), tmp.data(), rb2_mean2.data(), rb2_var2.data(), rb2_wt2.data(), rb2_bs2.data(), 1e-5f, batch, 64, h, w);
    for(size_t i=0;i<rb1_out.size();++i) rb1_out[i] += x[i];
    relu_avx2(x.data(), rb1_out.data(), rb1_out.size());

    // Global average pool
    avg_pool2d_ref(tmp.data(), x.data(), batch, 64, h, w, h, w, h, w, 0, 0);

    // Flatten
    std::vector<float> flat(batch*64);
    flatten(flat.data(), tmp.data(), batch, 64, 1, 1);

    // Fully connected
    std::vector<float> fc_w(10*64); init_vector(fc_w);
    std::vector<float> fc_b(10, 0.0f);
    std::vector<float> out(batch*10);
    fully_connected(out.data(), flat.data(), fc_w.data(), fc_b.data(), batch, 10, 64);

    std::cout << "Output:";
    for(int i=0;i<10;i++) std::cout << " " << out[i];
    std::cout << std::endl;
    return 0;
}
