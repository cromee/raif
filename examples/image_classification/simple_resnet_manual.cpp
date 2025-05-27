#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <functional>
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
    raif::init();
    const int batch = 1;
    const int in_c = 3;
    const int h = 32;
    const int w = 32;

    std::vector<float> input(batch*in_c*h*w, 1.0f);

    // Weights
    std::vector<float> conv_w(64*in_c*3*3); init_vector(conv_w);
    std::vector<float> conv_mean(64, 0.0f), conv_var(64, 1.0f);
    std::vector<float> conv_weight(64, 1.0f), conv_bias(64, 0.0f);

    std::vector<float> rb1_w1(64*64*3*3), rb1_w2(64*64*3*3);
    init_vector(rb1_w1); init_vector(rb1_w2);
    std::vector<float> rb1_mean1(64,0.0f), rb1_var1(64,1.0f), rb1_wt1(64,1.0f), rb1_bs1(64,0.0f);
    std::vector<float> rb1_mean2(64,0.0f), rb1_var2(64,1.0f), rb1_wt2(64,1.0f), rb1_bs2(64,0.0f);

    std::vector<float> rb2_w1(64*64*3*3), rb2_w2(64*64*3*3);
    init_vector(rb2_w1); init_vector(rb2_w2);
    std::vector<float> rb2_mean1(64,0.0f), rb2_var1(64,1.0f), rb2_wt1(64,1.0f), rb2_bs1(64,0.0f);
    std::vector<float> rb2_mean2(64,0.0f), rb2_var2(64,1.0f), rb2_wt2(64,1.0f), rb2_bs2(64,0.0f);

    std::vector<float> fc_w(10*64); init_vector(fc_w);
    std::vector<float> fc_b(10, 0.0f);

    auto forward = [&](auto conv) {
        std::vector<float> x(batch*64*h*w);
        conv(input.data(), conv_w.data(), x.data(), batch, in_c, 64, h, w);
        std::vector<float> tmp(batch*64*h*w);
        batchnorm_forward(tmp.data(), x.data(), conv_mean.data(), conv_var.data(),
                          conv_weight.data(), conv_bias.data(), 1e-5f,
                          batch, 64, h, w);
        relu_avx2(x.data(), tmp.data(), tmp.size());

        std::vector<float> rb1_out(batch*64*h*w);
        conv(x.data(), rb1_w1.data(), rb1_out.data(), batch, 64, 64, h, w);
        batchnorm_forward(tmp.data(), rb1_out.data(), rb1_mean1.data(), rb1_var1.data(),
                          rb1_wt1.data(), rb1_bs1.data(), 1e-5f,
                          batch, 64, h, w);
        relu_avx2(rb1_out.data(), tmp.data(), tmp.size());
        conv(rb1_out.data(), rb1_w2.data(), tmp.data(), batch, 64, 64, h, w);
        batchnorm_forward(rb1_out.data(), tmp.data(), rb1_mean2.data(), rb1_var2.data(),
                          rb1_wt2.data(), rb1_bs2.data(), 1e-5f,
                          batch, 64, h, w);
        for(size_t i=0;i<rb1_out.size();++i) rb1_out[i] += x[i];
        relu_avx2(x.data(), rb1_out.data(), rb1_out.size());

        conv(x.data(), rb2_w1.data(), rb1_out.data(), batch, 64, 64, h, w);
        batchnorm_forward(tmp.data(), rb1_out.data(), rb2_mean1.data(), rb2_var1.data(),
                          rb2_wt1.data(), rb2_bs1.data(), 1e-5f,
                          batch, 64, h, w);
        relu_avx2(rb1_out.data(), tmp.data(), tmp.size());
        conv(rb1_out.data(), rb2_w2.data(), tmp.data(), batch, 64, 64, h, w);
        batchnorm_forward(rb1_out.data(), tmp.data(), rb2_mean2.data(), rb2_var2.data(),
                          rb2_wt2.data(), rb2_bs2.data(), 1e-5f,
                          batch, 64, h, w);
        for(size_t i=0;i<rb1_out.size();++i) rb1_out[i] += x[i];
        relu_avx2(x.data(), rb1_out.data(), rb1_out.size());

        avg_pool2d_ref(tmp.data(), x.data(), batch, 64, h, w, h, w, h, w, 0, 0);

        std::vector<float> flat(batch*64);
        flatten(flat.data(), tmp.data(), batch, 64, 1, 1);

        std::vector<float> out(batch*10);
        fully_connected(out.data(), flat.data(), fc_w.data(), fc_b.data(), batch, 10, 64);
        return out;
    };

    auto winograd_conv = [&](const float* i,const float* f,float* o,int B,int IC,int OC,int H,int W){
        conv2d_3x3_winograd(i,f,o,B,IC,OC,H,W);
    };
    auto im2col_conv = [&](const float* i,const float* f,float* o,int B,int IC,int OC,int H,int W){
        conv2d_im2col(i,f,o,B,IC,OC,H,W,3,3,1,raif::PADDING_ZERO);
    };

    const int iters = 10;
    double winograd_ms = 0.0, im2col_ms = 0.0;
    std::vector<float> out_winograd, out_im2col;
    for(int i=0;i<iters;i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        out_winograd = forward(winograd_conv);
        auto t1 = std::chrono::high_resolution_clock::now();
        winograd_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    winograd_ms /= iters;

    for(int i=0;i<iters;i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        out_im2col = forward(im2col_conv);
        auto t1 = std::chrono::high_resolution_clock::now();
        im2col_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    im2col_ms /= iters;

    std::cout << "Winograd Output:";
    for(int i=0;i<10;i++) std::cout << " " << out_winograd[i];
    std::cout << "\nIm2col Output:";
    for(int i=0;i<10;i++) std::cout << " " << out_im2col[i];
    std::cout << "\n";
    std::cout << "Winograd time (ms): " << winograd_ms << "\n";
    std::cout << "Im2col time (ms): " << im2col_ms << std::endl;
    return 0;
}
