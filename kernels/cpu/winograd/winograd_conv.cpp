#include "kernels/cpu/winograd/winograd_conv.h"
#include <vector>
#include <cstring>

namespace raif {

namespace {

// Winograd F(2x2,3x3) transformation matrices
static const float G3[4][3] = {
    {1.f, 0.f, 0.f},
    {0.5f, 0.5f, 0.5f},
    {0.5f,-0.5f, 0.5f},
    {0.f, 0.f, 1.f}
};
static const float BT3[4][4] = {
    {1.f, 0.f,-1.f, 0.f},
    {0.f, 1.f, 1.f, 0.f},
    {0.f,-1.f, 1.f, 0.f},
    {0.f, 1.f, 0.f,-1.f}
};
static const float AT3[2][4] = {
    {1.f, 1.f, 1.f, 0.f},
    {0.f, 1.f,-1.f,-1.f}
};

inline void transform_filter_3x3(float U[4][4], const float g[3][3]) {
    float temp[4][3];
    for(int i=0;i<4;i++) {
        for(int j=0;j<3;j++) {
            temp[i][j] = G3[i][0]*g[0][j] + G3[i][1]*g[1][j] + G3[i][2]*g[2][j];
        }
    }
    for(int i=0;i<4;i++) {
        for(int j=0;j<4;j++) {
            U[i][j] = temp[i][0]*G3[j][0] + temp[i][1]*G3[j][1] + temp[i][2]*G3[j][2];
        }
    }
}

inline void transform_input_3x3(float V[4][4], const float d[4][4]) {
    float temp[4][4];
    for(int i=0;i<4;i++) {
        for(int j=0;j<4;j++) {
            temp[i][j] = BT3[i][0]*d[0][j] + BT3[i][1]*d[1][j] + BT3[i][2]*d[2][j] + BT3[i][3]*d[3][j];
        }
    }
    for(int i=0;i<4;i++) {
        for(int j=0;j<4;j++) {
            V[i][j] = temp[0][j]*BT3[i][0] + temp[1][j]*BT3[i][1] + temp[2][j]*BT3[i][2] + temp[3][j]*BT3[i][3];
        }
    }
}

inline void inverse_transform_3x3(float M[4][4], float Y[2][2]) {
    float temp[2][4];
    for(int i=0;i<2;i++) {
        for(int j=0;j<4;j++) {
            temp[i][j] = AT3[i][0]*M[0][j] + AT3[i][1]*M[1][j] + AT3[i][2]*M[2][j] + AT3[i][3]*M[3][j];
        }
    }
    for(int i=0;i<2;i++) {
        for(int j=0;j<2;j++) {
            Y[i][j] = temp[i][0]*AT3[j][0] + temp[i][1]*AT3[j][1] + temp[i][2]*AT3[j][2] + temp[i][3]*AT3[j][3];
        }
    }
}

}

void winograd_conv2d_3x3(const float* input, const float* filter, float* output,
                         int batches, int in_channels, int out_channels,
                         int height, int width) {
    const int tile_h = (height + 1) / 2;
    const int tile_w = (width + 1) / 2;
    const int tiles = tile_h * tile_w;

    std::vector<float> U(out_channels * in_channels * 16);
    for(int oc=0; oc<out_channels; ++oc) {
        for(int ic=0; ic<in_channels; ++ic) {
            const float* g = filter + (oc*in_channels + ic)*9;
            float g33[3][3];
            for(int i=0;i<3;i++)
                for(int j=0;j<3;j++)
                    g33[i][j] = g[i*3+j];
            float tmp[4][4];
            transform_filter_3x3(tmp, g33);
            float* dst = &U[(oc*in_channels + ic)*16];
            for(int i=0;i<4;i++)
                for(int j=0;j<4;j++)
                    dst[i*4+j] = tmp[i][j];
        }
    }

    std::vector<float> V(in_channels * tiles * 16);
    std::vector<float> M(out_channels * tiles * 16);

    for(int b=0; b<batches; ++b) {
        for(int ic=0; ic<in_channels; ++ic) {
            for(int th=0; th<tile_h; ++th) {
                for(int tw=0; tw<tile_w; ++tw) {
                    float d[4][4] = {0};
                    for(int i=0;i<4;i++) {
                        for(int j=0;j<4;j++) {
                            int y = th*2 + i - 1;
                            int x = tw*2 + j - 1;
                            if(y>=0 && y<height && x>=0 && x<width) {
                                int idx = ((b*in_channels + ic)*height + y)*width + x;
                                d[i][j] = input[idx];
                            } else {
                                d[i][j] = 0.f;
                            }
                        }
                    }
                    float Vpatch[4][4];
                    transform_input_3x3(Vpatch, d);
                    float* dst = &V[((ic*tiles) + th*tile_w+tw)*16];
                    for(int i=0;i<4;i++)
                        for(int j=0;j<4;j++)
                            dst[i*4+j] = Vpatch[i][j];
                }
            }
        }

        for(int oc=0; oc<out_channels; ++oc) {
            for(int th=0; th<tile_h; ++th) {
                for(int tw=0; tw<tile_w; ++tw) {
                    float Mpatch[4][4] = {0};
                    for(int ic=0; ic<in_channels; ++ic) {
                        const float* Udata = &U[(oc*in_channels + ic)*16];
                        const float* Vdata = &V[(ic*tiles + th*tile_w+tw)*16];
                        for(int i=0;i<4;i++)
                            for(int j=0;j<4;j++)
                                Mpatch[i][j] += Udata[i*4+j] * Vdata[i*4+j];
                    }
                    float Y[2][2];
                    inverse_transform_3x3(Mpatch, Y);
                    for(int i=0;i<2;i++) {
                        for(int j=0;j<2;j++) {
                            int y = th*2 + i;
                            int x = tw*2 + j;
                            if(y<height && x<width) {
                                int out_idx = ((b*out_channels + oc)*height + y)*width + x;
                                output[out_idx] += Y[i][j];
                            }
                        }
                    }
                }
            }
        }
    }
}

void winograd_conv2d_5x5(const float* input, const float* filter, float* output,
                         int batches, int in_channels, int out_channels,
                         int height, int width) {
    // Fallback simple convolution (5x5) with padding 2.
    for(int b=0; b<batches; ++b) {
        for(int oc=0; oc<out_channels; ++oc) {
            for(int y=0; y<height; ++y) {
                for(int x=0; x<width; ++x) {
                    float sum = 0.f;
                    for(int ic=0; ic<in_channels; ++ic) {
                        for(int ky=0; ky<5; ++ky) {
                            for(int kx=0; kx<5; ++kx) {
                                int iy = y + ky - 2;
                                int ix = x + kx - 2;
                                float val = 0.f;
                                if(iy>=0 && iy<height && ix>=0 && ix<width) {
                                    int in_idx = ((b*in_channels + ic)*height + iy)*width + ix;
                                    val = input[in_idx];
                                }
                                int f_idx = ((oc*in_channels + ic)*25) + ky*5 + kx;
                                sum += val * filter[f_idx];
                            }
                        }
                    }
                    int out_idx = ((b*out_channels + oc)*height + y)*width + x;
                    output[out_idx] = sum;
                }
            }
        }
    }
}

} // namespace raif

