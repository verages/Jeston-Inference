#pragma once
#include <cuda_runtime.h>
typedef unsigned char uint8_t;

struct Size {
    int width, height;
    Size(int w = 0, int h = 0) : width(w), height(h) {}
};

struct AffineMatrix {
    float i2d[6];
    float d2i[6];
    void invertAffineTransform(float imat[6], float omat[6]);
    void compute(const Size& from, const Size& to);
};

float warp_affine_bilinear(
    uint8_t* src, int src_line_size, int src_width, int src_height,
    float* dst, int dst_line_size, int dst_width, int dst_height,
    uint8_t fill_value, cudaStream_t stream
);
