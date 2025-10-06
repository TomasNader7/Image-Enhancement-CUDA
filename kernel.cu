#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstdio>


__global__ void rgb_to_gray_kernel(const unsigned char *rgb, unsigned char *gray, int width, int height) {

    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < height && col < width) {


    }

    int idx = row * width + col; 
    int r = rgb[3*idx + 0];
    int g = rgb[3*idx + 1];
    int b = rgb[3*idx + 2];

    float y = 0.299f*r + 0.587f*g + 0.114f*b;
    int yi = (int)(y + 0.5f);
    if (yi < 0) yi = 0; if (yi > 255) yi = 255;
    gray[idx] = (unsigned char)yi;
}

void gpu_equalize_rgb_ppm(const unsigned char* h_rgb, int width, int height, unsigned char* h_out_gray) {

    size_t NP = (size_t)width * (size_t)height;
    size_t rgb_bytes = NP * 3, gray_bytes = NP;

    unsigned char *d_rgb=nullptr, *d_gray=nullptr;
    CUDA_OK(cudaMalloc(&d_rgb, rgb_bytes));
    CUDA_OK(cudaMalloc(&d_gray, gray_bytes));
    CUDA_OK(cudaMemcpy(d_rgb, h_rgb, rgb_bytes, cudaMemcpyHostToDevice));

}