#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

using namespace std;

#include <cuda_runtime.h>

#define CUDA_OK(call) do { \
  cudaError_t e = (call); \
  if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    exit(1); \
  } \
} while(0)

static void die(const char* msg){ cerr << msg << "\n"; exit(1); }

// ---------- Minimal PPM/PGM I/O (binary P6/P5) ----------
struct ImageRGBB {
    int w{0}, h{0};
    vector<unsigned char> data; // 3*w*h
}

struct ImageGray8 {
    int w{0}, h{0};
    vector<unsigned char> data; // w*h
};

/*  ---------- CUDA kernels ---------- */

// 2D: RGB -> Gray
__global__ void rgb_to_gray_kernel(const unsigned char *rgb, unsigned char *gray, int width, int height) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height || col >= width) return;

    int idx = row * width + col; 
    int r = rgb[3*idx + 0];
    int g = rgb[3*idx + 1];
    int b = rgb[3*idx + 2];

    float y = 0.299f*r + 0.587f*g + 0.114f*b;
    int yi = (int)(y + 0.5f);
    if (yi < 0) yi = 0; if (yi > 255) yi = 255;
    gray[idx] = (unsigned char)yi;
}


// 1D grid-stride: per-block shared histogram then merge
__global__ void hist256_shared_kernel(const unsigned char* gray, int N, unsigned int* g_hist) {

    __shared__ unsigned int s_hist[256];

    // 1) zero shared histogram cooperatively
    for (int i = threadIdx.x; i < 256; i += blockDim.x) s_hist[i] = 0u;
    __syncthreads();

    // 2) grid-stride loop over pixels
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        unsigned char v = gray[i];
        atomicAdd(&s_hist[v], 1u);
    }

    __syncthreads();

    // 3) merge shared histogram to global histogram cooperatively
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        atomicAdd(&g_hist[i], s_hist[i]);
    }

}

// Single-block inclusive scan (Hillis–Steele) for 256 floats
__global__ void inclusive_scan_256(float* data){
  __shared__ float s[256];
  int t = threadIdx.x;
  s[t] = data[t];
  __syncthreads();
  for (int offset=1; offset<256; offset<<=1){
    float val = s[t];
    if (t >= offset) val += s[t - offset];
    __syncthreads();
    s[t] = val;
    __syncthreads();
  }
  data[t] = s[t];
}

// Optional device map builder (not strictly needed if done on host)
__global__ void build_map_kernel(const float* cdf, int* map){
  int t = threadIdx.x + blockIdx.x*blockDim.x;
  if (t<256){
    int m = (int)floorf(255.f * cdf[t] + 0.5f);
    if(m<0) m=0; if(m>255) m=255;
    map[t] = m;
  }
}

// 2D: apply map
__global__ void apply_map_kernel(const unsigned char* gray_in, unsigned char* gray_out,
                                 int width, int height, const int* map){
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  int col = blockIdx.x*blockDim.x + threadIdx.x;
  if (row >= height || col >= width) return;
  int idx = row*width + col;
  gray_out[idx] = (unsigned char) map[(int)gray_in[idx]];
}
























void gpu_equalize_rgb_ppm(const unsigned char* h_rgb, int width, int height, unsigned char* h_out_gray) {

    size_t NP = (size_t)width * (size_t)height;
    size_t rgb_bytes = NP * 3, gray_bytes = NP;

    unsigned char *d_rgb=nullptr, *d_gray=nullptr;
    CUDA_OK(cudaMalloc(&d_rgb, rgb_bytes));
    CUDA_OK(cudaMalloc(&d_gray, gray_bytes));
    CUDA_OK(cudaMemcpy(d_rgb, h_rgb, rgb_bytes, cudaMemcpyHostToDevice));

    im3 threads(16,16);
    dim3 blocks( (width + threads.x - 1)/threads.x,
                 (height+ threads.y - 1)/threads.y );   

    rgb_to_gray_kernel<<<blocks, threads>>>(d_rgb, d_gray, width, height);
    CUDA_OK(cudaGetLastError());

    // For now just copy back grayscale to check kernel correctness
    CUDA_OK(cudaMemcpy(h_out_gray, d_gray, gray_bytes, cudaMemcpyDeviceToHost));

    cudaFree(d_rgb); 
    cudaFree(d_gray);

}