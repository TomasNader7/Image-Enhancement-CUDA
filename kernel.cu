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
struct ImageRGB8 {
    int w{0}, h{0};
    vector<unsigned char> data; // 3*w*h
};

struct ImageGray8 {
    int w{0}, h{0};
    vector<unsigned char> data; // w*h
};

static ImageRGB8 readPPM(const string& path) {
  ifstream f(path, ios::binary);
  if(!f) die("Cannot open input file");
  string magic; f >> magic;
  if(magic != "P6") die("Not a P6 PPM");
  int w, h, maxv;
  // skip comments
  char c = f.peek();
  while (c == '#') { string line; getline(f, line); c = f.peek(); }
  f >> w >> h;
  f >> maxv;
  if(maxv != 255) die("Only 8-bit PPM supported");
  f.get(); // one whitespace after header

  ImageRGB8 img; img.w = w; img.h = h; img.data.resize((size_t)w*h*3);
  f.read(reinterpret_cast<char*>(img.data.data()), img.data.size());
  if(!f) die("Short read on PPM data");
  return img;
}

static void writePGM(const string& path, const ImageGray8& g) {
  ofstream f(path, ios::binary);
  if(!f) die("Cannot open output file");
  f << "P5\n" << g.w << " " << g.h << "\n255\n";
  f.write(reinterpret_cast<const char*>(g.data.data()), g.data.size());
  if(!f) die("Short write on PGM");
}


/* ---------- CPU reference pipeline ---------- */
static void cpu_rgb_to_gray(const ImageRGB8& rgb, ImageGray8& gray) {

  gray.w = rgb.w; gray.h = rgb.h; gray.data.resize((size_t)rgb.w*rgb.h);
  const unsigned char* p = rgb.data.data();
  unsigned char* q = gray.data.data();
  size_t N = (size_t)rgb.w * rgb.h;

  for (size_t i = 0; i < N; i++){
    int r = p[3*i+0], g = p[3*i+1], b = p[3*i+2];
    float y = 0.299f*r + 0.587f*g + 0.114f*b;
    int yi = (int)std::floor(y + 0.5f);
    if(yi < 0) yi = 0; if(yi > 255) yi = 255;
    q[i] = (unsigned char)yi;
  }
}

static void cpu_histogram(const ImageGray8& gray, vector<unsigned int>& hist) {
  hist.assign(256, 0);
  const unsigned char* g = gray.data.data();
  size_t N = (size_t)gray.w*gray.h;
  for (size_t i = 0; i  <N; i++) hist[(int)g[i]]++;
}

static void cpu_pdf_cdf(const vector<unsigned int>& hist, size_t N, vector<float>& pdf, vector<float>& cdf) {
  pdf.resize(256); 
  cdf.resize(256);

  for(int i = 0; i < 256; i++) pdf[i] = (float)hist[i]/(float)N;
  float acc = 0.f;
  for(int i = 0; i < 256; i++){ acc += pdf[i]; cdf[i]=acc; }
}

static void cpu_build_map(const vector<float>& cdf, vector<int>& map) {
  map.resize(256);
  for(int i = 0; i < 256; i++){
    int m = (int)std::floor(255.f * cdf[i] + 0.5f);
    if(m < 0) m = 0; if(m > 255) m = 255;
    map[i] = m;
  }
}

static void cpu_apply_map(const ImageGray8& in, const vector<int>& map, ImageGray8& out) {

  out.w = in.w; 
  out.h = in.h; 
  out.data.resize((size_t)in.w*in.h);

  size_t N = (size_t)in.w*in.h;

  for(size_t i = 0; i < N; i++) out.data[i] = (unsigned char) map[(int)in.data[i]];
}


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
  if (t < 256){
    int m = (int)floorf(255.f * cdf[t] + 0.5f);
    if(m < 0) m = 0; if(m > 255) m = 255;
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


/*---------- GPU pipeline ----------*/

static void gpu_equalize_rgb(const ImageRGB8& rgb,
                             ImageGray8& gray_gpu, ImageGray8& eq_gpu,
                             double& ms_total, double& ms_rgb2gray, double& ms_hist,
                             double& ms_scan, double& ms_apply)
{
  int W = rgb.w, H = rgb.h;
  size_t NP = (size_t)W*H;
  size_t rgb_bytes = NP*3, gray_bytes = NP, hist_bytes = 256*sizeof(unsigned int);

  unsigned char *d_rgb = nullptr, *d_gray = nullptr, *d_eq = nullptr;
  unsigned int *d_hist = nullptr;
  float *d_pdf = nullptr, //*d_cdf = nullptr;
  int *d_map = nullptr;

  cudaEvent_t e0,e1,e2,e3,e4,e5;
  CUDA_OK(cudaEventCreate(&e0)); CUDA_OK(cudaEventCreate(&e1));
  CUDA_OK(cudaEventCreate(&e2)); CUDA_OK(cudaEventCreate(&e3));
  CUDA_OK(cudaEventCreate(&e4)); CUDA_OK(cudaEventCreate(&e5));

  CUDA_OK(cudaEventRecord(e0));

  // Device alloc
  CUDA_OK(cudaMalloc(&d_rgb, rgb_bytes));
  CUDA_OK(cudaMalloc(&d_gray, gray_bytes));
  CUDA_OK(cudaMalloc(&d_eq,   gray_bytes));
  CUDA_OK(cudaMalloc(&d_hist, hist_bytes));
  CUDA_OK(cudaMalloc(&d_pdf,  256*sizeof(float)));
  // CUDA_OK(cudaMalloc(&d_cdf,  256*sizeof(float)));
  CUDA_OK(cudaMalloc(&d_map,  256*sizeof(int)));
  CUDA_OK(cudaMemset(d_hist, 0, hist_bytes));

  // H->D copy
  CUDA_OK(cudaMemcpy(d_rgb, rgb.data.data(), rgb_bytes, cudaMemcpyHostToDevice));

  dim3 threads2D(16,16);
  dim3 blocks2D( (W + threads2D.x - 1)/threads2D.x,
                 (H + threads2D.y - 1)/threads2D.y );

  // K1: RGB->Gray
  rgb_to_gray_kernel<<<blocks2D, threads2D>>>(d_rgb, d_gray, W, H);
  CUDA_OK(cudaGetLastError());
  CUDA_OK(cudaDeviceSynchronize());

  CUDA_OK(cudaEventRecord(e1));

  // K2: Histogram (1D launch)
  int t1D = 256;
  int b1D = (int)std::min<size_t>( (NP + t1D - 1)/t1D, 4096 );
  hist256_shared_kernel<<<b1D, t1D>>>(d_gray, (int)NP, d_hist);
  CUDA_OK(cudaGetLastError());
  CUDA_OK(cudaDeviceSynchronize());

  CUDA_OK(cudaEventRecord(e2));

  // Copy hist to host -> compute pdf on device and scan on device (for demo)
  vector<unsigned int> h_hist(256);
  CUDA_OK(cudaMemcpy(h_hist.data(), d_hist, hist_bytes, cudaMemcpyDeviceToHost));

  // Form pdf on host then upload (numerically nice & clear)
  vector<float> h_pdf(256), h_cdf(256);
  for(int i = 0; i < 256; i++) h_pdf[i] = (float)h_hist[i] / (float)NP;
  CUDA_OK(cudaMemcpy(d_pdf, h_pdf.data(), 256*sizeof(float), cudaMemcpyHostToDevice));

  // K3: inclusive scan (CDF) on GPU
  inclusive_scan_256<<<1,256>>>(d_pdf);
  CUDA_OK(cudaGetLastError());
  CUDA_OK(cudaDeviceSynchronize());
  CUDA_OK(cudaMemcpy(h_cdf.data(), d_pdf, 256*sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_OK(cudaEventRecord(e3));

  // Build map on device (could also do it on host)
  build_map_kernel<<<1,256>>>(d_pdf, d_map);
  CUDA_OK(cudaGetLastError());
  CUDA_OK(cudaDeviceSynchronize());

  // K5: apply map
  apply_map_kernel<<<blocks2D, threads2D>>>(d_gray, d_eq, W, H, d_map);
  CUDA_OK(cudaGetLastError());
  CUDA_OK(cudaDeviceSynchronize());
  CUDA_OK(cudaEventRecord(e4));

  // D->H gray (optional: to save the pre-equalization grayscale)
  gray_gpu.w = W; gray_gpu.h = H; gray_gpu.data.resize(NP);
  CUDA_OK(cudaMemcpy(gray_gpu.data.data(), d_gray, gray_bytes, cudaMemcpyDeviceToHost));

  // D->H equalized
  eq_gpu.w = W; eq_gpu.h = H; eq_gpu.data.resize(NP);
  CUDA_OK(cudaMemcpy(eq_gpu.data.data(), d_eq, gray_bytes, cudaMemcpyDeviceToHost));

  CUDA_OK(cudaEventRecord(e5));
  CUDA_OK(cudaEventSynchronize(e5));

  float t01=0, t12=0, t23=0, t34=0, t05=0;
  CUDA_OK(cudaEventElapsedTime(&t01, e0, e1)); // rgb->gray
  CUDA_OK(cudaEventElapsedTime(&t12, e1, e2)); // histogram
  CUDA_OK(cudaEventElapsedTime(&t23, e2, e3)); // scan
  CUDA_OK(cudaEventElapsedTime(&t34, e3, e4)); // build map + apply
  CUDA_OK(cudaEventElapsedTime(&t05, e0, e5)); // total

  ms_rgb2gray = t01; ms_hist = t12; ms_scan = t23; ms_apply = t34; ms_total = t05;

  cudaEventDestroy(e0); cudaEventDestroy(e1); cudaEventDestroy(e2);
  cudaEventDestroy(e3); cudaEventDestroy(e4); cudaEventDestroy(e5);

  cudaFree(d_rgb); cudaFree(d_gray); cudaFree(d_eq);
  cudaFree(d_hist); //cudaFree(d_cdf); 
  cudaFree(d_map);
}


/*  ---------- 2-norm difference ---------- */

static double diff_2norm(const ImageGray8& a, const ImageGray8& b){
  if(a.w != b.w || a.h != b.h) die("Dimension mismatch");
  size_t N = (size_t)a.w*a.h;

  double s=0.0;

  for(size_t i = 0; i < N; i++) 
  {
    int d = (int)a.data[i] - (int)b.data[i];
    s += (double)d*(double)d;
  }
  return std::sqrt(s);
}

// ---------- main ----------
int main(int argc, char** argv) {

  if(argc < 4){
    cerr << "Usage: " << argv[0] << " input.ppm out_cpu.pgm out_gpu.pgm [--cpu-only|--gpu-only]\n";
    return 1;
  }
  string inPath = argv[1];
  string cpuOut = argv[2];
  string gpuOut = argv[3];

  bool cpuOnly = false, gpuOnly = false;

  for(int i = 4; i < argc; i++){
    if(string(argv[i]) == "--cpu-only") cpuOnly = true;
    if(string(argv[i]) == "--gpu-only") gpuOnly = true;
  }

  ImageRGB8 rgb = readPPM(inPath);
  cout << "Loaded " << rgb.w << "x" << rgb.h << "\n";

  ImageGray8 gray_cpu, eq_cpu, gray_gpu, eq_gpu;

  // --- CPU path
  auto t0 = std::chrono::high_resolution_clock::now();
  if(!gpuOnly){
    cpu_rgb_to_gray(rgb, gray_cpu);

    vector<unsigned int> hist;
    cpu_histogram(gray_cpu, hist);

    vector<float> pdf, cdf;
    cpu_pdf_cdf(hist, (size_t)rgb.w*rgb.h, pdf, cdf);

    vector<int> map;
    cpu_build_map(cdf, map);

    cpu_apply_map(gray_cpu, map, eq_cpu);
    writePGM(cpuOut, eq_cpu);
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  double ms_cpu = std::chrono::duration<double,std::milli>(t1-t0).count();

  // --- GPU path
  double ms_total = 0, ms_rgb2gray = 0, ms_hist = 0, ms_scan = 0, ms_apply = 0;
  if(!cpuOnly){
    gpu_equalize_rgb(rgb, gray_gpu, eq_gpu, ms_total, ms_rgb2gray, ms_hist, ms_scan, ms_apply);
    writePGM(gpuOut, eq_gpu);
  }

  // --- Compare and report
  if(!gpuOnly && !cpuOnly){
    double n2 = diff_2norm(eq_cpu, eq_gpu);
    cout << "CPU time (ms)   : " << ms_cpu << "\n";
    cout << "GPU total (ms)  : " << ms_total << "\n";
    cout << "  rgb2gray (ms) : " << ms_rgb2gray << "\n";
    cout << "  histogram (ms): " << ms_hist << "\n";
    cout << "  scan (ms)     : " << ms_scan << "\n";
    cout << "  apply (ms)    : " << ms_apply << "\n";
    cout << "2-norm diff     : " << n2 << "\n";
  } else if (cpuOnly){
    cout << "CPU time (ms)   : " << ms_cpu << "\n";
    cout << "GPU not run (cpu-only).\n";
  } else {
    cout << "GPU total (ms)  : " << ms_total << "\n";
    cout << "  rgb2gray (ms) : " << ms_rgb2gray << "\n";
    cout << "  histogram (ms): " << ms_hist << "\n";
    cout << "  scan (ms)     : " << ms_scan << "\n";
    cout << "  apply (ms)    : " << ms_apply << "\n";
    cout << "CPU not run (gpu-only).\n";
  }

  return 0;
}