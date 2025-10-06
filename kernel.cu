#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <errno.h>
#include <time.h>


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
struct  {
    int w, h;
    unsigned char* data; // 3*w*h
} ImageRGB8;

typedef struct {
  int w, h;
  unsigned char* data; /* w*h */
} ImageGray8;

static void skip_ws_and_comments(FILE* f){
  int c;
  do {
    c = fgetc(f);
    if (c == '#'){
      while (c != '\n' && c != EOF) c = fgetc(f);
    }
  } while (c != EOF && (c==' ' || c=='\n' || c=='\r' || c=='\t'));
  if (c != EOF) ungetc(c, f);
}

static ImageRGB8 readPPM(const char& path) {
  ImageRGB8 img; img.w = img.h = 0; img.data = NULL;
  FILE* f = fopen(path, "rb");
  if(!f) die("Cannot open input file");

  char magic[3] = {0};
  if (fscanf(f, "%2s", magic) != 1 || strcmp(magic,"P6")!=0) die("Not a P6 PPM");

  skip_ws_and_comments(f);
  if (fscanf(f, "%d %d", &img.w, &img.h) != 2 || img.w<=0 || img.h<=0) die("Bad PPM size");

  skip_ws_and_comments(f);
  int maxv = 0;
  if (fscanf(f, "%d", &maxv) != 1 || maxv != 255) die("Bad maxval (must be 255)");

  /* consume single whitespace */
  int c = fgetc(f);
  if (c == EOF) die("Truncated PPM header");

  size_t N = (size_t)img.w * (size_t)img.h;
  img.data = (unsigned char*)malloc(3*N);
  if(!img.data) die("OOM");
  size_t got = fread(img.data, 1, 3*N, f);
  if (got != 3*N) die("Short read on PPM");
  fclose(f);
  return img;

}

static void writePGM(const char* path, const ImageGray8* g) {

  FILE* f = fopen(path, "wb");

  if(!f) die("Cannot open PGM for write");
  fprintf(f, "P5\n%d %d\n255\n", g->w, g->h);

  size_t N = (size_t)g->w * (size_t)g->h;
  if (fwrite(g->data, 1, N, f) != N) die("Short write on PGM");
  
  fclose(f);
}


/* ---------- CPU reference pipeline ---------- */
static void cpu_rgb_to_gray(const ImageRGB8* rgb, ImageGray8* gray) {

  gray->w = rgb->w; gray->h = rgb->h;
  size_t N = (size_t)rgb->w * (size_t)rgb->h;
  gray->data = (unsigned char*)malloc(N);

  if(!gray->data) die("OOM");

  const unsigned char* p = rgb->data;
  unsigned char* q = gray->data;

  for (size_t i = 0; i < N; i++) 
  {
    int r = p[3*i+0], g = p[3*i+1], b = p[3*i+2];
    float y = 0.299f*r + 0.587f*g + 0.114f*b;
    int yi = (int)floorf(y + 0.5f);
    if(yi < 0) yi = 0; if(yi > 255) yi = 255;
    q[i] = (unsigned char)yi;
  }
}

static void cpu_histogram(const ImageGray8* gray, unsigned int hist[256]) {
  for (int i = 0; i < 256; i++) hist[i] = 0;
  const unsigned char* g = gray->data;
  size_t N = (size_t)gray->w * (size_t)gray->h;
  for (size_t i = 0; i < N; i++) hist[(int)g[i]]++;
}

static void cpu_pdf_cdf(const unsigned int hist[256], int num_pixels, float cdf[256]){
  float accum = 0.f;

  for(int i=0;i<256;i++)
  {
    float pdf = (float)hist[i] / (float)num_pixels;
    accum += pdf;
    cdf[i] = accum;
  }
}

static void cpu_build_map_clipped(const float cdf[256], int map[256]) {
  float lower_cut = 0.01f, upper_cut = 0.99f;
  int floor_gray = 0;
  
  for (int i = 0; i < 256; i++){ if (cdf[i] >= lower_cut){ floor_gray = i; break; } }
  int ceil_gray = 255;
  for (int i = 255; i >= 0; i--){ if (cdf[i] <= upper_cut){ ceil_gray = i; break; } }
  float cdf_floor = cdf[floor_gray];
  float cdf_ceil  = cdf[ceil_gray];
  float denom = (cdf_ceil - cdf_floor);
  if (denom <= 0.f)
  {
    for (int i = 0; i < 256; i++)
    {
      int v = (int)floorf(255.f*cdf[i] + 0.5f);
      if (v<0) v=0; if (v>255) v=255; map[i]=v;
    }
    return;
  }
  for (int i = 0; i < 256; i++)
  {
    if (i < floor_gray) map[i] = 0;
    else if (i > ceil_gray) map[i] = 255;
    else {
      float norm = (cdf[i] - cdf_floor) / denom;
      int v = (int)floorf(255.f*norm + 0.5f);
      if (v<0) v=0; if (v>255) v=255;
      map[i] = v;
    }
  }
}

static void cpu_apply_map(const ImageGray8* in, const int map[256], ImageGray8* out) {
  out->w = in->w; out->h = in->h;
  size_t N = (size_t)in->w * (size_t)in->h;
  out->data = (unsigned char*)malloc(N);
  if(!out->data) die("OOM");
  for (size_t i=0;i<N;i++) out->data[i] = (unsigned char) map[(int)in->data[i]];
}


/*  ---------- CUDA kernels ---------- */
// 2D: RGB -> Gray
__global__ void rgb_to_gray_kernel(const unsigned char* rgb, unsigned char* gray, int N){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N){
    int r = rgb[3*idx+0], g = rgb[3*idx+1], b = rgb[3*idx+2];
    float y = 0.299f*r + 0.587f*g + 0.114f*b;
    int yi = (int)floorf(y + 0.5f);
    if(yi < 0) yi = 0; if(yi > 255) yi = 255;
    gray[idx] = (unsigned char)yi;
  }
}

__global__ void histogram_kernel_atomic(const unsigned char* gray, unsigned int* hist, int N){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N){
    atomicAdd(&hist[(int)gray[idx]], 1);
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
__global__ void apply_map_kernel(const unsigned char* in, unsigned char* out, int width, int height, const int* map){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height){
    int idx = y*width + x;
    out[idx] = (unsigned char) map[(int)in[idx]];
  }
}

/* ---------- CUDA pipeline ---------- */
static void run_cuda_pipeline(const ImageRGB8* in_rgb,
                              ImageGray8* out_eq,
                              float* ms_rgb2gray,
                              float* ms_hist,
                              float* ms_scan,
                              float* ms_apply,
                              float* ms_total)
{
  int width = in_rgb->w, height = in_rgb->h;
  int N = width * height;

  unsigned char *d_rgb = NULL, *d_gray = NULL, *d_eq = NULL;
  unsigned int *d_hist = NULL;
  int *d_map = NULL;

  float *d_cdf = NULL; /* optional if doing CDF on device */

  CUDA_OK(cudaMalloc((void**)&d_rgb,  3*(size_t)N));
  CUDA_OK(cudaMalloc((void**)&d_gray,     (size_t)N));
  CUDA_OK(cudaMalloc((void**)&d_eq,       (size_t)N));
  CUDA_OK(cudaMalloc((void**)&d_hist, 256*sizeof(unsigned int)));
  CUDA_OK(cudaMalloc((void**)&d_map,  256*sizeof(int)));
  CUDA_OK(cudaMalloc((void**)&d_cdf,  256*sizeof(float)));

  CUDA_OK(cudaMemset(d_hist, 0, 256*sizeof(unsigned int)));

  CUDA_OK(cudaMemcpy(d_rgb, in_rgb->data, 3*(size_t)N, cudaMemcpyHostToDevice));

  dim3 block1d(256);
  dim3 grid1d((N + block1d.x - 1) / block1d.x);

  dim3 block2d(16,16);
  dim3 grid2d((width  + block2d.x - 1)/block2d.x,
              (height + block2d.y - 1)/block2d.y);

  cudaEvent_t e0,e1,e2,e3,e4,e5;
  cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventCreate(&e2);
  cudaEventCreate(&e3); cudaEventCreate(&e4); cudaEventCreate(&e5);

  cudaEventRecord(e0);
  rgb_to_gray_kernel<<<grid1d, block1d>>>(d_rgb, d_gray, N);
  CUDA_OK(cudaGetLastError());
  cudaEventRecord(e1);

  histogram_kernel_atomic<<<grid1d, block1d>>>(d_gray, d_hist, N);
  CUDA_OK(cudaGetLastError());
  cudaEventRecord(e2);

  /* copy hist to host, compute PDF/CDF on host to preserve original logic */
  unsigned int h_hist[256];
  float h_cdf[256];
  CUDA_OK(cudaMemcpy(h_hist, d_hist, 256*sizeof(unsigned int), cudaMemcpyDeviceToHost));
  cpu_pdf_cdf(h_hist, N, h_cdf);
  int h_map[256];
  cpu_build_map_clipped(h_cdf, h_map);
  CUDA_OK(cudaMemcpy(d_map, h_map, 256*sizeof(int), cudaMemcpyHostToDevice));
  cudaEventRecord(e3);

  apply_map_kernel<<<grid2d, block2d>>>(d_gray, d_eq, width, height, d_map);
  CUDA_OK(cudaGetLastError());
  cudaEventRecord(e4);

  /* copy back equalized image */
  out_eq->w = width; out_eq->h = height;
  out_eq->data = (unsigned char*)malloc((size_t)N);
  if(!out_eq->data) die("OOM");
  CUDA_OK(cudaMemcpy(out_eq->data, d_eq, (size_t)N, cudaMemcpyDeviceToHost));
  cudaEventRecord(e5);
  cudaEventSynchronize(e5);

  float t01=0.f,t12=0.f,t23=0.f,t34=0.f,t05=0.f;
  cudaEventElapsedTime(&t01, e0, e1);
  cudaEventElapsedTime(&t12, e1, e2);
  cudaEventElapsedTime(&t23, e2, e3);
  cudaEventElapsedTime(&t34, e3, e4);
  cudaEventElapsedTime(&t05, e0, e5);

  *ms_rgb2gray = t01; *ms_hist = t12; *ms_scan = t23; *ms_apply = t34; *ms_total = t05;

  cudaEventDestroy(e0); cudaEventDestroy(e1); cudaEventDestroy(e2);
  cudaEventDestroy(e3); cudaEventDestroy(e4); cudaEventDestroy(e5);

  cudaFree(d_rgb); cudaFree(d_gray); cudaFree(d_eq);
  cudaFree(d_hist); cudaFree(d_map); cudaFree(d_cdf);
}


/* ---------- 2-norm difference ---------- */
static double two_norm_diff(const unsigned char* a, const unsigned char* b, size_t N){
  double s = 0.0;
  for (size_t i=0;i<N;i++){
    double d = (double)a[i] - (double)b[i];
    s += d*d;
  }
  return sqrt(s);
}

/* ---------- Simple CPU timer ---------- */
static double elapsed_ms(struct timespec a, struct timespec b){
  double ms = (b.tv_sec - a.tv_sec) * 1000.0;
  ms += (b.tv_nsec - a.tv_nsec) / 1.0e6;
  return ms;
}

// ---------- main ----------
int main(int argc, char** argv){
  const char* in_path  = (argc>=2)? argv[1] : "chest_x_rays.ppm";
  const char* out_gpu  = (argc>=3)? argv[2] : "output_gpu.pgm";
  const char* out_cpu  = (argc>=4)? argv[3] : "output_cpu.pgm";

  ImageRGB8 rgb = readPPM(in_path);

  /* CPU reference */
  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);

  ImageGray8 gray_cpu={0,0,NULL}, eq_cpu={0,0,NULL};
  cpu_rgb_to_gray(&rgb, &gray_cpu);
  unsigned int hist[256]; cpu_histogram(&gray_cpu, hist);
  float cdf[256]; cpu_pdf_cdf(hist, rgb.w*rgb.h, cdf);
  int map[256]; cpu_build_map_clipped(cdf, map);
  cpu_apply_map(&gray_cpu, map, &eq_cpu);

  clock_gettime(CLOCK_MONOTONIC, &t1);
  double cpu_ms = elapsed_ms(t0,t1);

  /* GPU pipeline */
  ImageGray8 eq_gpu={0,0,NULL};
  float ms_rgb2gray=0.f, ms_hist=0.f, ms_scan=0.f, ms_apply=0.f, ms_total=0.f;
  run_cuda_pipeline(&rgb, &eq_gpu, &ms_rgb2gray, &ms_hist, &ms_scan, &ms_apply, &ms_total);

  /* Write outputs */
  writePGM(out_gpu, &eq_gpu);
  writePGM(out_cpu, &eq_cpu);

  size_t N = (size_t)rgb.w * (size_t)rgb.h;
  double diff2 = two_norm_diff(eq_gpu.data, eq_cpu.data, N);

  printf("CPU time (ms): %.3f\n", cpu_ms);
  printf("GPU times (ms): rgb2gray=%.3f, hist=%.3f, cdf+map(host)=%.3f, apply=%.3f, total=%.3f\n",
         ms_rgb2gray, ms_hist, ms_scan, ms_apply, ms_total);
  printf("Speedup (CPU total / GPU total): %.3f x\n", (ms_total>0.f)? (cpu_ms/ms_total) : 0.0);
  printf("2-norm difference between CPU and GPU outputs: %.6f\n", diff2);

  /* cleanup */
  free(rgb.data);
  free(gray_cpu.data);
  free(eq_cpu.data);
  free(eq_gpu.data);

  return 0;
}
