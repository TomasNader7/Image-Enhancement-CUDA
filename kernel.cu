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

static void die(const char* msg){ fprintf(stderr, "%s\n", msg); exit(1); }

// ---------- High-resolution timer ----------
static double get_time_ms() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
}

// ---------- Minimal PPM/PGM I/O (binary P6/P5) ----------
typedef struct  {
    int w, h;
    unsigned char* data; /* 3*w*h */
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

static ImageRGB8 readPPM(const char* path) {
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
  int ch = fgetc(f);
  if (ch == EOF) die("Truncated PPM header");

  size_t N = (size_t)img.w * (size_t)img.h;
  img.data = (unsigned char*)malloc(3*N);
  if(!img.data){ fclose(f); die("OOM"); }

  size_t got = fread(img.data, 1, 3*N, f);
  fclose(f);
  if (got != 3*N) die("Short read on PPM data");

  return img;
}

static void writePPM_from_gray(const char* path, const ImageGray8* gray){
  FILE* f = fopen(path, "wb");
  if(!f) die("Cannot open output PPM");
  fprintf(f, "P6\n%d %d\n255\n", gray->w, gray->h);
  for (int i = 0; i < gray->w*gray->h; i++){
    unsigned char v = gray->data[i];
    fputc(v,f); fputc(v,f); fputc(v,f);
  }
  fclose(f);
}

static void cpu_rgb_to_gray(const ImageRGB8* rgb, ImageGray8* gray) {
  gray->w = rgb->w; gray->h = rgb->h;
  size_t N = (size_t)rgb->w * (size_t)rgb->h;
  gray->data = (unsigned char*)malloc(N);
  if (!gray->data) die("OOM gray");

  const unsigned char* p = rgb->data;
  for (size_t i = 0; i < N; i++) {
    int r = p[3*i+0];
    int g = p[3*i+1];
    int b = p[3*i+2];
    float y = 0.299f*r + 0.587f*g + 0.114f*b;
    int yi = (int)floorf(y + 0.5f);
    if (yi < 0) yi = 0; if (yi > 255) yi = 255;
    gray->data[i] = (unsigned char)yi;
  }
}

static void cpu_histogram(const ImageGray8* gray, int hist[256]){
  for (int i = 0; i < 256; i++) hist[i] = 0;
  size_t N = (size_t)gray->w*gray->h;
  for (size_t i = 0; i < N; i++) hist[ gray->data[i] ]++;
}

static void cpu_pdf_cdf(const int hist[256], int num_pixels, float cdf[256]){
  float accum = 0.0f;
  for (int i = 0; i < 256; i++){
    float p = (float)hist[i] / (float)num_pixels;
    accum += p;
    cdf[i] = accum;
  }
}

static void cpu_build_map_from_cdf(const float cdf[256], int map[256]){
  float lower_cut = 0.01f;
  float upper_cut = 0.99f;

  int floor_gray = 0;
  for (int i = 0; i < 256; i++){ if (cdf[i] >= lower_cut){ floor_gray = i; break; } }
  int ceil_gray = 255;
  for (int i = 255; i >= 0; i--){ if (cdf[i] <= upper_cut){ ceil_gray = i; break; } }

  float cdf_floor = cdf[floor_gray];
  float cdf_ceil  = cdf[ceil_gray];
  float denom = (cdf_ceil - cdf_floor);
  if (denom <= 0.0f)
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

static void cpu_apply_map(const ImageGray8* in, ImageGray8* out, const int map[256]){
  out->w = in->w; out->h = in->h;
  size_t N = (size_t)in->w * in->h;
  out->data = (unsigned char*)malloc(N);
  if(!out->data) die("OOM out");
  for (size_t i=0;i<N;i++) out->data[i] = (unsigned char)map[ in->data[i] ];
}

__global__ void rgb_to_gray_kernel(const unsigned char* rgb, unsigned char* gray, int N){
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= N) return;
  int r = rgb[3*idx+0], g = rgb[3*idx+1], b = rgb[3*idx+2];
  float y = 0.299f*r + 0.587f*g + 0.114f*b;
  int yi = (int)floorf(y + 0.5f);
  if (yi < 0) yi = 0; if (yi > 255) yi = 255;
  gray[idx] = (unsigned char)yi;
}

__global__ void hist_kernel_atomic(const unsigned char* gray, int* hist, int N){
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= N) return;
  unsigned char g = gray[idx];
  atomicAdd(&hist[(int)g], 1);
}

__global__ void apply_map_kernel(const unsigned char* in, unsigned char* out, const int* map, int N){
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= N) return;
  unsigned char t = in[idx];
  int m = map[(int)t];
  out[idx] = (unsigned char)m;
}

static void run_cuda_pipeline(const ImageRGB8* in_rgb,
                              ImageGray8* out_eq,
                              float *t_rgb2gray_ms,
                              float *t_hist_ms,
                              float *t_scan_ms,  /* host-side scan+map build time */
                              float *t_apply_ms,
                              float *t_total_ms)
{
  const int W = in_rgb->w, H = in_rgb->h;
  const int N = W*H;

  cudaEvent_t evStart, evA, evB, evC, evEnd;  
  CUDA_OK(cudaEventCreate(&evStart));
  CUDA_OK(cudaEventCreate(&evA));
  CUDA_OK(cudaEventCreate(&evB));
  CUDA_OK(cudaEventCreate(&evC));             
  CUDA_OK(cudaEventCreate(&evEnd));

  unsigned char *d_rgb = NULL, *d_gray = NULL, *d_out = NULL;
  int *d_hist = NULL, *d_map = NULL;

  const size_t bytes_rgb  = (size_t)3*N;
  const size_t bytes_gray = (size_t)N;

  CUDA_OK(cudaMalloc((void**)&d_rgb, bytes_rgb));
  CUDA_OK(cudaMalloc((void**)&d_gray, bytes_gray));
  CUDA_OK(cudaMalloc((void**)&d_out,  bytes_gray));
  CUDA_OK(cudaMalloc((void**)&d_hist, 256*sizeof(int)));
  CUDA_OK(cudaMalloc((void**)&d_map,  256*sizeof(int)));

  CUDA_OK(cudaMemcpy(d_rgb, in_rgb->data, bytes_rgb, cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemset(d_hist, 0, 256*sizeof(int)));

  dim3 block1d(256);
  dim3 grid1d((N + block1d.x - 1) / block1d.x);

  // RGB -> Gray
  CUDA_OK(cudaEventRecord(evStart));
  rgb_to_gray_kernel<<<grid1d, block1d>>>(d_rgb, d_gray, N);
  CUDA_OK(cudaEventRecord(evA));
  CUDA_OK(cudaEventSynchronize(evA));
  CUDA_OK(cudaGetLastError());

  // Histogram
  hist_kernel_atomic<<<grid1d, block1d>>>(d_gray, d_hist, N);
  CUDA_OK(cudaEventRecord(evB));
  CUDA_OK(cudaEventSynchronize(evB));
  CUDA_OK(cudaGetLastError());

  // Copy hist to host
  int hist[256];
  CUDA_OK(cudaMemcpy(hist, d_hist, sizeof(hist), cudaMemcpyDeviceToHost));

  // ---- host-side timing for CDF + Map ----
  double hs0 = get_time_ms();
  float cdf[256];  cpu_pdf_cdf(hist, N, cdf);
  int   map[256];  cpu_build_map_from_cdf(cdf, map);
  double hs1 = get_time_ms();
  if (t_scan_ms) *t_scan_ms = (float)(hs1 - hs0);
  // ----------------------------------------

  // Copy map H->D
  CUDA_OK(cudaMemcpy(d_map, map, sizeof(map), cudaMemcpyHostToDevice));

  // Apply mapping (kernel-only timing)
  CUDA_OK(cudaEventRecord(evC));  // start apply timing AFTER host work + H2D
  apply_map_kernel<<<grid1d, block1d>>>(d_gray, d_out, d_map, N);
  CUDA_OK(cudaEventRecord(evEnd));
  CUDA_OK(cudaEventSynchronize(evEnd));
  CUDA_OK(cudaGetLastError());

  // Timings
  float ms_rgb2gray=0.f, ms_hist=0.f, ms_apply=0.f, ms_total=0.f;
  CUDA_OK(cudaEventElapsedTime(&ms_rgb2gray, evStart, evA));   // rgb->gray kernel
  CUDA_OK(cudaEventElapsedTime(&ms_hist,     evA,     evB));   // hist kernel
  CUDA_OK(cudaEventElapsedTime(&ms_apply,    evC,     evEnd)); // apply kernel only
  CUDA_OK(cudaEventElapsedTime(&ms_total,    evStart, evEnd)); // total kernels

  if (t_rgb2gray_ms) *t_rgb2gray_ms = ms_rgb2gray;
  if (t_hist_ms)     *t_hist_ms     = ms_hist;
  if (t_apply_ms)    *t_apply_ms    = ms_apply;
  if (t_total_ms)    *t_total_ms    = ms_total;

  // Copy result back
  out_eq->w = W; out_eq->h = H;
  out_eq->data = (unsigned char*)malloc((size_t)N);
  if (!out_eq->data) die("OOM out_eq");
  CUDA_OK(cudaMemcpy(out_eq->data, d_out, (size_t)N, cudaMemcpyDeviceToHost));

  // Cleanup
  CUDA_OK(cudaEventDestroy(evStart));
  CUDA_OK(cudaEventDestroy(evA));
  CUDA_OK(cudaEventDestroy(evB));
  CUDA_OK(cudaEventDestroy(evC));
  CUDA_OK(cudaEventDestroy(evEnd));
  cudaFree(d_rgb); cudaFree(d_gray); cudaFree(d_out);
  cudaFree(d_hist); cudaFree(d_map);
}

// --------------- Utility: 2-norm diff ----------------

static double two_norm_diff(const ImageGray8* a, const ImageGray8* b){
  if (a->w != b->w || a->h != b->h) return -1.0;
  size_t N = (size_t)a->w * a->h;
  double acc = 0.0;
  for (size_t i=0;i<N;i++){
    double d = (double)a->data[i] - (double)b->data[i];
    acc += d*d;
  }
  return sqrt(acc);
}

// --------------- Main: serial + parallel in one program ---------------

int main(int argc, char** argv)
{
  const char* in_path  = (argc >= 2) ? argv[1] : "chest_x_rays.ppm";
  const char* out_cpu  = (argc >= 3) ? argv[2] : "output_serial.ppm";
  const char* out_gpu  = (argc >= 4) ? argv[3] : "output_cuda.ppm";
  int num_iterations   = (argc >= 5) ? atoi(argv[4]) : 200;

  ImageRGB8 rgb = readPPM(in_path);

  printf("Image size: %d x %d (%d pixels)\n", rgb.w, rgb.h, rgb.w*rgb.h);
  printf("Running %d iterations for timing accuracy...\n\n", num_iterations);

// --- SERIAL (multiple iterations for timing) ---
double t0 = get_time_ms();
ImageGray8 eq_cpu = {0,0,NULL};

for (int iter = 0; iter < num_iterations; iter++) {
    ImageGray8 gray_cpu = {0,0,NULL};
    cpu_rgb_to_gray(&rgb, &gray_cpu);

    int   hist[256];  cpu_histogram(&gray_cpu, hist);
    float cdf[256];   cpu_pdf_cdf(hist, rgb.w*rgb.h, cdf);
    int   map[256];   cpu_build_map_from_cdf(cdf, map);

    // prevent leak: cpu_apply_map allocates out->data each call
    if (eq_cpu.data) { free(eq_cpu.data); eq_cpu.data = NULL; }

    cpu_apply_map(&gray_cpu, &eq_cpu, map);

    free(gray_cpu.data);
}

double t1 = get_time_ms();
double cpu_ms = (t1 - t0) / (double)num_iterations;
writePPM_from_gray(out_cpu, &eq_cpu);

printf("--- SERIAL RESULTS ---\n");
printf("Total CPU time (%d iterations): %.3f ms\n", num_iterations, (t1 - t0));
printf("Average per iteration: %.3f ms\n\n", cpu_ms);
  // PARALLEL (CUDA) - single run
  float ms_rgb2gray=0.f, ms_hist=0.f, ms_scan=0.f, ms_apply=0.f, ms_total=0.f;
  ImageGray8 eq_gpu = {0,0,NULL};
  run_cuda_pipeline(&rgb, &eq_gpu, &ms_rgb2gray, &ms_hist, &ms_scan, &ms_apply, &ms_total);
  writePPM_from_gray(out_gpu, &eq_gpu);

  printf("--- PARALLEL (CUDA) RESULTS ---\n");
  printf("GPU times (ms):\n");
  printf("  RGB-to-Gray:           %.3f ms\n", ms_rgb2gray);
  printf("  Histogram (atomic):    %.3f ms\n", ms_hist);
  printf("  CDF+Map (host-side):   %.3f ms\n", ms_scan);
  printf("  Apply mapping:         %.3f ms\n", ms_apply);
  printf("  Total GPU time:        %.3f ms\n\n", ms_total);

  // Compare
  double diff2 = two_norm_diff(&eq_cpu, &eq_gpu);

  printf("--- PERFORMANCE COMPARISON ---\n");
  printf("CPU average time (per iteration): %.3f ms\n", cpu_ms);
  printf("GPU total time:                  %.3f ms\n", ms_total);
  printf("Speedup (CPU / GPU):             %.2f x\n", (ms_total>0.f) ? (cpu_ms/ms_total) : 0.0);
  printf("2-norm difference:               %.6f\n\n", diff2);

  if (diff2 < 0.001) {
    printf("✓ Results match: Serial and CUDA implementations are equivalent.\n");
  } else if (diff2 < 1.0) {
    printf("⚠ Minor differences detected (within acceptable tolerance)\n");
  } else {
    printf("✗ Significant differences detected. Check implementation.\n");
  }

  /* cleanup */
  free(rgb.data);
  free(eq_cpu.data);
  free(eq_gpu.data);

  return 0;
}