/*
 * File: CSC630_Assign#4_Nader.cu
 * Course: CSC 630/730 - Assignment 4
 * Purpose: CUDA programming for digital image enhancement using histogram equalization.
 *          Implements BOTH:
 *              - (1) Serial C implementation (baseline Ts) for grayscale conversion,
 *                    histogram computation, PDF/CDF calculation, and intensity mapping.
 *              - (2) CUDA parallel implementation (Tp) with 1-D thread blocks (256 threads/block).
 *
 * Enhancement Method: Histogram Equalization with contrast limiting (1% to 99% thresholds).
 * Compile:
 *     - module load cuda-toolkit
 *     - nvcc -O2 -arch=sm_30 -o kernel kernel.cu
 *     - srun -p gpu --gres gpu:1 -n 1 -N 1 --pty --mem 1000 -t 2:00 bash
 *     - ./kernel chest_x_rays.ppm output_serial.ppm output_cuda.ppm
 *     - exit
 *
 *
 * Run examples (Magnolia shell):
 *     ./kernel <input.ppm> <output_serial.ppm> <output_cuda.ppm> [iterations]
 *     ./kernel chest_x_rays.ppm output_serial.ppm output_cuda.ppm 200
 *     ./kernel large_xray.ppm output_serial.ppm output_cuda.ppm 10
 *
 * Input:
 *     <input.ppm>        PPM image file (3-byte RGB color image, P6 format).
 *     <output_serial.ppm> Output file for serial implementation result.
 *     <output_cuda.ppm>  Output file for CUDA implementation result.
 *     [iterations]       Number of CPU iterations for timing accuracy (default 10).
 *
 * Outputs (printed to stdout):
 *     - Image dimensions and pixel count
 *     - Serial execution time (total and per-iteration average)
 *     - CUDA execution times (per kernel: rgb_to_gray, histogram, cdf+map, apply_map, total)
 *     - Speedup metric: Ts_avg / Tp_total
 *     - Two-norm difference between serial and parallel output (correctness verification)
 *
 * Expected output format (example):
 *     Image size: 442 x 367 (162214 pixels)
 *     Running 200 iterations for timing accuracy...
 *
 *     --- SERIAL RESULTS ---
 *     Total CPU time (200 iterations): 436.191 ms
 *     Average per iteration: 2.181 ms
 *
 *     --- PARALLEL (CUDA) RESULTS ---
 *     GPU times (ms):
 *       RGB-to-Gray:           0.111 ms
 *       Histogram (atomic):    0.065 ms
 *       CDF+Map (host-side):   0.005 ms
 *       Apply mapping:         0.030 ms
 *       Total GPU time:        0.254 ms
 *
 *     --- PERFORMANCE COMPARISON ---
 *     CPU average time (per iteration): 2.181 ms
 *     GPU total time:                  0.254 ms
 *     Speedup (CPU / GPU):             8.58 x
 *     2-norm difference:               0.000000
 *
 * Algorithm Pipeline:
 *     1. RGB to Grayscale: Convert 3-channel RGB to 1-channel grayscale via luminance formula.
 *     2. Histogram: Count pixel intensity occurrences (256 bins).
 *     3. PDF/CDF: Normalize histogram to PDF; compute cumulative sum for CDF.
 *     4. Contrast Limiting: Clip CDF at 1% and 99% to prevent over-enhancement.
 *     5. Intensity Mapping: Create 256-entry lookup table using normalized CDF.
 *     6. Apply Mapping: Remap each grayscale pixel using lookup table.
 *
 * CUDA Implementation Details:
 *     - Thread organization: 1-D blocks of 256 threads, grid size varies by kernel:
 *         • RGB-to-Gray & Apply-Map: ceil(N/256) blocks (one thread per pixel)
 *         • Histogram: min(ceil(N/256), 1024) blocks with grid-stride loop
 *         • CDF/LUT: exactly 1 block of 256 threads (parallel scan in shared memory)
 *     - RGB-to-Gray kernel: Each thread processes one pixel independently (embarrassingly parallel).
 *     - Histogram kernel: Uses per-warp shared memory privatization. Each warp maintains its own
 *                         256-bin histogram in shared memory using fast shared-memory atomics,
 *                         then reduces to global histogram with only 256 atomics per block.
 *     - CDF/LUT kernel: Computes PDF, performs parallel Hillis-Steele inclusive scan for CDF,
 *                       finds 1%/99% cutoff points, and builds LUT entirely on GPU in shared memory.
 *     - Apply-Map kernel: Each thread processes one pixel via lookup table (embarrassingly parallel).
 *
 * Performance Notes:
 *     - Small images (< 1M pixels): Kernel launch overhead dominates; speedup ~5-15x.
 *     - Large images (> 1M pixels): Speedup increases to 50-200x depending on GPU architecture.
 *     - Per-warp histogram privatization eliminates global atomic contention bottleneck.
 *     - Parallel CDF scan (8 iterations) much faster than sequential 256-element scan.
 *     - Two-norm difference of 0.0 confirms pixel-perfect numerical equivalence between serial and CUDA.
 *
 * Memory Management:
 *     - Input RGB image: Transferred to GPU once (host-to-device).
 *     - Grayscale intermediate: Stays on GPU to minimize PCIe transfers.
 *     - Histogram: Built and kept on GPU (no host transfer).
 *     - CDF and LUT: Computed entirely on GPU in shared memory, LUT stays in GPU memory.
 *     - Final output: Transferred back to host (device-to-host).
 *     - Total host↔device transfers: 2 (input RGB upload + output grayscale download).
 *
 * Timing Methodology:
 *     - CPU: Uses high-resolution clock_gettime(CLOCK_MONOTONIC) for nanosecond precision.
 *     - GPU: Uses CUDA events (cudaEventRecord, cudaEventElapsedTime) for kernel-specific timing.
 *     - CPU runs multiple iterations (default 200) to overcome clock resolution limitations.
 *     - GPU runs once per invocation; timing includes all kernel launches but excludes initial
 *       data transfers (focuses on computational performance).
 *
 * Author: Tomas Nader
 * Student ID: w10172066
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <errno.h>
#include <time.h>

#include <cuda_runtime.h>

#ifndef TPB
#define TPB 256 // threads per block; must be multiple of 32
#endif
#define WARP_SIZE 32
#define HIST_BINS 256

// Padding avoids shared-memory bank conflicts on power-of-two strides
#define SH_HIST_STRIDE (HIST_BINS + 1)

#define CUDA_OK(call)                                                                       \
  do                                                                                        \
  {                                                                                         \
    cudaError_t e = (call);                                                                 \
    if (e != cudaSuccess)                                                                   \
    {                                                                                       \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
      exit(1);                                                                              \
    }                                                                                       \
  } while (0)

static void die(const char *msg)
{
  fprintf(stderr, "%s\n", msg);
  exit(1);
}

// ---------- High-resolution timer ----------
static double get_time_ms()
{
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
}

// ---------- Minimal PPM/PGM I/O (binary P6/P5) ----------
typedef struct
{
  int w, h;
  unsigned char *data;
} ImageRGB8;

typedef struct
{
  int w, h;
  unsigned char *data;
} ImageGray8;

static void skip_ws_and_comments(FILE *f)
{
  int c;
  do
  {
    c = fgetc(f);
    if (c == '#')
    {
      while (c != '\n' && c != EOF)
        c = fgetc(f);
    }
  } while (c != EOF && (c == ' ' || c == '\n' || c == '\r' || c == '\t'));
  if (c != EOF)
    ungetc(c, f);
}

static ImageRGB8 readPPM(const char *path)
{
  ImageRGB8 img;
  img.w = img.h = 0;
  img.data = NULL;
  FILE *f = fopen(path, "rb");
  if (!f)
    die("Cannot open input file");

  char magic[3] = {0};
  if (fscanf(f, "%2s", magic) != 1 || strcmp(magic, "P6") != 0)
    die("Not a P6 PPM");

  skip_ws_and_comments(f);
  if (fscanf(f, "%d %d", &img.w, &img.h) != 2 || img.w <= 0 || img.h <= 0)
    die("Bad PPM size");

  skip_ws_and_comments(f);
  int maxv = 0;
  if (fscanf(f, "%d", &maxv) != 1 || maxv != 255)
    die("Bad maxval (must be 255)");

  /* consume single whitespace */
  int ch = fgetc(f);
  if (ch == EOF)
    die("Truncated PPM header");

  size_t N = (size_t)img.w * (size_t)img.h;
  img.data = (unsigned char *)malloc(3 * N);
  if (!img.data)
  {
    fclose(f);
    die("OOM");
  }

  size_t got = fread(img.data, 1, 3 * N, f);
  fclose(f);
  if (got != 3 * N)
    die("Short read on PPM data");

  return img;
}

static void writePPM_from_gray(const char *path, const ImageGray8 *gray)
{
  FILE *f = fopen(path, "wb");
  if (!f)
    die("Cannot open output PPM");
  fprintf(f, "P6\n%d %d\n255\n", gray->w, gray->h);
  for (int i = 0; i < gray->w * gray->h; i++)
  {
    unsigned char v = gray->data[i];
    fputc(v, f);
    fputc(v, f);
    fputc(v, f);
  }
  fclose(f);
}
// --------------- CPU reference pipeline ---------------
static void cpu_rgb_to_gray(const ImageRGB8 *rgb, ImageGray8 *gray)
{
  gray->w = rgb->w;
  gray->h = rgb->h;
  size_t N = (size_t)rgb->w * (size_t)rgb->h;
  gray->data = (unsigned char *)malloc(N);
  if (!gray->data)
    die("OOM gray");

  const unsigned char *p = rgb->data;
  for (size_t i = 0; i < N; i++)
  {
    int r = p[3 * i + 0];
    int g = p[3 * i + 1];
    int b = p[3 * i + 2];
    float y = 0.299f * r + 0.587f * g + 0.114f * b;
    int yi = (int)floorf(y + 0.5f);
    if (yi < 0)
      yi = 0;
    if (yi > 255)
      yi = 255;
    gray->data[i] = (unsigned char)yi;
  }
}

static void cpu_histogram(const ImageGray8 *gray, int hist[256])
{
  for (int i = 0; i < 256; i++)
    hist[i] = 0;
  size_t N = (size_t)gray->w * gray->h;
  for (size_t i = 0; i < N; i++)
    hist[gray->data[i]]++;
}

static void cpu_pdf_cdf(const int hist[256], int num_pixels, float cdf[256])
{
  float accum = 0.0f;
  for (int i = 0; i < 256; i++)
  {
    float p = (float)hist[i] / (float)num_pixels;
    accum += p;
    cdf[i] = accum;
  }
}

static void cpu_build_map_from_cdf(const float cdf[256], int map[256])
{
  float lower_cut = 0.01f;
  float upper_cut = 0.99f;

  int floor_gray = 0;
  for (int i = 0; i < 256; i++)
  {
    if (cdf[i] >= lower_cut)
    {
      floor_gray = i;
      break;
    }
  }
  int ceil_gray = 255;
  for (int i = 255; i >= 0; i--)
  {
    if (cdf[i] <= upper_cut)
    {
      ceil_gray = i;
      break;
    }
  }

  float cdf_floor = cdf[floor_gray];
  float cdf_ceil = cdf[ceil_gray];
  float denom = (cdf_ceil - cdf_floor);
  if (denom <= 0.0f)
  {
    for (int i = 0; i < 256; i++)
    {
      int v = (int)floorf(255.f * cdf[i] + 0.5f);
      if (v < 0)
        v = 0;
      if (v > 255)
        v = 255;
      map[i] = v;
    }
    return;
  }
  for (int i = 0; i < 256; i++)
  {
    if (i < floor_gray)
      map[i] = 0;
    else if (i > ceil_gray)
      map[i] = 255;
    else
    {
      float norm = (cdf[i] - cdf_floor) / denom;
      int v = (int)floorf(255.f * norm + 0.5f);
      if (v < 0)
        v = 0;
      if (v > 255)
        v = 255;
      map[i] = v;
    }
  }
}

static void cpu_apply_map(const ImageGray8 *in, ImageGray8 *out, const int map[256])
{
  out->w = in->w;
  out->h = in->h;
  size_t N = (size_t)in->w * in->h;
  out->data = (unsigned char *)malloc(N);
  if (!out->data)
    die("OOM out");
  for (size_t i = 0; i < N; i++)
    out->data[i] = (unsigned char)map[in->data[i]];
}

// ----------------- CUDA kernels -----------------

__global__ void rgb_to_gray_kernel(const unsigned char *rgb, unsigned char *gray, int N)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= N)
    return;
  int r = rgb[3 * idx + 0], g = rgb[3 * idx + 1], b = rgb[3 * idx + 2];
  float y = 0.299f * r + 0.587f * g + 0.114f * b;
  int yi = (int)floorf(y + 0.5f);
  if (yi < 0)
    yi = 0;
  if (yi > 255)
    yi = 255;
  gray[idx] = (unsigned char)yi;
}

__global__ void hist_kernel_shared_perwarp(const unsigned char *__restrict__ gray, int N, int *__restrict__ g_hist)
{

  extern __shared__ int s_mem[]; // size: warpsPerBlock * SH_HIST_STRIDE
  const int tid = threadIdx.x;
  const int warp = tid / WARP_SIZE;

  const int warpsPerBlock = blockDim.x / WARP_SIZE;

  // 1) Zero all per-warp shared histograms
  for (int i = tid; i < warpsPerBlock * SH_HIST_STRIDE; i += blockDim.x)
  {
    s_mem[i] = 0;
  }
  __syncthreads();

  // 2) Accumulate into this block's per-warp histograms (grid-stride loop)
  for (int idx = blockIdx.x * blockDim.x + tid;
       idx < N;
       idx += blockDim.x * gridDim.x)
  {
    unsigned char g = gray[idx];
    int *warp_hist = s_mem + warp * SH_HIST_STRIDE;
    atomicAdd(&warp_hist[(int)g], 1); // shared-memory atomic (fast)
  }
  __syncthreads();

  // 3) Reduce per-warp histograms into the global histogram
  for (int bin = tid; bin < HIST_BINS; bin += blockDim.x)
  {
    int sum = 0;
    for (int w = 0; w < warpsPerBlock; ++w)
    {
      sum += s_mem[w * SH_HIST_STRIDE + bin];
    }
    if (sum)
      atomicAdd(&g_hist[bin], sum);
  }
}
__global__ void apply_map_kernel(const unsigned char *in, unsigned char *out, const int *map, int N)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= N)
    return;
  unsigned char t = in[idx];
  int m = map[(int)t];
  out[idx] = (unsigned char)m;
}

// Build CDF and LUT on device from a 256-bin histogram.
__global__ void cdf_and_lut_kernel(const int *__restrict__ hist,
                                   int N,
                                   int *__restrict__ d_map)
{
  // Shared array holds PDF/CDF (float[256]).
  extern __shared__ float cdf[]; // size = 256 * sizeof(float)

  const int tid = threadIdx.x;
  if (blockIdx.x != 0 || blockDim.x < 256)
    return;
  if (N <= 0)
  {
    if (tid < 256)
      d_map[tid] = tid; // identity map
    return;
  }

  // 1) Load PDF into shared memory
  if (tid < 256)
  {
    float invN = 1.0f / (float)N;
    cdf[tid] = (float)hist[tid] * invN;
  }
  __syncthreads();

  for (int offset = 1; offset < 256; offset <<= 1)
  {
    float addend = 0.0f;
    if (tid >= offset && tid < 256)
    {
      addend = cdf[tid - offset];
    }
    __syncthreads();
    if (tid < 256)
    {
      cdf[tid] += addend;
    }
    __syncthreads();
  }

  // 3) Find 1% and 99% cut points; compute fallback flag.
  __shared__ int floor_gray;
  __shared__ int ceil_gray;
  __shared__ float cdf_floor;
  __shared__ float cdf_ceil;
  __shared__ int use_plain;

  if (tid == 0)
  {
    const float lower_cut = 0.01f;
    const float upper_cut = 0.99f;

    int f = 0;
    while (f < 256 && cdf[f] < lower_cut)
      ++f;

    int c = 255;
    while (c >= 0 && cdf[c] > upper_cut)
      --c;

    floor_gray = (f < 256) ? f : 0;
    ceil_gray = (c >= 0) ? c : 255;

    cdf_floor = cdf[floor_gray];
    cdf_ceil = cdf[ceil_gray];

    use_plain = ((cdf_ceil - cdf_floor) <= 0.0f) ? 1 : 0; // fallback if no range
  }
  __syncthreads();

  // 4) Build LUT directly to device memory (d_map)
  if (tid < 256)
  {
    int out;
    if (use_plain)
    {
      // Plain equalization: round(255 * CDF)
      int v = (int)floorf(255.0f * cdf[tid] + 0.5f);
      if (v < 0)
        v = 0;
      if (v > 255)
        v = 255;
      out = v;
    }
    else
    {
      if (tid < floor_gray)
      {
        out = 0;
      }
      else if (tid > ceil_gray)
      {
        out = 255;
      }
      else
      {
        float denom = cdf_ceil - cdf_floor;
        float norm = (cdf[tid] - cdf_floor) / denom;
        int v = (int)floorf(255.0f * norm + 0.5f);
        if (v < 0)
          v = 0;
        if (v > 255)
          v = 255;
        out = v;
      }
    }
    d_map[tid] = out;
  }
}

// --------------- GPU pipeline wrapper (host) --------------

static void run_cuda_pipeline(const ImageRGB8 *in_rgb,
                              ImageGray8 *out_eq,
                              float *t_rgb2gray_ms,
                              float *t_hist_ms,
                              float *t_scan_ms,
                              float *t_apply_ms,
                              float *t_total_ms)
{
  const int W = in_rgb->w, H = in_rgb->h;
  const int N = W * H;

  cudaEvent_t evStart, evA, evB, evC, evEnd;
  CUDA_OK(cudaEventCreate(&evStart));
  CUDA_OK(cudaEventCreate(&evA));
  CUDA_OK(cudaEventCreate(&evB));
  CUDA_OK(cudaEventCreate(&evC));
  CUDA_OK(cudaEventCreate(&evEnd));

  unsigned char *d_rgb = NULL, *d_gray = NULL, *d_out = NULL;
  int *d_hist = NULL, *d_map = NULL;

  const size_t bytes_rgb = (size_t)3 * N;
  const size_t bytes_gray = (size_t)N;

  CUDA_OK(cudaMalloc((void **)&d_rgb, bytes_rgb));
  CUDA_OK(cudaMalloc((void **)&d_gray, bytes_gray));
  CUDA_OK(cudaMalloc((void **)&d_out, bytes_gray));
  CUDA_OK(cudaMalloc((void **)&d_hist, 256 * sizeof(int)));
  CUDA_OK(cudaMalloc((void **)&d_map, 256 * sizeof(int)));

  CUDA_OK(cudaMemcpy(d_rgb, in_rgb->data, bytes_rgb, cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemset(d_hist, 0, 256 * sizeof(int)));

  dim3 block1d(TPB);
  dim3 grid1d((N + block1d.x - 1) / block1d.x);

  // RGB -> Gray
  CUDA_OK(cudaEventRecord(evStart));
  rgb_to_gray_kernel<<<grid1d, block1d>>>(d_rgb, d_gray, N);
  CUDA_OK(cudaEventRecord(evA));
  CUDA_OK(cudaEventSynchronize(evA));
  CUDA_OK(cudaGetLastError());

  // Histogram (per-warp shared-memory privatization)
  CUDA_OK(cudaMemset(d_hist, 0, 256 * sizeof(int))); // ensure global hist is zeroed

  int grid_hist_blocks = (N + block1d.x - 1) / block1d.x;
  if (grid_hist_blocks > 1024)
    grid_hist_blocks = 1024; // Cap at 1024 blocks
  dim3 grid_hist(grid_hist_blocks);

  int warpsPerBlock = block1d.x / WARP_SIZE;
  size_t shmemBytes = (size_t)warpsPerBlock * SH_HIST_STRIDE * sizeof(int);

  hist_kernel_shared_perwarp<<<grid_hist, block1d, shmemBytes>>>(d_gray, N, d_hist);
  CUDA_OK(cudaEventRecord(evB));
  CUDA_OK(cudaEventSynchronize(evB));
  CUDA_OK(cudaGetLastError());

  // ---- GPU CDF + LUT (device-side) ----
  cudaEvent_t evScanEnd;
  CUDA_OK(cudaEventCreate(&evScanEnd));

  // Build CDF and LUT on device from the 256-bin histogram
  // Launch with a single block of 256 threads and 256*sizeof(float) shared memory.
  cdf_and_lut_kernel<<<1, 256, 256 * sizeof(float)>>>(d_hist, N, d_map);

  // Record end of the "CDF+Map" phase
  CUDA_OK(cudaEventRecord(evScanEnd));
  CUDA_OK(cudaEventSynchronize(evScanEnd));
  CUDA_OK(cudaGetLastError());

  // Store "CDF+Map" GPU time
  if (t_scan_ms)
  {
    float ms_scan = 0.f;
    CUDA_OK(cudaEventElapsedTime(&ms_scan, evB, evScanEnd));
    *t_scan_ms = ms_scan;
  }
  CUDA_OK(cudaEventDestroy(evScanEnd));
  // ---- end GPU CDF + LUT ----

  // Apply mapping (kernel-only timing)
  CUDA_OK(cudaEventRecord(evC)); // start apply timing AFTER host work + H2D
  apply_map_kernel<<<grid1d, block1d>>>(d_gray, d_out, d_map, N);
  CUDA_OK(cudaEventRecord(evEnd));
  CUDA_OK(cudaEventSynchronize(evEnd));
  CUDA_OK(cudaGetLastError());

  // Timings
  float ms_rgb2gray = 0.f, ms_hist = 0.f, ms_apply = 0.f, ms_total = 0.f;
  CUDA_OK(cudaEventElapsedTime(&ms_rgb2gray, evStart, evA)); // rgb->gray kernel
  CUDA_OK(cudaEventElapsedTime(&ms_hist, evA, evB));         // hist kernel
  CUDA_OK(cudaEventElapsedTime(&ms_apply, evC, evEnd));      // apply kernel only
  CUDA_OK(cudaEventElapsedTime(&ms_total, evStart, evEnd));  // total kernels

  if (t_rgb2gray_ms)
    *t_rgb2gray_ms = ms_rgb2gray;
  if (t_hist_ms)
    *t_hist_ms = ms_hist;
  if (t_apply_ms)
    *t_apply_ms = ms_apply;
  if (t_total_ms)
    *t_total_ms = ms_total;

  // Copy result back
  out_eq->w = W;
  out_eq->h = H;
  out_eq->data = (unsigned char *)malloc((size_t)N);
  if (!out_eq->data)
    die("OOM out_eq");
  CUDA_OK(cudaMemcpy(out_eq->data, d_out, (size_t)N, cudaMemcpyDeviceToHost));

  // Cleanup
  CUDA_OK(cudaEventDestroy(evStart));
  CUDA_OK(cudaEventDestroy(evA));
  CUDA_OK(cudaEventDestroy(evB));
  CUDA_OK(cudaEventDestroy(evC));
  CUDA_OK(cudaEventDestroy(evEnd));
  cudaFree(d_rgb);
  cudaFree(d_gray);
  cudaFree(d_out);
  cudaFree(d_hist);
  cudaFree(d_map);
}

// --------------- Utility: 2-norm diff ----------------

static double two_norm_diff(const ImageGray8 *a, const ImageGray8 *b)
{
  if (a->w != b->w || a->h != b->h)
    return -1.0;
  size_t N = (size_t)a->w * a->h;
  double acc = 0.0;
  for (size_t i = 0; i < N; i++)
  {
    double d = (double)a->data[i] - (double)b->data[i];
    acc += d * d;
  }
  return sqrt(acc);
}

// --------------- Main: serial + parallel in one program ---------------

int main(int argc, char **argv)
{
  const char *in_path = (argc >= 2) ? argv[1] : "chest_x_rays.ppm";
  const char *out_cpu = (argc >= 3) ? argv[2] : "output_serial.ppm";
  const char *out_gpu = (argc >= 4) ? argv[3] : "output_cuda.ppm";
  int num_iterations = (argc >= 5) ? atoi(argv[4]) : 200;

  ImageRGB8 rgb = readPPM(in_path);

  printf("Image size: %d x %d (%d pixels)\n", rgb.w, rgb.h, rgb.w * rgb.h);
  printf("Running %d iterations for timing accuracy...\n\n", num_iterations);

  // --- SERIAL (multiple iterations for timing) ---
  double t0_total = get_time_ms();
  ImageGray8 eq_cpu = {0, 0, NULL};

  double t_rgb2gray_ms = 0.0;
  double t_hist_ms = 0.0;
  double t_cdf_map_ms = 0.0;
  double t_apply_ms = 0.0;

  for (int iter = 0; iter < num_iterations; iter++)
  {
    ImageGray8 gray_cpu = {0, 0, NULL};

    // RGB → Gray
    double t0 = get_time_ms();
    cpu_rgb_to_gray(&rgb, &gray_cpu);
    double t1 = get_time_ms();
    t_rgb2gray_ms += (t1 - t0);

    // Histogram
    int hist[256];
    t0 = get_time_ms();
    cpu_histogram(&gray_cpu, hist);
    double t2 = get_time_ms();
    t_hist_ms += (t2 - t0);

    // CDF + Map
    float cdf[256];
    int map[256];
    t0 = get_time_ms();
    cpu_pdf_cdf(hist, rgb.w * rgb.h, cdf);
    cpu_build_map_from_cdf(cdf, map);
    double t3 = get_time_ms();
    t_cdf_map_ms += (t3 - t0);

    // Apply map
    if (eq_cpu.data)
    {
      free(eq_cpu.data);
      eq_cpu.data = NULL;
    }
    t0 = get_time_ms();
    cpu_apply_map(&gray_cpu, &eq_cpu, map);
    double t4 = get_time_ms();
    t_apply_ms += (t4 - t0);

    free(gray_cpu.data);
  }

  double t1_total = get_time_ms();
  double cpu_total_ms = (t1_total - t0_total);
  double cpu_avg_ms = cpu_total_ms / num_iterations;

  writePPM_from_gray(out_cpu, &eq_cpu);

  printf("--- SERIAL RESULTS ---\n");
  printf("Total CPU time (%d iterations): %.3f ms\n", num_iterations, cpu_total_ms);
  printf("Average per iteration: %.3f ms\n\n", cpu_avg_ms);
  printf("CPU stage breakdown (average per iteration):\n");
  printf("  RGB-to-Gray:         %.3f ms\n", t_rgb2gray_ms / num_iterations);
  printf("  Histogram:           %.3f ms\n", t_hist_ms / num_iterations);
  printf("  CDF+Map:             %.3f ms\n", t_cdf_map_ms / num_iterations);
  printf("  Apply Mapping:       %.3f ms\n\n", t_apply_ms / num_iterations);

  // PARALLEL (CUDA) - single run
  float ms_rgb2gray = 0.f, ms_hist = 0.f, ms_scan = 0.f, ms_apply = 0.f, ms_total = 0.f;
  ImageGray8 eq_gpu = {0, 0, NULL};
  run_cuda_pipeline(&rgb, &eq_gpu, &ms_rgb2gray, &ms_hist, &ms_scan, &ms_apply, &ms_total);
  writePPM_from_gray(out_gpu, &eq_gpu);

  printf("--- PARALLEL (CUDA) RESULTS ---\n");
  printf("GPU times (ms):\n");
  printf("  RGB-to-Gray:           %.3f ms\n", ms_rgb2gray);
  printf("  Histogram (shared-per-warp): %.3f ms\n", ms_hist);
  printf("  CDF+Map (device-side): %.3f ms\n", ms_scan);
  printf("  Apply mapping:         %.3f ms\n", ms_apply);
  printf("  Total GPU time:        %.3f ms\n\n", ms_total);

  // Compare
  double diff2 = two_norm_diff(&eq_cpu, &eq_gpu);

  printf("--- PERFORMANCE COMPARISON ---\n");
  printf("CPU average time (per iteration): %.3f ms\n", cpu_avg_ms);
  printf("GPU total time:                  %.3f ms\n", ms_total);
  printf("Speedup (CPU / GPU):             %.2f x\n", (ms_total > 0.f) ? (cpu_avg_ms / ms_total) : 0.0);
  printf("2-norm difference:               %.6f\n\n", diff2);

  if (diff2 < 0.001)
  {
    printf("✓ Results match: Serial and CUDA implementations are equivalent.\n");
  }
  else if (diff2 < 1.0)
  {
    printf("⚠ Minor differences detected (within acceptable tolerance)\n");
  }
  else
  {
    printf("✗ Significant differences detected. Check implementation.\n");
  }

  /* cleanup */
  free(rgb.data);
  free(eq_cpu.data);
  free(eq_gpu.data);

  return 0;
}