// All the needed library functions for this program
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

// Reads “P6”, width, height, maxval and returns 0 on success, nonzero on error
int read_ppm_header(FILE *fp, int *width, int *height, int *maxval);

// Reads rgb data (3 × num_pixels bytes) into rgb_buf; returns number of bytes read
size_t read_ppm_data(FILE *fp, unsigned char *rgb_buf, int num_pixels);

// Converts RGB (3 per pixel) to a single grayscale value per pixel
void rgb_to_grayscale(const unsigned char *rgb_buf, unsigned char *gray_buf, int num_pixels);

// Fills hist[] with counts of each gray value
void compute_histogram(const unsigned char *gray_buf, int num_pixels, int hist[256]);

// Computes probability (hist[i] / total_pixels) into pdf
void compute_pdf(const int hist[256], int num_pixels, float pdf[256]);

// Builds cumulative distribution (running sum) in cdf
void compute_cdf(const float pdf[256], float cdf[256]);

// Makes the mapping from old gray → new gray using cdf
void build_map(const float cdf[256], int map[256]);

// For each pixel, gray_out[i] = map[ gray_in[i] ]
void apply_mapping(const unsigned char *gray_in, unsigned char *gray_out, int num_pixels, const int map[256]);

// Writes a PPM file (with R=G=B = gray_buf) with given dimensions; returns 0 on success
int write_ppm_from_gray(const char *filename, const unsigned char *gray_buf, int width, int height);

// Enforce that map[] values lie between 0 and 255 — you can embed this logic in build_map too
void clamp_map(int map[256]);



int main() {



    return 0;
}