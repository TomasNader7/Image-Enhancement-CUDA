// All the needed library functions for this program
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

// Reads “P6”, width, height, maxval and returns 0 on success, nonzero on error
int read_ppm_header(FILE *fp, int *width, int *height, int *maxval);

// Skips comments and whitespace in PPM header
void skip_comments_and_whitespace(FILE *fp);

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

void skip_comments_and_whitespace(FILE *fp) {
    int c;
    while ((c = fgetc(fp)) != EOF) {
        if (isspace(c)) {
            continue; 
        } else if (c == '#') {
            while ((c = fgetc(fp)) != EOF && c != '\n');
        } else {
            ungetc(c, fp);
            break; 
        }
    }
}

int read_ppm_header(FILE *fp, int *width, int *height, int *maxval) {
    char magic[3];
    int c = fgetc(fp);

    skip_comments_and_whitespace(fp);
    if (fscanf(fp, "%2s", magic) != 1 || strcmp(magic, "P6") != 0) {
        fprintf(stderr, "Error: Not a PPM P6 file\n");
        return -1; 
    }

    skip_comments_and_whitespace(fp);
    if(fscanf(fp, "%d %d", width, height) != 2) {
        fprintf(stderr, "Error: Invalid image size\n");
        return -1; 
    }

    skip_comments_and_whitespace(fp);
    if(fscanf(fp, "%d", maxval) != 1 || *maxval != 255) {
        fprintf(stderr, "Error: Invalid maxval (must be 255)\n");
        return -1; 
    }

    skip_comments_and_whitespace(fp);
    if ( c == EOF || !isspace(c)) {
        fclose(fp);
        fprintf(stderr, "Error: Invalid PPM header\n");
        return -1;
    }

    return 0; 
}

size_t read_ppm_data(FILE *fp, unsigned char *rgb_buf, int num_pixels) {

    size_t total_bytes = num_pixels * 3;

    size_t bytes_read = fread(
        rgb_buf,
        sizeof(unsigned char),
        total_bytes,
        fp
    );

    if (bytes_read != total_bytes) {
        fprintf(stderr, "Error: Could not read pixel data\n");
        return 0; 
    }

    return bytes_read;

}

void rgb_to_grayscale(const unsigned char *rgb_buf, unsigned char *gray_buf, int num_pixels) {

    for (int i = 0; i < num_pixels; i++) {
        int r = rgb_buf[3 * i];
        int g = rgb_buf[3 * i + 1];
        int b = rgb_buf[3 * i + 2];
        float y = 0.299 * r + 0.587 * g + 0.114 * b;
        int y_int = (int)(y + 0.5); // rounding to nearest integer

        if (y_int < 0) y_int = 0;
        if (y_int > 255) y_int = 255;


        gray_buf[i] = (unsigned char)y_int;
    } 

}

void compute_histogram(const unsigned char *gray_buf, int num_pixels, int hist[256]) {
    for (int i = 0; i < 256; i++) {
        hist[i] = 0;
    }

    for (int i = 0; i < num_pixels; i++) {
        int gray_value = gray_buf[i];
        hist[gray_value]++;
    }
}


void compute_pdf(const int hist[256], int num_pixels, float pdf[256]) {
    for (int i = 0; i < 256; i++) {
        pdf[i] = (float)hist[i] / num_pixels;
    }
}

void compute_cdf(const float pdf[256], float cdf[256]) {
    float accum = 0.0f;

    for (int i = 0; i < 256; i++) {
        accum += pdf[i];
        cdf[i] = accum;
    }
}

void build_map(const float cdf[256], int map[256]) {
    for (int i = 0; i < 256; i++) {
        int new_value = (int)(cdf[i] * 255 + 0.5); 
        if (new_value < 0) new_value = 0;
        if (new_value > 255) new_value = 255;
        map[i] = new_value;
    }

}

void apply_mapping(const unsigned char *gray_in, unsigned char *gray_out, int num_pixels, const int map[256]) {
    for (int i = 0; i < num_pixels; i++) {
        int gray_value = gray_in[i];
        int new_value = map[gray_value];
        gray_out[i] = (unsigned char)new_value;
    }
}

int write_ppm_from_gray(const char *filename, const unsigned char *gray_buf, int width, int height) {
    FILE *fp = fopen(filename, "download");
    if (!fp) {
        fprintf(stderr, "Error: Could not open file for writing: %s\n", filename);
        return -1; 
    }

    fprintf(fp, "P6\n%d %d\n255\n", width, height);

    for (int i = 0; i < width * height; i++) {
        unsigned char gray = gray_buf[i];
        fputc(gray, fp); 
        fputc(gray, fp); 
        fputc(gray, fp); 
    }

    fclose(fp);
    return 0;
}
