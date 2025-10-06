// All the needed library functions for this program
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


int main(int argc, char *argv[]) {

    const char *input_filename  = (argc >= 2) ? argv[1] : "chest_x_rays.ppm";      
    const char *output_filename = (argc >= 3) ? argv[2] : "output1.ppm";

    // 1. Argument parsing
    if (argc != 1 && argc != 3) {
        fprintf(stderr, "Usage: %s [input.ppm output.ppm]\n", argv[0]);
        return -1;
    }

    // 2. Open input file
    FILE *fp = fopen(input_filename, "rb");
    if (!fp) {
        fprintf(stderr, "Error: Could not open file %s\n", input_filename);
        return -1;
    }

    // 3. Read PPM header
    int width, height, maxval;
    if (read_ppm_header(fp, &width, &height, &maxval) != 0) {
        fclose(fp);
        return -1;
    }
    int num_pixels = width * height;

    // 4. Allocate memory for image data
    unsigned char *rgb_buf = (unsigned char *)malloc(num_pixels * 3);
    unsigned char *gray_buf = (unsigned char *)malloc(num_pixels);
    unsigned char *equalized_gray_buf = (unsigned char *)malloc(num_pixels);
    if (!rgb_buf || !gray_buf || !equalized_gray_buf) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        fclose(fp);
        free(rgb_buf);
        free(gray_buf);
        free(equalized_gray_buf);
        return -1;
    }

    // 5. Read pixel data
    size_t expected = (size_t)num_pixels * 3;
    if (read_ppm_data(fp, rgb_buf, num_pixels) != expected) {
        fclose(fp);
        free(rgb_buf);
        free(gray_buf);
        free(equalized_gray_buf);
        return -1;
    }
    fclose(fp);

    // 6. Convert to grayscale
    rgb_to_grayscale(rgb_buf, gray_buf, num_pixels);
    free(rgb_buf); // No longer needed
    rgb_buf = NULL;

    // 7. Compute histogram
    int hist[256];
    compute_histogram(gray_buf, num_pixels, hist);

    // 8. Compute PDF
    float pdf[256];
    compute_pdf(hist, num_pixels, pdf);

    // 9. Compute CDF
    float cdf[256];
    compute_cdf(pdf, cdf);

    // 10. Build mapping
    int map[256];
    build_map(cdf, map);

    // 11. Apply mapping
    apply_mapping(gray_buf, equalized_gray_buf, num_pixels, map);
    free(gray_buf); // No longer needed

    // 12. Write output PPM file from grayscale data
    if (write_ppm_from_gray(output_filename, equalized_gray_buf, width, height) != 0) { 
        free(equalized_gray_buf);
        return -1;
    }

    free(equalized_gray_buf);
    return 0;
}

void skip_comments_and_whitespace(FILE *fp) {                              
    int ch;
    do {
        ch = fgetc(fp);
        if (ch == '#') {
            while ((ch = fgetc(fp)) != EOF && ch != '\n') {  }
        }
    } while (ch != EOF && isspace((unsigned char)ch));
    if (ch != EOF) ungetc(ch, fp);
}

int read_ppm_header(FILE *fp, int *width, int *height, int *maxval) {      
    char magic[3] = {0};

    skip_comments_and_whitespace(fp);
    if (fscanf(fp, "%2s", magic) != 1 || strcmp(magic, "P6") != 0) {
        fprintf(stderr, "Error: Not a PPM P6 file\n");
        return -1;
    }

    skip_comments_and_whitespace(fp);
    if (fscanf(fp, "%d %d", width, height) != 2 || *width <= 0 || *height <= 0) {
        fprintf(stderr, "Error: Invalid image size\n");
        return -1;
    }

    skip_comments_and_whitespace(fp);
    if (fscanf(fp, "%d", maxval) != 1 || *maxval != 255) {
        fprintf(stderr, "Error: Invalid maxval (must be 255)\n");
        return -1;
    }

    /* consume exactly one whitespace separator before binary payload */
    int ch = fgetc(fp);                                                    
    if (ch == EOF) {
        fprintf(stderr, "Error: Truncated PPM header\n");
        return -1;
    }
    return 0;
}

size_t read_ppm_data(FILE *fp, unsigned char *rgb_buf, int num_pixels) {

    size_t total_bytes = (size_t)num_pixels * 3;
    size_t bytes_read  = fread(rgb_buf, 1, total_bytes, fp);

    if (bytes_read != total_bytes) {
        fprintf(stderr, "Error: Could not read pixel data (got %zu of %zu bytes)\n",
                bytes_read, total_bytes);
    }

    return bytes_read;
}

void rgb_to_grayscale(const unsigned char *rgb_buf, unsigned char *gray_buf, int num_pixels) {

    for (int i = 0; i < num_pixels; i++) {
        int r = rgb_buf[3 * i];
        int g = rgb_buf[3 * i + 1];
        int b = rgb_buf[3 * i + 2];
        float y = 0.299f * r + 0.587f * g + 0.114f * b;
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

/* void build_map(const float cdf[256], int map[256]) {
    for (int i = 0; i < 256; i++) {
        int new_value = (int)(cdf[i] * 255 + 0.5); 
        if (new_value < 0) new_value = 0;
        if (new_value > 255) new_value = 255;
        map[i] = new_value;
    }
} */

void build_map(const float cdf[256], int map[256]) {
    // clipping thresholds (you may tune these)
    float lower_cut = 0.01f;   // 1%
    float upper_cut = 0.99f;   // 99%

    int floor_gray = 0;
    for (int i = 0; i < 256; i++) {
        if (cdf[i] >= lower_cut) {
            floor_gray = i;
            break;
        }
    }
    int ceil_gray = 255;
    for (int i = 255; i >= 0; i--) {
        if (cdf[i] <= upper_cut) {
            ceil_gray = i;
            break;
        }
    }

    float cdf_floor = cdf[floor_gray];
    float cdf_ceil  = cdf[ceil_gray];
    float denom = (cdf_ceil - cdf_floor);
    if (denom <= 0.0f) {
        // fallback to simple mapping if denom is zero
        for (int i = 0; i < 256; i++) {
            map[i] = (int)(cdf[i] * 255.0f + 0.5f);
            if (map[i] < 0) map[i] = 0;
            if (map[i] > 255) map[i] = 255;
        }
        return;
    }

    for (int i = 0; i < 256; i++) {
        if (i < floor_gray) {
            map[i] = 0;
        } else if (i > ceil_gray) {
            map[i] = 255;
        } else {
            float norm = (cdf[i] - cdf_floor) / denom;
            int v = (int)(norm * 255.0f + 0.5f);
            if (v < 0) v = 0;
            if (v > 255) v = 255;
            map[i] = v;
        }
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
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Error: Could not open file for writing: %s\n", filename);
        return -1; 
    }

    fprintf(fp, "P6\n%d %d\n255\n", width, height);

    for (int i = 0; i < width * height; i++) {
        unsigned char v = gray_buf[i];
        fputc(v, fp); 
        fputc(v, fp); 
        fputc(v, fp); 
    }

    fclose(fp);
    return 0;
}
