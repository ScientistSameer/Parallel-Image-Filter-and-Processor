/*
 * main_seq.c -- Sequential baseline for the Parallel Image Filter and
 * Processor (Hybrid MPI + OpenMP) project.
 *
 * Implements four image filters on real photographic input read from disk:
 *   1. Gaussian Blur        (5x5 weighted convolution, sum = 273)
 *   2. Sobel Edge Detection (3x3 gradient magnitude in X and Y)
 *   3. Image Sharpening     (3x3 unsharp-mask kernel)
 *   4. Grayscale Conversion (BT.601 weighted sum of RGB)
 *
 * Image I/O: binary NetPBM grayscale (P5) and RGB (P6) formats. No external
 * image libraries are used so the program has zero dependencies beyond libc.
 *
 * Usage:
 *   ./main_seq <gray.pgm> <rgb.ppm> <output_dir> [tag]
 *
 * Prints one CSV line per filter to stdout:
 *   RESULT,sequential,<filter>,1,<width>,<height>,<seconds>
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

/* Each filter is timed TRIALS times and the minimum is reported, which is
 * standard practice for wall-clock microbenchmarks: OS scheduling noise,
 * interrupts, and (here) WSL2 virtualization jitter can only ever make a
 * run slower than the machine's true capability, never faster, so the
 * minimum across repeated trials is the least noisy estimate. */
#define TRIALS 3

/* ------------------------------------------------------------------ */
/*  Convolution kernels (identical across all four implementations)    */
/* ------------------------------------------------------------------ */
#define GAUSS_K     5
#define GAUSS_HALF  (GAUSS_K / 2)
static const float GAUSS[GAUSS_K][GAUSS_K] = {
    { 1,  4,  7,  4,  1},
    { 4, 16, 26, 16,  4},
    { 7, 26, 41, 26,  7},
    { 4, 16, 26, 16,  4},
    { 1,  4,  7,  4,  1}
};
static const float GAUSS_SUM = 273.0f;

static const int SOBEL_X[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
static const int SOBEL_Y[3][3] = {{-1,-2,-1}, { 0, 0, 0}, { 1, 2, 1}};

static const int SHARPEN_K[3][3] = {
    { 0, -1,  0},
    {-1,  5, -1},
    { 0, -1,  0}
};

static inline unsigned char clamp_byte(float v)
{
    if (v <   0.0f) return 0;
    if (v > 255.0f) return 255;
    return (unsigned char)v;
}

/* ------------------------------------------------------------------ */
/*  Wall-clock timer (NOT clock() -- clock() reports CPU time, which   */
/*  is misleading once threads/processes run concurrently).            */
/* ------------------------------------------------------------------ */
static double now_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

/* ------------------------------------------------------------------ */
/*  Minimal binary NetPBM reader/writer (P5 grayscale, P6 RGB)         */
/* ------------------------------------------------------------------ */
static void skip_ws_comments(FILE *f)
{
    int c;
    for (;;) {
        c = fgetc(f);
        if (c == '#') { while (c != '\n' && c != EOF) c = fgetc(f); continue; }
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') continue;
        ungetc(c, f);
        return;
    }
}

static unsigned char *read_pnm(const char *path, int *w, int *h, int *channels)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "ERROR: cannot open %s\n", path); exit(1); }

    char magic[3] = {0};
    if (fscanf(f, "%2s", magic) != 1) { fprintf(stderr, "ERROR: bad header %s\n", path); exit(1); }

    if (strcmp(magic, "P5") == 0) *channels = 1;
    else if (strcmp(magic, "P6") == 0) *channels = 3;
    else { fprintf(stderr, "ERROR: unsupported format %s in %s\n", magic, path); exit(1); }

    int maxval;
    skip_ws_comments(f); if (fscanf(f, "%d", w) != 1)       { fprintf(stderr, "ERROR: bad header %s\n", path); exit(1); }
    skip_ws_comments(f); if (fscanf(f, "%d", h) != 1)       { fprintf(stderr, "ERROR: bad header %s\n", path); exit(1); }
    skip_ws_comments(f); if (fscanf(f, "%d", &maxval) != 1) { fprintf(stderr, "ERROR: bad header %s\n", path); exit(1); }
    fgetc(f); /* single whitespace byte before binary data */

    size_t n = (size_t)(*w) * (size_t)(*h) * (size_t)(*channels);
    unsigned char *buf = (unsigned char *)malloc(n);
    if (!buf) { fprintf(stderr, "ERROR: malloc failed reading %s\n", path); exit(1); }
    if (fread(buf, 1, n, f) != n) { fprintf(stderr, "ERROR: short read on %s\n", path); exit(1); }
    fclose(f);
    return buf;
}

static void write_pnm(const char *path, const unsigned char *buf,
                       int w, int h, int channels)
{
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "ERROR: cannot create %s\n", path); exit(1); }
    fprintf(f, "%s\n%d %d\n255\n", channels == 1 ? "P5" : "P6", w, h);
    fwrite(buf, 1, (size_t)w * h * channels, f);
    fclose(f);
}

/* ================================================================== */
/*  FILTER 1 -- Gaussian Blur                                          */
/* ================================================================== */
static void apply_gaussian_blur(const unsigned char *in, unsigned char *out,
                                 int width, int height)
{
    for (int y = GAUSS_HALF; y < height - GAUSS_HALF; y++) {
        for (int x = GAUSS_HALF; x < width - GAUSS_HALF; x++) {
            float sum = 0.0f;
            for (int ky = 0; ky < GAUSS_K; ky++)
                for (int kx = 0; kx < GAUSS_K; kx++)
                    sum += (float)in[(y + ky - GAUSS_HALF) * width + (x + kx - GAUSS_HALF)]
                           * GAUSS[ky][kx];
            out[y * width + x] = clamp_byte(sum / GAUSS_SUM);
        }
    }
    for (int x = 0; x < width; x++)
        for (int b = 0; b < GAUSS_HALF; b++) {
            out[b * width + x]            = in[b * width + x];
            out[(height-1-b) * width + x] = in[(height-1-b) * width + x];
        }
    for (int y = 0; y < height; y++)
        for (int b = 0; b < GAUSS_HALF; b++) {
            out[y * width + b]           = in[y * width + b];
            out[y * width + (width-1-b)] = in[y * width + (width-1-b)];
        }
}

/* ================================================================== */
/*  FILTER 2 -- Sobel Edge Detection                                    */
/* ================================================================== */
static void apply_sobel(const unsigned char *in, unsigned char *out,
                         int width, int height)
{
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int gx = 0, gy = 0;
            for (int ky = 0; ky < 3; ky++) {
                for (int kx = 0; kx < 3; kx++) {
                    int pixel = in[(y + ky - 1) * width + (x + kx - 1)];
                    gx += SOBEL_X[ky][kx] * pixel;
                    gy += SOBEL_Y[ky][kx] * pixel;
                }
            }
            out[y * width + x] = clamp_byte(sqrtf((float)(gx * gx + gy * gy)));
        }
    }
    for (int x = 0; x < width; x++) {
        out[0 * width + x]            = 0;
        out[(height-1) * width + x]   = 0;
    }
    for (int y = 0; y < height; y++) {
        out[y * width + 0]            = 0;
        out[y * width + (width - 1)]  = 0;
    }
}

/* ================================================================== */
/*  FILTER 3 -- Image Sharpening (unsharp mask)                        */
/* ================================================================== */
static void apply_sharpen(const unsigned char *in, unsigned char *out,
                           int width, int height)
{
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int sum = 0;
            for (int ky = 0; ky < 3; ky++)
                for (int kx = 0; kx < 3; kx++)
                    sum += SHARPEN_K[ky][kx] * (int)in[(y + ky - 1) * width + (x + kx - 1)];
            out[y * width + x] = clamp_byte((float)sum);
        }
    }
    for (int x = 0; x < width; x++) {
        out[0 * width + x]           = in[0 * width + x];
        out[(height-1) * width + x]  = in[(height-1) * width + x];
    }
    for (int y = 0; y < height; y++) {
        out[y * width + 0]           = in[y * width + 0];
        out[y * width + (width - 1)] = in[y * width + (width - 1)];
    }
}

/* ================================================================== */
/*  FILTER 4 -- Grayscale Conversion (BT.601)                          */
/* ================================================================== */
static void apply_grayscale(const unsigned char *rgb, unsigned char *gray,
                             int width, int height)
{
    int pixels = width * height;
    for (int i = 0; i < pixels; i++) {
        float r = rgb[i * 3 + 0], g = rgb[i * 3 + 1], b = rgb[i * 3 + 2];
        gray[i] = clamp_byte(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

/* ================================================================== */
/*  Real-world dataset throughput mode: loop over a directory of many   */
/*  real photographs (img_%05d.pgm / .ppm) instead of one image pair.   */
/*  Only the first WRITE_LIMIT images' outputs are written to disk --   */
/*  beyond that, writing thousands of result images would add disk I/O  */
/*  time that is not part of what we are trying to measure (filter      */
/*  throughput), since I/O is already excluded from every other timing  */
/*  in this project.                                                    */
/* ================================================================== */
#define WRITE_LIMIT 100

static void run_dataset_batch(const char *dir, int count, const char *out_dir)
{
    char path[1024];
    double t_gauss = 0, t_sobel = 0, t_sharp = 0, t_gray = 0;
    int gw = 0;

    for (int i = 0; i < count; i++) {
        int w, h, c, rw, rh, rc;
        snprintf(path, sizeof(path), "%s/img_%05d.pgm", dir, i);
        unsigned char *gray_in = read_pnm(path, &w, &h, &c);
        snprintf(path, sizeof(path), "%s/img_%05d.ppm", dir, i);
        unsigned char *rgb_in = read_pnm(path, &rw, &rh, &rc);
        gw = w;

        unsigned char *out_g = (unsigned char *)malloc((size_t)w * h);
        unsigned char *out_r = (unsigned char *)malloc((size_t)rw * rh);
        double t0, t1;
        int write_this = (i < WRITE_LIMIT);

        t0 = now_seconds(); apply_gaussian_blur(gray_in, out_g, w, h); t1 = now_seconds();
        t_gauss += t1 - t0;
        if (write_this) { snprintf(path, sizeof(path), "%s/batch_%05d_gaussian.pgm", out_dir, i); write_pnm(path, out_g, w, h, 1); }

        t0 = now_seconds(); apply_sobel(gray_in, out_g, w, h); t1 = now_seconds();
        t_sobel += t1 - t0;
        if (write_this) { snprintf(path, sizeof(path), "%s/batch_%05d_sobel.pgm", out_dir, i); write_pnm(path, out_g, w, h, 1); }

        t0 = now_seconds(); apply_sharpen(gray_in, out_g, w, h); t1 = now_seconds();
        t_sharp += t1 - t0;
        if (write_this) { snprintf(path, sizeof(path), "%s/batch_%05d_sharpen.pgm", out_dir, i); write_pnm(path, out_g, w, h, 1); }

        t0 = now_seconds(); apply_grayscale(rgb_in, out_r, rw, rh); t1 = now_seconds();
        t_gray += t1 - t0;
        if (write_this) { snprintf(path, sizeof(path), "%s/batch_%05d_grayscale.pgm", out_dir, i); write_pnm(path, out_r, rw, rh, 1); }

        free(gray_in); free(rgb_in); free(out_g); free(out_r);
    }

    printf("RESULT,sequential,gaussian_blur_batch,1,%d,%d,%.6f\n", gw, count, t_gauss);
    printf("RESULT,sequential,sobel_batch,1,%d,%d,%.6f\n",         gw, count, t_sobel);
    printf("RESULT,sequential,sharpen_batch,1,%d,%d,%.6f\n",       gw, count, t_sharp);
    printf("RESULT,sequential,grayscale_batch,1,%d,%d,%.6f\n",     gw, count, t_gray);
}

/* ================================================================== */
/*  Main                                                                */
/* ================================================================== */
int main(int argc, char *argv[])
{
    if (argc >= 5 && strcmp(argv[1], "--dataset") == 0) {
        const char *dir = argv[2];
        int count = atoi(argv[3]);
        const char *out_dir = argv[4];
        run_dataset_batch(dir, count, out_dir);
        return 0;
    }

    if (argc < 4) {
        fprintf(stderr, "Usage: %s <gray.pgm> <rgb.ppm> <output_dir> [tag]\n", argv[0]);
        fprintf(stderr, "       %s --dataset <dir> <count> <output_dir>\n", argv[0]);
        return 1;
    }
    const char *gray_path = argv[1];
    const char *rgb_path  = argv[2];
    const char *out_dir   = argv[3];

    int gw, gh, gc, rw, rh, rc;
    unsigned char *gray_in = read_pnm(gray_path, &gw, &gh, &gc);
    unsigned char *rgb_in  = read_pnm(rgb_path,  &rw, &rh, &rc);

    fprintf(stderr, "[seq] gray=%dx%d rgb=%dx%d\n", gw, gh, rw, rh);

    unsigned char *out_g = (unsigned char *)malloc((size_t)gw * gh);
    unsigned char *out_r = (unsigned char *)malloc((size_t)rw * rh);
    char path[1024];
    double t0, t1;

    double best;

    best = 1e18;
    for (int trial = 0; trial < TRIALS; trial++) {
        t0 = now_seconds();
        apply_gaussian_blur(gray_in, out_g, gw, gh);
        t1 = now_seconds();
        if (t1 - t0 < best) best = t1 - t0;
    }
    printf("RESULT,sequential,gaussian_blur,1,%d,%d,%.6f\n", gw, gh, best);
    snprintf(path, sizeof(path), "%s/seq_gaussian.pgm", out_dir);
    write_pnm(path, out_g, gw, gh, 1);

    best = 1e18;
    for (int trial = 0; trial < TRIALS; trial++) {
        t0 = now_seconds();
        apply_sobel(gray_in, out_g, gw, gh);
        t1 = now_seconds();
        if (t1 - t0 < best) best = t1 - t0;
    }
    printf("RESULT,sequential,sobel,1,%d,%d,%.6f\n", gw, gh, best);
    snprintf(path, sizeof(path), "%s/seq_sobel.pgm", out_dir);
    write_pnm(path, out_g, gw, gh, 1);

    best = 1e18;
    for (int trial = 0; trial < TRIALS; trial++) {
        t0 = now_seconds();
        apply_sharpen(gray_in, out_g, gw, gh);
        t1 = now_seconds();
        if (t1 - t0 < best) best = t1 - t0;
    }
    printf("RESULT,sequential,sharpen,1,%d,%d,%.6f\n", gw, gh, best);
    snprintf(path, sizeof(path), "%s/seq_sharpen.pgm", out_dir);
    write_pnm(path, out_g, gw, gh, 1);

    best = 1e18;
    for (int trial = 0; trial < TRIALS; trial++) {
        t0 = now_seconds();
        apply_grayscale(rgb_in, out_r, rw, rh);
        t1 = now_seconds();
        if (t1 - t0 < best) best = t1 - t0;
    }
    printf("RESULT,sequential,grayscale,1,%d,%d,%.6f\n", rw, rh, best);
    snprintf(path, sizeof(path), "%s/seq_grayscale.pgm", out_dir);
    write_pnm(path, out_r, rw, rh, 1);

    free(gray_in); free(rgb_in); free(out_g); free(out_r);
    return 0;
}
