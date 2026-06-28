/*
 * main_pth.c -- Pthreads (shared-memory) version of the Parallel Image
 * Filter and Processor project.
 *
 * Decomposition: each filter's row range is split into NTHREADS contiguous,
 * disjoint row-blocks. Because all threads share the same address space,
 * every thread can read any row of the (read-only) input buffer directly --
 * there is no need to copy halo/border rows as in the MPI version. Since
 * each thread writes only to its own disjoint slice of the output buffer,
 * no mutexes or atomics are required; the only synchronization point is the
 * implicit barrier formed by pthread_join() on all worker threads.
 *
 * Usage:
 *   ./main_pth <gray.pgm> <rgb.ppm> <output_dir> <num_threads>
 *
 * Prints one CSV line per filter to stdout:
 *   RESULT,pthreads,<filter>,<num_threads>,<width>,<height>,<seconds>
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <pthread.h>

/* See main_seq.c for why each filter call is timed TRIALS times and the
 * minimum is kept. */
#define TRIALS 3

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

static double now_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

/* ---- NetPBM I/O (identical to sequential version) ------------------ */
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
    fgetc(f);
    size_t n = (size_t)(*w) * (size_t)(*h) * (size_t)(*channels);
    unsigned char *buf = (unsigned char *)malloc(n);
    if (!buf) { fprintf(stderr, "ERROR: malloc failed reading %s\n", path); exit(1); }
    if (fread(buf, 1, n, f) != n) { fprintf(stderr, "ERROR: short read on %s\n", path); exit(1); }
    fclose(f);
    return buf;
}

static void write_pnm(const char *path, const unsigned char *buf, int w, int h, int channels)
{
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "ERROR: cannot create %s\n", path); exit(1); }
    fprintf(f, "%s\n%d %d\n255\n", channels == 1 ? "P5" : "P6", w, h);
    fwrite(buf, 1, (size_t)w * h * channels, f);
    fclose(f);
}

/* ---- Per-thread work descriptor -------------------------------------- */
typedef struct {
    const unsigned char *in;
    unsigned char *out;
    int width, height;
    int row_start, row_end;   /* [row_start, row_end) -- this thread's slice */
} task_t;

static void *worker_gaussian(void *arg)
{
    task_t *t = (task_t *)arg;
    int y0 = t->row_start < GAUSS_HALF ? GAUSS_HALF : t->row_start;
    int y1 = t->row_end > t->height - GAUSS_HALF ? t->height - GAUSS_HALF : t->row_end;
    for (int y = y0; y < y1; y++) {
        for (int x = GAUSS_HALF; x < t->width - GAUSS_HALF; x++) {
            float sum = 0.0f;
            for (int ky = 0; ky < GAUSS_K; ky++)
                for (int kx = 0; kx < GAUSS_K; kx++)
                    sum += (float)t->in[(y + ky - GAUSS_HALF) * t->width + (x + kx - GAUSS_HALF)]
                           * GAUSS[ky][kx];
            t->out[y * t->width + x] = clamp_byte(sum / GAUSS_SUM);
        }
    }
    return NULL;
}

static void *worker_sobel(void *arg)
{
    task_t *t = (task_t *)arg;
    int y0 = t->row_start < 1 ? 1 : t->row_start;
    int y1 = t->row_end > t->height - 1 ? t->height - 1 : t->row_end;
    for (int y = y0; y < y1; y++) {
        for (int x = 1; x < t->width - 1; x++) {
            int gx = 0, gy = 0;
            for (int ky = 0; ky < 3; ky++)
                for (int kx = 0; kx < 3; kx++) {
                    int pixel = t->in[(y + ky - 1) * t->width + (x + kx - 1)];
                    gx += SOBEL_X[ky][kx] * pixel;
                    gy += SOBEL_Y[ky][kx] * pixel;
                }
            t->out[y * t->width + x] = clamp_byte(sqrtf((float)(gx * gx + gy * gy)));
        }
    }
    return NULL;
}

static void *worker_sharpen(void *arg)
{
    task_t *t = (task_t *)arg;
    int y0 = t->row_start < 1 ? 1 : t->row_start;
    int y1 = t->row_end > t->height - 1 ? t->height - 1 : t->row_end;
    for (int y = y0; y < y1; y++) {
        for (int x = 1; x < t->width - 1; x++) {
            int sum = 0;
            for (int ky = 0; ky < 3; ky++)
                for (int kx = 0; kx < 3; kx++)
                    sum += SHARPEN_K[ky][kx] * (int)t->in[(y + ky - 1) * t->width + (x + kx - 1)];
            t->out[y * t->width + x] = clamp_byte((float)sum);
        }
    }
    return NULL;
}

static void *worker_grayscale(void *arg)
{
    task_t *t = (task_t *)arg;
    for (int y = t->row_start; y < t->row_end; y++) {
        for (int x = 0; x < t->width; x++) {
            int i = y * t->width + x;
            float r = t->in[i * 3 + 0], g = t->in[i * 3 + 1], b = t->in[i * 3 + 2];
            t->out[i] = clamp_byte(0.299f * r + 0.587f * g + 0.114f * b);
        }
    }
    return NULL;
}

/* Copies border rows/cols that worker_* deliberately skip (interior-only loops). */
static void copy_borders(const unsigned char *in, unsigned char *out, int width, int height, int half)
{
    for (int x = 0; x < width; x++)
        for (int b = 0; b < half; b++) {
            out[b * width + x]            = in[b * width + x];
            out[(height-1-b) * width + x] = in[(height-1-b) * width + x];
        }
    for (int y = 0; y < height; y++)
        for (int b = 0; b < half; b++) {
            out[y * width + b]           = in[y * width + b];
            out[y * width + (width-1-b)] = in[y * width + (width-1-b)];
        }
}

static void zero_borders(unsigned char *out, int width, int height)
{
    for (int x = 0; x < width; x++) { out[x] = 0; out[(height-1)*width + x] = 0; }
    for (int y = 0; y < height; y++) { out[y*width] = 0; out[y*width + width - 1] = 0; }
}

enum border_mode { BORDER_NONE, BORDER_COPY_1, BORDER_COPY_2, BORDER_ZERO };

/* Spawns nthreads workers over [0,height) using fn, joins them, applies the
 * border treatment, and times the whole thing -- mirroring exactly what the
 * sequential apply_*() functions do internally, so timings are comparable. */
static double run_parallel(void *(*fn)(void *), const unsigned char *in, unsigned char *out,
                            int width, int height, int nthreads, enum border_mode bm)
{
    pthread_t *tids = (pthread_t *)malloc(sizeof(pthread_t) * nthreads);
    task_t   *tasks = (task_t   *)malloc(sizeof(task_t)   * nthreads);
    int base = height / nthreads, rem = height % nthreads, row = 0;

    double t0 = now_seconds();
    for (int i = 0; i < nthreads; i++) {
        int rows = base + (i < rem ? 1 : 0);
        tasks[i] = (task_t){ in, out, width, height, row, row + rows };
        row += rows;
        pthread_create(&tids[i], NULL, fn, &tasks[i]);
    }
    for (int i = 0; i < nthreads; i++)
        pthread_join(tids[i], NULL);

    switch (bm) {
        case BORDER_COPY_1: copy_borders(in, out, width, height, 1); break;
        case BORDER_COPY_2: copy_borders(in, out, width, height, GAUSS_HALF); break;
        case BORDER_ZERO:   zero_borders(out, width, height); break;
        default: break;
    }
    double t1 = now_seconds();

    free(tids); free(tasks);
    return t1 - t0;
}

/* See main_seq.c's run_dataset_batch() for the rationale: loop over a
 * directory of many real photographs in one process, reusing the same
 * run_parallel() helper (and its per-call thread pool) per image, and
 * accumulate total filter time across the whole batch. Only the first
 * WRITE_LIMIT images' outputs are written, to keep batch I/O cheap. */
#define WRITE_LIMIT 100

static void run_dataset_batch(const char *dir, int count, const char *out_dir, int nthreads)
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
        int write_this = (i < WRITE_LIMIT);

        t_gauss += run_parallel(worker_gaussian, gray_in, out_g, w, h, nthreads, BORDER_COPY_2);
        if (write_this) { snprintf(path, sizeof(path), "%s/batch_%05d_gaussian.pgm", out_dir, i); write_pnm(path, out_g, w, h, 1); }

        t_sobel += run_parallel(worker_sobel, gray_in, out_g, w, h, nthreads, BORDER_ZERO);
        if (write_this) { snprintf(path, sizeof(path), "%s/batch_%05d_sobel.pgm", out_dir, i); write_pnm(path, out_g, w, h, 1); }

        t_sharp += run_parallel(worker_sharpen, gray_in, out_g, w, h, nthreads, BORDER_COPY_1);
        if (write_this) { snprintf(path, sizeof(path), "%s/batch_%05d_sharpen.pgm", out_dir, i); write_pnm(path, out_g, w, h, 1); }

        t_gray += run_parallel(worker_grayscale, rgb_in, out_r, rw, rh, nthreads, BORDER_NONE);
        if (write_this) { snprintf(path, sizeof(path), "%s/batch_%05d_grayscale.pgm", out_dir, i); write_pnm(path, out_r, rw, rh, 1); }

        free(gray_in); free(rgb_in); free(out_g); free(out_r);
    }

    printf("RESULT,pthreads,gaussian_blur_batch,%d,%d,%d,%.6f\n", nthreads, gw, count, t_gauss);
    printf("RESULT,pthreads,sobel_batch,%d,%d,%d,%.6f\n",         nthreads, gw, count, t_sobel);
    printf("RESULT,pthreads,sharpen_batch,%d,%d,%d,%.6f\n",       nthreads, gw, count, t_sharp);
    printf("RESULT,pthreads,grayscale_batch,%d,%d,%d,%.6f\n",     nthreads, gw, count, t_gray);
}

int main(int argc, char *argv[])
{
    if (argc >= 6 && strcmp(argv[1], "--dataset") == 0) {
        const char *dir = argv[2];
        int count = atoi(argv[3]);
        const char *out_dir = argv[4];
        int nthreads = atoi(argv[5]);
        if (nthreads < 1) nthreads = 1;
        run_dataset_batch(dir, count, out_dir, nthreads);
        return 0;
    }

    if (argc < 5) {
        fprintf(stderr, "Usage: %s <gray.pgm> <rgb.ppm> <output_dir> <num_threads>\n", argv[0]);
        fprintf(stderr, "       %s --dataset <dir> <count> <output_dir> <num_threads>\n", argv[0]);
        return 1;
    }
    const char *gray_path = argv[1];
    const char *rgb_path  = argv[2];
    const char *out_dir   = argv[3];
    int nthreads = atoi(argv[4]);
    if (nthreads < 1) nthreads = 1;

    int gw, gh, gc, rw, rh, rc;
    unsigned char *gray_in = read_pnm(gray_path, &gw, &gh, &gc);
    unsigned char *rgb_in  = read_pnm(rgb_path,  &rw, &rh, &rc);

    unsigned char *out_g = (unsigned char *)malloc((size_t)gw * gh);
    unsigned char *out_r = (unsigned char *)malloc((size_t)rw * rh);
    char path[1024];
    double dt, best;

    best = 1e18;
    for (int trial = 0; trial < TRIALS; trial++) {
        dt = run_parallel(worker_gaussian, gray_in, out_g, gw, gh, nthreads, BORDER_COPY_2);
        if (dt < best) best = dt;
    }
    printf("RESULT,pthreads,gaussian_blur,%d,%d,%d,%.6f\n", nthreads, gw, gh, best);
    snprintf(path, sizeof(path), "%s/pth_gaussian.pgm", out_dir);
    write_pnm(path, out_g, gw, gh, 1);

    best = 1e18;
    for (int trial = 0; trial < TRIALS; trial++) {
        dt = run_parallel(worker_sobel, gray_in, out_g, gw, gh, nthreads, BORDER_ZERO);
        if (dt < best) best = dt;
    }
    printf("RESULT,pthreads,sobel,%d,%d,%d,%.6f\n", nthreads, gw, gh, best);
    snprintf(path, sizeof(path), "%s/pth_sobel.pgm", out_dir);
    write_pnm(path, out_g, gw, gh, 1);

    best = 1e18;
    for (int trial = 0; trial < TRIALS; trial++) {
        dt = run_parallel(worker_sharpen, gray_in, out_g, gw, gh, nthreads, BORDER_COPY_1);
        if (dt < best) best = dt;
    }
    printf("RESULT,pthreads,sharpen,%d,%d,%d,%.6f\n", nthreads, gw, gh, best);
    snprintf(path, sizeof(path), "%s/pth_sharpen.pgm", out_dir);
    write_pnm(path, out_g, gw, gh, 1);

    best = 1e18;
    for (int trial = 0; trial < TRIALS; trial++) {
        dt = run_parallel(worker_grayscale, rgb_in, out_r, rw, rh, nthreads, BORDER_NONE);
        if (dt < best) best = dt;
    }
    printf("RESULT,pthreads,grayscale,%d,%d,%d,%.6f\n", nthreads, rw, rh, best);
    snprintf(path, sizeof(path), "%s/pth_grayscale.pgm", out_dir);
    write_pnm(path, out_r, rw, rh, 1);

    free(gray_in); free(rgb_in); free(out_g); free(out_r);
    return 0;
}
