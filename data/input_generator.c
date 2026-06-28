/*
 * input_generator.c -- builds the full benchmark dataset used by every
 * implementation (sequential/pthreads/openmp/mpi) from two real source
 * photographs checked in under data/base/:
 *
 *   data/base/camera_512.pgm      512x512 grayscale ("cameraman", a classic
 *                                  image-processing benchmark photo)
 *   data/base/astronaut_512.ppm   512x512 RGB photograph
 *
 * Both are real photographs distributed with scikit-image under a
 * BSD-style license (see README.md / paper for citation), not synthetic
 * noise -- this matters because Gaussian blur / Sobel / sharpen produce
 * meaningless results on uncorrelated random pixels.
 *
 * This program bilinearly resizes those two source images to every
 * resolution required for the strong-scaling sweep (N ~ 10^3 .. ~10^7
 * pixels) and the weak-scaling sweep (work-per-worker held constant as
 * worker count grows), and writes the results as binary PGM/PPM into
 * data/generated/. Resizing -- rather than generating noise at each
 * resolution -- means every test image still has real photographic
 * structure, at every scale.
 *
 * Usage:
 *   ./input_generator <base_gray.pgm> <base_rgb.ppm> <output_dir>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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

/* Standard pixel-center bilinear resize, works for channels = 1 or 3. */
static unsigned char *resize_bilinear(const unsigned char *src, int sw, int sh, int channels,
                                       int dw, int dh)
{
    unsigned char *dst = (unsigned char *)malloc((size_t)dw * dh * channels);
    float scale_x = (float)sw / dw;
    float scale_y = (float)sh / dh;

    for (int y = 0; y < dh; y++) {
        float sy = (y + 0.5f) * scale_y - 0.5f;
        if (sy < 0) sy = 0;
        if (sy > sh - 1) sy = (float)(sh - 1);
        int y0 = (int)sy, y1 = (y0 + 1 < sh) ? y0 + 1 : y0;
        float fy = sy - y0;

        for (int x = 0; x < dw; x++) {
            float sx = (x + 0.5f) * scale_x - 0.5f;
            if (sx < 0) sx = 0;
            if (sx > sw - 1) sx = (float)(sw - 1);
            int x0 = (int)sx, x1 = (x0 + 1 < sw) ? x0 + 1 : x0;
            float fx = sx - x0;

            for (int c = 0; c < channels; c++) {
                float p00 = src[(y0 * sw + x0) * channels + c];
                float p01 = src[(y0 * sw + x1) * channels + c];
                float p10 = src[(y1 * sw + x0) * channels + c];
                float p11 = src[(y1 * sw + x1) * channels + c];
                float top = p00 + (p01 - p00) * fx;
                float bot = p10 + (p11 - p10) * fx;
                float val = top + (bot - top) * fy;
                dst[(y * dw + x) * channels + c] = (unsigned char)(val + 0.5f);
            }
        }
    }
    return dst;
}

typedef struct { int w, h; const char *tag; } size_spec_t;

int main(int argc, char *argv[])
{
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <base_gray.pgm> <base_rgb.ppm> <output_dir>\n", argv[0]);
        return 1;
    }
    const char *gray_path = argv[1];
    const char *rgb_path  = argv[2];
    const char *out_dir   = argv[3];

    int gw, gh, gc, rw, rh, rc;
    unsigned char *gray_src = read_pnm(gray_path, &gw, &gh, &gc);
    unsigned char *rgb_src  = read_pnm(rgb_path,  &rw, &rh, &rc);
    if (gc != 1 || rc != 3) {
        fprintf(stderr, "ERROR: expected grayscale base (P5) and RGB base (P6)\n");
        return 1;
    }

    /* ---- Strong scaling sweep: fixed problem size, N ~ 10^3..10^7 px ---- */
    size_spec_t strong[] = {
        {  64,   64, "0000064x000064"},
        { 128,  128, "0000128x000128"},
        { 256,  256, "0000256x000256"},
        { 512,  512, "0000512x000512"},
        {1024, 1024, "0001024x001024"},
        {1920, 1080, "0001920x001080"},
        {2560, 1440, "0002560x001440"},
        {3840, 2160, "0003840x002160"},
        {5760, 3240, "0005760x003240"},
    };
    int n_strong = sizeof(strong) / sizeof(strong[0]);

    char path[1024];
    for (int i = 0; i < n_strong; i++) {
        unsigned char *g = resize_bilinear(gray_src, gw, gh, 1, strong[i].w, strong[i].h);
        unsigned char *r = resize_bilinear(rgb_src,  rw, rh, 3, strong[i].w, strong[i].h);
        snprintf(path, sizeof(path), "%s/strong_gray_%s.pgm", out_dir, strong[i].tag);
        write_pnm(path, g, strong[i].w, strong[i].h, 1);
        snprintf(path, sizeof(path), "%s/strong_rgb_%s.ppm", out_dir, strong[i].tag);
        write_pnm(path, r, strong[i].w, strong[i].h, 3);
        printf("generated %s : %dx%d (%ld px)\n", strong[i].tag, strong[i].w, strong[i].h,
               (long)strong[i].w * strong[i].h);
        free(g); free(r);
    }

    /* ---- Weak scaling sweep: width fixed, height grows with worker count
     *      so that rows-per-worker (and hence work-per-worker) stays fixed. */
    const int weak_width = 1920;
    const int rows_per_worker = 270;
    int worker_counts[] = {1, 2, 4, 8};
    int n_weak = sizeof(worker_counts) / sizeof(worker_counts[0]);

    for (int i = 0; i < n_weak; i++) {
        int p = worker_counts[i];
        int h = rows_per_worker * p;
        unsigned char *g = resize_bilinear(gray_src, gw, gh, 1, weak_width, h);
        unsigned char *r = resize_bilinear(rgb_src,  rw, rh, 3, weak_width, h);
        snprintf(path, sizeof(path), "%s/weak_gray_p%d.pgm", out_dir, p);
        write_pnm(path, g, weak_width, h, 1);
        snprintf(path, sizeof(path), "%s/weak_rgb_p%d.ppm", out_dir, p);
        write_pnm(path, r, weak_width, h, 3);
        printf("generated weak_p%d : %dx%d (%ld px, %d rows/worker)\n", p, weak_width, h,
               (long)weak_width * h, rows_per_worker);
        free(g); free(r);
    }

    free(gray_src); free(rgb_src);
    return 0;
}
