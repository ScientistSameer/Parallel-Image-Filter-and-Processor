/*
 * main_mpi.c -- MPI (distributed-memory) version of the Parallel Image
 * Filter and Processor project.
 *
 * Decomposition: rank 0 reads the input image(s) from disk and distributes
 * horizontal row-strips to every rank with MPI_Scatterv. Because processes
 * do NOT share memory, each convolution filter additionally needs "halo"
 * (ghost) rows from its neighbours before it can correctly compute pixels
 * near a strip boundary; these are exchanged with MPI_Sendrecv, mirroring
 * the boundary at the top/bottom edges of the whole image. After filtering,
 * MPI_Gatherv reassembles the full output on rank 0.
 *
 * MPI primitives used: MPI_Init/Finalize, MPI_Comm_rank/size, MPI_Bcast,
 * MPI_Scatterv, MPI_Sendrecv, MPI_Gatherv, MPI_Barrier, MPI_Wtime.
 *
 * Usage:
 *   mpirun -np P ./main_mpi <gray.pgm> <rgb.ppm> <output_dir>
 *
 * Rank 0 prints one CSV line per filter to stdout:
 *   RESULT,mpi,<filter>,<P>,<width>,<height>,<seconds>
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mpi.h>

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

#define TAG_UP   10
#define TAG_DOWN 20

static inline unsigned char clamp_byte(float v)
{
    if (v <   0.0f) return 0;
    if (v > 255.0f) return 255;
    return (unsigned char)v;
}

/* ---- NetPBM I/O (rank 0 only) --------------------------------------- */
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
    if (!f) { fprintf(stderr, "ERROR: cannot open %s\n", path); MPI_Abort(MPI_COMM_WORLD, 1); }
    char magic[3] = {0};
    if (fscanf(f, "%2s", magic) != 1) { fprintf(stderr, "ERROR: bad header %s\n", path); MPI_Abort(MPI_COMM_WORLD, 1); }
    if (strcmp(magic, "P5") == 0) *channels = 1;
    else if (strcmp(magic, "P6") == 0) *channels = 3;
    else { fprintf(stderr, "ERROR: unsupported format %s in %s\n", magic, path); MPI_Abort(MPI_COMM_WORLD, 1); }
    int maxval;
    skip_ws_comments(f); if (fscanf(f, "%d", w) != 1)       { fprintf(stderr, "ERROR: bad header %s\n", path); MPI_Abort(MPI_COMM_WORLD, 1); }
    skip_ws_comments(f); if (fscanf(f, "%d", h) != 1)       { fprintf(stderr, "ERROR: bad header %s\n", path); MPI_Abort(MPI_COMM_WORLD, 1); }
    skip_ws_comments(f); if (fscanf(f, "%d", &maxval) != 1) { fprintf(stderr, "ERROR: bad header %s\n", path); MPI_Abort(MPI_COMM_WORLD, 1); }
    fgetc(f);
    size_t n = (size_t)(*w) * (size_t)(*h) * (size_t)(*channels);
    unsigned char *buf = (unsigned char *)malloc(n);
    if (!buf) { fprintf(stderr, "ERROR: malloc failed reading %s\n", path); MPI_Abort(MPI_COMM_WORLD, 1); }
    if (fread(buf, 1, n, f) != n) { fprintf(stderr, "ERROR: short read on %s\n", path); MPI_Abort(MPI_COMM_WORLD, 1); }
    fclose(f);
    return buf;
}

static void write_pnm(const char *path, const unsigned char *buf, int w, int h, int channels)
{
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "ERROR: cannot create %s\n", path); MPI_Abort(MPI_COMM_WORLD, 1); }
    fprintf(f, "%s\n%d %d\n255\n", channels == 1 ? "P5" : "P6", w, h);
    fwrite(buf, 1, (size_t)w * h * channels, f);
    fclose(f);
}

/* ================================================================== */
/*  Filter kernels -- operate on an extended strip [halo|core|halo]    */
/* ================================================================== */
typedef void (*filter_fn_t)(const unsigned char *, unsigned char *, int, int);

static void apply_gaussian_blur(const unsigned char *in, unsigned char *out, int width, int height)
{
    for (int y = GAUSS_HALF; y < height - GAUSS_HALF; y++)
        for (int x = GAUSS_HALF; x < width - GAUSS_HALF; x++) {
            float sum = 0.0f;
            for (int ky = 0; ky < GAUSS_K; ky++)
                for (int kx = 0; kx < GAUSS_K; kx++)
                    sum += (float)in[(y + ky - GAUSS_HALF) * width + (x + kx - GAUSS_HALF)] * GAUSS[ky][kx];
            out[y * width + x] = clamp_byte(sum / GAUSS_SUM);
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

static void apply_sobel(const unsigned char *in, unsigned char *out, int width, int height)
{
    for (int y = 1; y < height - 1; y++)
        for (int x = 1; x < width - 1; x++) {
            int gx = 0, gy = 0;
            for (int ky = 0; ky < 3; ky++)
                for (int kx = 0; kx < 3; kx++) {
                    int pixel = in[(y + ky - 1) * width + (x + kx - 1)];
                    gx += SOBEL_X[ky][kx] * pixel;
                    gy += SOBEL_Y[ky][kx] * pixel;
                }
            out[y * width + x] = clamp_byte(sqrtf((float)(gx * gx + gy * gy)));
        }
    for (int x = 0; x < width; x++) { out[x] = 0; out[(height-1)*width + x] = 0; }
    for (int y = 0; y < height; y++) { out[y*width] = 0; out[y*width + width - 1] = 0; }
}

static void apply_sharpen(const unsigned char *in, unsigned char *out, int width, int height)
{
    for (int y = 1; y < height - 1; y++)
        for (int x = 1; x < width - 1; x++) {
            int sum = 0;
            for (int ky = 0; ky < 3; ky++)
                for (int kx = 0; kx < 3; kx++)
                    sum += SHARPEN_K[ky][kx] * (int)in[(y + ky - 1) * width + (x + kx - 1)];
            out[y * width + x] = clamp_byte((float)sum);
        }
    for (int x = 0; x < width; x++) {
        out[x] = in[x];
        out[(height-1)*width + x] = in[(height-1)*width + x];
    }
    for (int y = 0; y < height; y++) {
        out[y*width] = in[y*width];
        out[y*width + width - 1] = in[y*width + width - 1];
    }
}

static void apply_grayscale(const unsigned char *rgb, unsigned char *gray, int width, int height)
{
    int pixels = width * height;
    for (int i = 0; i < pixels; i++) {
        float r = rgb[i*3+0], g = rgb[i*3+1], b = rgb[i*3+2];
        gray[i] = clamp_byte(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

/* ================================================================== */
/*  exchange_halos -- fills top/bottom ghost rows via MPI_Sendrecv,    */
/*  mirroring the image edge at rank 0's top and the last rank's bottom*/
/* ================================================================== */
static void exchange_halos(unsigned char *ext, int local_rows, int halo,
                            int rank, int size, int width)
{
    if (halo == 0) return;
    int halo_bytes = halo * width;
    unsigned char *top_halo   = ext;
    unsigned char *first_core = ext + (size_t)halo * width;
    unsigned char *last_core_block = ext + (size_t)local_rows * width;
    unsigned char *bot_halo   = ext + (size_t)(halo + local_rows) * width;

    if (rank > 0)
        MPI_Sendrecv(first_core, halo_bytes, MPI_UNSIGNED_CHAR, rank - 1, TAG_UP,
                     top_halo,   halo_bytes, MPI_UNSIGNED_CHAR, rank - 1, TAG_DOWN,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    else
        for (int i = 0; i < halo; i++)
            memcpy(top_halo + (size_t)i * width, first_core, width);

    if (rank < size - 1)
        MPI_Sendrecv(last_core_block, halo_bytes, MPI_UNSIGNED_CHAR, rank + 1, TAG_DOWN,
                     bot_halo,        halo_bytes, MPI_UNSIGNED_CHAR, rank + 1, TAG_UP,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    else {
        unsigned char *last_row = ext + (size_t)(halo + local_rows - 1) * width;
        for (int i = 0; i < halo; i++)
            memcpy(bot_halo + (size_t)i * width, last_row, width);
    }
}

static void build_distribution(int height, int width, int size,
                                int *row_counts, int *row_displs, int *sendcounts, int *displs)
{
    int base = height / size, rem = height % size, offset = 0;
    for (int i = 0; i < size; i++) {
        row_counts[i] = base + (i < rem ? 1 : 0);
        row_displs[i] = offset;
        sendcounts[i] = row_counts[i] * width;
        displs[i]     = row_displs[i] * width;
        offset       += row_counts[i];
    }
}

enum border_mode { BORDER_COPY, BORDER_ZERO };

/* Image-edge (not inter-rank halo) border treatment, applied once on rank 0
 * after the gather. exchange_halos() mirrors rows at the true top/bottom
 * edge of the image so every rank can run the *same* interior convolution
 * loop with no special-casing -- but that mirroring is only a computational
 * convenience. The sequential/Pthreads/OpenMP versions all leave genuine
 * image-edge rows untouched (copied straight from the input, or zeroed for
 * Sobel) rather than convolving them against a mirrored reflection. Without
 * this fix-up the MPI version would disagree with the other three at every
 * image edge, even at -np 1. */
static void fix_global_borders(const unsigned char *full_in, unsigned char *full_out,
                                int width, int height, int halo, enum border_mode bm)
{
    if (bm == BORDER_ZERO) {
        for (int x = 0; x < width; x++) { full_out[x] = 0; full_out[(height-1)*width + x] = 0; }
        for (int y = 0; y < height; y++) { full_out[y*width] = 0; full_out[y*width + width - 1] = 0; }
        return;
    }
    for (int x = 0; x < width; x++)
        for (int b = 0; b < halo; b++) {
            full_out[b * width + x]            = full_in[b * width + x];
            full_out[(height-1-b) * width + x] = full_in[(height-1-b) * width + x];
        }
    for (int y = 0; y < height; y++)
        for (int b = 0; b < halo; b++) {
            full_out[y * width + b]           = full_in[y * width + b];
            full_out[y * width + (width-1-b)] = full_in[y * width + (width-1-b)];
        }
}

/* Runs one convolution filter end-to-end (scatter -> halo -> filter -> gather), timed. */
static double run_mpi_conv_filter(filter_fn_t fn, int halo, enum border_mode bm,
                                   const unsigned char *full_in, unsigned char *full_out,
                                   int width, int height, int rank, int size,
                                   const int *sendcounts, const int *displs, int local_rows)
{
    int ext_rows = local_rows + 2 * halo;
    unsigned char *local_core = (unsigned char *)malloc((size_t)local_rows * width);
    unsigned char *ext_in     = (unsigned char *)malloc((size_t)ext_rows   * width);
    unsigned char *ext_out    = (unsigned char *)malloc((size_t)ext_rows   * width);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    MPI_Scatterv(full_in, sendcounts, displs, MPI_UNSIGNED_CHAR,
                 local_core, local_rows * width, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    memcpy(ext_in + (size_t)halo * width, local_core, (size_t)local_rows * width);
    exchange_halos(ext_in, local_rows, halo, rank, size, width);

    fn(ext_in, ext_out, width, ext_rows);

    MPI_Gatherv(ext_out + (size_t)halo * width, local_rows * width, MPI_UNSIGNED_CHAR,
                full_out, sendcounts, displs, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    if (rank == 0)
        fix_global_borders(full_in, full_out, width, height, halo, bm);

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    free(local_core); free(ext_in); free(ext_out);
    return t1 - t0;
}

/* See main_seq.c's run_dataset_batch() for the rationale. Every rank loops
 * over the same directory of real photographs collectively -- each
 * iteration is one ordinary call to run_mpi_conv_filter() (a full
 * Scatterv -> halo exchange -> filter -> Gatherv -> border-fix round
 * trip), exactly as in the single-image path, just repeated count times
 * inside one mpirun session instead of relaunching mpirun per image
 * (which would make the per-launch MPI startup cost dominate the
 * measurement instead of the filter work itself). Only rank 0 writes the
 * first WRITE_LIMIT images' outputs, to keep batch I/O cheap. */
#define WRITE_LIMIT 100

static void run_dataset_batch(const char *dir, int count, const char *out_dir,
                               int rank, int size)
{
    char path[1024];
    double t_gauss = 0, t_sobel = 0, t_sharp = 0, t_gray = 0;
    int gw = 0;

    for (int i = 0; i < count; i++) {
        int gw_i = 0, gh_i = 0, rw_i = 0, rh_i = 0;
        unsigned char *full_gray = NULL, *full_rgb = NULL, *full_out = NULL;

        if (rank == 0) {
            int gc, rc;
            snprintf(path, sizeof(path), "%s/img_%05d.pgm", dir, i);
            full_gray = read_pnm(path, &gw_i, &gh_i, &gc);
            snprintf(path, sizeof(path), "%s/img_%05d.ppm", dir, i);
            full_rgb  = read_pnm(path, &rw_i, &rh_i, &rc);
        }
        MPI_Bcast(&gw_i, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&gh_i, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&rw_i, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&rh_i, 1, MPI_INT, 0, MPI_COMM_WORLD);
        gw = gw_i;
        if (rank == 0) full_out = (unsigned char *)malloc((size_t)gw_i * gh_i);

        int *row_counts = malloc(size * sizeof(int)), *row_displs = malloc(size * sizeof(int));
        int *sendcounts = malloc(size * sizeof(int)), *displs     = malloc(size * sizeof(int));
        build_distribution(gh_i, gw_i, size, row_counts, row_displs, sendcounts, displs);
        int local_rows = row_counts[rank];
        int write_this = (i < WRITE_LIMIT);

        t_gauss += run_mpi_conv_filter(apply_gaussian_blur, GAUSS_HALF, BORDER_COPY, full_gray, full_out, gw_i, gh_i, rank, size, sendcounts, displs, local_rows);
        if (rank == 0 && write_this) { snprintf(path, sizeof(path), "%s/batch_%05d_gaussian.pgm", out_dir, i); write_pnm(path, full_out, gw_i, gh_i, 1); }

        t_sobel += run_mpi_conv_filter(apply_sobel, 1, BORDER_ZERO, full_gray, full_out, gw_i, gh_i, rank, size, sendcounts, displs, local_rows);
        if (rank == 0 && write_this) { snprintf(path, sizeof(path), "%s/batch_%05d_sobel.pgm", out_dir, i); write_pnm(path, full_out, gw_i, gh_i, 1); }

        t_sharp += run_mpi_conv_filter(apply_sharpen, 1, BORDER_COPY, full_gray, full_out, gw_i, gh_i, rank, size, sendcounts, displs, local_rows);
        if (rank == 0 && write_this) { snprintf(path, sizeof(path), "%s/batch_%05d_sharpen.pgm", out_dir, i); write_pnm(path, full_out, gw_i, gh_i, 1); }

        /* Grayscale: no halo, separate RGB scatter/gather (same as single-image path). */
        {
            int *scR = malloc(size * sizeof(int)), *diR = malloc(size * sizeof(int));
            int *rcR = malloc(size * sizeof(int)), *rdR = malloc(size * sizeof(int));
            build_distribution(rh_i, rw_i, size, rcR, rdR, scR, diR);
            int *sc_rgb = malloc(size * sizeof(int)), *di_rgb = malloc(size * sizeof(int));
            for (int k = 0; k < size; k++) { sc_rgb[k] = scR[k] * 3; di_rgb[k] = diR[k] * 3; }
            int local_rows_r = rcR[rank];

            unsigned char *local_rgb  = malloc((size_t)local_rows_r * rw_i * 3);
            unsigned char *local_gray = malloc((size_t)local_rows_r * rw_i);
            unsigned char *full_gray_out = (rank == 0) ? malloc((size_t)rw_i * rh_i) : NULL;

            MPI_Barrier(MPI_COMM_WORLD);
            double t0 = MPI_Wtime();
            MPI_Scatterv(full_rgb, sc_rgb, di_rgb, MPI_UNSIGNED_CHAR,
                         local_rgb, local_rows_r * rw_i * 3, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
            apply_grayscale(local_rgb, local_gray, rw_i, local_rows_r);
            MPI_Gatherv(local_gray, local_rows_r * rw_i, MPI_UNSIGNED_CHAR,
                        full_gray_out, scR, diR, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);
            double t1 = MPI_Wtime();
            t_gray += t1 - t0;

            if (rank == 0) {
                if (write_this) { snprintf(path, sizeof(path), "%s/batch_%05d_grayscale.pgm", out_dir, i); write_pnm(path, full_gray_out, rw_i, rh_i, 1); }
                free(full_gray_out);
            }
            free(local_rgb); free(local_gray);
            free(sc_rgb); free(di_rgb); free(rcR); free(rdR); free(scR); free(diR);
        }

        if (rank == 0) { free(full_gray); free(full_rgb); free(full_out); }
        free(row_counts); free(row_displs); free(sendcounts); free(displs);
    }

    if (rank == 0) {
        printf("RESULT,mpi,gaussian_blur_batch,%d,%d,%d,%.6f\n", size, gw, count, t_gauss);
        printf("RESULT,mpi,sobel_batch,%d,%d,%d,%.6f\n",         size, gw, count, t_sobel);
        printf("RESULT,mpi,sharpen_batch,%d,%d,%d,%.6f\n",       size, gw, count, t_sharp);
        printf("RESULT,mpi,grayscale_batch,%d,%d,%d,%.6f\n",     size, gw, count, t_gray);
    }
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc >= 5 && strcmp(argv[1], "--dataset") == 0) {
        const char *dir = argv[2];
        int count = atoi(argv[3]);
        const char *out_dir = argv[4];
        run_dataset_batch(dir, count, out_dir, rank, size);
        MPI_Finalize();
        return 0;
    }

    if (argc < 4) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s <gray.pgm> <rgb.ppm> <output_dir>\n", argv[0]);
            fprintf(stderr, "       %s --dataset <dir> <count> <output_dir>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }
    const char *gray_path = argv[1];
    const char *rgb_path  = argv[2];
    const char *out_dir   = argv[3];

    int gw = 0, gh = 0, rw = 0, rh = 0;
    unsigned char *full_gray = NULL, *full_rgb = NULL, *full_out = NULL;

    if (rank == 0) {
        int gc, rc;
        full_gray = read_pnm(gray_path, &gw, &gh, &gc);
        full_rgb  = read_pnm(rgb_path,  &rw, &rh, &rc);
    }
    MPI_Bcast(&gw, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&gh, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&rw, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&rh, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) full_out = (unsigned char *)malloc((size_t)gw * gh);

    int *row_counts = malloc(size * sizeof(int)), *row_displs = malloc(size * sizeof(int));
    int *sendcounts = malloc(size * sizeof(int)), *displs     = malloc(size * sizeof(int));
    build_distribution(gh, gw, size, row_counts, row_displs, sendcounts, displs);
    int local_rows = row_counts[rank];

    double dt, best;
    char path[1024];

    best = 1e18;
    for (int trial = 0; trial < TRIALS; trial++) {
        dt = run_mpi_conv_filter(apply_gaussian_blur, GAUSS_HALF, BORDER_COPY, full_gray, full_out, gw, gh, rank, size, sendcounts, displs, local_rows);
        if (dt < best) best = dt;
    }
    if (rank == 0) {
        printf("RESULT,mpi,gaussian_blur,%d,%d,%d,%.6f\n", size, gw, gh, best);
        snprintf(path, sizeof(path), "%s/mpi_gaussian.pgm", out_dir);
        write_pnm(path, full_out, gw, gh, 1);
    }

    best = 1e18;
    for (int trial = 0; trial < TRIALS; trial++) {
        dt = run_mpi_conv_filter(apply_sobel, 1, BORDER_ZERO, full_gray, full_out, gw, gh, rank, size, sendcounts, displs, local_rows);
        if (dt < best) best = dt;
    }
    if (rank == 0) {
        printf("RESULT,mpi,sobel,%d,%d,%d,%.6f\n", size, gw, gh, best);
        snprintf(path, sizeof(path), "%s/mpi_sobel.pgm", out_dir);
        write_pnm(path, full_out, gw, gh, 1);
    }

    best = 1e18;
    for (int trial = 0; trial < TRIALS; trial++) {
        dt = run_mpi_conv_filter(apply_sharpen, 1, BORDER_COPY, full_gray, full_out, gw, gh, rank, size, sendcounts, displs, local_rows);
        if (dt < best) best = dt;
    }
    if (rank == 0) {
        printf("RESULT,mpi,sharpen,%d,%d,%d,%.6f\n", size, gw, gh, best);
        snprintf(path, sizeof(path), "%s/mpi_sharpen.pgm", out_dir);
        write_pnm(path, full_out, gw, gh, 1);
    }

    /* ---- Grayscale conversion: no halo, RGB scatter/gather ---- */
    {
        int *sc_rgb = malloc(size * sizeof(int)), *di_rgb = malloc(size * sizeof(int));
        int *rcR = malloc(size * sizeof(int)), *rdR = malloc(size * sizeof(int));
        int *scR = malloc(size * sizeof(int)), *diR = malloc(size * sizeof(int));
        build_distribution(rh, rw, size, rcR, rdR, scR, diR);
        for (int i = 0; i < size; i++) { sc_rgb[i] = scR[i] * 3; di_rgb[i] = diR[i] * 3; }
        int local_rows_r = rcR[rank];

        unsigned char *local_rgb  = malloc((size_t)local_rows_r * rw * 3);
        unsigned char *local_gray = malloc((size_t)local_rows_r * rw);
        unsigned char *full_gray_out = (rank == 0) ? malloc((size_t)rw * rh) : NULL;

        double gray_best = 1e18;
        for (int trial = 0; trial < TRIALS; trial++) {
            MPI_Barrier(MPI_COMM_WORLD);
            double t0 = MPI_Wtime();
            MPI_Scatterv(full_rgb, sc_rgb, di_rgb, MPI_UNSIGNED_CHAR,
                         local_rgb, local_rows_r * rw * 3, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
            apply_grayscale(local_rgb, local_gray, rw, local_rows_r);
            MPI_Gatherv(local_gray, local_rows_r * rw, MPI_UNSIGNED_CHAR,
                        full_gray_out, scR, diR, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);
            double t1 = MPI_Wtime();
            if (t1 - t0 < gray_best) gray_best = t1 - t0;
        }

        if (rank == 0) {
            printf("RESULT,mpi,grayscale,%d,%d,%d,%.6f\n", size, rw, rh, gray_best);
            snprintf(path, sizeof(path), "%s/mpi_grayscale.pgm", out_dir);
            write_pnm(path, full_gray_out, rw, rh, 1);
            free(full_gray_out);
        }
        free(local_rgb); free(local_gray);
        free(sc_rgb); free(di_rgb); free(rcR); free(rdR); free(scR); free(diR);
    }

    if (rank == 0) { free(full_gray); free(full_rgb); free(full_out); }
    free(row_counts); free(row_displs); free(sendcounts); free(displs);

    MPI_Finalize();
    return 0;
}
