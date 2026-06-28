#!/bin/bash
# run_experiments.sh -- builds all four implementations (sequential, Pthreads,
# OpenMP, MPI), regenerates the benchmark dataset from the real source
# photographs in data/base/, runs the full strong-scaling / problem-size /
# weak-scaling experiment matrix, verifies cross-implementation correctness,
# and writes every raw result under results/raw_outputs/.
#
# Usage:
#   ./scripts/run_experiments.sh
#
# Only relative paths are used, so this script can be run from any working
# directory and on any machine that has gcc, mpicc/mpirun, and a POSIX shell.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

BIN="$ROOT/scripts/bin"
GEN="$ROOT/data/generated"
OUT="$ROOT/results/raw_outputs"
FIG="$ROOT/results/figures/sample_outputs"
SCRATCH="$OUT/scratch"
LOGS="$OUT/logs"

mkdir -p "$BIN" "$GEN" "$OUT" "$FIG" "$SCRATCH" "$LOGS"

NP_LIST="1 2 4 8"
MPIRUN_FLAGS="--oversubscribe"

echo "=============================================================="
echo " [1/7] Compiling sequential, Pthreads, OpenMP, MPI, generator"
echo "=============================================================="
gcc   -O3 -Wall -Wextra -o "$BIN/input_generator" data/input_generator.c -lm
gcc   -O3 -Wall -Wextra -o "$BIN/main_seq"        src/sequential/main_seq.c -lm
gcc   -O3 -Wall -Wextra -pthread  -o "$BIN/main_pth" src/pthreads/main_pth.c -lm
gcc   -O3 -Wall -Wextra -fopenmp  -o "$BIN/main_omp" src/openmp/main_omp.c  -lm
mpicc -O3 -Wall -Wextra -o "$BIN/main_mpi"        src/mpi/main_mpi.c -lm

echo "=============================================================="
echo " [2/7] Generating benchmark dataset from real base photographs"
echo "=============================================================="
"$BIN/input_generator" data/base/camera_512.pgm data/base/astronaut_512.ppm "$GEN" \
    | tee "$LOGS/00_dataset_generation.log"

RESULTS_CSV="$OUT/results.csv"
echo "experiment,model,filter,workers,width,height,seconds" > "$RESULTS_CSV"

# Runs "$@", tees raw stdout+stderr to $LOGS/<tag>.log, and appends any
# "RESULT,..." line (emitted by the C programs) to results.csv tagged with
# the experiment name.
capture() {
    local experiment="$1" tag="$2"; shift 2
    "$@" 2>&1 | tee "$LOGS/${tag}.log" \
        | grep '^RESULT,' | sed "s/^RESULT,/${experiment},/" >> "$RESULTS_CSV" || true
}

echo "=============================================================="
echo " [3/7] Strong scaling: fixed 4K image (3840x2160), workers = $NP_LIST"
echo "=============================================================="
GRAY4K="$GEN/strong_gray_0003840x002160.pgm"
RGB4K="$GEN/strong_rgb_0003840x002160.ppm"

capture strong_scaling seq_4k "$BIN/main_seq" "$GRAY4K" "$RGB4K" "$SCRATCH"
for n in $NP_LIST; do
    capture strong_scaling pth_4k_n${n} "$BIN/main_pth" "$GRAY4K" "$RGB4K" "$SCRATCH" "$n"
    capture strong_scaling omp_4k_n${n} "$BIN/main_omp" "$GRAY4K" "$RGB4K" "$SCRATCH" "$n"
    capture strong_scaling mpi_4k_n${n} mpirun $MPIRUN_FLAGS -np "$n" "$BIN/main_mpi" "$GRAY4K" "$RGB4K" "$SCRATCH"
done

echo "=============================================================="
echo " [4/7] Problem-size sweep: workers fixed at 8, N ~ 10^3..10^7 px"
echo "=============================================================="
for f in "$GEN"/strong_gray_*.pgm; do
    tag=$(basename "$f" .pgm | sed 's/strong_gray_//')
    GRAY="$f"
    RGB="$GEN/strong_rgb_${tag}.ppm"
    capture size_sweep seq_${tag}      "$BIN/main_seq" "$GRAY" "$RGB" "$SCRATCH"
    capture size_sweep pth_${tag}_n8   "$BIN/main_pth" "$GRAY" "$RGB" "$SCRATCH" 8
    capture size_sweep omp_${tag}_n8   "$BIN/main_omp" "$GRAY" "$RGB" "$SCRATCH" 8
    capture size_sweep mpi_${tag}_n8   mpirun $MPIRUN_FLAGS -np 8 "$BIN/main_mpi" "$GRAY" "$RGB" "$SCRATCH"
done

echo "=============================================================="
echo " [5/7] Weak scaling: rows/worker fixed, workers = $NP_LIST"
echo "=============================================================="
for n in $NP_LIST; do
    GRAYW="$GEN/weak_gray_p${n}.pgm"
    RGBW="$GEN/weak_rgb_p${n}.ppm"
    capture weak_scaling pth_weak_n${n} "$BIN/main_pth" "$GRAYW" "$RGBW" "$SCRATCH" "$n"
    capture weak_scaling omp_weak_n${n} "$BIN/main_omp" "$GRAYW" "$RGBW" "$SCRATCH" "$n"
    capture weak_scaling mpi_weak_n${n} mpirun $MPIRUN_FLAGS -np "$n" "$BIN/main_mpi" "$GRAYW" "$RGBW" "$SCRATCH"
done

echo "=============================================================="
echo " [6/7] Correctness check + qualitative sample gallery"
echo "=============================================================="
REPORT="$OUT/correctness_report.txt"
{
    echo "Correctness report -- generated $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo "All four implementations run on data/generated/strong_gray_0000512x000512.pgm"
    echo "and the matching RGB image, compared byte-for-byte against the sequential"
    echo "baseline output (cmp -s). PASS means zero byte difference. If the optional"
    echo "real-dataset experiment (step 7/7) is enabled, its batch-mode code path is"
    echo "checked the same way, on a 50-image subset of the real photographs."
    echo ""
} > "$REPORT"

GRAY512="$GEN/strong_gray_0000512x000512.pgm"
RGB512="$GEN/strong_rgb_0000512x000512.ppm"
CHECK="$SCRATCH/correctness"
mkdir -p "$CHECK"
"$BIN/main_seq" "$GRAY512" "$RGB512" "$CHECK" > /dev/null

overall_pass=1
for n in $NP_LIST; do
    for model in pth omp; do
        bin_var="main_${model}"
        mkdir -p "$CHECK/${model}${n}"
        "$BIN/$bin_var" "$GRAY512" "$RGB512" "$CHECK/${model}${n}" "$n" > /dev/null
    done
    mkdir -p "$CHECK/mpi${n}"
    mpirun $MPIRUN_FLAGS -np "$n" "$BIN/main_mpi" "$GRAY512" "$RGB512" "$CHECK/mpi${n}" > /dev/null

    for filt in gaussian sobel sharpen grayscale; do
        for model in pth omp; do
            a="$CHECK/seq_${filt}.pgm"; b="$CHECK/${model}${n}/${model}_${filt}.pgm"
            if cmp -s "$a" "$b"; then status="PASS"; else status="FAIL"; overall_pass=0; fi
            echo "[$status] $model workers=$n filter=$filt" >> "$REPORT"
        done
        a="$CHECK/seq_${filt}.pgm"; b="$CHECK/mpi${n}/mpi_${filt}.pgm"
        if cmp -s "$a" "$b"; then status="PASS"; else status="FAIL"; overall_pass=0; fi
        echo "[$status] mpi workers=$n filter=$filt" >> "$REPORT"
    done
done
echo "" >> "$REPORT"
if [ "$overall_pass" -eq 1 ]; then
    echo "OVERALL: ALL OUTPUTS BYTE-IDENTICAL TO THE SEQUENTIAL BASELINE" >> "$REPORT"
else
    echo "OVERALL: CORRECTNESS FAILURE -- see FAIL lines above" >> "$REPORT"
fi
cat "$REPORT"

# Qualitative before/after gallery for the paper, generated straight from the
# 512x512 real photographs (kept; everything in $SCRATCH is throwaway).
"$BIN/main_seq" data/base/camera_512.pgm data/base/astronaut_512.ppm "$FIG" > /dev/null
cp data/base/camera_512.pgm data/base/astronaut_512.ppm "$FIG/"

echo "=============================================================="
echo " [7/7] Real-world dataset throughput (Imagenette, real photographs)"
echo "=============================================================="
REALDS="$ROOT/data/real_dataset"
DATASET_COUNT=5000
n_real=$(ls "$REALDS"/img_*.pgm 2>/dev/null | wc -l || echo 0)

if [ "$n_real" -lt "$DATASET_COUNT" ]; then
    echo "SKIPPED: $REALDS has only $n_real prepared images (need $DATASET_COUNT)."
    echo "  This is an optional, additional experiment on top of the mandated"
    echo "  strong/weak/size-sweep experiments above, which do not require it."
    echo "  To enable it (one-time, real photographs, Apache-2.0, fastai/imagenette):"
    echo "    curl -fSL -o /tmp/imagenette2-160.tgz https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
    echo "    tar xzf /tmp/imagenette2-160.tgz -C /tmp"
    echo "    python3 data/prepare_real_dataset.py /tmp/imagenette2-160/train data/real_dataset $DATASET_COUNT"
    echo "  then re-run this script."
else
    BCHECK="$SCRATCH/batch_correctness"
    mkdir -p "$BCHECK/seq"
    "$BIN/main_seq" --dataset "$REALDS" 50 "$BCHECK/seq" > /dev/null

    for n in $NP_LIST; do
        for model in pth omp; do
            mkdir -p "$BCHECK/${model}${n}"
            "$BIN/main_${model}" --dataset "$REALDS" 50 "$BCHECK/${model}${n}" "$n" > /dev/null
        done
        mkdir -p "$BCHECK/mpi${n}"
        mpirun $MPIRUN_FLAGS -np "$n" "$BIN/main_mpi" --dataset "$REALDS" 50 "$BCHECK/mpi${n}" > /dev/null

        for filt in gaussian sobel sharpen grayscale; do
            for model in pth omp; do
                a="$BCHECK/seq/batch_00000_${filt}.pgm"; b="$BCHECK/${model}${n}/batch_00000_${filt}.pgm"
                if cmp -s "$a" "$b"; then status="PASS"; else status="FAIL"; overall_pass=0; fi
                echo "[$status] dataset-batch $model workers=$n filter=$filt" >> "$REPORT"
            done
            a="$BCHECK/seq/batch_00000_${filt}.pgm"; b="$BCHECK/mpi${n}/batch_00000_${filt}.pgm"
            if cmp -s "$a" "$b"; then status="PASS"; else status="FAIL"; overall_pass=0; fi
            echo "[$status] dataset-batch mpi workers=$n filter=$filt" >> "$REPORT"
        done
    done

    capture dataset_throughput seq_real "$BIN/main_seq" --dataset "$REALDS" "$DATASET_COUNT" "$SCRATCH"
    for n in $NP_LIST; do
        capture dataset_throughput pth_real_n${n} "$BIN/main_pth" --dataset "$REALDS" "$DATASET_COUNT" "$SCRATCH" "$n"
        capture dataset_throughput omp_real_n${n} "$BIN/main_omp" --dataset "$REALDS" "$DATASET_COUNT" "$SCRATCH" "$n"
        capture dataset_throughput mpi_real_n${n} mpirun $MPIRUN_FLAGS -np "$n" "$BIN/main_mpi" --dataset "$REALDS" "$DATASET_COUNT" "$SCRATCH"
    done
    echo "Real-dataset throughput run complete: $n_real real photographs, $DATASET_COUNT used per run."
fi

echo ""
echo "Done. Combined timing data: $RESULTS_CSV"
echo "Correctness report:        $REPORT"
echo "Sample filtered images:    $FIG"
if [ "$overall_pass" -ne 1 ]; then
    echo "WARNING: correctness check reported failures, see $REPORT" >&2
    exit 1
fi
