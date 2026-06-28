#!/usr/bin/env python3
"""
prepare_real_dataset.py -- builds the real-world, multi-thousand-image
benchmark corpus used by the "dataset throughput" experiment.

Source: Imagenette (https://github.com/fastai/imagenette), a 10-class,
real-photograph subset of ImageNet distributed by fastai under the
Apache-2.0 license. This script does NOT generate or guess any image
content -- it converts already-downloaded real JPEG photographs into the
project's existing binary NetPBM format (P5 grayscale / P6 RGB), the same
format every C implementation already reads.

This is a one-time, Python-only data-preparation step (mirrors how
data/base/*.pgm|ppm were produced from scikit-image earlier in this
project) -- the timed C/MPI/Pthreads/OpenMP pipeline itself has zero
Python dependency, only this offline conversion does.

Usage:
    # 1) Download once (Apache-2.0, fastai/imagenette, ~99MB):
    curl -fSL -o imagenette2-160.tgz \
        https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz
    tar xzf imagenette2-160.tgz

    # 2) Build the balanced 5,000-image benchmark corpus:
    python3 data/prepare_real_dataset.py imagenette2-160/train data/real_dataset 5000

Produces data/real_dataset/img_NNNNN.pgm (grayscale) and .ppm (RGB) for
NNNNN in [0, count), each resized to a fixed IMG_SIZE x IMG_SIZE so every
image in the batch costs the same amount of filter work -- the point of
this experiment is to measure throughput over many real images, not to
re-run the existing single-image resolution sweep.
"""
import os
import sys
from PIL import Image

IMG_SIZE = 160  # fixed square size: uniform per-image cost across the batch

# The 10 Imagenette classes, in the canonical order used by fastai's
# README (https://github.com/fastai/imagenette) -- kept here only for the
# manifest/log, not used by the C pipeline.
CLASS_NAMES = {
    "n01440764": "tench",
    "n02102040": "English springer",
    "n02979186": "cassette player",
    "n03000684": "chainsaw",
    "n03028079": "church",
    "n03394916": "French horn",
    "n03417042": "garbage truck",
    "n03425413": "gas pump",
    "n03445777": "golf ball",
    "n03888257": "parachute",
}


def write_pgm(path, im_gray):
    w, h = im_gray.size
    with open(path, "wb") as f:
        f.write(f"P5\n{w} {h}\n255\n".encode("ascii"))
        f.write(im_gray.tobytes())


def write_ppm(path, im_rgb):
    w, h = im_rgb.size
    with open(path, "wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
        f.write(im_rgb.tobytes())


def main():
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <imagenette_train_dir> <out_dir> <count>")
        sys.exit(1)

    src_dir, out_dir, count = sys.argv[1], sys.argv[2], int(sys.argv[3])
    os.makedirs(out_dir, exist_ok=True)

    classes = sorted(d for d in os.listdir(src_dir) if d in CLASS_NAMES)
    if not classes:
        print(f"ERROR: no known Imagenette class folders found under {src_dir}")
        sys.exit(1)

    per_class = count // len(classes)
    remainder = count % len(classes)

    manifest = []
    idx = 0
    for ci, cls in enumerate(classes):
        cls_dir = os.path.join(src_dir, cls)
        files = sorted(f for f in os.listdir(cls_dir) if f.lower().endswith((".jpg", ".jpeg")))
        take = per_class + (1 if ci < remainder else 0)
        if len(files) < take:
            print(f"ERROR: class {cls} only has {len(files)} images, need {take}")
            sys.exit(1)
        for fname in files[:take]:
            src_path = os.path.join(cls_dir, fname)
            with Image.open(src_path) as im:
                im = im.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
                gray = im.convert("L")
                tag = f"img_{idx:05d}"
                write_pgm(os.path.join(out_dir, f"{tag}.pgm"), gray)
                write_ppm(os.path.join(out_dir, f"{tag}.ppm"), im)
                manifest.append(f"{tag},{cls},{CLASS_NAMES[cls]},{fname}")
            idx += 1

    with open(os.path.join(out_dir, "manifest.csv"), "w") as f:
        f.write("tag,wnid,class_name,source_file\n")
        f.write("\n".join(manifest) + "\n")

    print(f"Prepared {idx} real photographs ({IMG_SIZE}x{IMG_SIZE}) from "
          f"{len(classes)} Imagenette classes into {out_dir}")


if __name__ == "__main__":
    main()
