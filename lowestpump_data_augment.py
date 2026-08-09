import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
 
import numpy as np
from PIL import Image
 
# ---------------------------------------------------------------------------
# Combination definitions
# ---------------------------------------------------------------------------
 
FLIPS = {
    "NoF": None,
    "Fx": Image.FLIP_LEFT_RIGHT,
    # "Fy" (vertical flip) dropped -- with 5 rotations below this keeps the
    # total near 100,000 images. Add it back (3 flips x 5 rotations = 15/image)
    # if you'd rather have more variety and don't mind ~150,000 images instead.
}

ROTATIONS = [0, 72, 144, 216, 288]  # 5 evenly-spaced angles
TONE_LEVELS = [0]  # no tone/color augmentation -- keeps combos low and simple
TONE_STEP = 15  # pixel-value shift per level, tune to taste

# Combos/image = len(FLIPS) x len(ROTATIONS) x len(TONE_LEVELS)^2
#              = 2 x 5 x 1 x 1 = 10
# Total images = 10 x (number of source images, e.g. 10,015 for HAM10000)
#              = ~100,150 -- adjust FLIPS/ROTATIONS/TONE_LEVELS above to retarget.
 
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
 
 
def apply_flip(img, flip_name):
    op = FLIPS[flip_name]
    return img if op is None else img.transpose(op)
 
 
def apply_rotation(img, degrees):
    if degrees == 0:
        return img
    # expand=False keeps original canvas size so every output is uniform;
    # change to expand=True if you'd rather keep full rotated content.
    return img.rotate(degrees, resample=Image.BICUBIC, expand=False)
 
 
def apply_tone(img, cool_level, warm_level):
    if cool_level == 0 and warm_level == 0:
        return img
    arr = np.asarray(img).astype(np.int16)
    net_warmth = (warm_level - cool_level) * TONE_STEP
    arr[..., 0] = np.clip(arr[..., 0] + net_warmth, 0, 255)   # Red
    arr[..., 2] = np.clip(arr[..., 2] - net_warmth, 0, 255)   # Blue
    return Image.fromarray(arr.astype(np.uint8))
 
 
def build_combinations():
    combos = []
    for flip_name in FLIPS:
        for deg in ROTATIONS:
            for cool in TONE_LEVELS:
                for warm in TONE_LEVELS:
                    combos.append((flip_name, deg, cool, warm))
    return combos
 
 
COMBINATIONS = build_combinations()
 
 
# ---------------------------------------------------------------------------
# Per-image worker
# ---------------------------------------------------------------------------
 
def process_one_image(args):
    src_path, output_dir, resize, quality, label = args
    image_id = src_path.stem
    out_subdir = output_dir / image_id
    out_subdir.mkdir(parents=True, exist_ok=True)
 
    rows = []
    try:
        with Image.open(src_path) as im:
            im = im.convert("RGB")
            if resize:
                im = im.resize(resize, Image.LANCZOS)
 
            for flip_name, deg, cool, warm in COMBINATIONS:
                out = apply_flip(im, flip_name)
                out = apply_rotation(out, deg)
                out = apply_tone(out, cool, warm)
 
                fname = f"{image_id}_{flip_name}_R{deg}_C{cool}_W{warm}.jpg"
                out_path = out_subdir / fname
                out.save(out_path, "JPEG", quality=quality)
 
                rows.append({
                    "filename": str(out_path),
                    "original_image": src_path.name,
                    "flip": flip_name,
                    "rotation_deg": deg,
                    "cool_level": cool,
                    "warm_level": warm,
                    "label": label if label is not None else "",
                })
    except Exception as e:
        print(f"[WARN] Failed on {src_path.name}: {e}", file=sys.stderr)
        return []
 
    return rows
 
 
# ---------------------------------------------------------------------------
# Metadata loading
# ---------------------------------------------------------------------------
 
def load_labels(metadata_csv, id_col, label_col):
    labels = {}
    if not metadata_csv:
        return labels
    with open(metadata_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if id_col not in reader.fieldnames or label_col not in reader.fieldnames:
            print(f"[WARN] metadata_csv missing columns '{id_col}' / '{label_col}'. "
                  f"Found columns: {reader.fieldnames}. Continuing without labels.",
                  file=sys.stderr)
            return labels
        for row in reader:
            labels[row[id_col]] = row[label_col]
    return labels
 
 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
 
def main():
    parser = argparse.ArgumentParser(description="Augment images into a large training set.")
    parser.add_argument("--input_dir", required=True, help="Folder of source images")
    parser.add_argument("--output_dir", default="all_augment_100K",
                         help='Output root folder (default: "all_augment_100K")')
    parser.add_argument("--metadata_csv", default=None, help="Optional CSV with image id + label columns")
    parser.add_argument("--id_col", default="image_id")
    parser.add_argument("--label_col", default="dx")
    parser.add_argument("--resize", default=None, help='Optional "W,H", e.g. "224,224"')
    parser.add_argument("--quality", type=int, default=90)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--limit", type=int, default=None, help="Only process first N source images (testing)")
    args = parser.parse_args()
 
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"[ERROR] --output_dir '{output_dir}' already exists and is not empty. "
              f"Refusing to run to avoid silently overwriting/mixing files from a "
              f"previous run (this is exactly what caused the all-augment mix-up "
              f"before). Pick a new --output_dir, or delete/rename the existing "
              f"folder first if you're sure you want to regenerate into it.",
              file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
 
    resize = None
    if args.resize:
        w, h = (int(x) for x in args.resize.split(","))
        resize = (w, h)
 
    src_paths = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in IMG_EXTENSIONS
    )
    if args.limit:
        src_paths = src_paths[: args.limit]
 
    n_images = len(src_paths)
    n_combos = len(COMBINATIONS)
    total_out = n_images * n_combos
 
    print(f"Source images found : {n_images}")
    print(f"Combinations/image  : {n_combos} "
          f"({len(FLIPS)} flips x {len(ROTATIONS)} rotations x "
          f"{len(TONE_LEVELS)} cool x {len(TONE_LEVELS)} warm)")
    print(f"Total output images : {total_out:,}")
    print(f"Output folder       : {output_dir.resolve()}")
    print(f"Workers             : {args.workers}")
    print("-" * 60)
 
    labels = load_labels(args.metadata_csv, args.id_col, args.label_col)
 
    meta_path = output_dir / "lowmeta.csv"
    fieldnames = ["filename", "original_image", "flip", "rotation_deg",
                  "cool_level", "warm_level", "label"]
 
    tasks = []
    for src_path in src_paths:
        label = labels.get(src_path.stem)
        tasks.append((src_path, output_dir, resize, args.quality, label))
 
    start = time.time()
    written = 0
 
    with open(meta_path, "w", newline="", encoding="utf-8") as meta_f:
        writer = csv.DictWriter(meta_f, fieldnames=fieldnames)
        writer.writeheader()
 
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_one_image, t): t[0] for t in tasks}
            done_images = 0
            for future in as_completed(futures):
                rows = future.result()
                for row in rows:
                    writer.writerow(row)
                written += len(rows)
                done_images += 1
 
                if done_images % 50 == 0 or done_images == n_images:
                    elapsed = time.time() - start
                    rate = done_images / elapsed if elapsed > 0 else 0
                    remaining = (n_images - done_images) / rate if rate > 0 else float("inf")
                    print(f"[{done_images}/{n_images} source images] "
                          f"{written:,} output images written | "
                          f"{elapsed/60:.1f} min elapsed | "
                          f"~{remaining/60:.1f} min remaining", flush=True)
 
    print("-" * 60)
    print(f"Done. {written:,} images written to '{output_dir}'.")
    print(f"Metadata written to '{meta_path}'.")
 
 
if __name__ == "__main__":
    main()