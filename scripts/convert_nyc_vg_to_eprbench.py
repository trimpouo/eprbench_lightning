"""Convert NYC-Event-VPR_VG flat filenames into EPRBench-style place-based structure.

The VG images have filenames like:
    @584433.829@4507163.828@18@T@lat@lon@@@heading@@@@timestamp@hash@.jpg
where the first two @-fields are UTM easting and northing (meters).

We cluster images whose UTM coordinates are within `cluster_threshold_meters`
and put them into directories named `@NYC@<easting>@<northing>@<sub>`,
copying/symlinking the images as `@NYC@<easting>@<northing>@<sub>vNN.jpg`.

Usage:
    python scripts/convert_nyc_vg_to_eprbench.py \
        --src data/NYC-Event-VPR-VG \
        --dst data/NYC-Event-VPR-converted \
        --modalities Event RGB \
        --threshold 25.0
"""
import argparse
import re
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm


def parse_vg_filename(filename):
    """Return (easting, northing) from a VG filename, or None."""
    parts = filename.split("@")
    if len(parts) < 3:
        return None
    try:
        easting = float(parts[1])
        northing = float(parts[2])
    except ValueError:
        return None
    return easting, northing


def cluster_images(image_paths, threshold):
    """Greedy spatial clustering by UTM distance (meters)."""
    image_paths = sorted(image_paths)
    coords = []
    for p in image_paths:
        c = parse_vg_filename(p.name)
        if c is None:
            print(f"Warning: cannot parse {p.name}")
        coords.append(c)

    clusters = []  # list of (centroid_e, centroid_n, list_of_paths)
    for p, c in zip(image_paths, coords):
        if c is None:
            continue
        assigned = False
        for i, (ce, cn, paths) in enumerate(clusters):
            if np.hypot(c[0] - ce, c[1] - cn) <= threshold:
                paths.append(p)
                # update centroid
                new_ce = np.mean([parse_vg_filename(x.name)[0] for x in paths])
                new_cn = np.mean([parse_vg_filename(x.name)[1] for x in paths])
                clusters[i] = (new_ce, new_cn, paths)
                assigned = True
                break
        if not assigned:
            clusters.append((c[0], c[1], [p]))
    return clusters


def extract_split(src_split_dir):
    """Extract database.zip and queries.zip if not already extracted."""
    src_split_dir = Path(src_split_dir)
    for sub, zip_name in [("database", "database.zip"), ("queries", "queries.zip")]:
        sub_dir = src_split_dir / sub
        zip_path = src_split_dir / zip_name
        if not sub_dir.exists() and zip_path.exists():
            print(f"  Extracting {zip_path} ...")
            shutil.unpack_archive(str(zip_path), str(src_split_dir))


def convert_split(src_split_dir, dst_split_dir, threshold, symlink=True, min_images=1):
    """Convert one split (containing database/ and queries/) to place-based dirs."""
    src_split_dir = Path(src_split_dir)
    dst_split_dir = Path(dst_split_dir)

    extract_split(src_split_dir)

    # Gather all images from database and queries
    all_images = []
    for sub in ["database", "queries"]:
        sub_dir = src_split_dir / sub
        if not sub_dir.exists():
            print(f"Warning: {sub_dir} does not exist, skipping")
            continue
        all_images.extend(list(sub_dir.glob("*.jpg")) + list(sub_dir.glob("*.png")))

    print(f"  Found {len(all_images)} images in {src_split_dir}")
    clusters = cluster_images(all_images, threshold)
    print(f"  Clustered into {len(clusters)} places (threshold={threshold}m)")

    kept = 0
    num_kept_places = 0
    for cluster_idx, (ce, cn, paths) in enumerate(clusters):
        if len(paths) < min_images:
            continue
        num_kept_places += 1
        # Deterministic place name
        place_name = f"@NYC@{int(round(ce)):06d}@{int(round(cn)):07d}@{cluster_idx:04d}"
        # Split images into database and queries so validation recalls can be computed.
        # Use a fixed ratio: first half database, second half queries.
        sorted_paths = sorted(paths)
        mid = max(1, len(sorted_paths) // 2)
        db_paths = sorted_paths[:mid]
        q_paths = sorted_paths[mid:]
        for j, src in enumerate(db_paths):
            place_dir = dst_split_dir / "database" / place_name
            place_dir.mkdir(parents=True, exist_ok=True)
            dst = place_dir / f"{place_name}v{j+1:02d}{src.suffix.lower()}"
            if not dst.exists():
                if symlink:
                    dst.symlink_to(src.resolve())
                else:
                    shutil.copy2(src, dst)
            kept += 1
        for j, src in enumerate(q_paths):
            place_dir = dst_split_dir / "queries" / place_name
            place_dir.mkdir(parents=True, exist_ok=True)
            dst = place_dir / f"{place_name}v{j+mid+1:02d}{src.suffix.lower()}"
            if not dst.exists():
                if symlink:
                    dst.symlink_to(src.resolve())
                else:
                    shutil.copy2(src, dst)
            kept += 1

    print(f"  Kept {kept} images in {num_kept_places} places")
    return kept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Path to NYC-Event-VPR-VG root")
    parser.add_argument("--dst", required=True, help="Output converted dataset root")
    parser.add_argument("--modalities", nargs="+", default=["Event", "RGB"])
    parser.add_argument("--threshold", type=float, default=25.0)
    parser.add_argument("--symlink", action="store_true", default=True, help="Symlink instead of copy")
    parser.add_argument("--min-images", type=int, default=1)
    args = parser.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)

    modality_map = {
        "Event": "NYC-Event-VPR_Event",
        "RGB": "NYC-Event-VPR_RGB",
    }

    for mod in args.modalities:
        vg_mod_name = modality_map.get(mod, mod)
        print(f"\n=== Converting {mod} ({vg_mod_name}) ===")
        for split in ["train", "val", "test"]:
            src_split = src_root / vg_mod_name / "images" / split
            dst_split = dst_root / mod.lower() / split
            if not src_split.exists():
                print(f"  Skip {src_split} (not found)")
                continue
            print(f"\n-- Split: {split} --")
            convert_split(src_split, dst_split, args.threshold, args.symlink, args.min_images)

    print(f"\nDone. Converted dataset at {dst_root}")


if __name__ == "__main__":
    main()
