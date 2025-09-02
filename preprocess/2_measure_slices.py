#%%
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
# from preprocessingutils import pwr_transform
import os
from pathlib import Path
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed


from argparse import ArgumentParser

#%%


def features_from_mask(img_hu, mask, pixel_spacing=(0.5, 0.5), hist_bins=32, hist_window=None):
    sx, sy = pixel_spacing
    px_area = sx * sy

    vals = img_hu[mask > 0].astype(np.float64)
    feats = {}

    # --- intensity ---
    feats['n_px'] = vals.size
    feats['area_mm2'] = vals.size * px_area
    feats['hu_mean'] = vals.mean()
    feats['hu_var'] = vals.var()
    feats['hu_median'] = np.median(vals)
    feats['hu_iqr'] = np.percentile(vals, 75) - np.percentile(vals, 25)
    mad = np.median(np.abs(vals - feats['hu_median']))
    feats['hu_mad'] = mad
    p = np.percentile(vals, [10,25,75,90,95])
    feats.update(dict(hu_p10=p[0], hu_p25=p[1], hu_p75=p[2], hu_p90=p[3], hu_p95=p[4]))

    # histogram / entropy (dtype-aware defaults)
    if hist_window is None:
        # If the input looks like 8-bit windowed PNGs, use [0,255]; otherwise assume HU window [-1000,1000]
        low, high = (0.0, 255.0) if (np.issubdtype(img_hu.dtype, np.integer) and img_hu.dtype == np.uint8) else (-1000.0, 1000.0)
    else:
        low, high = float(hist_window[0]), float(hist_window[1])
    # Ensure deterministic fixed-width bins
    bins = np.linspace(low, high, int(hist_bins) + 1)
    hist, _ = np.histogram(vals, bins=bins)
    if hist.sum() > 0:
        p_hist = hist.astype(np.float64) / hist.sum()
        feats['hu_entropy'] = -np.sum(p_hist[p_hist>0] * np.log(p_hist[p_hist>0]))
    else:
        feats['hu_entropy'] = np.nan

    # gradients (inside mask)
    gx, gy = np.gradient(img_hu.astype(np.float64))
    gmag = np.hypot(gx, gy)
    feats['grad_mean_in'] = gmag[mask>0].mean()

    # --- shape from coordinates ---
    ys, xs = np.nonzero(mask)
    cx, cy = xs.mean(), ys.mean()
    feats['centroid_px_x'] = cx
    feats['centroid_px_y'] = cy
    feats['centroid_rel_x'] = cx / img_hu.shape[1]
    feats['centroid_rel_y'] = cy / img_hu.shape[0]

    # covariance -> major/minor axes, orientation
    X = np.vstack([(xs - cx)*sx, (ys - cy)*sy])  # mm-coords
    C = (X @ X.T) / (X.shape[1] - 1)
    evals, evecs = np.linalg.eigh(C)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]; evecs = evecs[:, idx]
    major, minor = 2*np.sqrt(evals[0]), 2*np.sqrt(evals[1])  # ~2*std as a scale proxy
    feats['axis_major_mm'] = major
    feats['axis_minor_mm'] = minor
    feats['elongation'] = (major / minor) if minor>0 else np.inf
    feats['orientation_rad'] = np.arctan2(evecs[1,0], evecs[0,0])

    # perimeter (simple 4-neigh transitions)
    # A pixel is interior iff all 4 neighbors are foreground; boundary = foreground & ~interior
    pad = np.pad(mask.astype(bool), 1, constant_values=False)
    up = pad[:-2, 1:-1]
    dn = pad[2:, 1:-1]
    lf = pad[1:-1, :-2]
    rt = pad[1:-1, 2:]
    interior = up & dn & lf & rt
    boundary = mask.astype(bool) & (~interior)
    perim_px = int(boundary.sum())
    feats['perimeter_mm_approx'] = perim_px * 0.5 * (sx + sy)

    # circularity
    area = feats['area_mm2']
    perim = feats['perimeter_mm_approx']
    if perim > 0:
        feats['circularity'] = 4.0 * np.pi * area / (perim ** 2)
    else:
        feats['circularity'] = np.nan

    return feats

# ----------------------------
# I/O helpers
# ----------------------------
def load_png_as_array(path: Path) -> np.ndarray:
    img = Image.open(path)
    # If RGB/RGBA, convert to single-channel grayscale
    if img.mode not in ("I;16", "I", "L"):
        img = img.convert("L")
    arr = np.array(img, dtype=np.uint8)
    return arr


def binarize_mask(mask_arr: np.ndarray) -> np.ndarray:
    # Any nonzero -> True
    if mask_arr.dtype != bool:
        return mask_arr.astype(np.uint16) > 0
    return mask_arr



def pair_files(img_dir: Path, mask_dir: Path):
    """
    Pair files by filename stem. Returns list of (stem, img_path, mask_path).
    """
    imgs = {p.stem: p for p in img_dir.glob("*.png")}
    masks = {p.stem: p for p in mask_dir.glob("*.png")}
    common = sorted(set(imgs.keys()) & set(masks.keys()))
    pairs = [(k, imgs[k], masks[k]) for k in common]
    missing_imgs = sorted(set(masks.keys()) - set(imgs.keys()))
    missing_masks = sorted(set(imgs.keys()) - set(masks.keys()))
    return pairs, missing_imgs, missing_masks


def process_one(stem: str, img_path: Path, mask_path: Path, spacing, hist_bins, hist_window):
    img = load_png_as_array(img_path).astype(np.float64)  # safe for gradients/statistics
    mask = binarize_mask(load_png_as_array(mask_path))
    feats = features_from_mask(img, mask, pixel_spacing=spacing, hist_bins=hist_bins, hist_window=hist_window)
    return {"id": stem, **feats}


def main(args):
    if args.data_dir is None:
        data_dir = Path(Path.cwd().parent / 'data' / 'nodules2d')
    else:
        data_dir = args.data_dir

    # parse spacing string like "0.5,0.5"
    if isinstance(args.spacing, str):
        spacing = tuple(float(x) for x in args.spacing.split(","))
    else:
        spacing = tuple(args.spacing)
    if len(spacing) != 2:
        raise ValueError("--spacing must be 'sx,sy'")

    # parse histogram window if provided
    hist_window = None
    if args.window is not None:
        lo, hi = args.window.split(",")
        hist_window = (float(lo), float(hi))
    hist_bins = int(args.bins)

    # find pairs of img / mask
    img_dir = data_dir / "imgs"
    mask_dir = data_dir / "masks"

    pairs, missing_imgs, missing_masks = pair_files(img_dir, mask_dir)
    if not pairs:
        raise SystemExit("No matching (image, mask) filename stems found.")
    else:
        print(f"Found {len(pairs)} matching (image, mask) pairs.")

    if missing_imgs:
        print(f"Warning: {len(missing_imgs)} mask(s) have no matching image: {missing_imgs[:5]}{'...' if len(missing_imgs)>5 else ''}")
    if missing_masks:
        print(f"Warning: {len(missing_masks)} image(s) have no matching mask: {missing_masks[:5]}{'...' if len(missing_masks)>5 else ''}")

    if args.test_run:
        pairs = pairs[:100]

    results = []
    if args.workers and args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [
                ex.submit(process_one, stem, ip, mp, spacing, hist_bins, hist_window)
                for stem, ip, mp in pairs
            ]
            for fut in tqdm(as_completed(futs), total=len(futs), desc="Processing", unit="img"):
                results.append(fut.result())
    else:
        for stem, ip, mp in tqdm(pairs, desc="Processing", unit="img"):
            results.append(process_one(stem, ip, mp, spacing, hist_bins, hist_window))

    # Write CSV using pandas
    if not results:
        raise SystemExit("No results computed.")
    all_keys = set().union(*[r.keys() for r in results])
    preferred = [
        "id", "n_px", "area_mm2", "perimeter_mm_approx", "circularity",
        "axis_major_mm", "axis_minor_mm", "elongation", "orientation_rad",
        "centroid_px_x", "centroid_px_y", "centroid_rel_x", "centroid_rel_y",
        "hu_mean", "hu_var", "hu_median", "hu_iqr", "hu_mad",
        "hu_p10", "hu_p25", "hu_p75", "hu_p90", "hu_p95", "hu_entropy",
        "grad_mean_in"
    ]
    remaining = [k for k in sorted(all_keys) if k not in preferred]
    fieldnames = preferred + remaining

    df = pd.DataFrame(results)
    # Ensure all columns are present and in the right order
    for col in fieldnames:
        if col not in df.columns:
            df[col] = ""
    df = df[fieldnames]

    # id has the following structure: <patient id>n<nodule id>a<annotiation id>s<slide number>
    # e.g. 0002n03a2s086
    # first, create patient id, nodule id, annotation id and slice_id columns
    df["patient_id"] = df["id"].str.extract(r"^(\d+)n")[0]
    df["nodule_idx"] = df["id"].str.extract(r"n(\d+)a")[0]
    # create 'nodule_id' as combination of patientid and nodule id (i.e. everything before a.s...)
    df["nodule_id"] = df["id"].str.extract(r"^(\d+n\d+)a")[0]
    df["annotation_idx"] = df["id"].str.extract(r"a(\d+)s")[0]
    # create 'annotation_id' as combination of nodule_id and annotation_idx
    df["annotation_id"] = df["nodule_id"] + "a" + df["annotation_idx"]
    df["slice_id"] = df["id"].str.extract(r"s(\d+)$")[0]

    # for each nodule and annotation, count the number of slices, add to dataframe
    df["slice_count"] = df.groupby(["patient_id", "nodule_idx", "annotation_idx"])["id"].transform("count")

    # add a counter for every unique combination of patient, nodule and annotation
    ## first, sort by id
    df = df.sort_values("id")
    df["slice_index"] = df.groupby(["patient_id", "nodule_idx", "annotation_idx"]).cumcount()

    ## add a column that is True if the slice is the middle slice of the nodule
    df["is_middle_slice"] = df["slice_index"] == (df["slice_count"] - 1) // 2

    ## for nodules with an even number of slices, add a column that is True if the slice is the second middle slice
    df["is_second_middle_slice"] = df["slice_index"] == df["slice_count"] // 2

    ## for each nodule, add a column that says if a certain slice id is found in every annotation for that nodule
    ## first, count the number of annotations per patient_id and nodule_idx
    df["num_annotations"] = df.groupby(["patient_id", "nodule_idx"])["annotation_idx"].transform("nunique")

    # for each nodule and slice_id, count the number of annotations that have that slice
    slice_counts = df.groupby(["patient_id", "nodule_idx", "slice_id"]).size().reset_index(name="num_occurrences")
    # merge this into the data
    df = pd.merge(df, slice_counts, on=["patient_id", "nodule_idx", "slice_id"])
    # then, check for each slice id if it occurs num_annotations times
    df["slice_found_in_all_annotations"] = df["num_occurrences"] == df["num_annotations"]

    ## re-do the 'slice_index' and 'is_middle_slice' for those that are in all annotations
    df_consensus = df[df["slice_found_in_all_annotations"]]
    df_consensus = df_consensus.sort_values("id")
    df_consensus["slice_index"] = df_consensus.groupby(["patient_id", "nodule_idx", "annotation_idx"]).cumcount()
    df_consensus["is_middle_slice"] = df_consensus["slice_index"] == (df_consensus["slice_count"] - 1) // 2 

    df.to_csv(data_dir / "measurements.csv", index=False)
    df_consensus.to_csv(data_dir / "measurements_consensus.csv", index=False)
    print(f"Wrote {len(results)} rows to {data_dir / 'measurements.csv'}")
    print(f"Wrote {len(df_consensus)} rows to {data_dir / 'measurements_consensus.csv'}")

    # optionally export stacks of slices
    if args.save_stacks:
        print("exporting as stacked arrays")
        img_paths = [p for _, p, _ in pairs]
        mask_paths = [p for _, _, p in pairs]
        img_list = [load_png_as_array(p) for p in img_paths]
        mask_list = [binarize_mask(load_png_as_array(p)) for p in mask_paths]
        imgs = np.stack(img_list, axis=0, dtype=np.uint8)
        masks = np.stack(mask_list, axis=0, dtype=bool)

        np.savez_compressed(data_dir / "imgs_masks.npz", imgs=imgs, masks=masks)

    return


#%%
if __name__ == "__main__":
    ap = ArgumentParser(description="Compute simple features from CT PNGs + mask PNGs.")
    ap.add_argument("--data_dir", type=Path, help="Directory with data, has subdirectories 'imgs' and 'masks'")
    ap.add_argument("--spacing", default="0.5,0.5",
                    help="Pixel spacing in mm as 'sx,sy' (default: 0.5,0.5)")
    ap.add_argument("--bins", type=int, default=32,
                    help="Number of histogram bins (default: 32)")
    ap.add_argument("--window", type=str, default=None,
                    help="Histogram window 'low,high'. If omitted, uses [0,255] for uint8, else [-1000,1000].")
    ap.add_argument("--workers", type=int, default=12,
                    help="Number of parallel workers (0 or 1 = no parallelism).")
    ap.add_argument("--test_run", action="store_true",
                    help="If set, process only 100 images for testing.")
    ap.add_argument("--save_stacks", action="store_true", help="save stacks of imgs and masks as npy files")
    # args = ap.parse_known_args()[0]
    args = ap.parse_args()

    main(args)

# %%
