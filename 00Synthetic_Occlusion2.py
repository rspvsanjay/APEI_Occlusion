import os
import glob
import random
from typing import List, Tuple, Optional

import cv2
import numpy as np

# ---------------- occluder generators (unchanged) ----------------

def draw_rectangle_mask(h: int, w: int, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (x1, y1), (x2, y2), 1, thickness=-1)
    return mask

def draw_ellipse_mask(h: int, w: int, center: Tuple[int,int], axes: Tuple[int,int], angle: float=0.0) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, center, axes, angle, 0, 360, 1, -1)
    return mask

def draw_blob_mask(h:int, w:int, center:Tuple[int,int], scale:float=0.5, noise=0.25) -> np.ndarray:
    axes = (max(2, int(w * scale * 0.4)), max(2, int(h * scale * 0.6)))
    base = draw_ellipse_mask(h, w, center, axes, angle=0.0)
    k = max(3, int((h + w) * noise * 0.01))
    if random.random() < 0.5:
        base = cv2.dilate(base, np.ones((k,k), np.uint8))
    else:
        base = cv2.erode(base, np.ones((k,k), np.uint8))
    return (base > 0).astype(np.uint8)

def draw_part_based_human_mask(h:int, w:int, center:Tuple[int,int], scale:float=1.0, flip:bool=False) -> np.ndarray:
    canvas = np.zeros((h, w), dtype=np.uint8)
    cx, cy = center
    head_r = max(2, int(w * 0.05 * scale))
    head_c = (cx, max(0, cy - int(h * 0.18 * scale)))
    cv2.circle(canvas, head_c, head_r, 1, -1)
    torso_w = max(4, int(w * 0.12 * scale))
    torso_h = max(6, int(h * 0.28 * scale))
    top_left = (max(0, cx - torso_w//2), max(0, cy - torso_h//2))
    bottom_right = (min(w-1, cx + torso_w//2), min(h-1, cy + torso_h//2))
    cv2.rectangle(canvas, top_left, bottom_right, 1, -1)
    limb_w = max(3, int(w * 0.03 * scale))
    limb_len_upper = int(h * 0.14 * scale)
    limb_len_lower = int(h * 0.18 * scale)
    arm_y = top_left[1] + int(torso_h*0.2)
    # left arm
    lx1 = max(0, cx - torso_w//2 - limb_len_upper//2)
    lx2 = max(0, lx1 + limb_w)
    ly1 = arm_y
    ly2 = min(h-1, ly1 + limb_len_upper)
    canvas[ly1:ly2, lx1:lx2] = 1
    # right arm
    rx2 = min(w-1, cx + torso_w//2 + limb_len_upper//2)
    rx1 = max(0, rx2 - limb_w)
    ry1 = arm_y
    ry2 = min(h-1, ry1 + limb_len_upper)
    canvas[ry1:ry2, rx1:rx2] = 1
    # legs
    leg_top = bottom_right[1]
    llx1 = max(0, cx - int(torso_w*0.4))
    llx2 = llx1 + limb_w
    lly1 = leg_top
    lly2 = min(h-1, lly1 + limb_len_lower)
    canvas[lly1:lly2, llx1:llx2] = 1
    rlx2 = min(w-1, cx + int(torso_w*0.4))
    rlx1 = max(0, rlx2 - limb_w)
    rly1 = leg_top
    rly2 = min(h-1, rly1 + limb_len_lower)
    canvas[rly1:rly2, rlx1:rlx2] = 1
    canvas = cv2.dilate(canvas, np.ones((3,3), np.uint8))
    canvas = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    return (canvas > 0).astype(np.uint8)

def generate_occluder_mask(h:int, w:int,
                           occ_type: str,
                           size_scale: float,
                           position: Tuple[int,int],
                           orientation: float = 0.0) -> np.ndarray:
    if occ_type == "rectangle":
        occ_w = int(w * size_scale * random.uniform(0.6, 1.0))
        occ_h = int(h * size_scale * random.uniform(0.6, 1.0))
        x1 = max(0, position[0] - occ_w//2)
        y1 = max(0, position[1] - occ_h//2)
        x2 = min(w-1, x1 + occ_w)
        y2 = min(h-1, y1 + occ_h)
        return draw_rectangle_mask(h, w, x1, y1, x2, y2)
    if occ_type == "ellipse":
        axes = (max(2, int(w * size_scale * random.uniform(0.3, 0.6))),
                max(2, int(h * size_scale * random.uniform(0.3, 0.6))))
        return draw_ellipse_mask(h, w, position, axes, angle=orientation)
    if occ_type == "blob":
        return draw_blob_mask(h, w, position, scale=size_scale, noise=random.uniform(0.1, 0.4))
    if occ_type == "human_like":
        return draw_part_based_human_mask(h, w, position, scale=size_scale*1.1, flip=(random.random()<0.5))
    return draw_rectangle_mask(h, w, max(0,position[0]-5), max(0,position[1]-5), min(w-1,position[0]+5), min(h-1,position[1]+5))

def union_silhouette(seq: List[np.ndarray]) -> np.ndarray:
    U = np.zeros_like(seq[0], dtype=np.uint8)
    for s in seq:
        U = np.logical_or(U, (s>0)).astype(np.uint8)
    return U

def compute_overlap_fraction(occl_mask: np.ndarray, silhouette_mask: np.ndarray) -> float:
    inter = np.logical_and(occl_mask, (silhouette_mask>0)).sum()
    denom = (silhouette_mask>0).sum()
    if denom == 0:
        return 0.0
    return float(inter) / float(denom)

# ---------------- noise injection (new) ----------------

def add_noise_patches(
    img: np.ndarray,
    min_patches: int = 10,
    max_patches: int = 60,
    min_patch_frac: float = 0.02,
    max_patch_frac: float = 0.25,
    rect_prob: float = 0.6,
    patch_value: Optional[int] = 0,  # use 0 for black occlusion; 255 for white; None -> random per patch
    preserve_binary: bool = True,
    binary_thresh: int = 127,
    apply_to: str = "both",  # "both" | "foreground" | "background"
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Add random rectangle/ellipse patches to a grayscale/binary image.

    - img: uint8 single channel image (e.g., binary silhouette 0/255)
    - apply_to: "both" (anywhere), "foreground" (only where img>0), "background" (only where img==0)
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random

    h, w = img.shape[:2]
    out = img.copy().astype(np.uint8)

    # absolute patch size bounds
    min_pw = max(1, int(min_patch_frac * w))
    min_ph = max(1, int(min_patch_frac * h))
    max_pw = max(1, int(max_patch_frac * w))
    max_ph = max(1, int(max_patch_frac * h))
    if max_pw < min_pw: max_pw = min_pw
    if max_ph < min_ph: max_ph = min_ph

    num_patches = int(rng.randint(min_patches, max_patches + 1))

    for _ in range(num_patches):
        pw = int(rng.randint(min_pw, max_pw + 1))
        ph = int(rng.randint(min_ph, max_ph + 1))

        # choose a candidate top-left; if we need to restrict to foreground/background, sample until valid or max trials
        max_trials = 50
        chosen = False
        for trial in range(max_trials):
            x = int(rng.randint(0, max(1, w - pw + 1)))
            y = int(rng.randint(0, max(1, h - ph + 1)))
            patch_region = out[y:y+ph, x:x+pw]
            if apply_to == "both":
                chosen = True
                break
            elif apply_to == "foreground":
                # require at least some foreground pixel inside candidate (so we occlude person)
                if (patch_region > 0).any():
                    chosen = True
                    break
            elif apply_to == "background":
                # require at least some background pixel inside candidate (so we occlude background)
                if (patch_region == 0).any():
                    chosen = True
                    break
            else:
                chosen = True
                break
        if not chosen:
            # fallback to generic placement
            x = int(rng.randint(0, max(1, w - pw + 1)))
            y = int(rng.randint(0, max(1, h - ph + 1)))

        val = patch_value if patch_value is not None else (0 if rng.rand() < 0.5 else 255)

        if rng.rand() < rect_prob:
            out[y:y + ph, x:x + pw] = val
        else:
            center = (int(x + pw/2), int(y + ph/2))
            axes = (max(1, int(pw/2)), max(1, int(ph/2)))
            angle = int(rng.randint(0, 360))
            cv2.ellipse(out, center, axes, angle, 0, 360, int(val), -1)

    if preserve_binary:
        _, out = cv2.threshold(out, binary_thresh, 255, cv2.THRESH_BINARY)

    return out

# ---------------- main occluder application (modified) ----------------

def apply_0_to_many_occluders(
    seq: List[np.ndarray],
    occluder_type_choices: List[str] = ["rectangle","ellipse","blob","human_like"],
    mode: str = "blind_random",
    max_occluders: int = 3,
    min_occluders: int = 0,
    overlap_thresh: float = 0.25,
    ensure_overlap_trials: int = 80,
    force_overlap: bool = True,            # NEW: ensure at least one occluder overlaps silhouette if possible
    random_seed: Optional[int] = None
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Improved occluder placement that locates the person more robustly.

    Key changes:
      - Uses the union silhouette and extracts the Largest Connected Component (LC) as the person's region.
      - Samples occluder centers from pixels within that LC (so placements are on/near the person).
      - Tests multiple sizes/orientations and enforces overlap fraction threshold.
      - If force_overlap=True, attempts extra retries to guarantee at least one occluder overlaps.

    Returns (occluded_sequence, occluder_union_mask).
    """
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    if len(seq) == 0:
        return [], np.zeros((0,0), dtype=np.uint8)

    h, w = seq[0].shape
    sil_union = union_silhouette(seq)  # existing helper

    # --- find largest connected component in sil_union for robust person location ---
    # label components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(sil_union.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        # no foreground found
        lc_mask = sil_union.copy()
    else:
        # pick largest non-background component (skip label 0)
        comp_areas = stats[1:, cv2.CC_STAT_AREA]
        largest_idx = int(np.argmax(comp_areas)) + 1
        lc_mask = (labels == largest_idx).astype(np.uint8)

    # fallback: if lc_mask empty, use sil_union
    if lc_mask.sum() == 0:
        lc_mask = sil_union.copy()

    # precompute silhouette area for overlap fraction denominator
    sil_area = float((sil_union > 0).sum())
    if sil_area == 0:
        sil_area = 1.0

    # sample number of occluders
    min_o = max(0, int(min_occluders))
    max_o = max(0, int(max_occluders))
    if max_o < min_o:
        max_o = min_o
    num_occluders = random.randint(min_o, max_o)

    if num_occluders == 0:
        return seq.copy(), np.zeros((h, w), dtype=np.uint8)

    occluder_union = np.zeros((h, w), dtype=np.uint8)
    placed_any_overlap = False

    for i in range(num_occluders):
        placed = False

        # choose a type for this occluder (repetition allowed)
        typ = random.choice(occluder_type_choices)

        if mode == "blind_random":
            pos = (random.randint(0, w-1), random.randint(0, h-1))
            size = random.uniform(0.15, 0.7)
            orient = random.uniform(-30, 30)
            occ = generate_occluder_mask(h, w, typ, size, pos, orient)
            occluder_union = np.logical_or(occluder_union, occ).astype(np.uint8)
            placed = True
            # check if it overlapped silhouette (for tracking)
            if compute_overlap_fraction(occ, sil_union) >= overlap_thresh:
                placed_any_overlap = True

        elif mode == "ensure_overlap":
            # We will attempt different strategies in order:
            # 1) Sample center directly from LC mask pixels (most likely to overlap)
            # 2) Sample from centroid ± jitter
            # 3) Random fallback
            trials = ensure_overlap_trials
            for t in range(trials):
                # strategy: sample center from LC pixels with probability 0.85 else random
                if lc_mask.sum() > 0 and random.random() < 0.85:
                    ys, xs = np.where(lc_mask)
                    idx = random.randint(0, len(xs)-1)
                    cx = int(xs[idx] + random.randint(-int(w*0.05), int(w*0.05)))
                    cy = int(ys[idx] + random.randint(-int(h*0.05), int(h*0.05)))
                    pos = (np.clip(cx, 0, w-1), np.clip(cy, 0, h-1))
                elif sil_union.sum() > 0 and random.random() < 0.5:
                    # sample near silhouette centroid with jitter
                    ys, xs = np.where(sil_union)
                    cx = int(np.mean(xs) + random.randint(-int(w*0.15), int(w*0.15)))
                    cy = int(np.mean(ys) + random.randint(-int(h*0.15), int(h*0.15)))
                    pos = (np.clip(cx, 0, w-1), np.clip(cy, 0, h-1))
                else:
                    pos = (random.randint(0, w-1), random.randint(0, h-1))

                # choose size scaled to silhouette extents: sample size_scale relative to silhouette bbox
                ys_s, xs_s = np.where(lc_mask)
                if len(xs_s) > 0:
                    sil_w = max(1, int(xs_s.max() - xs_s.min()))
                    sil_h = max(1, int(ys_s.max() - ys_s.min()))
                    # sample size relative to silhouette bbox to increase chance of overlap
                    base_scale = float(max(sil_w / float(w), sil_h / float(h)))
                    # jitter around base_scale
                    size = float(max(0.12, min(0.9, base_scale * random.uniform(0.7, 1.6))))
                else:
                    size = random.uniform(0.2, 0.6)

                orient = random.uniform(-30, 30)
                occ = generate_occluder_mask(h, w, typ, size, pos, orient)
                frac = compute_overlap_fraction(occ, sil_union)

                if frac >= overlap_thresh:
                    occluder_union = np.logical_or(occluder_union, occ).astype(np.uint8)
                    placed = True
                    placed_any_overlap = True
                    break

            # if trials failed and force_overlap requested, try an aggressive placement centered on LC centroid
            if (not placed) and force_overlap and lc_mask.sum() > 0:
                ys_lc, xs_lc = np.where(lc_mask)
                centroid_x = int(np.mean(xs_lc))
                centroid_y = int(np.mean(ys_lc))
                # try a few scales centered exactly at LC centroid
                for scale_try in [0.9, 0.7, 0.5, 0.35, 0.25]:
                    pos = (centroid_x, centroid_y)
                    occ = generate_occluder_mask(h, w, typ, float(scale_try), pos, random.uniform(-30,30))
                    frac = compute_overlap_fraction(occ, sil_union)
                    if frac >= overlap_thresh * 0.6:  # accept slightly lower if forced
                        occluder_union = np.logical_or(occluder_union, occ).astype(np.uint8)
                        placed = True
                        placed_any_overlap = True
                        break

            # final fallback: blind random placement
            if not placed:
                pos = (random.randint(0, w-1), random.randint(0, h-1))
                size = random.uniform(0.12, 0.6)
                orient = random.uniform(-30, 30)
                occ = generate_occluder_mask(h, w, typ, size, pos, orient)
                occluder_union = np.logical_or(occluder_union, occ).astype(np.uint8)
                # we don't mark placed_any_overlap here unless it overlaps
                if compute_overlap_fraction(occ, sil_union) >= overlap_thresh:
                    placed_any_overlap = True
                placed = True

        else:
            raise ValueError("mode must be 'blind_random' or 'ensure_overlap'")

    # After placing all occluders, if force_overlap=True and none overlapped, try to force one occluder
    if force_overlap and (not placed_any_overlap) and (lc_mask.sum() > 0):
        # make one occluder that covers a chunk of LC
        ys_lc, xs_lc = np.where(lc_mask)
        centroid_x = int(np.mean(xs_lc))
        centroid_y = int(np.mean(ys_lc))
        typ = random.choice(occluder_type_choices)
        # size big enough to cover a chunk
        occ = generate_occluder_mask(h, w, typ, 0.5, (centroid_x, centroid_y), random.uniform(-20,20))
        occluder_union = np.logical_or(occluder_union, occ).astype(np.uint8)

    # apply occluder union (static across sequence)
    occluded_seq = []
    for s in seq:
        s_bin = (s > 0).astype(np.uint8)
        occl = s_bin.copy()
        occl[occluder_union == 1] = 0
        occluded_seq.append(occl)

    return occluded_seq, occluder_union

# ----------------- Example usage -----------------
if __name__ == "__main__":
    # ---------- user-configurable params ----------
    INPUT_ROOT = r"E:\Gait_IIT_BHU_Data_D\Inbuilt_1\Merged_Normalized_256x256_Vertically_Horizontally_Aligned_Refind_Images"
    OUTPUT_ROOT = r"E:\Gait_IIT_BHU_Data_D\Inbuilt_1\Merged_Normalized_256x256_Vertically_Horizontally_Aligned_Refind_Images_Noisy"
    PROCESS_RECURSIVELY = True
    MODE = "ensure_overlap"
    MAX_OCCLUDERS = 4
    MIN_OCCLUDERS = 0
    RANDOM_SEED = 42
    OVERLAP_THRESH = 0.25

    # ---- noise params ----
    ENABLE_NOISE = True
    NOISE_MIN_PATCHES = 8
    NOISE_MAX_PATCHES = 50
    NOISE_MIN_PATCH_FRAC = 0.02
    NOISE_MAX_PATCH_FRAC = 0.20
    NOISE_RECT_PROB = 0.7
    NOISE_PATCH_VALUE = 0
    NOISE_PRESERVE_BINARY = True
    NOISE_APPLY_TO = "both"
    # ---------------------------------------------

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    def list_sequence_folders(root, recursive=True):
        """
        Recursively collect all folders containing image sequences (multiple frames per sequence)
        while keeping the folder structure.
        """
        seq_folders = []
        if recursive:
            for subdir, _, _ in os.walk(root):
                image_glob = glob.glob(os.path.join(subdir, "*.png")) + \
                             glob.glob(os.path.join(subdir, "*.jpg")) + \
                             glob.glob(os.path.join(subdir, "*.bmp"))
                if len(image_glob) > 0:
                    seq_folders.append(subdir)
        else:
            seq_folders.append(root)
        return seq_folders

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    seq_folders = list_sequence_folders(INPUT_ROOT, recursive=PROCESS_RECURSIVELY)
    if len(seq_folders) == 0:
        print(f"No image sequences found under {INPUT_ROOT}.")
    else:
        print(f"Found {len(seq_folders)} sequence folders. Processing...")

    for seq_idx, seq_folder in enumerate(seq_folders):
        img_paths = sorted(
            glob.glob(os.path.join(seq_folder, "*.png")) +
            glob.glob(os.path.join(seq_folder, "*.jpg")) +
            glob.glob(os.path.join(seq_folder, "*.bmp"))
        )
        if len(img_paths) == 0:
            print(f"Skipping {seq_folder} (no images).")
            continue

        # load sequence
        seq = []
        for p in img_paths:
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Warning: failed to read {p}, skipping.")
                continue
            _, binary = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
            seq.append(binary.astype(np.uint8))

        if len(seq) == 0:
            print(f"No valid images loaded for {seq_folder}, skipping.")
            continue

        # apply occluders
        occl_seq, occluder_union = apply_0_to_many_occluders(
            seq,
            occluder_type_choices=["rectangle", "ellipse", "blob", "human_like"],
            mode=MODE,
            max_occluders=MAX_OCCLUDERS,
            min_occluders=MIN_OCCLUDERS,
            overlap_thresh=OVERLAP_THRESH,
            ensure_overlap_trials=50,
            random_seed=RANDOM_SEED
        )

        # optionally add noise
        if ENABLE_NOISE:
            noisy_seq = []
            for idx, frame in enumerate(occl_seq):
                frame_seed = RANDOM_SEED + seq_idx * 1000 + idx
                noisy = add_noise_patches(
                    (frame * 255).astype(np.uint8),
                    min_patches=NOISE_MIN_PATCHES,
                    max_patches=NOISE_MAX_PATCHES,
                    min_patch_frac=NOISE_MIN_PATCH_FRAC,
                    max_patch_frac=NOISE_MAX_PATCH_FRAC,
                    rect_prob=NOISE_RECT_PROB,
                    patch_value=NOISE_PATCH_VALUE,
                    preserve_binary=NOISE_PRESERVE_BINARY,
                    binary_thresh=127,
                    apply_to=NOISE_APPLY_TO,
                    seed=frame_seed
                )
                _, noisy_bin = cv2.threshold(noisy, 127, 1, cv2.THRESH_BINARY)
                noisy_seq.append(noisy_bin.astype(np.uint8))
            occl_seq = noisy_seq

        # prepare output folder preserving subject/sequence structure
        rel_path = os.path.relpath(seq_folder, INPUT_ROOT)
        out_seq_dir = os.path.join(OUTPUT_ROOT, rel_path)
        os.makedirs(out_seq_dir, exist_ok=True)

        # save occluded/noisy frames
        for i, (orig, occ) in enumerate(zip(seq, occl_seq)):
            out_occ  = os.path.join(out_seq_dir, f"frame_{i:04d}_occl.png")
            cv2.imwrite(out_occ, (occ * 255).astype(np.uint8))

        print(f"[{seq_idx+1}/{len(seq_folders)}] Saved occluded sequence to {out_seq_dir}")


    print("All done.")
