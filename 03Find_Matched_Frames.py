import os
import glob
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ======================================================================================
# OUTPUT
# ======================================================================================
SHOW_TOPK_COS = 10
PRINT_FRAMES = True

SAVE_MIN_LEN = 4
SAVE_DIRNAME = "_mappings"
SAVE_FILENAME = "mapped_pairs_len_ge_4.txt"

# SUBJECT RANGE (as you said)
SUBJECT_ID_MIN = 201
SUBJECT_ID_MAX = 333

# ======================================================================================
# Windows long-path helper
# ======================================================================================
def win_long_path(p: str) -> str:
    p = os.path.abspath(p)
    if p.startswith("\\\\?\\"):
        return p
    if p.startswith("\\\\"):
        return "\\\\?\\UNC\\" + p[2:]
    return "\\\\?\\" + p


# ======================================================================================
# Parsing helpers (no regex)
# ======================================================================================
def extract_frame_index(path_or_name: str) -> Optional[int]:
    base = os.path.basename(path_or_name)
    name, _ = os.path.splitext(base)
    parts = name.split("_")
    for p in parts:
        if p.startswith("frame"):
            num = p[len("frame"):]
            digits = "".join(ch for ch in num if ch.isdigit())
            return int(digits) if digits else None
    return None


def extract_pose_number(path_or_name: str) -> Optional[int]:
    base = os.path.basename(path_or_name)
    name, _ = os.path.splitext(base)
    parts = name.split("_")
    for tok in reversed(parts):
        tok = tok.strip()
        trailing = []
        for ch in reversed(tok):
            if ch.isdigit():
                trailing.append(ch)
            else:
                break
        if trailing:
            return int("".join(reversed(trailing)))
    return None


def list_frames_sorted(folder: str) -> List[str]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.webp")
    files: List[str] = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, e)))

    def _key(p: str):
        idx = extract_frame_index(p)
        return (idx if idx is not None else 10**12, os.path.basename(p))

    return sorted(files, key=_key)


# ======================================================================================
# Robust image read: OpenCV + PIL fallback
# ======================================================================================
def imread_gray_mixed(img_path: str) -> Optional[np.ndarray]:
    p = win_long_path(img_path)
    if not os.path.exists(p):
        return None

    # --- Try OpenCV first ---
    try:
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    except Exception:
        img = None

    if img is not None and getattr(img, "size", 0) > 0:
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if img.dtype == np.uint16:
            img = (img / 256).astype(np.uint8)
        elif img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        return img

    # --- Fallback to PIL ---
    try:
        with Image.open(p) as im:
            im.load()
            im = im.convert("L")
            arr = np.array(im)
            if arr is None or arr.size == 0:
                return None
            if arr.dtype != np.uint8:
                arr = arr.astype(np.uint8)
            return arr
    except Exception:
        return None


# ======================================================================================
# Cosine feature utilities
# ======================================================================================
def load_feature_vector(img_path: str, cache: Dict[str, np.ndarray], size: int = 64) -> Optional[np.ndarray]:
    if img_path in cache:
        return cache[img_path]

    img = imread_gray_mixed(img_path)
    if img is None:
        return None

    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    v = img.astype(np.float32).reshape(-1)
    v = v / float(np.linalg.norm(v) + 1e-8)

    cache[img_path] = v
    return v


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def get_noisy_subfolder_name(noisy_path: str) -> str:
    return os.path.basename(os.path.dirname(noisy_path))


# ======================================================================================
# DP: best contiguous pose match with masks
# ======================================================================================
def best_contiguous_pose_match_with_masks(
    clean_pose: List[Optional[int]],
    noisy_pose: List[Optional[int]],
    clean_ok: List[bool],
    noisy_ok: List[bool],
) -> Tuple[int, int, int]:
    n, m = len(clean_pose), len(noisy_pose)
    dp = [0] * (m + 1)

    best_len = 0
    best_end_i = -1
    best_end_j = -1

    for i in range(1, n + 1):
        prev_diag = 0
        ci = i - 1
        for j in range(1, m + 1):
            tmp = dp[j]
            nj = j - 1

            if (
                clean_ok[ci] and noisy_ok[nj]
                and clean_pose[ci] is not None and noisy_pose[nj] is not None
                and clean_pose[ci] == noisy_pose[nj]
            ):
                dp[j] = prev_diag + 1
                if dp[j] > best_len:
                    best_len = dp[j]
                    best_end_i = ci
                    best_end_j = nj
            else:
                dp[j] = 0

            prev_diag = tmp

    if best_len == 0:
        return 0, 0, 0

    clean_start = best_end_i - best_len + 1
    noisy_start = best_end_j - best_len + 1
    return clean_start, noisy_start, best_len


# ======================================================================================
# Folder discovery inside a sequence
# ======================================================================================
def list_subsequence_folders(sequence_root: str) -> Tuple[List[str], List[str]]:
    all_dirs = [d for d in glob.glob(os.path.join(sequence_root, "*")) if os.path.isdir(d)]
    clean, noisy = [], []
    for d in all_dirs:
        name = os.path.basename(d).lower()
        if name.startswith("subsequence_"):
            clean.append(d)
        elif name.startswith("noisy_subsequence_"):
            noisy.append(d)

    def suffix_num(folder: str) -> int:
        base = os.path.basename(folder)
        tok = base.split("_")[-1]
        digits = "".join(ch for ch in tok if ch.isdigit())
        return int(digits) if digits else 10**12

    clean.sort(key=lambda x: (suffix_num(x), os.path.basename(x)))
    noisy.sort(key=lambda x: (suffix_num(x), os.path.basename(x)))
    return clean, noisy


def build_global_noisy_pool(noisy_folders: List[str]) -> Dict[int, List[str]]:
    pool: Dict[int, List[str]] = {}
    for nf in noisy_folders:
        files = list_frames_sorted(nf)
        for f in files:
            p = extract_pose_number(f)
            if p is None:
                continue
            pool.setdefault(p, []).append(f)
    return pool


# ======================================================================================
# Printing helpers
# ======================================================================================
def print_pairs(title: str, pairs: List[Tuple[str, str]]) -> None:
    print(title)
    for cf, nf in pairs:
        pose = extract_pose_number(cf)
        print(f"  [pose={pose}] {os.path.basename(cf)}  <-->  {get_noisy_subfolder_name(nf)}/{os.path.basename(nf)}")


# ======================================================================================
# Save mapping (per sequence)
# ======================================================================================
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def mapping_out_path(sequence_root: str) -> str:
    out_dir = os.path.join(sequence_root, SAVE_DIRNAME)
    ensure_dir(out_dir)
    return os.path.join(out_dir, SAVE_FILENAME)


def append_mapping_block(
    out_txt: str,
    clean_subseq: str,
    noisy_core_subseq: str,
    c_start: int,
    n_start: int,
    before_len: int,
    after_len: int,
    pairs: List[Tuple[str, str]],
    ext_detail: Optional[Dict[str, Any]],
) -> None:
    with open(out_txt, "a", encoding="utf-8") as f:
        f.write("=" * 120 + "\n")
        f.write(f"CLEAN_SUBSEQ        : {clean_subseq}\n")
        f.write(f"NOISY_CORE_SUBSEQ   : {noisy_core_subseq}\n")
        f.write(f"CORE_STARTS         : clean_start={c_start} noisy_start={n_start}\n")
        f.write(f"LENGTH              : before={before_len} after={after_len}\n")

        if ext_detail and ext_detail.get("status") == "ADDED":
            f.write("EXTENSION           : ADDED\n")
            f.write(f"  side              : {ext_detail.get('side')}\n")
            f.write(f"  pose              : {ext_detail.get('pose')}\n")
            f.write(f"  clean_frame       : {ext_detail.get('clean_frame')}\n")
            f.write(f"  noisy_frame       : {ext_detail.get('noisy_subseq')}/{ext_detail.get('noisy_frame')}\n")
            f.write(f"  cosine            : {ext_detail.get('cosine'):.6f}\n")
        else:
            f.write("EXTENSION           : NONE\n")

        f.write("\nPAIRS (clean_path <TAB> noisy_path):\n")
        for cf, nf in pairs:
            f.write(f"{cf}\t{nf}\n")
        f.write("\n")


# ======================================================================================
# RIGHT extension using GLOBAL noisy pool (within SAME sequence)
# ======================================================================================
def extend_right_global(
    clean_files: List[str],
    clean_pose: List[Optional[int]],
    clean_ok: List[bool],
    c_start: int,
    L: int,
    core_pairs: List[Tuple[str, str]],
    core_pose_seq: List[int],
    global_noisy_pool: Dict[int, List[str]],
    global_used_noisy: set,
    feat_cache: Dict[str, np.ndarray],
    feat_size: int = 64,
    topk_cos: int = 10,
) -> Dict[str, Any]:
    matched_pairs = list(core_pairs)
    matched_pose_seq = list(core_pose_seq)

    right_ci = c_start + L

    if not (0 <= right_ci < len(clean_files)):
        return {"matched_pairs": matched_pairs, "matched_pose_seq": matched_pose_seq, "right_added": False, "right_detail": None}
    if not clean_ok[right_ci]:
        return {"matched_pairs": matched_pairs, "matched_pose_seq": matched_pose_seq, "right_added": False, "right_detail": None}

    pose_needed = clean_pose[right_ci]
    if pose_needed is None:
        return {"matched_pairs": matched_pairs, "matched_pose_seq": matched_pose_seq, "right_added": False, "right_detail": None}

    clean_vec = load_feature_vector(clean_files[right_ci], feat_cache, size=feat_size)
    if clean_vec is None:
        return {
            "matched_pairs": matched_pairs,
            "matched_pose_seq": matched_pose_seq,
            "right_added": False,
            "right_detail": {"status": "FAIL_CLEAN_UNREADABLE", "clean_frame": os.path.basename(clean_files[right_ci]), "pose": pose_needed},
        }

    candidates_all = global_noisy_pool.get(pose_needed, [])
    candidates_free = [p for p in candidates_all if p not in global_used_noisy]
    if not candidates_free:
        return {
            "matched_pairs": matched_pairs,
            "matched_pose_seq": matched_pose_seq,
            "right_added": False,
            "right_detail": {"status": "FAIL_NO_FREE_CANDIDATES", "clean_frame": os.path.basename(clean_files[right_ci]), "pose": pose_needed, "total_candidates": len(candidates_all)},
        }

    scored: List[Tuple[float, str]] = []
    for npth in candidates_free:
        nvec = load_feature_vector(npth, feat_cache, size=feat_size)
        if nvec is None:
            continue
        scored.append((float(cosine_sim(clean_vec, nvec)), npth))

    if not scored:
        return {
            "matched_pairs": matched_pairs,
            "matched_pose_seq": matched_pose_seq,
            "right_added": False,
            "right_detail": {"status": "FAIL_ALL_CANDIDATES_UNREADABLE", "clean_frame": os.path.basename(clean_files[right_ci]), "pose": pose_needed, "free_candidates": len(candidates_free)},
        }

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_path = scored[0]

    matched_pairs.append((clean_files[right_ci], best_path))
    matched_pose_seq.append(pose_needed)
    global_used_noisy.add(best_path)
    clean_ok[right_ci] = False

    ext_detail = {
        "status": "ADDED",
        "side": "RIGHT",
        "pose": pose_needed,
        "clean_frame": os.path.basename(clean_files[right_ci]),
        "noisy_frame": os.path.basename(best_path),
        "noisy_subseq": get_noisy_subfolder_name(best_path),
        "cosine": float(best_score),
        "topk": [(float(sc), get_noisy_subfolder_name(p), os.path.basename(p)) for sc, p in scored[:max(1, topk_cos)]],
    }

    return {"matched_pairs": matched_pairs, "matched_pose_seq": matched_pose_seq, "right_added": True, "right_detail": ext_detail}


# ======================================================================================
# Run mapping for ONE SEQUENCE (your existing logic)
# ======================================================================================
def run_one_sequence(
    sequence_root: str,
    feat_size: int = 64,
    min_core_len: int = 1,
    topk_cos: int = 10,
    save_min_len: int = 4,
) -> int:
    clean_folders, noisy_folders = list_subsequence_folders(sequence_root)
    if not clean_folders or not noisy_folders:
        return 0

    global_pool = build_global_noisy_pool(noisy_folders)
    global_used_noisy = set()

    out_txt = mapping_out_path(sequence_root)
    try:
        if os.path.exists(out_txt):
            os.remove(out_txt)
    except Exception:
        pass

    saved_blocks = 0

    for clean_folder in clean_folders:
        for noisy_folder in noisy_folders:
            clean_files = list_frames_sorted(clean_folder)
            noisy_files = list_frames_sorted(noisy_folder)

            clean_pose = [extract_pose_number(f) for f in clean_files]
            noisy_pose = [extract_pose_number(f) for f in noisy_files]

            clean_ok = [True] * len(clean_files)
            noisy_ok = [True] * len(noisy_files)

            c_start, n_start, L = best_contiguous_pose_match_with_masks(clean_pose, noisy_pose, clean_ok, noisy_ok)
            if L < min_core_len:
                continue

            core_pairs = list(zip(clean_files[c_start:c_start + L], noisy_files[n_start:n_start + L]))
            core_pose_seq = list(clean_pose[c_start:c_start + L])

            # Mark core noisy frames globally used (within this sequence)
            for _, npth in core_pairs:
                global_used_noisy.add(npth)

            # BEFORE
            print("\n" + "=" * 110)
            print(f"SEQUENCE: {os.path.basename(sequence_root)}")
            print(f"CLEAN: {os.path.basename(clean_folder)}")
            print(f"NOISY (core match): {os.path.basename(noisy_folder)}")
            print("-" * 110)
            print(f"BEFORE (core) len={L}  (clean_start={c_start}, noisy_start={n_start})")
            print(f"BEFORE pose_seq = {core_pose_seq}")
            if PRINT_FRAMES:
                print_pairs("BEFORE frames (CLEAN <--> NOISY_CORE):", core_pairs)

            # AFTER (+1)
            feat_cache: Dict[str, np.ndarray] = {}
            ext = extend_right_global(
                clean_files=clean_files,
                clean_pose=clean_pose,
                clean_ok=clean_ok,
                c_start=c_start,
                L=L,
                core_pairs=core_pairs,
                core_pose_seq=core_pose_seq,
                global_noisy_pool=global_pool,
                global_used_noisy=global_used_noisy,
                feat_cache=feat_cache,
                feat_size=feat_size,
                topk_cos=topk_cos,
            )

            after_pairs = ext["matched_pairs"]
            after_pose = ext["matched_pose_seq"]
            after_len = len(after_pairs)

            print("\nAFTER (extended) "
                  f"{'RIGHT+1' if ext['right_added'] else 'no_extension'}  len={after_len}")
            print(f"AFTER pose_seq = {after_pose}")
            if PRINT_FRAMES:
                print_pairs("AFTER frames (CLEAN <--> NOISY_ANY):", after_pairs)

            # Extension summary
            detail = ext.get("right_detail")
            if ext["right_added"] and detail and detail.get("status") == "ADDED":
                print("\nEXTENSION USED:")
                print(f"  clean     : {detail['clean_frame']}  (pose={detail['pose']})")
                print(f"  noisy     : {detail['noisy_subseq']}/{detail['noisy_frame']}")
                print(f"  cosine    : {detail['cosine']:.6f}")

            # SAVE only if after_len >= save_min_len
            if after_len >= save_min_len:
                append_mapping_block(
                    out_txt=out_txt,
                    clean_subseq=os.path.basename(clean_folder),
                    noisy_core_subseq=os.path.basename(noisy_folder),
                    c_start=c_start,
                    n_start=n_start,
                    before_len=L,
                    after_len=after_len,
                    pairs=after_pairs,
                    ext_detail=detail if (detail and isinstance(detail, dict)) else None,
                )
                saved_blocks += 1
                print(f"\n[SAVED] {out_txt}  (len={after_len} >= {save_min_len})")
            else:
                print(f"\n[NOT SAVED] len={after_len} < {save_min_len}")

            print("=" * 110)

    if saved_blocks > 0:
        print(f"\n[SEQ DONE] Saved {saved_blocks} block(s) -> {out_txt}")
    return saved_blocks


# ======================================================================================
# MULTI-SUBJECT, MULTI-SEQUENCE runner
# - mapping stays inside each sequence
# - subject folders assumed like: <root>\203\Sequence_05_right_to_left
# ======================================================================================
def list_sequences_for_subject(subject_dir: str) -> List[str]:
    # You can restrict here if needed (e.g., only Sequence_*)
    seq_dirs = [d for d in glob.glob(os.path.join(subject_dir, "*")) if os.path.isdir(d)]
    seq_dirs = [d for d in seq_dirs if os.path.basename(d).lower().startswith("sequence_")]
    return sorted(seq_dirs, key=lambda x: os.path.basename(x).lower())


def run_all_subjects_and_sequences(
    dataset_root: str,
    subject_min: int = 201,
    subject_max: int = 333,
    feat_size: int = 64,
    min_core_len: int = 1,
    topk_cos: int = 10,
    save_min_len: int = 4,
):
    total_saved = 0
    total_sequences = 0
    total_subjects_found = 0

    for sid in range(subject_min, subject_max + 1):
        subject_dir = os.path.join(dataset_root, str(sid))
        if not os.path.isdir(subject_dir):
            continue

        total_subjects_found += 1
        seq_dirs = list_sequences_for_subject(subject_dir)
        if not seq_dirs:
            continue

        print("\n" + "#" * 120)
        print(f"SUBJECT {sid}  |  sequences={len(seq_dirs)}")
        print("#" * 120)

        for seq_root in seq_dirs:
            total_sequences += 1
            saved = run_one_sequence(
                sequence_root=seq_root,
                feat_size=feat_size,
                min_core_len=min_core_len,
                topk_cos=topk_cos,
                save_min_len=save_min_len,
            )
            total_saved += saved

    print("\n" + "=" * 120)
    print("ALL DONE")
    print(f"subjects_found   : {total_subjects_found}")
    print(f"sequences_scanned: {total_sequences}")
    print(f"mapping_blocks   : {total_saved} (saved when after_len >= {save_min_len})")
    print("=" * 120)


# ======================================================================================
# MAIN
# ======================================================================================
if __name__ == "__main__":
    dataset_root = r"D:\Aggregated-Pose-Energy-Image\GaitTemporalCompletion\data\Merged_Normalized_256x256_Vertically_Horizontally_Aligned_Refind_Images_DGEI_17_Renamed_IntraCorrected_WithNoisySubseq"

    run_all_subjects_and_sequences(
        dataset_root=dataset_root,
        subject_min=SUBJECT_ID_MIN,
        subject_max=SUBJECT_ID_MAX,
        feat_size=64,
        min_core_len=1,
        topk_cos=SHOW_TOPK_COS,
        save_min_len=SAVE_MIN_LEN,
    )
