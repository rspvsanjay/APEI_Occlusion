import os
import re
import glob
import cv2
import numpy as np

# =============================
# Config
# =============================
THRESHOLD = 127

# Pose correction
RESIZE_HW = (256, 256)
POSE_WINDOW = 2                 # only allow pose change within ±2 (cyclic)
MIN_SIM_TO_CHANGE = 0.10        # if similarity < this, keep original pose

# Segment rule
MIN_CONSEC_LEN = 5              # "more than 4" => >= 5

# Roots
CLEAN_ROOT = r"D:\Aggregated-Pose-Energy-Image\GaitTemporalCompletion\data\Merged_Normalized_256x256_Vertically_Horizontally_Aligned_Refind_Images_DGEI_17_Renamed"
OUT_ROOT   = r"D:\Aggregated-Pose-Energy-Image\GaitTemporalCompletion\data\Merged_Normalized_256x256_Vertically_Horizontally_Aligned_Refind_Images_DGEI_17_Renamed_selected"
FINAL_ROOT = r"D:\Aggregated-Pose-Energy-Image\GaitTemporalCompletion\data\Merged_Normalized_256x256_Vertically_Horizontally_Aligned_Refind_Images_DGEI_17_Renamed_IntraCorrected_WithNoisySubseq"

SUBJECT_START = 201
SUBJECT_END = 333
FIXED_NUM_POSES = 17


# =============================
# Helpers
# =============================
def get_file_list(folder: str):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff")
    files = []
    for e in exts:
        files += glob.glob(os.path.join(folder, e))
    return sorted(files)

def get_subfolders(path: str):
    return sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def imwrite_gray(dst_path: str, src_path: str):
    img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    ensure_dir(os.path.dirname(dst_path))
    return bool(cv2.imwrite(dst_path, img))


# =============================
# Pose parsing / renaming
# =============================
POSE_SUFFIX_RE = re.compile(r"_(\d{2})(?=\.[^.]+$)")

def parse_pose_suffix(path_or_name: str):
    base = os.path.basename(path_or_name)
    m = POSE_SUFFIX_RE.search(base)
    if not m:
        return None
    return int(m.group(1))

def replace_pose_suffix(path_or_name: str, new_pose: int):
    base = os.path.basename(path_or_name)
    pose2 = f"{new_pose:02d}"
    if POSE_SUFFIX_RE.search(base):
        return POSE_SUFFIX_RE.sub(f"_{pose2}", base)
    b, e = os.path.splitext(base)
    return f"{b}_{pose2}{e}"


# =============================
# Image -> vector + similarity
# =============================
def img_to_vec(path: str, size=(256, 256)):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    if (img.shape[1], img.shape[0]) != size:
        img = cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)

    v = img.astype(np.float32).reshape(-1)
    v = v - v.mean()
    n = np.linalg.norm(v) + 1e-8
    return v / n

def cosine_sim(v1, v2):
    return float(np.dot(v1, v2))


# =============================
# Cyclic pose window
# =============================
def allowed_poses(cur_pose: int, K: int, window: int):
    allowed = set()
    for delta in range(-window, window + 1):
        p = ((cur_pose - 1 + delta) % K) + 1
        allowed.add(p)
    return sorted(allowed)


# =============================
# Selected set from subsequences (also preserve subseq structure)
# =============================
def load_selected_from_subsequences(selected_seq_root: str):
    """
    Returns:
      selected_basenames: set of filenames (basename only)
      pose_to_vecs: dict pose -> list of vectors (from selected frames only)
      selected_pose_set: set of pose numbers present among selected frames
      selected_subseq_map: dict subseq_name -> list of basenames in that subseq
    """
    selected_basenames = set()
    pose_to_vecs = {}
    selected_pose_set = set()
    selected_subseq_map = {}

    if not os.path.isdir(selected_seq_root):
        return selected_basenames, pose_to_vecs, selected_pose_set, selected_subseq_map

    for sub in get_subfolders(selected_seq_root):
        if not sub.lower().startswith("subsequence_"):
            continue
        sub_dir = os.path.join(selected_seq_root, sub)

        for f in get_file_list(sub_dir):
            base = os.path.basename(f)
            selected_basenames.add(base)
            selected_subseq_map.setdefault(sub, []).append(base)

            pose = parse_pose_suffix(base)
            if pose is None:
                continue

            v = img_to_vec(f, size=RESIZE_HW)
            if v is None:
                continue

            selected_pose_set.add(pose)
            pose_to_vecs.setdefault(pose, []).append(v)

    return selected_basenames, pose_to_vecs, selected_pose_set, selected_subseq_map


# =============================
# Consecutive segments on indices
# =============================
def consecutive_segments(idxs, min_len=5):
    if not idxs:
        return []
    idxs = sorted(set(int(x) for x in idxs))
    segs = []
    s = idxs[0]
    p = idxs[0]
    for v in idxs[1:]:
        if v == p + 1:
            p = v
        else:
            if (p - s + 1) >= min_len:
                segs.append((s, p))
            s = v
            p = v
    if (p - s + 1) >= min_len:
        segs.append((s, p))
    return segs


# =============================
# Process one sequence
#   1) Copy selected subsequences as-is (same folder names)
#   2) For non-selected frames:
#        - pose correct within ±2 using selected exemplars
#        - keep only frames whose final pose is in selected_pose_set
#        - create noisy_subsequence_XX for consecutive noisy frames (len>=5)
# =============================
def process_one_sequence(subject: str, seq: str):
    seq_in = os.path.join(CLEAN_ROOT, subject, seq)
    if not os.path.isdir(seq_in):
        print(f"[SKIP] missing input: {seq_in}")
        return

    files = get_file_list(seq_in)
    if not files:
        print(f"[SKIP] no frames: {seq_in}")
        return

    selected_seq_root = os.path.join(OUT_ROOT, subject, seq)
    selected_basenames, pose_to_vecs, selected_pose_set, selected_subseq_map = load_selected_from_subsequences(selected_seq_root)

    out_seq_root = os.path.join(FINAL_ROOT, subject, seq)
    ensure_dir(out_seq_root)

    if not selected_basenames or not pose_to_vecs or not selected_pose_set:
        print(f"[{subject}/{seq}] No selected subsequences/poses found. Saving NOTHING for this sequence.")
        return

    # -------------------------
    # (A) Copy selected subsequences as-is (same subsequence folder names)
    # -------------------------
    copied_selected = 0
    for sub_name, base_list in selected_subseq_map.items():
        dst_sub_dir = os.path.join(out_seq_root, sub_name)
        ensure_dir(dst_sub_dir)
        for base in base_list:
            src = os.path.join(seq_in, base)
            if os.path.exists(src):
                if imwrite_gray(os.path.join(dst_sub_dir, base), src):
                    copied_selected += 1

    # -------------------------
    # (B) Process non-selected frames -> decide if they are saved as noisy frames
    # -------------------------
    K = FIXED_NUM_POSES
    noisy_saved_candidates = []  # list of (frame_index, final_name, src_path)
    changed = 0
    skipped_no_pose = 0
    skipped_no_candidates = 0
    skipped_pose_not_selected = 0

    # Build quick map from basename -> index (to form consecutive segments)
    base_to_idx = {os.path.basename(p): i for i, p in enumerate(files)}

    for i, f in enumerate(files):
        base = os.path.basename(f)

        # Skip selected frames here (already copied with subseq structure)
        if base in selected_basenames:
            continue

        cur_pose = parse_pose_suffix(base)
        if cur_pose is None:
            skipped_no_pose += 1
            continue

        final_pose = cur_pose
        final_name = base

        # try correction within ±2 (using only selected exemplars)
        cand_poses = allowed_poses(cur_pose, K, POSE_WINDOW)
        candidate_vecs = []
        for p in cand_poses:
            if p in pose_to_vecs:
                candidate_vecs.extend([(p, v) for v in pose_to_vecs[p]])

        if not candidate_vecs:
            skipped_no_candidates += 1
        else:
            v = img_to_vec(f, size=RESIZE_HW)
            if v is not None:
                best_pose = None
                best_score = -1.0
                for p, ref in candidate_vecs:
                    s = cosine_sim(v, ref)
                    if s > best_score:
                        best_score = s
                        best_pose = p

                if best_pose is not None and best_pose != cur_pose and best_score >= MIN_SIM_TO_CHANGE:
                    final_pose = best_pose
                    final_name = replace_pose_suffix(base, best_pose)
                    changed += 1

        # Only keep if final pose is one of the selected poses
        if final_pose not in selected_pose_set:
            skipped_pose_not_selected += 1
            continue

        # Candidate noisy frame to be saved later into noisy subsequences (if consecutive len>=5)
        noisy_saved_candidates.append((i, final_name, f))

    # -------------------------
    # (C) Create noisy subsequences from consecutive noisy candidates (len>=5)
    # -------------------------
    noisy_indices = [i for (i, _, _) in noisy_saved_candidates]
    noisy_segs = consecutive_segments(noisy_indices, min_len=MIN_CONSEC_LEN)

    noisy_subseq_count = 0
    noisy_frames_written = 0

    # For quick lookup (index -> (final_name, src_path))
    idx_to_info = {i: (final_name, src_path) for (i, final_name, src_path) in noisy_saved_candidates}

    for (s, e) in noisy_segs:
        noisy_subseq_count += 1
        noisy_dir = os.path.join(out_seq_root, f"noisy_subsequence_{noisy_subseq_count:02d}")
        ensure_dir(noisy_dir)

        for idx in range(s, e + 1):
            if idx not in idx_to_info:
                continue
            final_name, src_path = idx_to_info[idx]
            if imwrite_gray(os.path.join(noisy_dir, final_name), src_path):
                noisy_frames_written += 1

    print(
        f"[{subject}/{seq}] selected_subseq={len(selected_subseq_map)} | copied_selected_frames={copied_selected} "
        f"| noisy_candidates={len(noisy_saved_candidates)} | noisy_subseq_saved={noisy_subseq_count} "
        f"| noisy_frames_saved={noisy_frames_written} | pose_changed_on_noisy={changed} "
        f"| skipped_no_pose={skipped_no_pose} | skipped_no_candidates_pm2={skipped_no_candidates} "
        f"| skipped_pose_not_selected={skipped_pose_not_selected}"
    )


# =============================
# Batch
# =============================
def process_all():
    total = 0
    for sid in range(SUBJECT_START, SUBJECT_END + 1):
        subject = f"{sid:03d}"
        subj_in = os.path.join(CLEAN_ROOT, subject)
        if not os.path.isdir(subj_in):
            continue
        sequences = get_subfolders(subj_in)
        for seq in sequences:
            total += 1
            print("\n=================================================")
            print(f"Subject: {subject} | Sequence: {seq}")
            process_one_sequence(subject, seq)

    print("\n==================== SUMMARY ====================")
    print(f"Total sequences processed: {total}")
    print(f"Input root: {CLEAN_ROOT}")
    print(f"Selected subseq root: {OUT_ROOT}")
    print(f"Output root: {FINAL_ROOT}")
    print("=================================================")

if __name__ == "__main__":
    process_all()