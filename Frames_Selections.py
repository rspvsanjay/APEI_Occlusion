import os
import glob
import cv2
import numpy as np


# =============================
# Config
# =============================
THRESHOLD = 127
MIN_ITERS = 3                  # (1) minimum three iterations compulsory
DIFF_THRESH = 1.1              # (2) stop if abs Δ%change <= 1.1 after 3 iters
MIN_CONSEC_LEN = 5             # (3) "more than 4" => >= 5
MAX_ITERS = 50                 # safety cap

CLEAN_ROOT = r"D:\Gait_IIT_BHU_Data_D\Inbuilt_1\Merged_Normalized_256x256_Vertically_Horizontally_Aligned_Refind_Images_DGEI_17_Renamed"
OUT_ROOT = r"D:\Gait_IIT_BHU_Data_D\Inbuilt_1\Merged_Normalized_256x256_Vertically_Horizontally_Aligned_Refind_Images_DGEI_17_Renamed_selected"


# =============================
# I/O utilities
# =============================
def list_subject_sequence_folders(clean_root: str):
    """
    Expects:
      clean_root/<subject>/<sequence>/*.png|*.jpg|...
    Yields:
      (subject_id, sequence_name, sequence_folder_path)
    """
    for subject in sorted(os.listdir(clean_root)):
        subj_path = os.path.join(clean_root, subject)
        if not os.path.isdir(subj_path):
            continue
        for seq in sorted(os.listdir(subj_path)):
            seq_path = os.path.join(subj_path, seq)
            if os.path.isdir(seq_path):
                yield subject, seq, seq_path


def load_frames_and_files(folder: str, threshold: int = THRESHOLD):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    files = []
    for e in exts:
        files += glob.glob(os.path.join(folder, e))
    files = sorted(files)

    if not files:
        raise RuntimeError(f"No images found in {folder}")

    frames01 = []
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to read {f}")
        _, bw = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
        frames01.append((bw > 0).astype(np.uint8))

    return frames01, files


def compute_area_curve(frames01):
    return np.array([int(f.sum()) for f in frames01], dtype=np.int64)


# =============================
# Core refinement logic
# =============================
def refine_with_rules(area_curve: np.ndarray,
                      diff_thresh: float = DIFF_THRESH,
                      min_iters: int = MIN_ITERS,
                      max_iters: int = MAX_ITERS):
    """
    Runs iterative selection:
      selected = {t | area[t] > (mean(selected)-std(selected))}
    Tracks:
      pct_change(it) = % change in lower vs previous iteration (starts it=2)
      pct_diff(it)   = abs(pct_change(it) - pct_change(it-1)) (starts it=3)

    Stop rule:
      - must compute at least `min_iters` iterations
      - at it>=3, if pct_diff <= diff_thresh, choose PREVIOUS iteration selection

    Returns:
      history: list of dict
      chosen_iter: int
      chosen_idx: np.ndarray
    """
    T = len(area_curve)
    selected = np.arange(T, dtype=np.int64)

    history = []
    pct_change_prev = None

    for it in range(1, max_iters + 1):
        if len(selected) == 0:
            break

        vals = area_curve[selected]
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals))
        lower = mean_val - std_val

        new_selected = selected[area_curve[selected] > lower]

        entry = {
            "iter": it,
            "lower": float(lower),
            "selected_idx": new_selected.copy(),
            "pct_change": None,  # starts it=2
            "pct_diff": None,    # starts it=3
            "count_after": int(len(new_selected)),
        }

        if it >= 2 and len(history) >= 1:
            prev_lower = history[-1]["lower"]
            pct_change = ((lower - prev_lower) / (abs(prev_lower) + 1e-8)) * 100.0
            entry["pct_change"] = float(pct_change)

            if it >= 3 and pct_change_prev is not None:
                entry["pct_diff"] = float(abs(pct_change - pct_change_prev))

            pct_change_prev = pct_change

        history.append(entry)

        # Apply stop condition only after minimum iterations
        if it >= min_iters and it >= 3:
            pct_diff = entry["pct_diff"]
            if pct_diff is not None and pct_diff <= diff_thresh:
                chosen_iter = it - 1
                chosen_idx = history[-2]["selected_idx"].copy()
                return history, chosen_iter, chosen_idx

        selected = new_selected

    # Fallback: choose last non-empty selection
    for h in reversed(history):
        if len(h["selected_idx"]) > 0:
            return history, h["iter"], h["selected_idx"].copy()

    return history, None, np.array([], dtype=np.int64)


# =============================
# Save logic: consecutive segments to subsequences
# =============================
def consecutive_segments(indices: np.ndarray):
    if indices is None or len(indices) == 0:
        return []
    idx = np.unique(indices.astype(np.int64))
    idx.sort()

    segs = []
    s = idx[0]
    prev = idx[0]
    for v in idx[1:]:
        if v == prev + 1:
            prev = v
        else:
            segs.append((s, prev))
            s = v
            prev = v
    segs.append((s, prev))
    return segs


def save_subsequences(files, selected_idx, out_seq_root, min_consec_len=MIN_CONSEC_LEN):
    """
    Saves consecutive segments of selected_idx of length >= min_consec_len
    into:
      out_seq_root/subsequence_01/
      out_seq_root/subsequence_02/
      ...

    Returns: number of saved subsequences
    """
    os.makedirs(out_seq_root, exist_ok=True)
    segs = consecutive_segments(selected_idx)

    saved = 0
    for (s, e) in segs:
        L = e - s + 1
        if L < min_consec_len:
            continue

        saved += 1
        sub_dir = os.path.join(out_seq_root, f"subsequence_{saved:02d}")
        os.makedirs(sub_dir, exist_ok=True)

        for t in range(s, e + 1):
            src = files[t]
            dst = os.path.join(sub_dir, os.path.basename(src))
            img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise RuntimeError(f"Failed to read {src}")
            cv2.imwrite(dst, img)

    return saved


# =============================
# Batch processing: all subjects, all sequences
# =============================
def process_all(clean_root=CLEAN_ROOT, out_root=OUT_ROOT):
    total_sequences = 0
    total_saved_subseq = 0
    total_skipped_no_frames = 0

    for subject, seq, seq_path in list_subject_sequence_folders(clean_root):
        total_sequences += 1
        print("\n=================================================")
        print(f"Subject: {subject} | Sequence: {seq}")
        print(f"Path: {seq_path}")

        try:
            frames01, files = load_frames_and_files(seq_path, threshold=THRESHOLD)
        except RuntimeError as e:
            print(f"[SKIP] {e}")
            total_skipped_no_frames += 1
            continue

        area_curve = compute_area_curve(frames01)
        history, chosen_iter, chosen_idx = refine_with_rules(
            area_curve,
            diff_thresh=DIFF_THRESH,
            min_iters=MIN_ITERS,
            max_iters=MAX_ITERS
        )

        # Log iterations
        for h in history:
            it = h["iter"]
            msg = f"Iter {it:02d} | selected={h['count_after']:3d} | lower={h['lower']:.6f}"
            if h["pct_change"] is not None:
                msg += f" | %chg={h['pct_change']:.4f}"
            if h["pct_diff"] is not None:
                msg += f" | Δ%chg(abs)={h['pct_diff']:.4f}"
            print(msg)

        if chosen_iter is None or chosen_idx is None or len(chosen_idx) == 0:
            print("[RESULT] No frames selected after refinement.")
            continue

        print(f"[RESULT] Chosen iteration: {chosen_iter} | chosen_frames={len(chosen_idx)}")
        print(f"[RESULT] Chosen indices: {chosen_idx.tolist()}")

        # Output path preserving structure:
        # OUT_ROOT/<subject>/<sequence>/subsequence_XX/...
        out_seq_root = os.path.join(out_root, subject, seq)

        saved = save_subsequences(
            files=files,
            selected_idx=chosen_idx,
            out_seq_root=out_seq_root,
            min_consec_len=MIN_CONSEC_LEN
        )

        total_saved_subseq += saved
        print(f"[SAVE] out: {out_seq_root} | saved_subsequences(len>={MIN_CONSEC_LEN})={saved}")

        if saved == 0:
            print("[SAVE] No consecutive segments of length >= 5. Nothing saved for this sequence.")

    print("\n==================== SUMMARY ====================")
    print(f"Total sequences scanned: {total_sequences}")
    print(f"Total subsequences saved: {total_saved_subseq}")
    print(f"Sequences skipped (no frames): {total_skipped_no_frames}")
    print(f"Output root: {OUT_ROOT}")
    print("=================================================")


if __name__ == "__main__":
    process_all()


