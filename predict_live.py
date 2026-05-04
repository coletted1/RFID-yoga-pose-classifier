"""
Real-time RFID yoga pose predictor.
Watches the ItemTest log folder and predicts pose from the latest CSV.

Usage:
    python predict_live.py
"""

import time
import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

# ── Config ────────────────────────────────────────────────────────────────────
LOG_FOLDER  = Path(r"C:\rfid_yoga\logs")
MODEL_PATH  = Path(r"C:\rfid_yoga\model\rfid_pose_classifier.joblib")
META_PATH   = Path(r"C:\rfid_yoga\model\rfid_pose_classifier_meta.json")

POLL_SEC    = 2.0    # predict every 2 seconds
WINDOW_SEC  = 4.0    # use last 4 seconds of reads
MIN_READS   = 30     # require at least 30 reads before predicting
CONFIRM_COUNT = 3    # require 3 consecutive agreeing windows

COL_NAMES = [
    "Timestamp", "EPC", "TID", "Antenna",
    "RSSI", "Frequency", "Hostname",
    "PhaseAngle", "DopplerFrequency", "CRHandle"
]

_FEAT_KEYS = [
    "rssi_mean", "rssi_std", "rssi_min", "rssi_max", "rssi_range",
    "rssi_iqr", "rssi_skew", "rssi_kurt", "read_rate",
    "phase_mean", "phase_std", "phase_range", "phase_sin_mean", "phase_cos_mean"
]

# ── Feature extraction ────────────────────────────────────────────────────────
def load_rfid_csv(path):
    df = pd.read_csv(
        path, comment="/", names=COL_NAMES,
        dtype={"EPC": str, "TID": str},
    )
    df.dropna(subset=["EPC", "RSSI"], inplace=True)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)
    df.sort_values("Timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def window_features(group, window_sec):
    rssi  = group["RSSI"].values.astype(float)
    phase = group["PhaseAngle"].dropna().values.astype(float)
    n = len(rssi)
    feats = {
        "rssi_mean"  : np.mean(rssi),
        "rssi_std"   : np.std(rssi)   if n > 1 else 0.0,
        "rssi_min"   : np.min(rssi),
        "rssi_max"   : np.max(rssi),
        "rssi_range" : np.ptp(rssi),
        "rssi_iqr"   : float(np.percentile(rssi, 75) - np.percentile(rssi, 25)) if n > 1 else 0.0,
        "rssi_skew"  : float(stats.skew(rssi))     if n > 2 else 0.0,
        "rssi_kurt"  : float(stats.kurtosis(rssi)) if n > 3 else 0.0,
        "read_rate"  : n / window_sec,
    }
    if len(phase) > 0:
        feats["phase_mean"]     = np.mean(phase)
        feats["phase_std"]      = np.std(phase)  if len(phase) > 1 else 0.0
        feats["phase_range"]    = np.ptp(phase)  if len(phase) > 1 else 0.0
        feats["phase_sin_mean"] = np.mean(np.sin(phase))
        feats["phase_cos_mean"] = np.mean(np.cos(phase))
    else:
        feats.update({
            "phase_mean": 0.0, "phase_std": 0.0, "phase_range": 0.0,
            "phase_sin_mean": 0.0, "phase_cos_mean": 0.0
        })
    return feats


def extract_single_window(df, window_sec, known_tags):
    """Extract features from the most recent window_sec of reads."""
    t_end   = df["Timestamp"].iloc[-1]
    t_start = t_end - pd.Timedelta(seconds=window_sec)
    win_df  = df[df["Timestamp"] >= t_start]

    row = {}
    for tag in known_tags:
        pfx = tag[:8]
        tdf = win_df[win_df["EPC"] == tag]
        if len(tdf) == 0:
            row.update({f"{pfx}_{k}": 0.0 for k in _FEAT_KEYS})
        else:
            row.update({
                f"{pfx}_{k}": v
                for k, v in window_features(tdf, window_sec).items()
            })
    return row


def get_latest_log(folder):
    """Return the most recently modified CSV in the log folder."""
    csvs = list(folder.glob("*.csv"))
    if not csvs:
        return None
    return max(csvs, key=lambda p: p.stat().st_mtime)


def get_probas(pipe, X, classes):
    """
    Safely get per-class probability estimates regardless of model type.
    Uses predict_proba if available, otherwise falls back to softmax of
    decision_function scores (works for SVM without probability=True).
    """
    if hasattr(pipe, "predict_proba"):
        try:
            return pipe.predict_proba(X)[0]
        except Exception:
            pass

    if hasattr(pipe, "decision_function"):
        decision = pipe.decision_function(X)[0]
        # Softmax to convert raw scores into a probability-like distribution
        exp_d  = np.exp(decision - decision.max())
        return exp_d / exp_d.sum()

    # Last resort: one-hot from hard prediction
    pred = pipe.predict(X)[0]
    one_hot = np.zeros(len(classes))
    one_hot[pred] = 1.0
    return one_hot


def confidence_bar(proba, width=20):
    filled = int(proba * width)
    return "[" + "█" * filled + "░" * (width - filled) + f"] {proba:.0%}"


def clear_line():
    print("\033[A\033[K", end="")


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  RFID Yoga Pose Predictor")
    print("=" * 60)
    print(f"  Model  : {MODEL_PATH}")
    print(f"  Logs   : {LOG_FOLDER}")
    print(f"  Window : {WINDOW_SEC}s   Poll: {POLL_SEC}s")
    print("=" * 60)

    # Load model and metadata
    if not MODEL_PATH.exists():
        print(f"\n  ERROR: Model not found at {MODEL_PATH}")
        print("  Run the notebook first to train and save the model.")
        return
    if not META_PATH.exists():
        print(f"\n  ERROR: Metadata not found at {META_PATH}")
        return

    print("\nLoading model...", end=" ")
    pipe = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text())
    classes        = meta["classes"]
    canonical_tags = meta["canonical_tags"]
    feature_cols   = meta["feature_cols"]
    print(f"OK  ({meta['model']})")
    print(f"Classes : {classes}")
    print(f"Tags    : {[t[:8] for t in canonical_tags]}")
    print(f"\nStart the Inventory in ItemTest, then hold a pose.\n")
    print("Ctrl+C to stop.\n")
    print("-" * 60)

    last_path       = None
    last_label      = None
    consecutive     = 0          # how many consecutive windows agree

    while True:
        try:
            log_path = get_latest_log(LOG_FOLDER)

            # No log file yet
            if log_path is None:
                print("  Waiting for ItemTest to start logging...   ", end="\r")
                time.sleep(POLL_SEC)
                continue

            # New log file detected
            if log_path != last_path:
                print(f"\n  New log file: {log_path.name}")
                last_path   = log_path
                last_label  = None
                consecutive = 0

            # Load the CSV
            try:
                df = load_rfid_csv(log_path)
            except Exception as e:
                print(f"  Could not read log: {e}   ", end="\r")
                time.sleep(POLL_SEC)
                continue

            # Not enough data yet
            if len(df) < MIN_READS:
                print(f"  Warming up — {len(df)} reads so far...   ", end="\r")
                time.sleep(POLL_SEC)
                continue

            # Extract features from the latest window
            row = extract_single_window(df, WINDOW_SEC, canonical_tags)
            X   = np.array([[row[c] for c in feature_cols]])

            # Predict
            pred_idx = pipe.predict(X)[0]
            label    = classes[pred_idx]
            probas   = get_probas(pipe, X, classes)
            conf     = probas.max()

            # Stability filter: only confirm if N consecutive windows agree
            if label == last_label:
                consecutive += 1
            else:
                consecutive = 1
                last_label  = label

            # Build output
            print("\n" + "=" * 60)
            stable_marker = "  ✓ STABLE" if consecutive >= CONFIRM_COUNT else f"  (confirming {consecutive}/{CONFIRM_COUNT})"
            print(f"  POSE: {label.upper():<16} {confidence_bar(conf)}{stable_marker}")
            print("-" * 60)
            print("  Per-class breakdown:")
            for cls, p in sorted(zip(classes, probas), key=lambda x: -x[1]):
                marker = " ◄" if cls == label else ""
                bar    = "█" * int(p * 15)
                print(f"    {cls:<16} {bar:<15} {p:.1%}{marker}")
            print("-" * 60)

            # Show which tags were seen in the window
            t_end   = df["Timestamp"].iloc[-1]
            t_start = t_end - pd.Timedelta(seconds=WINDOW_SEC)
            win_df  = df[df["Timestamp"] >= t_start]
            tag_counts = win_df.groupby("EPC").size()
            print("  Tags in window:")
            for tag in canonical_tags:
                count = tag_counts.get(tag, 0)
                bar   = "█" * min(count, 20)
                print(f"    {tag[:8]}...  {bar:<20} {count} reads")
            print("=" * 60)

        except KeyboardInterrupt:
            print("\n\nStopped.")
            break
        except Exception as e:
            print(f"  Unexpected error: {e}")

        time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()