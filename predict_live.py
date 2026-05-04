"""
Real-time RFID yoga pose predictor.
Watches the ItemTest log folder and predicts pose from the latest CSV.

Usage:
    python predict_live.py
"""

import time
import glob
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

POLL_SEC    = 1.5    # how often to predict (seconds)
WINDOW_SEC  = 2.0    # how many seconds of recent reads to use
MIN_READS   = 10     # skip prediction if fewer reads in window

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

# ── Feature extraction (copied from notebook) ─────────────────────────────────
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
        "rssi_iqr"   : float(np.percentile(rssi,75)-np.percentile(rssi,25)) if n>1 else 0.0,
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
        feats.update({"phase_mean": 0.0, "phase_std": 0.0, "phase_range": 0.0,
                      "phase_sin_mean": 0.0, "phase_cos_mean": 0.0})
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
            row.update({f"{pfx}_{k}": v
                        for k, v in window_features(tdf, window_sec).items()})
    return row


def get_latest_log(folder):
    """Return the most recently modified CSV in the log folder."""
    csvs = list(folder.glob("*.csv"))
    if not csvs:
        return None
    return max(csvs, key=lambda p: p.stat().st_mtime)


def confidence_bar(proba, width=20):
    filled = int(proba * width)
    return "[" + "█" * filled + "░" * (width - filled) + f"] {proba:.0%}"


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    print("Loading model...")
    pipe = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text())
    classes      = meta["classes"]
    canonical_tags = meta["canonical_tags"]
    feature_cols = meta["feature_cols"]
    print(f"Model loaded: {meta['model']}  |  Classes: {classes}")
    print(f"Watching: {LOG_FOLDER}")
    print("-" * 60)

    last_path = None

    while True:
        try:
            log_path = get_latest_log(LOG_FOLDER)

            if log_path is None:
                print("  No log file found yet — start Inventory in ItemTest")
                time.sleep(POLL_SEC)
                continue

            if log_path != last_path:
                print(f"  Reading: {log_path.name}")
                last_path = log_path

            df = load_rfid_csv(log_path)

            if len(df) < MIN_READS:
                print(f"  Only {len(df)} reads so far, waiting...")
                time.sleep(POLL_SEC)
                continue

            row = extract_single_window(df, WINDOW_SEC, canonical_tags)
            X   = np.array([[row[c] for c in feature_cols]])

            pred   = pipe.predict(X)[0]
            probas = pipe.predict_proba(X)[0]
            label  = classes[pred]
            conf   = probas.max()

            # Print prediction
            print(f"\n  POSE: {label.upper():<16} {confidence_bar(conf)}")
            print("  Per-class probabilities:")
            for cls, p in sorted(zip(classes, probas), key=lambda x: -x[1]):
                marker = " ◄" if cls == label else ""
                print(f"    {cls:<16} {p:.1%}{marker}")

        except KeyboardInterrupt:
            print("\nStopped.")
            break
        except Exception as e:
            print(f"  Error: {e}")

        time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()