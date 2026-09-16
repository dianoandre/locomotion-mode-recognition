"""Project configuration: paths and constants shared by the notebook and the scripts.

The cache location can be changed with the HAR_CACHE_DIR environment variable.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Paths ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = PROJECT_ROOT / "Dataset_Camargo"
PAPERS_DIR = PROJECT_ROOT / "Paper"
REPORTS_DIR = PROJECT_ROOT / "reports"
SRC_DIR = PROJECT_ROOT / "src"

_default_cache = Path.home() / "har-dm-cache"
CACHE_ROOT = Path(os.environ.get("HAR_CACHE_DIR", _default_cache))

PARQUET_DIR = CACHE_ROOT / "parquet"
WINDOWS_DIR = CACHE_ROOT / "windows"
FEATURES_DIR = CACHE_ROOT / "features"
MODELS_DIR = CACHE_ROOT / "models"
FIGURES_DIR = CACHE_ROOT / "figures"

# --- Signals and windows -------------------------------------------------------

RANDOM_SEED = 42

FS_TARGET_HZ = 200          # IMU and labels are recorded at 200 Hz
FS_GON_HZ = 1000            # goniometers are resampled onto the IMU clock
MERGE_TOLERANCE_S = 0.0025  # half an IMU sample, for merge_asof on Header

# Measured: labels share the IMU time base (the dataset README states 1000 Hz).
FS_LABELS_HZ = 200

WINDOW_MS = 250
WINDOW_SAMPLES = int(WINDOW_MS * FS_TARGET_HZ / 1000)   # 50

# Overlapping windows do not leak, because the split is by subject.
STRIDE_MS_CV = 125          # cross-validation and tuning (50% overlap)
STRIDE_MS_EVAL = 50         # finer stride, kept for evaluation scripts

STRIDE_MS = STRIDE_MS_EVAL
STRIDE_SAMPLES = int(STRIDE_MS * FS_TARGET_HZ / 1000)          # 10
STRIDE_SAMPLES_CV = int(STRIDE_MS_CV * FS_TARGET_HZ / 1000)    # 25

# Share of `idle` windows kept, on training rows only: validation and test
# keep the real class distribution.
IDLE_KEEP_FRACTION = 0.40

MODES = ["levelground", "stair", "ramp"]        # treadmill is excluded
MODE_TREADMILL = "treadmill"

N_TEST_SUBJECTS = 4         # held-out subjects

# --- Labels --------------------------------------------------------------------
# Raw labels -> six classes. Each transition takes the label of the class it
# leads into, so the controller can anticipate the next locomotion mode.
LABEL_MAP = {
    # steady states
    "idle": "idle",
    "stand": "idle",
    "walk": "walk",
    "turn1": "walk",
    "turn2": "walk",
    "stairascent": "stair_ascent",
    "stairdescent": "stair_descent",
    "rampascent": "ramp_ascent",
    "rampdescent": "ramp_descent",
    # entry transitions
    "stand-walk": "walk",
    "walk-stairascent": "stair_ascent",
    "walk-stairdescent": "stair_descent",
    "walk-rampascent": "ramp_ascent",
    "walk-rampdescent": "ramp_descent",
    # exit transitions
    "walk-stand": "idle",
    "stairascent-walk": "walk",
    "stairdescent-walk": "walk",
    "rampascent-walk": "walk",
    "rampdescent-walk": "walk",
}

TRANSITION_ENTER = [
    "stand-walk",
    "walk-stairascent",
    "walk-stairdescent",
    "walk-rampascent",
    "walk-rampdescent",
]

TRANSITION_EXIT = [
    "walk-stand",
    "stairascent-walk",
    "stairdescent-walk",
    "rampascent-walk",
    "rampdescent-walk",
]

TRANSITION_LABELS = TRANSITION_ENTER + TRANSITION_EXIT


def transition_direction(label_raw: str) -> str:
    """Return 'enter', 'exit', or '' when the label is not a transition."""
    if label_raw in TRANSITION_ENTER:
        return "enter"
    if label_raw in TRANSITION_EXIT:
        return "exit"
    return ""


CLASSES = [
    "idle",
    "walk",
    "stair_ascent",
    "stair_descent",
    "ramp_ascent",
    "ramp_descent",
]

# --- Sensors and channels ------------------------------------------------------
# Only sensors an exoskeleton can carry: EMG, force plates, motion capture and
# inverse kinematics/dynamics are excluded.
SENSORS_ALLOWED = ["imu", "gon"]
SENSOR_LABELS = "conditions"

IMU_SEGMENTS = ["trunk", "thigh", "shank", "foot"]      # right side
IMU_CHANNELS = [
    f"{seg}_{kind}_{axis}"
    for seg in IMU_SEGMENTS
    for kind in ("Accel", "Gyro")
    for axis in ("X", "Y", "Z")
]                                                        # 24 channels

GON_CHANNELS = [
    "hip_sagittal", "hip_frontal",
    "knee_sagittal",
    "ankle_sagittal", "ankle_frontal",
]                                                        # 5 channels, right side

ALL_CHANNELS = IMU_CHANNELS + GON_CHANNELS               # 29 channels

# --- Valid samples -------------------------------------------------------------
# A value is valid when it is finite AND within the physical limit of its
# channel (np.isnan alone misses infinities). The limits are wide on purpose:
# they catch sensor faults, not physiological extremes.
GON_LIMIT_DEG = 200.0        # degrees
ACCEL_LIMIT_G = 16.0         # typical IMU full scale, in g
GYRO_LIMIT_RADS = 35.0       # about 2000 deg/s


def channel_limit(channel: str) -> float:
    """Largest plausible absolute value for a channel."""
    if "Accel" in channel:
        return ACCEL_LIMIT_G
    if "Gyro" in channel:
        return GYRO_LIMIT_RADS
    return GON_LIMIT_DEG


def valid_mask(block, channels=None):
    """Boolean mask with one entry per cell of `block` (n_samples x n_channels).

    True where the value is finite and within its channel's limit. This is the
    single definition of a valid sample used throughout the project.
    """
    import numpy as _np
    channels = channels or ALL_CHANNELS
    arr = _np.asarray(block, dtype=float)
    ok = _np.isfinite(arr)
    limits = _np.array([channel_limit(c) for c in channels])
    ok &= _np.abs(_np.nan_to_num(arr, nan=0.0, posinf=_np.inf,
                                 neginf=_np.inf)) <= limits
    return ok


# Trial metadata kept for analysis only: used as features, it would leak the label.
CONDITION_META_FORBIDDEN_AS_FEATURES = [
    "speed", "turn", "stairHeight", "rampIncline",
    "leadingLegStart", "leadingLegStop", "transLegAscent", "transLegDescent",
]


def ensure_cache_dirs() -> None:
    """Create the cache folders if they do not exist."""
    for d in (PARQUET_DIR, WINDOWS_DIR, FEATURES_DIR, MODELS_DIR, FIGURES_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_cache_dirs()
    print(f"PROJECT_ROOT : {PROJECT_ROOT}")
    print(f"DATASET_ROOT : {DATASET_ROOT}   (exists: {DATASET_ROOT.is_dir()})")
    print(f"CACHE_ROOT   : {CACHE_ROOT}")
    print(f"window       : {WINDOW_MS} ms = {WINDOW_SAMPLES} samples")
    print(f"stride CV    : {STRIDE_MS_CV} ms = {STRIDE_SAMPLES_CV} samples "
          f"(overlap {100 * (1 - STRIDE_SAMPLES_CV / WINDOW_SAMPLES):.0f}%)")
    print(f"stride eval  : {STRIDE_MS_EVAL} ms = {STRIDE_SAMPLES} samples "
          f"(overlap {100 * (1 - STRIDE_SAMPLES / WINDOW_SAMPLES):.0f}%)")
    print(f"idle kept    : {IDLE_KEEP_FRACTION:.0%}")
    print(f"classes      : {CLASSES}")
