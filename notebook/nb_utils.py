"""Support module for the project notebook.

Path bootstrap, the notebook's output and cache folders, the class colour
palette, plotting style, figure export and small helpers. Imported once, at
the top of the notebook, with ``from nb_utils import *``.
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import timedelta
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# --- Paths ---------------------------------------------------------------------
# The project root is the first folder above this file that contains
# src/config.py, so the notebook runs whichever folder Jupyter was started in.

NOTEBOOK_DIR = Path(__file__).resolve().parent

for _base in (NOTEBOOK_DIR, *NOTEBOOK_DIR.parents):
    if (_base / "src" / "config.py").is_file():
        PROJECT_ROOT = _base
        break
else:
    raise FileNotFoundError(
        f"src/config.py was not found in any folder above {NOTEBOOK_DIR}. "
        f"nb_utils.py must live somewhere inside the project folder."
    )

SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import config as C  # noqa: E402  (needs SRC_DIR on sys.path first)

FIG_DIR = NOTEBOOK_DIR / "figures"
TAB_DIR = NOTEBOOK_DIR / "tables"
CKPT_DIR = NOTEBOOK_DIR / "checkpoints"

# Computed cache (about 4 GB). Deleting it forces a cold run; HAR_CACHE_DIR
# moves it elsewhere.
CACHE_DIR = Path(os.environ.get("HAR_CACHE_DIR", NOTEBOOK_DIR / "cache"))
PARQUET_DIR = CACHE_DIR / "parquet"
WINDOWS_DIR = CACHE_DIR / "windows"
FEATURES_DIR = CACHE_DIR / "features"
MODELS_DIR = CACHE_DIR / "models"

for _d in (FIG_DIR, TAB_DIR, CKPT_DIR,
           PARQUET_DIR, WINDOWS_DIR, FEATURES_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Class colours -------------------------------------------------------------
# One colour per class, shared by every figure, chosen to stay distinguishable
# for colour-blind readers.

CLASS_COLORS = {
    "idle":          "#934a06",   # brown
    "walk":          "#2159bf",   # blue
    "stair_ascent":  "#8f1d6f",   # magenta
    "stair_descent": "#7b80f4",   # periwinkle
    "ramp_ascent":   "#24a39c",   # teal
    "ramp_descent":  "#a48f00",   # olive
}

assert list(CLASS_COLORS) == list(C.CLASSES), (
    "CLASS_COLORS must list exactly the classes of config.CLASSES, in order"
)

SURFACE = "#ffffff"
TEXT_PRIMARY = "#111111"
TEXT_SECONDARY = "#555555"
GRID = "#d8d8d8"


def class_color(label: str) -> str:
    """Colour of one class label; raises KeyError on an unknown label."""
    return CLASS_COLORS[label]


# --- Plot style ----------------------------------------------------------------

def set_plot_style() -> None:
    """Apply the notebook-wide matplotlib style."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "text.color": TEXT_PRIMARY,
        "axes.labelcolor": TEXT_PRIMARY,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.7,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "legend.frameon": False,
        "figure.autolayout": False,
    })


def save_figure(fig, name: str) -> Path:
    """Save a figure as figures/<name>.png (300 dpi) and figures/<name>.pdf.

    Returns the path of the PNG.
    """
    png = FIG_DIR / f"{name}.png"
    pdf = FIG_DIR / f"{name}.pdf"
    fig.savefig(png)
    fig.savefig(pdf)
    print(f"saved  {png.relative_to(PROJECT_ROOT)}  and  {pdf.name}")
    return png


# --- Expected-value checks -----------------------------------------------------

def check_expected(rows, title: str = "Verification") -> None:
    """Print an expected/found table and raise ValueError on any mismatch.

    Each row is (label, expected, found), (label, expected, found, tol) for a
    numeric tolerance, or (label, None, found) for an informational row.
    """
    print(f"\n{title}")
    print("-" * 78)
    print(f"  {'quantity':<34}{'expected':>16}{'found':>16}   verdict")

    failures = []
    for row in rows:
        label, expected, found = row[0], row[1], row[2]
        tol = row[3] if len(row) > 3 else 0

        if expected is None:
            verdict = "info"
        else:
            try:
                ok = abs(float(found) - float(expected)) <= float(tol)
            except (TypeError, ValueError):
                ok = found == expected
            verdict = "OK" if ok else "MISMATCH"
            if not ok:
                failures.append((label, expected, found))

        exp_txt = "-" if expected is None else _fmt(expected)
        print(f"  {label:<34}{exp_txt:>16}{_fmt(found):>16}   {verdict}")

    print("-" * 78)
    if failures:
        detail = "; ".join(f"{lab}: expected {exp}, found {fnd}"
                           for lab, exp, fnd in failures)
        raise ValueError(
            f"{title}: {len(failures)} value(s) do not match what was expected "
            f"-> {detail}. Do not continue: a number that does not add up means "
            f"something upstream is wrong."
        )
    print("  all checks passed")


def _fmt(v) -> str:
    """Compact rendering of a value for the check table."""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        return f"{v:,.4f}".rstrip("0").rstrip(".") if abs(v) < 1e6 else f"{v:.4g}"
    return str(v)


# --- Output helpers ------------------------------------------------------------

def section(title: str) -> None:
    """Print a section banner."""
    print("=" * 78)
    print(title)
    print("=" * 78)


def checkpoint_note(path: Path, n_done: int, unit: str = "units") -> None:
    """Report that a checkpoint was reloaded instead of recomputed."""
    print(f"  checkpoint found: {path.name} -> reusing {n_done:,} {unit} "
          f"already computed; they will not be recomputed")


# --- Parallel SHAP -------------------------------------------------------------

_SHAP_EXPLAINER = None


def _shap_init(clf) -> None:
    """Build one explainer per worker process."""
    global _SHAP_EXPLAINER
    import shap
    _SHAP_EXPLAINER = shap.TreeExplainer(clf)


def _shap_normalise(sv) -> np.ndarray:
    """Return a (rows, features, classes) array, whatever the shap version."""
    if isinstance(sv, list):
        return np.stack(sv, axis=-1)
    return sv if sv.ndim == 3 else sv[..., None]


def _shap_chunk(block: np.ndarray) -> np.ndarray:
    return _shap_normalise(_SHAP_EXPLAINER.shap_values(block))


def hms(seconds: float) -> str:
    """Format seconds as h:mm:ss."""
    return str(timedelta(seconds=int(max(0, seconds))))


def progress_bar(done: int, total: int, t0: float, width: int = 30) -> None:
    """Single-line progress bar, rewritten in place."""
    q = done / total if total else 1.0
    filled = int(q * width)
    elapsed = time.time() - t0
    remaining = elapsed / q * (1 - q) if q > 0 else 0.0
    print(f"\r  [{'#' * filled}{'.' * (width - filled)}] "
          f"{done:>6,}/{total:,}  {q:>5.1%}   "
          f"elapsed {hms(elapsed)}   left ~{hms(remaining)}   ",
          end="", flush=True)
    if done >= total:
        print(flush=True)


def shap_parallel(clf, B: np.ndarray, jobs: int = 8, backend: str = "process",
                  chunk: int = 150, show: bool = True) -> np.ndarray:
    """Shapley values of every row of ``B``, computed in parallel chunks.

    Returns an array shaped (rows, features, classes), identical to a
    sequential run.
    """
    global _SHAP_EXPLAINER
    blocks = [B[i:i + chunk] for i in range(0, len(B), chunk)]
    results: list[np.ndarray | None] = [None] * len(blocks)
    t0 = time.time()
    done = 0

    if jobs <= 1:
        import shap
        _SHAP_EXPLAINER = shap.TreeExplainer(clf)
        for i, b in enumerate(blocks):
            results[i] = _shap_chunk(b)
            done += len(b)
            if show:
                progress_bar(done, len(B), t0)
        return np.concatenate(results, axis=0)

    if backend == "thread":
        import shap
        _SHAP_EXPLAINER = shap.TreeExplainer(clf)
        pool = ThreadPoolExecutor(max_workers=jobs)
    else:
        pool = ProcessPoolExecutor(max_workers=jobs, initializer=_shap_init,
                                   initargs=(clf,))
    with pool as ex:
        futures = {ex.submit(_shap_chunk, b): i for i, b in enumerate(blocks)}
        for f in as_completed(futures):
            i = futures[f]
            results[i] = f.result()
            done += len(blocks[i])
            if show:
                progress_bar(done, len(B), t0)

    return np.concatenate(results, axis=0)


# --- XGBoost wrapper -----------------------------------------------------------

class XGBClasses:
    """`XGBClassifier` with the same interface as the scikit-learn models.

    - Labels are encoded in the order of `C.CLASSES` (not alphabetically), so
      `classes_` matches the other models.
    - Class rebalancing uses balanced sample weights, since XGBoost has no
      `class_weight` for multiclass problems.
    - `predict` returns class names instead of integer codes.

    `iteration_range` is passed through: the model at n rounds is an exact
    prefix of the full model.
    """

    def __init__(self, classes, **params):
        from xgboost import XGBClassifier
        self.classes_ = np.asarray(list(classes))
        self._index = {c: i for i, c in enumerate(self.classes_)}
        self._clf = XGBClassifier(
            objective="multi:softprob", num_class=len(self.classes_),
            tree_method="hist", **params)

    def encode(self, y):
        """Class names to integer codes, in the order of `C.CLASSES`."""
        unknown = set(np.unique(y)) - set(self._index)
        if unknown:
            raise ValueError(f"labels outside C.CLASSES: {sorted(unknown)}")
        return np.array([self._index[v] for v in y], dtype=np.int8)

    def fit(self, X, y, sample_weight=None):
        """Fit on class names or integer codes; XGBoost always receives codes."""
        from sklearn.utils.class_weight import compute_sample_weight
        y = np.asarray(y)
        y_code = y.astype(np.int8) if y.dtype.kind in "iu" else self.encode(y)
        assert y_code.dtype.kind in "iu", "XGBoost needs integer labels"
        if sample_weight is None:
            sample_weight = compute_sample_weight("balanced", y_code)
        self._clf.fit(X, y_code, sample_weight=sample_weight)
        return self

    def predict(self, X, iteration_range=None):
        code = self._clf.predict(X, iteration_range=iteration_range)
        return self.classes_[np.asarray(code, dtype=int)]

    def predict_proba(self, X, iteration_range=None):
        return self._clf.predict_proba(X, iteration_range=iteration_range)

    @property
    def booster(self):
        """The underlying XGBClassifier, used by shap.TreeExplainer."""
        return self._clf


__all__ = [
    "C", "PROJECT_ROOT", "NOTEBOOK_DIR", "SRC_DIR",
    "FIG_DIR", "TAB_DIR", "CKPT_DIR",
    "CACHE_DIR", "PARQUET_DIR", "WINDOWS_DIR", "FEATURES_DIR", "MODELS_DIR",
    "CLASS_COLORS", "class_color",
    "SURFACE", "TEXT_PRIMARY", "TEXT_SECONDARY", "GRID",
    "set_plot_style", "save_figure",
    "check_expected", "section", "checkpoint_note",
    "shap_parallel", "progress_bar", "hms",
    "XGBClasses",
    "plt", "mpl",
]
