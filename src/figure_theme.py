"""
Figure theme selection for pipeline visualisations.

Two palettes are supported:

  dark  (default) - charcoal + teal, matching the deployed dashboard.
                    Written to outputs/figures/, which the dashboard reads.
  light           - white background for print. Written to
                    outputs/figures_light/, leaving the dashboard set intact.

Select with the FIGURE_THEME environment variable:

    FIGURE_THEME=light python -m src.evaluation
    FIGURE_THEME=light python -m src.explainability

With no variable set the behaviour is unchanged from before this module
existed: dark figures, same filenames, same directory.
"""

import os

import matplotlib.pyplot as plt

from src.config import FIGURES_DIR as _DARK_FIGURES_DIR, OUTPUTS_DIR

THEME = os.getenv("FIGURE_THEME", "dark").strip().lower()
if THEME not in {"dark", "light"}:
    raise ValueError(
        f"FIGURE_THEME must be 'dark' or 'light' (got {THEME!r})"
    )

_IS_LIGHT = THEME == "light"

# ---------------------------------------------------------------------------
# Palettes
# ---------------------------------------------------------------------------
if _IS_LIGHT:
    FIGURE_BG = "#FFFFFF"
    PANEL_BG = "#FFFFFF"
    GRID = "#D0D5DD"
    TEXT = "#101828"
    MUTED = "#475467"
    # Sequential map that stays legible on white; low cells are near-white,
    # so cell text must be contrasted per-cell rather than forced light.
    CONFUSION_CMAP = "Blues"
    CONFUSION_TEXT_COLOR = None  # None -> let matplotlib auto-contrast
    DPI = 300
else:
    FIGURE_BG = "#111827"
    PANEL_BG = "#1F2937"
    GRID = "#374151"
    TEXT = "#F9FAFB"
    MUTED = "#94A3B8"
    CONFUSION_CMAP = "cividis"
    CONFUSION_TEXT_COLOR = "#F9FAFB"
    DPI = 150

# Where this run's figures are written.
OUTPUT_DIR = (OUTPUTS_DIR / "figures_light") if _IS_LIGHT else _DARK_FIGURES_DIR


def apply_theme() -> None:
    """Apply the selected palette to matplotlib's global rcParams."""
    plt.rcParams.update({
        "figure.facecolor": FIGURE_BG,
        "savefig.facecolor": FIGURE_BG,
        "axes.facecolor": PANEL_BG,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT,
        "axes.titlecolor": TEXT,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": TEXT,
        "legend.facecolor": PANEL_BG,
        "legend.edgecolor": GRID,
        "legend.labelcolor": TEXT,
        # Opaque: the default 0.8 alpha lets bars/curves show through.
        "legend.framealpha": 1.0,
        "grid.color": GRID,
        "grid.alpha": 0.4 if not _IS_LIGHT else 0.6,
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
