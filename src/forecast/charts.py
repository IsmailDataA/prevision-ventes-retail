"""Graphiques statiques (README) — palette validée daltonisme."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def _style(ax, title, subtitle=None):
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=INK, pad=22 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9.2, color=INK2)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def models_bar(summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.2))
    s = summary.iloc[::-1]
    bars = ax.barh(s["modele"], 100 * s["wape"], color=[BLUE if "Boosting" in m else MUTED for m in s["modele"]],
                   height=0.55)
    ax.bar_label(bars, labels=[f"{100 * v:.1f} %".replace(".", ",") for v in s["wape"]], padding=4,
                 color=INK2, fontsize=9.5)
    _style(ax, "Erreur de prévision (WAPE) à 4 semaines — plus bas = meilleur",
           "Moyenne sur 6 fenêtres de backtest, 48 séries magasin × famille")
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", colors=INK2, labelsize=10)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0f} %")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor="#fcfcfb")
    plt.close(fig)


def example_series(df: pd.DataFrame, gbm: str, ma: str, title: str, path: Path) -> None:
    d = df.sort_values("date")
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(d["date"], d["ca"], color=INK2, linewidth=2, label="Ventes réelles")
    ax.plot(d["date"], d[gbm], color=BLUE, linewidth=2, label="Gradient Boosting")
    ax.plot(d["date"], d[ma], color=ORANGE, linewidth=2, linestyle="--", label="Moyenne 4 semaines")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:,.0f} €".replace(",", " "))
    _style(ax, title, "La moyenne glissante ne voit pas venir le pic de fin d'année ; le modèle l'anticipe")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor="#fcfcfb")
    plt.close(fig)
