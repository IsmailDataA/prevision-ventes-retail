"""Ventes quotidiennes synthétiques d'un réseau de magasins (magasin × famille).

Composantes réalistes : tendance, saisonnalités hebdomadaire et annuelle propres
à chaque famille, jours fériés (magasins fermés le 1er mai, 25 décembre et
1er janvier), veilles de fêtes, opérations promotionnelles planifiées, bruit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .calendar_fr import jours_feries

STORES = {
    "Orléans Centre": 1.00, "Tours Nord": 0.85, "Blois": 0.55, "Chartres": 0.70,
    "Bourges": 0.50, "Châteauroux": 0.45, "Le Mans Sud": 0.90, "Vierzon": 0.35,
}
# famille : (CA moyen / jour / magasin en €, amplitude saisonnière annuelle, pic (jour de l'année), marge)
FAMILIES = {
    "Épicerie":           (5200, 0.06, 350, 0.24),
    "Frais":              (4300, 0.08, 200, 0.28),
    "Boissons":           (2100, 0.30, 200, 0.30),
    "Hygiène-Beauté":     (1300, 0.08, 340, 0.36),
    "Maison":             (900,  0.20, 120, 0.40),
    "Jouets-Loisirs":     (500,  0.90, 355, 0.38),
}
WEEKLY = np.array([0.82, 0.86, 0.92, 0.96, 1.12, 1.38, 0.0])  # lundi → dimanche (fermé)
CLOSED = {(1, 1), (5, 1), (12, 25)}


def promo_calendar(dates: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    """Opérations commerciales nationales : connues à l'avance (utilisables en prévision)."""
    rows = []
    for fam in FAMILIES:
        start = dates[0]
        while start < dates[-1]:
            start += pd.Timedelta(days=int(rng.integers(35, 70)))
            rows.append((fam, start, start + pd.Timedelta(days=int(rng.integers(6, 13))),
                         float(rng.uniform(0.15, 0.45))))
    return pd.DataFrame(rows, columns=["famille", "debut", "fin", "uplift"])


def generate(start: str = "2023-01-02", end: str = "2026-09-27", seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    feries = set().union(*[jours_feries(y) for y in range(dates[0].year, dates[-1].year + 1)])
    promos = promo_calendar(pd.date_range(start, dates[-1] + pd.Timedelta(days=120)), rng)
    doy = dates.dayofyear.values
    t = np.arange(len(dates)) / 365.25
    closed = np.array([(d.month, d.day) in CLOSED or d.dayofweek == 6 for d in dates])
    holiday = np.array([d in feries for d in dates])
    eve = np.array([(d + pd.Timedelta(days=1)) in feries or (d.month == 12 and d.day in (23, 24, 31))
                    for d in dates])
    frames = []
    for store, size in STORES.items():
        store_trend = rng.normal(0.025, 0.02)
        for fam, (base, amp, peak, _marge) in FAMILIES.items():
            season = 1 + amp * np.cos(2 * np.pi * (doy - peak) / 365.25)
            if fam == "Jouets-Loisirs":  # pic très concentré avant Noël
                season = 1 + 2.6 * np.exp(-((doy - 352) ** 2) / (2 * 12 ** 2)) + 0.15 * np.cos(2 * np.pi * (doy - 200) / 365.25)
            mean = base * size * season * WEEKLY[dates.dayofweek] * (1 + store_trend) ** t
            mean = mean * np.where(holiday, 0.55, 1.0) * np.where(eve, 1.35, 1.0)
            uplift = np.ones(len(dates))
            promo_flag = np.zeros(len(dates), dtype=int)
            for _, p in promos[promos["famille"] == fam].iterrows():
                m = (dates >= p["debut"]) & (dates <= p["fin"])
                uplift[m] *= 1 + p["uplift"]
                promo_flag[m] = 1
            noise = rng.lognormal(0, 0.11, len(dates))
            sales = np.where(closed, 0.0, mean * uplift * noise)
            frames.append(pd.DataFrame({"date": dates, "magasin": store, "famille": fam,
                                        "promo": promo_flag, "ferie": holiday.astype(int),
                                        "veille_fete": eve.astype(int), "ferme": closed.astype(int),
                                        "ca": sales.round(2)}))
    return pd.concat(frames, ignore_index=True), promos


def future_frame(promos: pd.DataFrame, start: pd.Timestamp, days: int = 28) -> pd.DataFrame:
    """Lignes à prévoir : calendrier et promotions planifiées, CA inconnu."""
    dates = pd.date_range(start, periods=days, freq="D")
    feries = set().union(*[jours_feries(y) for y in {dates[0].year, dates[-1].year, dates[-1].year + 1}])
    rows = []
    for store in STORES:
        for fam in FAMILIES:
            p = promos[promos["famille"] == fam]
            for d in dates:
                rows.append({
                    "date": d, "magasin": store, "famille": fam,
                    "promo": int(((p["debut"] <= d) & (p["fin"] >= d)).any()),
                    "ferie": int(d in feries),
                    "veille_fete": int((d + pd.Timedelta(days=1)) in feries
                                       or (d.month == 12 and d.day in (23, 24, 31))),
                    "ferme": int((d.month, d.day) in CLOSED or d.dayofweek == 6),
                    "ca": np.nan,
                })
    return pd.DataFrame(rows)
