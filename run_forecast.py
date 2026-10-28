"""Prévision des ventes à 4 semaines par magasin × famille + tableau de bord.

Usage :
    python run_forecast.py
"""
from __future__ import annotations

import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from forecast import charts  # noqa: E402
from forecast.generate_sales import FAMILIES, STORES, future_frame, generate  # noqa: E402
from forecast.models import HORIZON, MODELS, add_features, backtest, forecast_future, wape  # noqa: E402

REPO_URL = "https://github.com/ismaildataa/prevision-ventes-retail"
LEAD_TIME_DAYS = 2      # délai de réapprovisionnement d'un magasin
Z_SERVICE = 1.65        # taux de service cible 95 %
GBM, MA = "Gradient Boosting", "Moyenne 4 semaines"


def fr(x: float, d: int = 0) -> str:
    return f"{x:,.{d}f}".replace(",", " ").replace(".", ",")


def monday(s: pd.Series) -> pd.Series:
    return s - pd.to_timedelta(s.dt.dayofweek, unit="D")


def main() -> None:
    t0 = time.perf_counter()
    docs, out = ROOT / "docs", ROOT / "outputs"
    (docs / "img").mkdir(parents=True, exist_ok=True)
    out.mkdir(exist_ok=True)

    print("1/5  Génération des ventes (8 magasins × 6 familles, 3,7 ans)…")
    sales, promos = generate()
    end = sales["date"].max()

    print("2/5  Backtest à origine glissante (6 × 28 jours)…")
    feat = add_features(sales)
    bt, bt_pred = backtest(feat, n_folds=6)

    print("3/5  Test spécifique sur la période de Noël 2025…")
    xmas, xmas_pred = backtest(feat, cutoffs=[pd.Timestamp("2025-11-27")])

    print("4/5  Prévision des 4 prochaines semaines…")
    fut = future_frame(promos, end + pd.Timedelta(days=1), HORIZON)
    allf = add_features(pd.concat([sales, fut], ignore_index=True))
    hist, fut_f = allf[allf["date"] <= end], allf[allf["date"] > end].copy()
    fut_f["prevision"] = forecast_future(hist, fut_f)

    print("5/5  Indicateurs, graphiques et tableau de bord…")
    # ---- synthèse des modèles
    summary = bt.groupby("modele", sort=False).agg(wape=("wape", "mean"), biais=("biais", "mean")).reset_index()
    xm = xmas.set_index("modele")["wape"]
    summary["wape_noel"] = summary["modele"].map(xm)

    # ---- stock de sécurité (au coût d'achat) nécessaire pour 95 % de service
    marge = {f: v[3] for f, v in FAMILIES.items()}
    stock = {}
    for name in MODELS:
        err = bt_pred.assign(e=bt_pred[name] - bt_pred["ca"])
        sd = err.groupby(["magasin", "famille"])["e"].std().reset_index()
        sd["stock"] = Z_SERVICE * sd["e"] * np.sqrt(LEAD_TIME_DAYS) * (1 - sd["famille"].map(marge))
        stock[name] = sd["stock"].sum()
    summary["stock_securite"] = summary["modele"].map(stock)
    gain_stock = stock[MA] - stock[GBM]
    gain_wape = 1 - summary.set_index("modele").loc[GBM, "wape"] / summary.set_index("modele").loc[MA, "wape"]

    # ---- graphiques statiques (README / rapport)
    charts.models_bar(summary, docs / "img" / "comparaison_modeles.png")
    ex = xmas_pred[(xmas_pred["magasin"] == "Orléans Centre") & (xmas_pred["famille"] == "Jouets-Loisirs")]
    charts.example_series(ex, GBM, MA, "Orléans Centre · Jouets-Loisirs — 4 semaines avant Noël 2025",
                          docs / "img" / "exemple_noel.png")

    # ---- données du tableau de bord (agrégats hebdomadaires)
    sales["semaine"] = monday(sales["date"])
    weeks = sorted(sales["semaine"].unique())[-104:]
    wk = sales[sales["semaine"].isin(weeks)].groupby(["magasin", "famille", "semaine"])["ca"].sum()
    bt_pred["semaine"] = monday(bt_pred["date"])
    bt_wk = bt_pred.groupby(["magasin", "famille", "semaine"])[[GBM, "ca"]].sum()
    fut_f["semaine"] = monday(fut_f["date"])
    fut_wk = fut_f.groupby(["magasin", "famille", "semaine"])["prevision"].sum()
    bt_weeks = sorted(bt_pred["semaine"].unique())
    fut_weeks = sorted(fut_f["semaine"].unique())

    series = {}
    for m in STORES:
        for f in FAMILIES:
            key = f"{m}|{f}"
            b = bt_wk.loc[(m, f)]
            full = b[(b["ca"] > 0)]
            pe = ((full[GBM] - full["ca"]) / full["ca"]).abs()
            q80 = float(np.quantile(pe, 0.8)) if len(pe) else 0.1
            errs = {name: [float(np.abs(g[name] - g["ca"]).sum()), float(g["ca"].sum())]
                    for name in MODELS for g in [bt_pred[(bt_pred["magasin"] == m) & (bt_pred["famille"] == f)]]}
            series[key] = {
                "h": [round(float(wk.get((m, f, w), 0.0))) for w in weeks],
                "b": [round(float(b[GBM].get(w, 0.0))) for w in bt_weeks],
                "f": [round(float(fut_wk.get((m, f, w), 0.0))) for w in fut_weeks],
                "q": round(q80, 4),
                "e": errs,
            }
    payload = {
        "weeks": [pd.Timestamp(w).strftime("%Y-%m-%d") for w in weeks],
        "btWeeks": [pd.Timestamp(w).strftime("%Y-%m-%d") for w in bt_weeks],
        "futWeeks": [pd.Timestamp(w).strftime("%Y-%m-%d") for w in fut_weeks],
        "stores": list(STORES), "families": list(FAMILIES), "models": list(MODELS),
        "series": series,
    }
    env = Environment(loader=FileSystemLoader(ROOT / "templates"))
    s = summary.set_index("modele")
    ctx = {
        "run_date": date.today().strftime("%d/%m/%Y"), "repo_url": REPO_URL,
        "data_json": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        "wape_gbm": fr(100 * s.loc[GBM, "wape"], 1), "wape_ma": fr(100 * s.loc[MA, "wape"], 1),
        "wape_naive": fr(100 * s.loc["Naïf saisonnier (J-28)", "wape"], 1),
        "noel_gbm": fr(100 * s.loc[GBM, "wape_noel"], 1), "noel_ma": fr(100 * s.loc[MA, "wape_noel"], 1),
        "gain_wape": fr(100 * gain_wape, 0), "gain_stock_pct": fr(100 * gain_stock / stock[MA], 0), "gain_stock": fr(gain_stock / 1000, 0),
        "stock_gbm": fr(stock[GBM] / 1000, 0), "stock_ma": fr(stock[MA] / 1000, 0),
        "biais_gbm": fr(100 * s.loc[GBM, "biais"], 1),
        "n_series": len(STORES) * len(FAMILIES), "debut": sales["date"].min().strftime("%d/%m/%Y"),
        "fin": end.strftime("%d/%m/%Y"),
        "summary": [{"modele": r.modele, "wape": fr(100 * r.wape, 1), "noel": fr(100 * r.wape_noel, 1),
                     "biais": fr(100 * r.biais, 1), "stock": fr(r.stock_securite / 1000, 0)}
                    for r in summary.itertuples()],
    }
    (docs / "index.html").write_text(env.get_template("dashboard.html.j2").render(**ctx), encoding="utf-8")
    fut_f[["date", "magasin", "famille", "promo", "prevision"]].to_csv(out / "previsions_28j.csv", index=False)
    summary.to_csv(out / "comparaison_modeles.csv", index=False)

    print(f"\nTerminé en {time.perf_counter() - t0:.0f} s")
    print(summary.round(4).to_string(index=False))
    print(f"Erreur réduite de {gain_wape:.0%} vs moyenne 4 semaines ; stock de sécurité −{gain_stock:,.0f} €")


if __name__ == "__main__":
    main()
