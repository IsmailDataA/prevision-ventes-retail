"""Modèles de prévision à J+1…J+28 et backtest à origine glissante.

Approche « directe » : toutes les variables utilisées par le modèle sont connues
au moment de prévoir (retards ≥ 28 jours, calendrier, promotions planifiées).
Pas de fuite d'information du futur vers le passé.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

HORIZON = 28
KEYS = ["magasin", "famille"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(KEYS + ["date"]).copy()
    g = df.groupby(KEYS)["ca"]
    for lag in (28, 35, 42, 364):
        df[f"lag_{lag}"] = g.shift(lag)
    df["moy_4sem_meme_jour"] = (df["lag_28"] + df["lag_35"] + df["lag_42"] + g.shift(49)) / 4
    df["moy_28j"] = g.transform(lambda s: s.shift(28).rolling(28, min_periods=14).mean())
    df["jour_semaine"] = df["date"].dt.dayofweek
    df["jour_annee"] = df["date"].dt.dayofyear
    df["mois"] = df["date"].dt.month
    df["semaine"] = df["date"].dt.isocalendar().week.astype(int)
    df["jours_avant_noel"] = (pd.to_datetime(df["date"].dt.year.astype(str) + "-12-25") - df["date"]).dt.days.clip(-10, 60)
    df["magasin_code"] = df["magasin"].astype("category").cat.codes
    df["famille_code"] = df["famille"].astype("category").cat.codes
    return df


FEATURES = ["lag_28", "lag_35", "lag_42", "lag_364", "moy_4sem_meme_jour", "moy_28j",
            "jour_semaine", "jour_annee", "mois", "semaine", "jours_avant_noel", "promo", "ferie",
            "veille_fete", "ferme", "magasin_code", "famille_code"]


def naive_seasonal(test: pd.DataFrame) -> np.ndarray:
    """Même jour, 4 semaines plus tôt (dernière valeur connue à J-28)."""
    return test["lag_28"].fillna(0).values


def moving_average(test: pd.DataFrame) -> np.ndarray:
    """Moyenne des 4 mêmes jours de semaine connus — pratique courante en magasin."""
    return test["moy_4sem_meme_jour"].fillna(test["lag_28"]).fillna(0).values


def gbm_fit_predict(train: pd.DataFrame, test: pd.DataFrame, seed: int = 0) -> np.ndarray:
    tr = train.dropna(subset=["lag_364"])
    model = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, max_leaf_nodes=48,
                                          l2_regularization=1.0, random_state=seed)
    model.fit(tr[FEATURES], np.log1p(tr["ca"]))
    pred = np.expm1(model.predict(test[FEATURES]))
    return np.where(test["ferme"].values == 1, 0.0, np.clip(pred, 0, None))


MODELS = {
    "Naïf saisonnier (J-28)": lambda tr, te: naive_seasonal(te),
    "Moyenne 4 semaines": lambda tr, te: moving_average(te),
    "Gradient Boosting": gbm_fit_predict,
}


def wape(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.abs(y - p).sum() / np.abs(y).sum())


def bias(y: np.ndarray, p: np.ndarray) -> float:
    return float((p - y).sum() / y.sum())


def backtest(df: pd.DataFrame, n_folds: int = 6,
             cutoffs: list[pd.Timestamp] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Origine glissante : n_folds fenêtres de 28 jours à la fin de l'historique
    (ou fenêtres démarrant après chaque date de `cutoffs`)."""
    end = df["date"].max()
    if cutoffs is None:
        cutoffs = [end - pd.Timedelta(days=HORIZON * k) for k in range(n_folds, 0, -1)]
    rows, preds = [], []
    for i, cutoff in enumerate(cutoffs, start=1):
        train = df[df["date"] <= cutoff]
        test = df[(df["date"] > cutoff) & (df["date"] <= cutoff + pd.Timedelta(days=HORIZON))]
        test = test[test["ferme"] == 0]
        out = test[KEYS + ["date", "ca"]].copy()
        out["fold"] = i
        for name, fn in MODELS.items():
            p = fn(train, test)
            out[name] = p
            rows.append({"fold": i, "debut": cutoff + pd.Timedelta(days=1),
                         "modele": name, "wape": wape(test["ca"].values, p),
                         "biais": bias(test["ca"].values, p)})
        preds.append(out)
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True)


def forecast_future(df: pd.DataFrame, future: pd.DataFrame) -> np.ndarray:
    """Entraîne sur tout l'historique et prévoit les 28 prochains jours."""
    return gbm_fit_predict(df, future)
