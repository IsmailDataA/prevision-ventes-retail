import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forecast.generate_sales import future_frame, generate  # noqa: E402
from forecast.models import HORIZON, add_features, backtest, bias, wape  # noqa: E402


@pytest.fixture(scope="module")
def data():
    sales, promos = generate(start="2024-01-01", end="2025-12-28")
    return sales, promos, add_features(sales)


def test_magasins_fermes_le_dimanche_et_jours_speciaux(data):
    sales, _, _ = data
    assert (sales.loc[sales["date"].dt.dayofweek == 6, "ca"] == 0).all()
    assert (sales.loc[sales["date"] == "2025-12-25", "ca"] == 0).all()


def test_pas_de_fuite_du_futur(data):
    """Le retard le plus court utilisé est de 28 jours (= horizon de prévision)."""
    _, _, f = data
    s = f[(f["magasin"] == "Blois") & (f["famille"] == "Frais")].set_index("date")
    d = pd.Timestamp("2025-06-30")
    assert s.loc[d, "lag_28"] == s.loc[d - pd.Timedelta(days=28), "ca"]


def test_metriques():
    y, p = np.array([100.0, 200.0]), np.array([110.0, 180.0])
    assert wape(y, p) == pytest.approx(30 / 300)
    assert bias(y, p) == pytest.approx(-10 / 300)


def test_le_modele_bat_les_methodes_simples(data):
    _, _, f = data
    res, _ = backtest(f, n_folds=2)
    m = res.groupby("modele")["wape"].mean()
    assert m["Gradient Boosting"] < m["Moyenne 4 semaines"] < m["Naïf saisonnier (J-28)"]


def test_future_frame(data):
    _, promos, _ = data
    fut = future_frame(promos, pd.Timestamp("2025-12-29"), HORIZON)
    assert fut["date"].nunique() == HORIZON
    assert fut["ca"].isna().all()
    assert fut.loc[fut["date"] == "2026-01-01", "ferme"].eq(1).all()
