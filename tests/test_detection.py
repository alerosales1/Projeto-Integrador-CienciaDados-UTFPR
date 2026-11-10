import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from detection import (  # noqa: E402
    anomaly_scores,
    evaluate_detection,
    in_failure,
    post_failure_mask,
    window_features,
)

FAILS = [
    {"id": 1, "start": datetime(2020, 3, 1, 12, 0), "end": datetime(2020, 3, 1, 18, 0)},
    {"id": 2, "start": datetime(2020, 3, 3, 0, 0), "end": datetime(2020, 3, 3, 6, 0)},
]


def _series():
    ts = pd.date_range("2020-02-20", "2020-03-05", freq="min")
    return pd.Series(ts)


def test_in_failure_and_post_failure_mask():
    ts = _series()
    lab = in_failure(ts, FAILS)
    assert lab.sum() == (6 * 60 + 1) * 2          # janelas inclusivas de 6 h
    post = post_failure_mask(ts, FAILS, hours=24)
    assert not (post & (lab == 1)).any()          # recuperação não inclui a falha
    assert post.sum() > 0


def test_models_detect_a_shifted_regime_and_ignore_normal_data():
    rng = np.random.default_rng(0)
    ts = _series()
    X = pd.DataFrame(rng.normal(size=(len(ts), 5)), columns=list("abcde"))
    lab = in_failure(ts, FAILS)
    X.loc[lab == 1, ["a", "b"]] += 6.0            # regime anômalo dentro das falhas
    train = X[ts < "2020-02-26"]
    val = X[(ts >= "2020-02-26") & (ts < "2020-02-28")]
    for model in ("IsolationForest", "Mahalanobis"):
        sc = anomaly_scores(model, train, {"val": val, "test": X}, seed=0)
        thr = np.quantile(sc["val"], 0.999)
        t = pd.DataFrame({"ts": ts, "score": sc["test"], "label": lab})
        t = t[t["ts"] >= "2020-02-28"]
        events, glob = evaluate_detection(t, thr, FAILS)
        assert events["detectada"].tolist() == [1.0, 1.0]
        assert (events["atraso_h"] < 1).all()
        assert (events["recall_minutos"] > 0.1).all()
        assert glob["falsos_alarmes_por_dia"] < 3


def test_window_features_respect_coverage_and_use_only_past():
    ts = pd.date_range("2020-03-01", periods=600, freq="min")
    sil = pd.DataFrame({"ts": ts, "n_readings": 6, "x": np.arange(600, dtype=float), "d": 1.0})
    sil = sil[~((sil.ts >= "2020-03-01 05:00") & (sil.ts < "2020-03-01 06:00"))]  # lacuna de 1 h
    feat = window_features(sil, ["x"], ["d"], [60], 0.8)
    assert "x_mean_60m" in feat.columns and "d_duty_60m" in feat.columns
    first = feat.iloc[0]
    assert not first["coverage_ok"]               # sem histórico suficiente no começo
    ok_rows = feat[feat["coverage_ok"]]
    assert (ok_rows["ts"] > pd.Timestamp("2020-03-01 06:00")).any()
    # média 60m no ts = média dos últimos 60 minutos (só passado): sem a lacuna, a janela cheia
    row = feat[feat["ts"] == pd.Timestamp("2020-03-01 04:30")].iloc[0]
    assert abs(row["x_mean_60m"] - np.arange(211, 271).mean()) < 1e-9
