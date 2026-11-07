import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluation import (  # noqa: E402
    alarm_episodes,
    circular_shift_null,
    event_metrics,
    fixed_fpr_threshold,
)


def test_fixed_fpr_threshold_uses_only_negatives():
    scores = np.concatenate([np.linspace(0, 1, 1001), [5.0, 6.0]])  # dois positivos altos
    labels = np.concatenate([np.zeros(1001), [1, 1]])
    thr = fixed_fpr_threshold(scores, labels, 0.01)
    assert abs(thr - 0.99) < 1e-6


def test_alarm_episodes_merge_close_minutes():
    ts = pd.to_datetime(["2020-01-01 00:00", "2020-01-01 00:01", "2020-01-01 00:30",
                         "2020-01-01 05:00"])
    eps = alarm_episodes(ts, gap="60min")
    assert len(eps) == 2
    assert eps[0] == (ts[0], ts[2])


def _toy():
    ts = pd.date_range("2020-03-01", periods=300, freq="min")
    df = pd.DataFrame({"ts": ts, "label": 0, "score": 0.0})
    df.loc[100:159, "label"] = 1            # horizonte pré-falha
    df.loc[120, "score"] = 0.95             # acerto dentro do horizonte
    df.loc[[10, 11, 12], "score"] = 0.95    # um episódio de falso alarme
    df.loc[250, "score"] = 0.95             # outro episódio
    return df, ts


def test_event_metrics_detection_lead_and_false_alarms():
    df, ts = _toy()
    failure_start = ts[160]
    m = event_metrics(df, "score", 0.9, failure_start)
    assert m["detected"] == 1.0
    assert abs(m["lead_h"] - 40 / 60) < 1e-9    # alarme no minuto 120, falha no 160
    assert m["false_alarm_episodes"] == 2
    assert abs(m["false_alarms_per_day"] - 2 / (300 / 1440)) < 1e-9


def test_event_without_positives_gives_nan_detection():
    df, ts = _toy()
    df["label"] = 0
    m = event_metrics(df, "score", 0.9, ts[160])
    assert np.isnan(m["detected"]) and np.isnan(m["lead_h"])


def test_circular_shift_null_is_near_prevalence_for_unaligned_scores():
    rng = np.random.default_rng(1)
    n = 3000
    df = pd.DataFrame({"ts": pd.date_range("2020-03-01", periods=n, freq="min"),
                       "label": 0, "score": rng.random(n)})
    df.loc[1000:1299, "label"] = 1                       # prevalência 10%
    df["score"] = np.where(df["label"] == 1, df["score"] + 5, df["score"])  # alinhado: PR-AUC = 1
    null = circular_shift_null(df, "score", fpr=0.01, n_shifts=100, seed=0)
    assert list(null.columns) == ["pr_auc", "detected"] and len(null) == 100
    assert null["pr_auc"].median() < 0.3                 # sem alinhamento, perto da prevalência
    again = circular_shift_null(df, "score", fpr=0.01, n_shifts=100, seed=0)
    assert null.equals(again)                            # reprodutível pela semente
