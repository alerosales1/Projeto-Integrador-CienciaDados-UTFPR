import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_timing_opt_in_registra_e_nao_altera(tmp_path, monkeypatch):
    monkeypatch.setenv("PI_FIT_TIMING", "1")
    monkeypatch.setenv("PI_TIMING_FILE", str(tmp_path / "t.csv"))
    monkeypatch.setenv("PI_NOTEBOOK", "teste")
    import timing
    from sklearn.ensemble import IsolationForest

    timing._installed = False
    assert timing.install()
    X = np.random.default_rng(0).normal(size=(200, 3))
    m = IsolationForest(n_estimators=10, random_state=0).fit(X)
    m.score_samples(X)
    rows = list(csv.DictReader((tmp_path / "t.csv").open(encoding="utf-8")))
    assert {r["metodo"] for r in rows} == {"fit", "score_samples"}
    assert rows[0]["linhas"] == "200" and rows[0]["notebook"] == "teste"


def test_timing_desligado_por_padrao(monkeypatch):
    monkeypatch.delenv("PI_FIT_TIMING", raising=False)
    import timing

    timing._installed = False
    assert timing.install() is False
