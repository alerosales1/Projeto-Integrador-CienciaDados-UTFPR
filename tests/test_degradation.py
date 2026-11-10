import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from degradation import add_noise, drop_minutes  # noqa: E402


def _sil():
    ts = pd.date_range("2020-03-01", periods=5000, freq="min")
    rng = np.random.default_rng(0)
    return pd.DataFrame({"ts": ts, "x": rng.normal(10, 2, 5000), "d": rng.integers(0, 2, 5000) * 1.0})


def test_drop_minutes_fraction_and_reproducibility():
    sil = _sil()
    a = drop_minutes(sil, 0.3, seed=1)
    b = drop_minutes(sil, 0.3, seed=1)
    c = drop_minutes(sil, 0.3, seed=2)
    assert abs(len(a) / len(sil) - 0.7) < 0.03
    assert a.equals(b) and not a.equals(c)
    assert a["ts"].is_monotonic_increasing and len(sil) == 5000   # original intacto
    with pytest.raises(ValueError):
        drop_minutes(sil, 1.0, seed=0)


def test_add_noise_scale_untouched_columns_and_original():
    sil = _sil()
    ref = pd.Series({"x": 2.0})
    out = add_noise(sil, ["x"], k=0.3, ref_std=ref, seed=0)
    diff = out["x"] - sil["x"]
    assert abs(diff.std() - 0.6) < 0.05           # 0,3 * 2,0
    assert abs(diff.mean()) < 0.05
    assert out["d"].equals(sil["d"])              # coluna fora da lista não muda
    assert sil["x"].equals(_sil()["x"])           # original não foi alterado
    assert add_noise(sil, ["x"], 0.0, ref, 0)["x"].equals(sil["x"])
