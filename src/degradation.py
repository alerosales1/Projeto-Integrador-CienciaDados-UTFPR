"""Degradação proposital de dados (P2): minutos ausentes e ruído nos sensores analógicos.

Aplicada a uma CÓPIA da Silver de 1 min (a base original nunca é alterada). Cada função é
reprodutível pela semente.
"""
import numpy as np
import pandas as pd


def drop_minutes(sil: pd.DataFrame, frac: float, seed: int) -> pd.DataFrame:
    """Remove aleatoriamente a fração `frac` das linhas (minutos), de forma independente."""
    if not 0 <= frac < 1:
        raise ValueError("frac deve estar em [0, 1)")
    rng = np.random.default_rng(seed)
    keep = rng.random(len(sil)) >= frac
    return sil[keep].reset_index(drop=True)


def add_noise(
    sil: pd.DataFrame, columns: list[str], k: float, ref_std: pd.Series, seed: int
) -> pd.DataFrame:
    """Soma ruído gaussiano de desvio-padrão k * ref_std[coluna] nas colunas indicadas."""
    rng = np.random.default_rng(seed)
    out = sil.copy()
    for c in columns:
        out[c] = out[c].to_numpy() + rng.normal(0.0, k * float(ref_std[c]), len(out))
    return out
