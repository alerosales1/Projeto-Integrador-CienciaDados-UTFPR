"""Medição opt-in do tempo de treino/predição dos modelos.

Ativação:  PI_FIT_TIMING=1  (o run_pipeline.py faz isso). Sem a variável, não faz nada.
Variáveis: PI_NOTEBOOK (rótulo do notebook), PI_TIMING_FILE (CSV de saída;
           padrão docs/tempos_modelos.csv).

Envolve fit / predict_proba / score_samples / mahalanobis dos estimadores usados e
acrescenta uma linha por chamada: notebook, rotulo, fold, estimador, metodo,
segundos, linhas, colunas. Não altera nenhum resultado.
"""
import csv
import os
import time
from pathlib import Path

import config

FIELDS = ["notebook", "rotulo", "fold", "estimador", "metodo", "segundos", "linhas", "colunas"]
_installed = False
CONTEXT = {"rotulo": "", "fold": ""}   # os notebooks podem preencher (opcional)


def _out() -> Path:
    return Path(os.environ.get("PI_TIMING_FILE", config.DOCS_DIR / "tempos_modelos.csv"))


def _shape(args):
    for a in args:
        if hasattr(a, "shape") and len(a.shape) == 2:
            return int(a.shape[0]), int(a.shape[1])
    return "", ""


def _wrap(cls, method):
    original = getattr(cls, method, None)
    if original is None or getattr(original, "_pi_timed", False):
        return

    def timed(self, *args, **kwargs):
        t0 = time.perf_counter()
        try:
            return original(self, *args, **kwargs)
        finally:
            dt = time.perf_counter() - t0
            rows, cols = _shape(args)
            path = _out()
            path.parent.mkdir(parents=True, exist_ok=True)
            new = not path.exists()
            with path.open("a", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                if new:
                    w.writerow(FIELDS)
                w.writerow([
                    os.environ.get("PI_NOTEBOOK", ""), CONTEXT["rotulo"], CONTEXT["fold"],
                    cls.__name__, method, f"{dt:.4f}", rows, cols,
                ])

    timed._pi_timed = True
    timed.__wrapped__ = original
    setattr(cls, method, timed)


def install() -> bool:
    """Instala os wrappers se PI_FIT_TIMING=1. Idempotente."""
    global _installed
    if _installed or os.environ.get("PI_FIT_TIMING") != "1":
        return False
    from lightgbm import LGBMClassifier
    from sklearn.covariance import LedoitWolf
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    for cls, methods in (
        (RandomForestClassifier, ("fit", "predict_proba")),
        (LogisticRegression, ("fit", "predict_proba")),
        (LGBMClassifier, ("fit", "predict_proba")),
        (IsolationForest, ("fit", "score_samples")),
        (LedoitWolf, ("fit", "mahalanobis")),
    ):
        for m in methods:
            _wrap(cls, m)
    _installed = True
    return True
