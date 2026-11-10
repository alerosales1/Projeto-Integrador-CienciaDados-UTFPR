"""Detecção de condição anormal sem rótulos (treino só com dados normais).

Fluxo: treinar no início da base (sem falhas), definir o limiar em uma janela normal de
validação (percentil dos scores, ANTES de olhar o teste) e avaliar nas falhas reportadas.
Score: quanto maior, mais anômalo.
"""
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from evaluation import alarm_episodes, circular_shift_null


def window_features(
    sil: pd.DataFrame, analog: list[str], digital: list[str], windows: list[int], min_coverage: float
) -> pd.DataFrame:
    """Features pontuais + janelas por tempo (só passado) a partir da Silver de 1 min.

    Retorna ts, features e `coverage_ok` (janelas com cobertura mínima de minutos).
    Requer a coluna `n_readings` da Silver.
    """
    s = sil.set_index("ts")
    sensors = analog + digital
    parts = [s[sensors]]
    ok = pd.Series(True, index=s.index)
    for w in windows:
        mean = s[sensors].rolling(f"{w}min").mean()
        mean.columns = [f"{c}_mean_{w}m" if c in analog else f"{c}_duty_{w}m" for c in sensors]
        std = s[analog].rolling(f"{w}min").std()
        std.columns = [f"{c}_std_{w}m" for c in analog]
        n = s["n_readings"].rolling(f"{w}min").count()
        ok &= (n / w) >= min_coverage
        parts += [mean, std]
    feat = pd.concat(parts, axis=1)
    feat = feat.loc[:, ~feat.columns.duplicated()]
    feat["coverage_ok"] = ok.to_numpy()
    return feat.reset_index()


def in_failure(ts: pd.Series, failures: list[dict]) -> np.ndarray:
    """1 para minutos dentro de uma falha reportada."""
    out = np.zeros(len(ts), dtype=int)
    for f in failures:
        out |= ((ts >= pd.Timestamp(f["start"])) & (ts <= pd.Timestamp(f["end"]))).to_numpy().astype(int)
    return out


def post_failure_mask(ts: pd.Series, failures: list[dict], hours: float) -> np.ndarray:
    """True para minutos nas `hours` horas depois do fim de uma falha (recuperação pós-reparo)."""
    out = np.zeros(len(ts), dtype=bool)
    for f in failures:
        end = pd.Timestamp(f["end"])
        out |= ((ts > end) & (ts <= end + pd.Timedelta(hours=hours))).to_numpy()
    return out


def anomaly_scores(model_name: str, X_train, others: dict, seed: int = 0) -> dict:
    """Ajusta o modelo no treino (normal) e devolve scores (maior = mais anômalo) por conjunto."""
    if model_name == "IsolationForest":
        model = IsolationForest(n_estimators=200, random_state=seed, n_jobs=-1).fit(X_train)
        scorer = lambda X: -model.score_samples(X)  # noqa: E731
    elif model_name == "Mahalanobis":
        scaler = StandardScaler().fit(X_train)
        cov = LedoitWolf().fit(scaler.transform(X_train))
        scorer = lambda X: cov.mahalanobis(scaler.transform(X))  # noqa: E731
    else:
        raise ValueError(f"modelo desconhecido: {model_name}")
    return {name: scorer(X) for name, X in others.items()}


def evaluate_detection(
    t: pd.DataFrame, threshold: float, failures: list[dict], gap: str = "60min"
) -> tuple[pd.DataFrame, dict]:
    """Avalia um limiar já fixado. `t`: ts, score, label (1 = em falha), só minutos avaliados.

    Por evento: detectada (algum alarme dentro da falha), atraso até o primeiro alarme e
    fração de minutos da falha em alarme. Global: falsos alarmes (episódios) por dia de
    minutos normais e fração de minutos normais em alarme.
    """
    alarms = t[t["score"] >= threshold]
    rows = []
    for f in failures:
        start, end = pd.Timestamp(f["start"]), pd.Timestamp(f["end"])
        n_in = int(((t["ts"] >= start) & (t["ts"] <= end)).sum())
        a = alarms[(alarms["ts"] >= start) & (alarms["ts"] <= end)]
        rows.append({
            "falha": f["id"],
            "minutos_em_falha": n_in,
            "minutos_em_alarme": len(a),
            "detectada": float(len(a) > 0) if n_in else np.nan,
            "atraso_h": (a["ts"].min() - start).total_seconds() / 3600 if len(a) else np.nan,
            "recall_minutos": len(a) / n_in if n_in else np.nan,
        })
    normal = t[t["label"] == 0]
    false_alarms = alarms[alarms["label"] == 0]
    episodes = alarm_episodes(false_alarms["ts"], gap)
    days = len(normal) / 1440
    glob = {
        "limiar": float(threshold),
        "minutos_normais": len(normal),
        "falsos_alarmes_episodios": len(episodes),
        "falsos_alarmes_por_dia": len(episodes) / days if days else np.nan,
        "frac_minutos_normais_em_alarme": len(false_alarms) / len(normal) if len(normal) else np.nan,
    }
    return pd.DataFrame(rows), glob


def pr_auc_with_null(t: pd.DataFrame, n_shifts: int = 200, seed: int = 0) -> dict:
    """PR-AUC, lift e p-valor contra o baseline de rotação circular dos scores."""
    if t["label"].sum() == 0:
        return {"pr_auc": np.nan, "lift": np.nan, "p_pr_auc": np.nan}
    pr = average_precision_score(t["label"], t["score"])
    prev = t["label"].mean()
    null = circular_shift_null(t, "score", fpr=0.001, n_shifts=n_shifts, seed=seed)
    return {"pr_auc": pr, "lift": pr / prev, "p_pr_auc": float((null["pr_auc"] >= pr).mean())}
