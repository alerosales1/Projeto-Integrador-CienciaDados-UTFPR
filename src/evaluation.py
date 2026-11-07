"""Métricas por evento para predição de falhas (pandas/numpy, sem Spark).

Cada linha avaliada é um minuto (1 min), com colunas: ts, label, score.

Ponto de operação: o limiar é escolhido para uma taxa de falsos positivos fixa (FPR) nos
minutos normais do teste. É uma ferramenta de COMPARAÇÃO entre modelos e variantes (todos
operam com o mesmo orçamento de falsos alarmes), não uma regra de implantação.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def fixed_fpr_threshold(scores, labels, fpr: float) -> float:
    """Limiar tal que a fração de minutos normais com score >= limiar é ~fpr."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    negatives = scores[labels == 0]
    if negatives.size == 0:
        return float("nan")
    return float(np.quantile(negatives, 1.0 - fpr))


def alarm_episodes(alarm_ts, gap: str = "60min") -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Agrupa minutos em alarme em episódios; separa quando o intervalo excede `gap`."""
    ts = pd.Series(pd.to_datetime(alarm_ts)).sort_values().reset_index(drop=True)
    if ts.empty:
        return []
    episode_id = (ts.diff() > pd.Timedelta(gap)).cumsum()
    return [(g.iloc[0], g.iloc[-1]) for _, g in ts.groupby(episode_id)]


def event_metrics(
    df: pd.DataFrame,
    score_col: str,
    threshold: float,
    failure_start: pd.Timestamp,
    gap: str = "60min",
) -> dict:
    """Métricas do evento testado em um fold.

    - detected: 1 se algum minuto do horizonte pré-falha (label = 1) passou do limiar
    - lead_h: horas entre o primeiro alarme no horizonte e o início da falha
    - false_alarm_episodes: episódios de alarme em minutos normais (label = 0)
    - false_alarms_per_day: episódios por dia de dados (minutos de dados / 1440)
    """
    alarms = df[df[score_col] >= threshold]
    positives_alarmed = alarms[alarms["label"] == 1]
    false_alarms = alarms[alarms["label"] == 0]
    has_positives = bool((df["label"] == 1).any())

    detected = float(len(positives_alarmed) > 0) if has_positives else np.nan
    lead_h = (
        (failure_start - positives_alarmed["ts"].min()).total_seconds() / 3600
        if len(positives_alarmed)
        else np.nan
    )
    episodes = alarm_episodes(false_alarms["ts"], gap)
    days = df["ts"].nunique() / 1440
    return {
        "detected": detected,
        "lead_h": lead_h,
        "false_alarm_episodes": len(episodes),
        "false_alarms_per_day": len(episodes) / days if days else np.nan,
        "alarm_minutes": int(len(alarms)),
    }


def circular_shift_null(
    df: pd.DataFrame, score_col: str, fpr: float, n_shifts: int = 200, seed: int = 0
) -> pd.DataFrame:
    """Baseline de acaso por rotação circular dos scores em relação aos rótulos.

    A rotação preserva a autocorrelação dos scores e dos rótulos (que são blocos
    contínuos no tempo), mas destrói o alinhamento entre eles. Para cada rotação,
    recalcula a PR-AUC e se haveria detecção (algum alarme dentro do horizonte
    pré-falha) com o mesmo orçamento de falsos positivos.

    Retorna um DataFrame com colunas pr_auc e detected (uma linha por rotação).
    """
    rng = np.random.default_rng(seed)
    scores = df[score_col].to_numpy(dtype=float)
    labels = df["label"].to_numpy()
    n = len(df)
    rows = []
    for k in rng.integers(1, n, size=n_shifts):
        rolled = np.roll(scores, int(k))
        thr = fixed_fpr_threshold(rolled, labels, fpr)
        rows.append(
            (
                average_precision_score(labels, rolled),
                float(((rolled >= thr) & (labels == 1)).any()),
            )
        )
    return pd.DataFrame(rows, columns=["pr_auc", "detected"])
