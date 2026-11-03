"""Camada Gold: Silver -> conjunto pronto para modelagem (rótulos e features).

Rotulagem (horizonte X, em horas):
  label = 1  se uma falha COMEÇA em (t, t + X]   (pré-falha)
  label = 0  caso contrário
  Leituras DENTRO de uma falha são removidas: senão o modelo só aprenderia a reconhecer
  uma falha em curso (diagnóstico) e não a antecipá-la.

Features:
  e0 / e1  valores pontuais dos sensores
  e2       + janelas deslizantes por TEMPO (somente passado, incluindo o instante atual):
           média e desvio dos analógicos, ciclo de trabalho (média) dos digitais.
           Linhas sem cobertura mínima na janela são descartadas (sem preencher lacunas).

Uso:
  python src/gold.py --variant e2 --horizon 24
  python src/gold.py --variant e2 --all-horizons
"""
import argparse
import json
from datetime import datetime

import pandas as pd
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

import config
from spark_utils import get_spark, lit_ts, load_columns


def load_failures(path=config.FAILURE_REPORT) -> list[dict]:
    """Lê o relatório de falhas e une intervalos sobrepostos."""
    pdf = pd.read_csv(path, parse_dates=["start_time", "end_time"])
    if pdf.empty:
        raise ValueError(f"{path} está vazio. Preencha o relatório de falhas.")
    pdf = pdf.sort_values("start_time")
    merged: list[dict] = []
    for row in pdf.itertuples():
        start: datetime = row.start_time.to_pydatetime()
        end: datetime = row.end_time.to_pydatetime()
        if merged and start <= merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], end)
        else:
            merged.append({"id": int(row.failure_id), "start": start, "end": end})
    return merged


def add_labels(df: DataFrame, failures: list[dict], horizon_h: int) -> DataFrame:
    ts = F.col("ts")
    horizon = F.expr(f"INTERVAL {horizon_h} HOURS")
    in_failure = F.lit(False)
    ahead = F.lit(None).cast("int")
    for f in failures:
        start, end = lit_ts(f["start"]), lit_ts(f["end"])
        in_failure = in_failure | ((ts >= start) & (ts <= end))
        ahead = F.coalesce(ahead, F.when((ts < start) & (ts >= start - horizon), F.lit(f["id"])))
    return (
        df.withColumn("in_failure", in_failure)
        .withColumn("failure_id_ahead", ahead)
        .withColumn(
            "label", (F.col("failure_id_ahead").isNotNull() & ~F.col("in_failure")).cast("int")
        )
    )


def add_rolling_features_naive(
    df: DataFrame, analog: list[str], digital: list[str], windows_min: list[int]
) -> DataFrame:
    """Versão simples: janelas por tempo com rangeBetween (custo cresce com o tamanho da janela).

    Mantida para validação e para comparar tempos no artigo (P3).

    Cada linha é um balde de 1 min, então a janela de W minutos tem no máximo W linhas.
    Obs.: janela sem partição move os dados para uma partição; é aceitável aqui porque
    a Silver de 1 min tem cerca de 250 mil linhas.
    """
    secs = F.col("ts").cast("long")
    cols = []
    for w in windows_min:
        win = Window.orderBy(secs).rangeBetween(-(w - 1) * 60, 0)
        cols.append(F.count(F.lit(1)).over(win).alias(f"n_{w}m"))
        cols += [F.avg(c).over(win).alias(f"{c}_mean_{w}m") for c in analog]
        cols += [F.stddev_samp(c).over(win).alias(f"{c}_std_{w}m") for c in analog]
        cols += [F.avg(c).over(win).alias(f"{c}_duty_{w}m") for c in digital]
    return df.select("*", *cols)


def add_rolling_features(
    df: DataFrame, analog: list[str], digital: list[str], windows_min: list[int]
) -> DataFrame:
    """Janelas por TEMPO com somas acumuladas (prefix sums): custo independe da janela.

    1. Cria uma grade regular com todos os minutos entre o primeiro e o último balde.
    2. Acumula n, soma(x) e soma(x^2) sobre a grade (frame "unbounded preceding", incremental).
    3. A soma da janela de W minutos é acumulado(m) - acumulado(m - W), com lag de W linhas,
       já que a grade é regular. Minutos sem dado contribuem com zero (lacunas não são preenchidas).
    Equivale a rangeBetween(-(W-1) min, 0): só passado e instante atual, sem vazamento.
    """
    sensors = analog + digital
    base = df.withColumn("m", (F.col("ts").cast("long") / 60).cast("long"))
    lo, hi = base.agg(F.min("m"), F.max("m")).first()
    grid = df.sparkSession.range(lo, hi + 1).withColumnRenamed("id", "m")
    g = grid.join(base, "m", "left")

    present = F.col("ts").isNotNull()
    cum = Window.orderBy("m").rowsBetween(Window.unboundedPreceding, 0)
    order = Window.orderBy("m")

    cs = [F.sum(present.cast("double")).over(cum).alias("cs_n")]
    for c in sensors:
        x = F.coalesce(F.col(c), F.lit(0.0))
        cs.append(F.sum(x).over(cum).alias(f"cs_{c}"))
        if c in analog:
            cs.append(F.sum(x * x).over(cum).alias(f"cs2_{c}"))
    g = g.select("m", "ts", *sensors, *[F.col("n_readings")] if "n_readings" in df.columns else [], *cs).cache()

    feats = []
    for w in windows_min:
        lag = lambda name: F.lag(F.col(name), w, 0.0).over(order)  # noqa: E731
        n_w = F.col("cs_n") - lag("cs_n")
        feats.append(n_w.alias(f"n_{w}m"))
        for c in sensors:
            s1 = F.col(f"cs_{c}") - lag(f"cs_{c}")
            mean = F.when(n_w > 0, s1 / n_w)
            if c in analog:
                s2 = F.col(f"cs2_{c}") - lag(f"cs2_{c}")
                var = F.when(n_w > 1, (s2 - n_w * mean * mean) / (n_w - 1))
                feats.append(F.sqrt(F.greatest(var, F.lit(0.0))).alias(f"{c}_std_{w}m"))
                feats.append(mean.alias(f"{c}_mean_{w}m"))
            else:
                feats.append(mean.alias(f"{c}_duty_{w}m"))
    keep = ["ts", *sensors] + (["n_readings"] if "n_readings" in df.columns else [])
    return g.select(*keep, *feats).filter(F.col("ts").isNotNull())


def build_features(
    silver: DataFrame, analog: list[str], digital: list[str], variant: str, naive: bool = False
) -> DataFrame:
    """Features independem do horizonte: calcule uma vez e reutilize em todos os X."""
    df = silver
    if variant == "e2":
        fn = add_rolling_features_naive if naive else add_rolling_features
        df = fn(silver, analog, digital, config.ROLLING_WINDOWS_MIN)
    return df


def build_gold(
    features: DataFrame,
    variant: str,
    analog: list[str],
    digital: list[str],
    failures: list[dict],
    horizon_h: int,
) -> tuple[DataFrame, dict]:
    log: dict = {"variante": variant, "horizonte_h": horizon_h}

    df = add_labels(features, failures, horizon_h)
    log["linhas_silver"] = features.count()

    log["linhas_removidas_em_falha"] = df.filter("in_failure").count()
    df = df.filter(~F.col("in_failure"))

    if variant == "e2":
        for w in config.ROLLING_WINDOWS_MIN:
            df = df.filter(F.col(f"n_{w}m") / w >= config.MIN_WINDOW_COVERAGE)
        log["linhas_removidas_sem_cobertura"] = (
            log["linhas_silver"] - log["linhas_removidas_em_falha"] - df.count()
        )
        log["cobertura_minima_janela"] = config.MIN_WINDOW_COVERAGE
        log["janelas_min"] = config.ROLLING_WINDOWS_MIN

    meta = {"ts", "in_failure", "failure_id_ahead", "label", "n_readings"}
    feature_cols = [
        c for c in df.columns
        if c not in meta and not (c.startswith("n_") and c.endswith("m"))
    ]
    df = df.select("ts", *feature_cols, "failure_id_ahead", "label")

    total = df.count()
    positives = df.filter("label = 1").count()
    log["linhas_gold"] = total
    log["positivos"] = positives
    log["taxa_positivos"] = round(positives / total, 6) if total else None
    log["positivos_por_falha"] = {
        str(r["failure_id_ahead"]): r["count"]
        for r in df.filter("label = 1").groupBy("failure_id_ahead").count().collect()
    }
    log["n_features"] = len(feature_cols)
    log["sensores_analogicos"] = analog
    log["sensores_digitais"] = digital
    return df, log


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=list(config.VARIANTS), required=True)
    parser.add_argument("--horizon", type=int, default=config.HORIZON_HOURS)
    parser.add_argument("--all-horizons", action="store_true")
    parser.add_argument(
        "--windows", type=str, default=None,
        help="janelas em minutos separadas por vírgula (ex.: 60,360); padrão: config.ROLLING_WINDOWS_MIN",
    )
    parser.add_argument(
        "--min-coverage", type=float, default=None,
        help="fração mínima de minutos presentes na janela (padrão: config.MIN_WINDOW_COVERAGE)",
    )
    parser.add_argument(
        "--naive-rolling", action="store_true",
        help="usa rangeBetween em vez de somas acumuladas (mais lento; para comparar tempos)",
    )
    args = parser.parse_args()

    suffix = ""
    if args.windows:
        config.ROLLING_WINDOWS_MIN = [int(w) for w in args.windows.split(",")]
        suffix += "_w" + "-".join(str(w) for w in config.ROLLING_WINDOWS_MIN)
    if args.min_coverage is not None:
        config.MIN_WINDOW_COVERAGE = args.min_coverage
        suffix += f"_c{int(round(args.min_coverage * 100))}"

    spark = get_spark(f"metropt3-gold-{args.variant}")
    silver_variant = "e0" if args.variant == "e0" else "e1"
    silver_dir = config.SILVER_DIR / silver_variant
    silver = spark.read.parquet(str(silver_dir))
    analog, digital = load_columns(silver_dir)
    failures = load_failures()
    print("Falhas (após unir sobreposições):", failures)

    features = build_features(silver, analog, digital, args.variant, args.naive_rolling).cache()

    horizons = config.HORIZON_SENSITIVITY if args.all_horizons else [args.horizon]
    config.DOCS_DIR.mkdir(exist_ok=True)
    for h in horizons:
        df, log = build_gold(features, args.variant, analog, digital, failures, h)
        out = config.GOLD_DIR / f"{args.variant}_h{h}{suffix}"
        df.write.mode("overwrite").parquet(str(out))
        log["saida"] = str(out)
        report = config.DOCS_DIR / f"gold_{args.variant}_h{h}{suffix}_resumo.json"
        report.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(log, indent=2, ensure_ascii=False))
    spark.stop()


if __name__ == "__main__":
    main()
