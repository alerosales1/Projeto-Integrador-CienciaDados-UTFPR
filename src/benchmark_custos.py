"""P3: custos de processamento do pipeline (tempo e disco).

Mede, na máquina em que roda (os resultados dependem do hardware):
  A. Armazenamento: tamanho em disco do CSV bruto, da Bronze (Parquet) e da Silver.
  B. Leitura: CSV × Parquet, com pandas e com Spark.
  C. Silver de 1 min: Spark × pandas (mesmas regras; confere o número de linhas).
  D. Janelas por tempo (1 h e 6 h): pandas × Spark com somas acumuladas × Spark com
     rangeBetween (versão simples). A versão simples só roda em um subconjunto
     (--naive-rows), porque seu custo cresce com o tamanho da janela.

Uso (depois de bronze.py, silver.py --variant e1):
  python src/benchmark_custos.py
  python src/benchmark_custos.py --repeats 5 --naive-rows 100000
  python src/benchmark_custos.py --skip-naive
Saídas: docs/custos_p3.csv e docs/custos_p3.json
"""
import argparse
import json
import os
import platform
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

import config
from spark_utils import detect_columns, get_spark  # configura o winutils no Windows antes do pyspark

import pyspark  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

import gold  # noqa: E402
from detection import window_features  # noqa: E402

rows: list[dict] = []


def size_mb(path) -> float:
    p = Path(path)
    files = [p] if p.is_file() else [f for f in p.rglob("*") if f.is_file()]
    return sum(f.stat().st_size for f in files) / 1e6


def timed(fn, repeats: int):
    times, out = [], None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    return times, out


def record(experimento: str, metodo: str, linhas: int, times: list[float], **extra) -> None:
    med = statistics.median(times)
    rows.append({
        "experimento": experimento, "metodo": metodo, "linhas": linhas,
        "tempo_mediano_s": round(med, 3), "tempo_min_s": round(min(times), 3),
        "tempo_max_s": round(max(times), 3), "repeticoes": len(times),
        "linhas_por_s": round(linhas / med) if med > 0 else None, **extra,
    })
    print(f"  {experimento:<22} {metodo:<34} {linhas:>10,} linhas  {med:8.2f} s  (n={len(times)})")


def noop(df) -> None:
    """Força a execução completa do plano sem gravar nada."""
    df.write.format("noop").mode("overwrite").save()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--naive-rows", type=int, default=60_000)
    ap.add_argument("--naive-repeats", type=int, default=1)
    ap.add_argument("--skip-naive", action="store_true")
    args = ap.parse_args()

    csv_path = config.find_raw_csv()
    bronze = config.BRONZE_DIR / "metropt3"
    silver_e1 = config.SILVER_DIR / "e1"
    spark = get_spark("metropt3-custos")

    # ------------------------------------------------------------------ A. Disco
    print("\nA. Armazenamento")
    disk = {
        "csv_bruto_mb": size_mb(csv_path),
        "bronze_parquet_mb": size_mb(bronze),
        "silver_e1_parquet_mb": size_mb(silver_e1),
    }
    disk["bronze_sobre_csv"] = round(disk["bronze_parquet_mb"] / disk["csv_bruto_mb"], 3)
    for k, v in disk.items():
        print(f"  {k:<24} {v:,.3f}")

    # ------------------------------------------------------------------ B. Leitura
    print("\nB. Leitura completa")
    n_rows = len(pd.read_parquet(bronze))
    # aquecimento do Spark (JVM, caches), fora da medição
    noop(spark.read.parquet(str(bronze)))

    t, _ = timed(lambda: pd.read_csv(csv_path), args.repeats)
    record("B_leitura", "pandas read_csv", n_rows, t)
    t, _ = timed(lambda: pd.read_parquet(bronze), args.repeats)
    record("B_leitura", "pandas read_parquet", n_rows, t)
    t, _ = timed(lambda: noop(spark.read.option("header", True).option("inferSchema", True).csv(str(csv_path))), args.repeats)
    record("B_leitura", "Spark csv (inferSchema)", n_rows, t)
    t, _ = timed(lambda: noop(spark.read.parquet(str(bronze))), args.repeats)
    record("B_leitura", "Spark parquet", n_rows, t)

    # ------------------------------------------------------------------ C. Silver 1 min
    print("\nC. Silver de 1 min (Bronze -> baldes de 1 min)")
    b = spark.read.parquet(str(bronze))
    time_col, analog, digital = detect_columns(b)
    sensors = analog + digital
    min_n = config.MIN_READINGS_PER_BUCKET

    def silver_spark():
        df = (
            spark.read.parquet(str(bronze))
            .select(F.col(time_col).cast("timestamp").alias("ts"), *sensors)
            .dropDuplicates(["ts"]).dropna(subset=sensors)
        )
        agg = df.groupBy(F.window("ts", config.RESAMPLE)).agg(
            F.count(F.lit(1)).alias("n_readings"), *[F.avg(c).alias(c) for c in sensors]
        )
        return agg.select(F.col("window.start").alias("ts"), "n_readings", *sensors).filter(F.col("n_readings") >= min_n)

    def silver_pandas():
        df = pd.read_parquet(bronze, columns=[time_col] + sensors)
        df["ts"] = pd.to_datetime(df[time_col])
        df = df.drop_duplicates("ts").dropna(subset=sensors)
        g = df.groupby(df["ts"].dt.floor("min"))
        out = g[sensors].mean()
        out["n_readings"] = g.size()
        return out[out["n_readings"] >= min_n].reset_index()

    n_spark = silver_spark().count()
    t, out_pd = timed(silver_pandas, args.repeats)
    n_pandas = len(out_pd)
    record("C_silver_1min", "pandas (leitura + agregação)", n_rows, t, linhas_saida=n_pandas)
    t, _ = timed(lambda: noop(silver_spark()), args.repeats)
    record("C_silver_1min", "Spark (leitura + agregação)", n_rows, t, linhas_saida=n_spark)
    check_c = {"linhas_saida_spark": n_spark, "linhas_saida_pandas": n_pandas, "iguais": n_spark == n_pandas}
    print("  conferência do número de linhas:", check_c)

    # ------------------------------------------------------------------ D. Janelas
    print("\nD. Janelas por tempo (1 h e 6 h) sobre a Silver de 1 min")
    windows = config.ROLLING_WINDOWS_MIN
    sil = pd.read_parquet(silver_e1)
    sil["ts"] = pd.to_datetime(sil["ts"]).dt.tz_localize(None)
    sil = sil.sort_values("ts").reset_index(drop=True)
    tmp = Path(tempfile.mkdtemp(prefix="p3_"))
    check_d = {}
    try:
        def run_size(sub: pd.DataFrame, label: str, include_naive: bool) -> None:
            path = tmp / f"sub_{label}"
            path.mkdir(parents=True, exist_ok=True)
            # cópia para o Spark: timestamp em UTC (como a Silver gravada pelo Spark) e em microssegundos
            sub.assign(ts=sub["ts"].dt.tz_localize("UTC")).to_parquet(
                path / "part-0.parquet", coerce_timestamps="us", allow_truncated_timestamps=True
            )
            n = len(sub)

            t, feat_pd = timed(lambda: window_features(sub, analog, digital, windows, config.MIN_WINDOW_COVERAGE), args.repeats)
            record("D_janelas", f"pandas rolling [{label}]", n, t)

            def prefix():
                sdf = spark.read.parquet(str(path))
                noop(gold.add_rolling_features(sdf, analog, digital, windows))
                spark.catalog.clearCache()

            t, _ = timed(prefix, args.repeats)
            record("D_janelas", f"Spark somas acumuladas [{label}]", n, t)

            if include_naive:
                def naive():
                    sdf = spark.read.parquet(str(path))
                    noop(gold.add_rolling_features_naive(sdf, analog, digital, windows))

                t, _ = timed(naive, args.naive_repeats)
                record("D_janelas", f"Spark rangeBetween [{label}]", n, t)

                # conferência numérica: somas acumuladas × pandas (média de 1 h)
                col = f"{analog[0]}_mean_{windows[0]}m"
                sp = (gold.add_rolling_features(spark.read.parquet(str(path)), analog, digital, windows)
                      .select("ts", col).toPandas().sort_values("ts").set_index("ts")[col])
                pdv = feat_pd.set_index("ts")[col]
                common = sp.index.intersection(pdv.index)
                check_d[label] = {"coluna": col, "max_abs_diff": float((sp.loc[common] - pdv.loc[common]).abs().max())}

        n_naive = min(args.naive_rows, len(sil))
        run_size(sil.head(n_naive), f"{n_naive} linhas", include_naive=not args.skip_naive)
        if len(sil) > n_naive:
            run_size(sil, f"{len(sil)} linhas (completa)", include_naive=False)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  conferência numérica:", check_d)

    # ------------------------------------------------------------------ Saídas
    env = {
        "so": platform.platform(), "python": sys.version.split()[0], "pandas": pd.__version__,
        "pyspark": pyspark.__version__, "cpus": os.cpu_count(), "spark_master": spark.sparkContext.master,
        "windows_min": windows, "min_window_coverage": config.MIN_WINDOW_COVERAGE,
    }
    config.DOCS_DIR.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(config.DOCS_DIR / "custos_p3.csv", index=False)
    (config.DOCS_DIR / "custos_p3.json").write_text(
        json.dumps({"ambiente": env, "disco": disk, "conferencias": {"silver_1min": check_c, "janelas": check_d},
                    "medicoes": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nResumo:\n", df[["experimento", "metodo", "linhas", "tempo_mediano_s", "linhas_por_s"]].to_string(index=False))
    print("\nSalvo em docs/custos_p3.csv e docs/custos_p3.json")
    spark.stop()


if __name__ == "__main__":
    main()
