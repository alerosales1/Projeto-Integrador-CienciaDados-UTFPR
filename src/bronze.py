"""Camada Bronze: CSV bruto -> Parquet, sem alterar os valores.

Únicas mudanças: nomes de colunas compatíveis com Parquet e colunas de
metadados de ingestão (_source_file, _ingested_at).

Uso:  python src/bronze.py
"""
import re

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from config import BRONZE_DIR, find_raw_csv


def sanitize(name: str) -> str:
    """Parquet não aceita espaços e alguns símbolos nos nomes de coluna."""
    return re.sub(r"[ ,;{}()\n\t=:]+", "_", name.strip()).strip("_")


def main() -> None:
    raw = find_raw_csv()
    spark = SparkSession.builder.appName("metropt3-bronze").getOrCreate()

    df = (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .csv(str(raw))
    )
    df = df.toDF(*[sanitize(c) for c in df.columns])
    df = df.withColumn("_source_file", F.lit(raw.name)).withColumn(
        "_ingested_at", F.current_timestamp()
    )

    out = BRONZE_DIR / "metropt3"
    df.write.mode("overwrite").parquet(str(out))
    print(f"Bronze gravada em {out} | linhas: {df.count()} | colunas: {len(df.columns)}")
    spark.stop()


if __name__ == "__main__":
    main()
