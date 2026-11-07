"""Utilitários compartilhados de Spark (sessão e colunas)."""
import os
import re
import sys

# No Windows, o Spark precisa do winutils (C:\hadoop\bin). Deve vir ANTES do pyspark.
if sys.platform == "win32":
    os.environ.setdefault("HADOOP_HOME", r"C:\hadoop")
    os.environ["PATH"] += os.pathsep + r"C:\hadoop\bin"

from pyspark.sql import DataFrame, SparkSession  # noqa: E402
from pyspark.sql.types import NumericType  # noqa: E402


def get_spark(app_name: str) -> SparkSession:
    """Sessão local em UTC, para que timestamps e janelas não dependam do fuso da máquina."""
    spark = (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


# Colunas que nunca são sensores: índice sem nome do CSV (o Spark o chama de _c0 e o
# bronze.py o renomeia para c0), "Unnamed: 0" do pandas e colunas de metadados (prefixo "_").
# O índice cresce com o tempo: usá-lo como feature vazaria a posição temporal.
IGNORED_COLUMN_PATTERN = re.compile(r"^(c\d+|Unnamed.*|_.*)$")


def detect_columns(
    df: DataFrame, time_col: str | None = None
) -> tuple[str, list[str], list[str]]:
    """Retorna (coluna_de_tempo, analógicas, digitais).

    Digital = no máximo 2 valores distintos. Ignora colunas de índice
    ("Unnamed...") e de metadados de ingestão (prefixo "_").
    """
    time_col = time_col or next(c for c in df.columns if "time" in c.lower())
    numeric = [
        f.name
        for f in df.schema.fields
        if isinstance(f.dataType, NumericType)
        and f.name != time_col
        and f.name != "n_readings"
        and not IGNORED_COLUMN_PATTERN.match(f.name)
    ]
    from pyspark.sql import functions as F

    counts = df.agg(*[F.countDistinct(c).alias(c) for c in numeric]).first()
    digital = [c for c in numeric if counts[c] <= 2]
    analog = [c for c in numeric if c not in digital]
    return time_col, analog, digital


def lit_ts(dt):
    """Literal de timestamp interpretado no fuso da sessão (UTC), não no do Python."""
    from pyspark.sql import functions as F

    return F.lit(dt.strftime("%Y-%m-%d %H:%M:%S")).cast("timestamp")


def save_columns(out_dir, analog: list[str], digital: list[str]) -> None:
    """Grava os grupos de colunas ao lado do Parquet (arquivos com "_" são ignorados pelo Spark)."""
    import json
    from pathlib import Path

    (Path(out_dir) / "_columns.json").write_text(
        json.dumps({"analog": analog, "digital": digital}), encoding="utf-8"
    )


def load_columns(silver_dir) -> tuple[list[str], list[str]]:
    """Lê os grupos de colunas gravados pela Silver.

    Necessário porque, na Silver de 1 min, os digitais viram ciclo de trabalho
    (valores fracionários) e deixam de ser detectáveis pelo número de valores distintos.
    """
    import json
    from pathlib import Path

    meta = json.loads((Path(silver_dir) / "_columns.json").read_text(encoding="utf-8"))
    return meta["analog"], meta["digital"]
