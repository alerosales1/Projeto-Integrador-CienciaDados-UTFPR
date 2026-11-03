"""Divisão temporal por eventos de falha (forward-chaining com embargo).

Com poucas falhas, uma divisão aleatória (ou "primeiro mês para treino", que não
contém nenhuma falha) não serve para classificação supervisionada. Aqui cada fold
testa em uma falha e treina somente com o passado:

  fold da falha 2: treino = tudo até o fim da falha 1
  fold da falha 3: treino = tudo até o fim da falha 2
  fold da falha 4: treino = tudo até o fim da falha 3

O teste começa `embargo_hours` depois do fim da falha anterior, para que a
dinâmica logo após o reparo não contamine a avaliação.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True)
class Fold:
    name: str
    train_end: datetime      # treino: ts <= train_end
    test_start: datetime     # teste:  test_start < ts <= test_end
    test_end: datetime
    test_failure_id: int


def forward_chaining_folds(failures: list[dict], embargo_hours: float) -> list[Fold]:
    """`failures`: lista de dicts com id, start, end (datetime), em qualquer ordem."""
    ordered = sorted(failures, key=lambda f: f["start"])
    folds = []
    for prev, cur in zip(ordered, ordered[1:]):
        folds.append(
            Fold(
                name=f"fold_falha_{cur['id']}",
                train_end=prev["end"],
                test_start=prev["end"] + timedelta(hours=embargo_hours),
                test_end=cur["end"],
                test_failure_id=cur["id"],
            )
        )
    return folds


def split_dataframe(df, fold: Fold, ts_col: str = "ts"):
    """Aplica um fold a um DataFrame do Spark e devolve (treino, teste)."""
    from pyspark.sql import functions as F

    from spark_utils import lit_ts

    ts = F.col(ts_col)
    train = df.filter(ts <= lit_ts(fold.train_end))
    test = df.filter((ts > lit_ts(fold.test_start)) & (ts <= lit_ts(fold.test_end)))
    return train, test
