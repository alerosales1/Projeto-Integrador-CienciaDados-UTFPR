"""Configuração central do projeto: caminhos e parâmetros dos experimentos."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"
EXTERNAL_DIR = DATA_DIR / "external"
DOCS_DIR = ROOT / "docs"

# Relatório de falhas (copiado da página da base na UCI)
FAILURE_REPORT = EXTERNAL_DIR / "failure_reports.csv"

# ---------------------------------------------------------------------------
# Parâmetros dos experimentos
# ---------------------------------------------------------------------------
SEED = 42

# Silver -------------------------------------------------------------------
# duração do balde da reamostragem (sintaxe do Spark)
RESAMPLE = "1 minute"
READINGS_PER_BUCKET = 6        # cadência observada: 10 s => 6 leituras por minuto
# cobertura mínima do balde; abaixo disso o minuto é descartado
MIN_READINGS_PER_BUCKET = 4

# Margem de 0,05 para ruído de offset dos sensores (mínimos observados ~ -0,03).
# Faixas largas de propósito: os valores altos ocorrem nas falhas e não podem ser removidos.
SENSOR_RANGES: dict[str, tuple[float, float]] = {
    "TP2": (-0.05, 11.0),
    "TP3": (-0.05, 11.0),
    "H1": (-0.05, 11.0),
    "DV_pressure": (-0.05, 11.0),
    "Reservoirs": (-0.05, 11.0),
    "Oil_temperature": (0.0, 100.0),
    "Motor_current": (0.0, 10.0),
}

# Gold ---------------------------------------------------------------------
HORIZON_HOURS = 24                      # X em "haverá falha nas próximas X horas?"
# valores de X para a análise de sensibilidade
HORIZON_SENSITIVITY = [6, 12, 24, 48]
# janelas por TEMPO, em minutos (1 h, 6 h, 24 h)
# 1 h e 6 h (a de 24 h elimina os positivos da falha 4)
ROLLING_WINDOWS_MIN = [60, 360]
# fração mínima de minutos presentes na janela
MIN_WINDOW_COVERAGE = 0.8
# intervalo de segurança entre treino e teste
EMBARGO_HOURS = HORIZON_HOURS

# Variantes do pipeline (E3 e E4 acontecem na etapa de modelagem)
VARIANTS = {
    "e0": "10 s, sem regras de qualidade, features pontuais",
    "e1": "+ Silver: 1 min, cobertura, validação, features pontuais",
    "e2": "+ features em janelas por tempo",
}


def find_raw_csv() -> Path:
    """Retorna o CSV bruto da MetroPT-3 dentro de data/raw."""
    candidates = sorted(RAW_DIR.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"Nenhum CSV em {RAW_DIR}. Rode `python src/download_data.py` "
            "ou coloque o arquivo manualmente nessa pasta."
        )
    return candidates[0]
