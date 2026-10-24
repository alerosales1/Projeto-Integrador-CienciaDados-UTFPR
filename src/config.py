"""Configuração central do projeto: caminhos e parâmetros dos experimentos."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"
EXTERNAL_DIR = DATA_DIR / "external"

# Relatório de falhas (preencher a partir da página da base na UCI)
FAILURE_REPORT = EXTERNAL_DIR / "failure_reports.csv"

# Parâmetros dos experimentos (viram variáveis do pipeline E0-E4)
SEED = 42
RESAMPLE_FREQ = "1min"      # frequência da reamostragem na camada Silver
HORIZON_HOURS = 24          # X em "haverá falha nas próximas X horas?"
ROLLING_WINDOWS = ["1h", "6h", "24h"]  # janelas das features na camada Gold


def find_raw_csv() -> Path:
    """Retorna o CSV bruto da MetroPT-3 dentro de data/raw."""
    candidates = sorted(RAW_DIR.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"Nenhum CSV em {RAW_DIR}. Rode `python src/download_data.py` "
            "ou coloque o arquivo manualmente nessa pasta."
        )
    return candidates[0]
