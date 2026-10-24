"""Baixa a base MetroPT-3 (UCI) para data/raw.

Fonte: Davari et al., MetroPT-3 Dataset, UCI Machine Learning Repository.
DOI: 10.24432/C5VW3R | Licença: CC BY 4.0

Uso:  python src/download_data.py
Se o download automático falhar, baixe manualmente em
https://archive.ics.uci.edu/dataset/791/metropt+3+dataset
e coloque o CSV em data/raw/.
"""
import sys
import urllib.request
import zipfile

from config import RAW_DIR

URL = "https://archive.ics.uci.edu/static/public/791/metropt+3+dataset.zip"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / "metropt3.zip"
    try:
        print(f"Baixando {URL} ...")
        urllib.request.urlretrieve(URL, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(RAW_DIR)
        zip_path.unlink()
        print("Arquivos em data/raw:", [p.name for p in RAW_DIR.iterdir()])
    except Exception as exc:  # noqa: BLE001
        print(f"Falha no download automático: {exc}", file=sys.stderr)
        print(
            "Baixe manualmente em https://archive.ics.uci.edu/dataset/791/"
            "metropt+3+dataset e coloque o CSV em data/raw/",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
