# Manutenção preditiva com a MetroPT-3: efeito das decisões de engenharia de dados

Projeto Integrador – Especialização em Ciência de Dados (UTFPR), foco técnico.

## Pergunta de pesquisa

> Como decisões de engenharia de dados (qualidade, janelas temporais, horizonte de rotulagem e balanceamento) afetam o desempenho de modelos de predição de falhas do compressor de um trem de metrô, e a que custo de processamento?

- **P1 (desempenho):** quanto cada etapa do pipeline (variantes E0–E4) melhora as métricas?
- **P2 (robustez):** como cada modelo degrada com dados removidos ou corrompidos?
- **P3 (eficiência):** quanto custam tempo e disco em CSV × Parquet × Delta e em Spark × pandas?

## Base de dados

MetroPT-3 (UCI): leituras de pressão, temperatura, corrente do motor e válvulas do compressor (Air Production Unit) de um trem do metrô do Porto.
Davari et al., *MetroPT-3 Dataset*, UCI Machine Learning Repository, DOI [10.24432/C5VW3R](https://doi.org/10.24432/C5VW3R). Licença CC BY 4.0.

A base não vem rotulada. As falhas estão no relatório de manutenção da página da UCI, que deve ser copiado para `data/external/failure_reports.csv` (colunas `failure_id,start_time,end_time,failure_type,severity`).

## Arquitetura (medalhão)

| Camada | Conteúdo | Código |
|---|---|---|
| Bronze | CSV bruto em Parquet, sem alterar valores | `src/bronze.py` |
| Silver | Tipagem, deduplicação, qualidade, reamostragem | `src/silver.py` (a implementar) |
| Gold | Rótulos, features em janelas, divisão temporal | `src/gold.py` (a implementar) |

Parâmetros dos experimentos (semente, horizonte, janelas, frequência) ficam em `src/config.py`.

## Estrutura

```
data/
  raw/ bronze/ silver/ gold/   # não versionados (ver .gitignore)
  external/                    # relatório de falhas (versionado)
notebooks/                     # análise exploratória e experimentos
src/                           # pipeline e utilitários
docs/                          # figuras e anotações para o artigo
```

## Como reproduzir

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/download_data.py       # baixa a base para data/raw
python src/bronze.py              # gera a camada Bronze
jupyter notebook notebooks/01_eda_metropt3.ipynb
```

O PySpark exige Java instalado (JDK 11 ou 17).

## Status

- [x] Estrutura do repositório
- [x] Download e Bronze
- [x] Notebook de análise exploratória
- [ ] Preencher o relatório de falhas
- [ ] Silver
- [ ] Gold
- [ ] Modelagem e experimentos E0–E4
- [ ] Artigo
