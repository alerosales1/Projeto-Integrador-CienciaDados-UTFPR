# Detecção de condição anormal em um compressor de trem (MetroPT-3): efeito das decisões de engenharia de dados

Projeto Integrador – Especialização em Ciência de Dados (UTFPR), foco técnico.

## Perguntas de pesquisa

> Como decisões de engenharia de dados (resolução temporal, qualidade, janelas por tempo, robustez a dados degradados e formato de armazenamento) afetam a detecção de condições anormais em um compressor de trem, e a que custo de processamento? É possível antecipar a falha com 2 a 24 h de antecedência?

- **P1, desempenho:** quanto cada versão do pipeline (E0–E4) muda a detecção?
- **P2, robustez:** como as versões e os modelos degradam com minutos ausentes e ruído?
- **P3, eficiência:** custo de tempo e disco (CSV × Parquet, Spark × pandas, `rangeBetween` × somas acumuladas).
- **P4, prognóstico (resultado negativo):** há evidência de antecipação em 2, 6, 12 e 24 h?

## Base de dados

MetroPT-3 (UCI): pressões, temperatura do óleo, corrente do motor e válvulas do compressor de um trem do metrô do Porto. Davari et al., *MetroPT-3 Dataset*, UCI Machine Learning Repository, DOI [10.24432/C5VW3R](https://doi.org/10.24432/C5VW3R), licença CC BY 4.0.

A base não vem rotulada. As 4 falhas (vazamentos de ar) estão no relatório de manutenção da página da UCI, copiado para `data/external/failure_reports.csv` (versionado).

## Arquitetura (medalhão) e variantes

| Camada | Conteúdo | Código |
|---|---|---|
| Bronze | CSV bruto em Parquet, sem alterar valores | `src/bronze.py` |
| Silver | `e0`: 10 s, sem regras de qualidade. `e1`: deduplicação, nulos, faixas dos sensores, baldes de 1 min (média; nos digitais, ciclo de trabalho) e cobertura mínima | `src/silver.py` |
| Gold | Rótulo de pré-falha com horizonte X, leituras dentro de falha removidas, features pontuais ou em janelas por tempo | `src/gold.py` |
| Divisão | Forward-chaining por falha, com embargo | `src/splits.py` |
| Avaliação | Métricas por evento, baseline de acaso por rotação circular | `src/evaluation.py` |
| Detecção sem rótulos | Treino só com dados normais; limiar no percentil da validação | `src/detection.py` |
| Degradação (P2) | Minutos ausentes e ruído gaussiano | `src/degradation.py` |
| Custos (P3) | Tempo e disco | `src/benchmark_custos.py` |
| Tabelas e figuras do artigo | Consolida os CSVs de `docs/` em `docs/artigo/` | `src/make_artigo_assets.py` |

| Variante | O que muda |
|---|---|
| E0 | Leituras de 10 s, sem regras de qualidade, features pontuais |
| E1 | + Silver de 1 min, qualidade e cobertura |
| E2 | + janelas por tempo de 1 h e 6 h (cobertura mínima de 80%) |
| E3 | + `class_weight` (só braço supervisionado) |
| E4 | + seleção de features por grupos de sensores (exploratória e pós-hoc) |

Decisões de projeto: janelas por **tempo** (há lacunas de até dois dias; lacunas não são preenchidas), rótulo de pré-falha com leituras em falha removidas, avaliação por evento com forward-chaining, todas as variantes avaliadas nos mesmos minutos, e quantil de referência do limiar (99,9) definido antes de ver o teste.

## Como reproduzir

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

python src/download_data.py          # baixa a base para data/raw
python src/bronze.py
python src/silver.py --variant e0
python src/silver.py --variant e1
python src/gold.py --variant e0      # e1, e2: prognóstico (horizonte padrão de 24 h)
python src/gold.py --variant e1
python src/gold.py --variant e2
python -m pytest tests
python src/benchmark_custos.py       # P3
# (rodar os notebooks 02 a 05; ver abaixo)
python src/make_artigo_assets.py     # tabelas e figuras do artigo em docs/artigo/
```

Requisitos: Java (JDK 17 ou superior) para o PySpark. No Windows, também o `winutils` em `C:\hadoop\bin`. Use `pandas<3`.

## Notebooks (em `notebooks/`, nesta ordem)

| Notebook | Conteúdo |
|---|---|
| `01_eda_metropt3` | Análise exploratória |
| `02_modelagem_metropt3` | Prognóstico (resultado negativo), com baseline de acaso; rodar com `HORIZON` = 2, 6, 12 e 24 (embargo fixo de 24 h) |
| `03_diagnostico_metropt3_v2` | Braço supervisionado, detecção da falha em curso, variantes E0–E4 |
| `04_deteccao_sem_rotulos_v2` | Detecção sem rótulos (Isolation Forest e Mahalanobis), deriva e ablação |
| `05_p2_degradacao` | Robustez a minutos ausentes e ruído |

As versões anteriores dos notebooks 03 e 04 foram substituídas pelas v2.

## Resultados e rastreabilidade

Cada execução grava resumos e tabelas em `docs/` (contagens por etapa de qualidade, positivos por falha, resultados por notebook, custos). O texto do artigo se apoia nesses arquivos.

## Limitações

4 eventos de um tipo de falha em um compressor: os resultados são descritivos e não generalizam. As configurações só com analógicos e a ablação surgiram dos resultados no teste e são exploratórias.
