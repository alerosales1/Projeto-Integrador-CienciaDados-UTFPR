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
| Silver | `e0`: 10 s, sem regras de qualidade. `e1`: deduplicação, nulos, faixas dos sensores, baldes de 1 min (média; nos digitais, ciclo de trabalho) e cobertura mínima | `src/silver.py` |
| Gold | Rótulo de pré-falha com horizonte X, remoção das leituras dentro de falha, features pontuais (`e0`, `e1`) e em janelas por tempo (`e2`) | `src/gold.py` |
| Divisão | Forward-chaining por falha, com embargo | `src/splits.py` |

Variantes do pipeline: `e0` (10 s, sem qualidade), `e1` (+ Silver) e `e2` (+ janelas por tempo). Balanceamento (E3) e seleção de features (E4) ficam na etapa de modelagem.

Decisões de projeto:
- Janelas por **tempo**, nunca por número de linhas, porque há lacunas (a maior tem dois dias). Lacunas não são preenchidas; linhas sem cobertura mínima na janela são descartadas.
- Rótulo: positivo se uma falha **começa** nas próximas X horas. Leituras dentro de uma falha são removidas (senão o modelo só reconheceria falha em curso).
- Com 4 falhas, a avaliação é por evento (forward-chaining): treina nas falhas anteriores e testa na seguinte, com embargo de X horas.

Parâmetros (semente, horizonte, janelas, cobertura, faixas dos sensores) ficam em `src/config.py`.

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
python src/bronze.py              # Bronze
python src/silver.py --variant e0
python src/silver.py --variant e1
python src/gold.py --variant e0   # horizonte padrão: 24 h
python src/gold.py --variant e1
python src/gold.py --variant e2
python src/gold.py --variant e2 --all-horizons   # sensibilidade: 6, 12, 24, 48 h

python -m pytest tests            # testes da divisão por falhas
jupyter notebook notebooks/01_eda_metropt3.ipynb
```

Cada execução grava um resumo em `docs/` (contagens por etapa de qualidade, taxa de positivos), útil para o artigo.

O PySpark exige Java instalado (JDK 17 ou superior). No Windows, também é preciso o `winutils` em `C:\\hadoop\\bin`.

## Status

- [x] Estrutura do repositório
- [x] Download e Bronze
- [x] Análise exploratória
- [x] Relatório de falhas
- [x] Silver (e0, e1) e Gold (e0, e1, e2) com rótulos e janelas por tempo
- [x] Divisão por falhas (forward-chaining)
- [ ] Modelagem e experimentos E0–E4
- [ ] Artigo
