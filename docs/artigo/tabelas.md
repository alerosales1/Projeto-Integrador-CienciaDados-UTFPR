# Tabelas do artigo (geradas por make_artigo_assets.py)

## t_prognostico

Prognóstico (E0–E2, 3 modelos, 3 falhas de teste): resultados por horizonte contra o acaso. Células vazias: arquivo sem baseline de acaso (reexecutar o notebook 02).

| horizonte_h | combinações | com_lift_maior_que_1 | com_p_menor_que_0,10 | menor_p | maior_lift | detecções_observadas | detecções_esperadas_por_acaso |
|---|---|---|---|---|---|---|---|
| 2.000 | 27.000 | 8.000 | 1.000 | 0.070 | 3.974 | 10.000 | 6.170 |
| 6.000 | 27.000 | 3.000 | 0.000 | 0.125 | 1.550 | 6.000 | 9.845 |
| 12.000 | 27.000 | 6.000 | 0.000 | 0.110 | 1.909 | 4.000 | 10.265 |
| 24.000 | 27.000 | 7.000 | 0.000 | 0.100 | 6.377 | 7.000 | 12.705 |

## t_supervisionado

Braço supervisionado (detecção da falha em curso): PR-AUC média dos 3 modelos por falha, detecções sustentadas (alarme em pelo menos 10% dos minutos da falha) e falsos alarmes por dia. E4* é exploratória (só analógicos).

| variante | PR-AUC falha 2 | PR-AUC falha 3 | PR-AUC falha 4 | detecções_sustentadas | combinações | falsos_alarmes_por_dia |
|---|---|---|---|---|---|---|
| E0 | 0.652 | 0.947 | 0.729 | 8.000 | 9 | 1.263 |
| E1 | 0.377 | 0.957 | 0.530 | 7.000 | 9 | 0.962 |
| E2 | 0.107 | 0.988 | 0.278 | 6.000 | 9 | 0.387 |
| E3 | 0.143 | 0.931 | 0.113 | 5.000 | 9 | 0.494 |
| E4* | 0.285 | 0.974 | 0.037 | 5.000 | 9 | 0.954 |

## t_supervisionado_modelos

Braço supervisionado: PR-AUC por variante, modelo e falha testada.

| variante | modelo | PR-AUC falha 2 | PR-AUC falha 3 | PR-AUC falha 4 |
|---|---|---|---|---|
| E0 | LightGBM | 0.933 | 0.957 | 0.527 |
| E0 | LogReg | 0.089 | 0.913 | 0.826 |
| E0 | RandomForest | 0.934 | 0.972 | 0.834 |
| E1 | LightGBM | 0.141 | 0.949 | 0.316 |
| E1 | LogReg | 0.086 | 0.939 | 0.571 |
| E1 | RandomForest | 0.905 | 0.984 | 0.703 |
| E2 | LightGBM | 0.128 | 0.996 | 0.338 |
| E2 | LogReg | 0.090 | 0.983 | 0.011 |
| E2 | RandomForest | 0.102 | 0.985 | 0.484 |
| E3 | LightGBM | 0.105 | 0.970 | 0.041 |
| E3 | LogReg | 0.080 | 0.847 | 0.067 |
| E3 | RandomForest | 0.245 | 0.976 | 0.231 |
| E4* | LightGBM | 0.568 | 0.986 | 0.018 |
| E4* | LogReg | 0.087 | 0.972 | 0.026 |
| E4* | RandomForest | 0.200 | 0.966 | 0.067 |

## t_sem_rotulos

Detecção sem rótulos no quantil de referência (percentil 99,9 da validação), ordenada pela PR-AUC. Variantes com sufixo -A (só analógicos) são exploratórias.

| variante | modelo | pr_auc | lift | eventos_detectados | eventos_sustentados | recall_medio | atraso_mediano_h | falsos_alarmes_por_dia | frac_minutos_normais_em_alarme |
|---|---|---|---|---|---|---|---|---|---|
| E2-A | IsolationForest | 0.784 | 22.637 | 0 | 0 | 0.000 |  | 0.000 | 0.000 |
| E2 | IsolationForest | 0.725 | 20.955 | 1 | 0 | 0.000 | 5.833 | 0.000 | 0.000 |
| E2-A | Mahalanobis | 0.608 | 17.569 | 3 | 3 | 0.528 | 3.967 | 0.022 | 0.013 |
| E1-A | IsolationForest | 0.580 | 16.754 | 3 | 2 | 0.490 | 0.333 | 0.335 | 0.015 |
| E0 | IsolationForest | 0.550 | 15.874 | 4 | 3 | 0.404 | 1.592 | 0.034 | 0.015 |
| E1-A | Mahalanobis | 0.526 | 15.181 | 1 | 1 | 0.151 | 0.000 | 0.168 | 0.009 |
| E1 | IsolationForest | 0.354 | 10.221 | 3 | 0 | 0.019 | 0.733 | 2.984 | 0.004 |
| E1 | Mahalanobis | 0.284 | 8.203 | 1 | 0 | 0.000 | 0.083 | 0.246 | 0.020 |
| E2 | Mahalanobis | 0.270 | 7.789 | 4 | 4 | 0.628 | 3.233 | 0.156 | 0.084 |
| E0 | Mahalanobis | 0.069 | 1.983 | 1 | 0 | 0.000 | 0.083 | 0.089 | 0.020 |

## t_ablacao

Ablação por grupos de sensores (E1, quantil 99,9; exploratória e pós-hoc).

| cenario | n_features | modelo | eventos_detectados | eventos_sustentados | atraso_mediano_h | recall_medio | falsos_alarmes_por_dia |
|---|---|---|---|---|---|---|---|
| sem_corrente | 14 | IsolationForest | 3 | 0 | 0.733 | 0.010 | 2.839 |
| sem_digitais | 7 | IsolationForest | 3 | 2 | 0.333 | 0.490 | 0.335 |
| sem_pressao | 10 | IsolationForest | 3 | 0 | 0.733 | 0.004 | 2.917 |
| sem_temperatura | 14 | IsolationForest | 3 | 0 | 0.733 | 0.006 | 2.604 |
| somente_corrente | 1 | IsolationForest | 0 | 0 |  | 0.000 | 1.107 |
| somente_digitais | 8 | IsolationForest | 3 | 0 | 0.733 | 0.005 | 2.191 |
| somente_pressao | 5 | IsolationForest | 1 | 1 | 0.883 | 0.114 | 2.012 |
| somente_temperatura | 1 | IsolationForest | 0 | 0 |  | 0.000 | 0.000 |
| todos | 15 | IsolationForest | 3 | 0 | 0.733 | 0.019 | 2.984 |
| sem_corrente | 14 | Mahalanobis | 2 | 0 | 1.533 | 0.015 | 0.246 |
| sem_digitais | 7 | Mahalanobis | 1 | 1 | 0.000 | 0.151 | 0.168 |
| sem_pressao | 10 | Mahalanobis | 1 | 1 | 2.850 | 0.079 | 0.514 |
| sem_temperatura | 14 | Mahalanobis | 1 | 0 | 0.083 | 0.000 | 0.246 |
| somente_corrente | 1 | Mahalanobis | 0 | 0 |  | 0.000 | 0.335 |
| somente_digitais | 8 | Mahalanobis | 1 | 1 | 2.850 | 0.073 | 0.514 |
| somente_pressao | 5 | Mahalanobis | 2 | 1 | 0.142 | 0.168 | 0.168 |
| somente_temperatura | 1 | Mahalanobis | 1 | 1 | 0.017 | 0.156 | 0.000 |
| todos | 15 | Mahalanobis | 1 | 0 | 0.083 | 0.000 | 0.246 |

## t_deriva

Falsos alarmes (episódios por dia) por mês, quantil 99,9.

| variante | modelo | 2020-04 | 2020-05 | 2020-06 | 2020-07 | 2020-08 |
|---|---|---|---|---|---|---|
| E0 | IsolationForest | 0.000 | 0.100 | 0.000 | 0.050 | 0.000 |
| E0 | Mahalanobis | 0.290 | 0.050 | 0.100 | 0.100 | 0.050 |
| E1 | IsolationForest | 0.290 | 1.390 | 2.910 | 3.450 | 4.910 |
| E1 | Mahalanobis | 0.440 | 0.620 | 0.100 | 0.190 | 0.050 |
| E1-A | IsolationForest | 0.150 | 0.150 | 0.140 | 0.870 | 0.230 |
| E1-A | Mahalanobis | 0.290 | 0.410 | 0.050 | 0.150 | 0.050 |
| E2 | IsolationForest | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| E2 | Mahalanobis | 0.440 | 0.210 | 0.140 | 0.150 | 0.050 |
| E2-A | IsolationForest | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| E2-A | Mahalanobis | 0.000 | 0.100 | 0.000 | 0.000 | 0.000 |

## t_p2

P2: média entre as sementes por configuração e cenário (nível 0 = referência).

| config | cenario | nivel | eventos_sustentados | recall_medio | pr_auc | minutos_de_referencia_com_score |
|---|---|---|---|---|---|---|
| E1-A IF | base | 0.000 | 2.000 | 0.490 | 0.580 | 1.000 |
| E1-A IF | faltantes | 0.100 | 1.000 | 0.205 | 0.490 | 0.900 |
| E1-A IF | faltantes | 0.300 | 1.600 | 0.231 | 0.397 | 0.700 |
| E1-A IF | ruido | 0.100 | 2.000 | 0.383 | 0.559 | 1.000 |
| E1-A IF | ruido | 0.300 | 2.000 | 0.217 | 0.532 | 1.000 |
| E2-A Mah (cob 50%) | base | 0.000 | 3.000 | 0.530 | 0.608 | 1.000 |
| E2-A Mah (cob 50%) | faltantes | 0.100 | 3.000 | 0.476 | 0.554 | 0.900 |
| E2-A Mah (cob 50%) | faltantes | 0.300 | 3.000 | 0.370 | 0.437 | 0.700 |
| E2-A Mah (cob 50%) | ruido | 0.100 | 3.000 | 0.525 | 0.608 | 1.000 |
| E2-A Mah (cob 50%) | ruido | 0.300 | 3.000 | 0.511 | 0.608 | 1.000 |
| E2-A Mah (cob 80%) | base | 0.000 | 3.000 | 0.528 | 0.608 | 1.000 |
| E2-A Mah (cob 80%) | faltantes | 0.100 | 3.000 | 0.446 | 0.509 | 0.860 |
| E2-A Mah (cob 80%) | faltantes | 0.300 | 0.000 | 0.000 |  | 0.000 |
| E2-A Mah (cob 80%) | ruido | 0.100 | 3.000 | 0.524 | 0.609 | 1.000 |
| E2-A Mah (cob 80%) | ruido | 0.300 | 3.000 | 0.511 | 0.610 | 1.000 |

## t_custos

P3: tempo mediano por etapa e método (depende do hardware; ver custos_p3.json).

| experimento | metodo | linhas | tempo_mediano_s | linhas_por_s | vezes_o_mais_rápido |
|---|---|---|---|---|---|
| B_leitura | pandas read_csv | 1516948 | 2.603 | 582733 | 20.500 |
| B_leitura | pandas read_parquet | 1516948 | 0.127 | 11916782 | 1.000 |
| B_leitura | Spark csv (inferSchema) | 1516948 | 1.966 | 771719 | 15.500 |
| B_leitura | Spark parquet | 1516948 | 0.281 | 5390070 | 2.200 |
| C_silver_1min | pandas (leitura + agregação) | 1516948 | 0.578 | 2626022 | 1.000 |
| C_silver_1min | Spark (leitura + agregação) | 1516948 | 3.103 | 488837 | 5.400 |
| D_janelas | pandas rolling [60000 linhas] | 60000 | 0.130 | 462835 | 1.000 |
| D_janelas | Spark somas acumuladas [60000 linhas] | 60000 | 2.172 | 27628 | 16.700 |
| D_janelas | Spark rangeBetween [60000 linhas] | 60000 | 26.358 | 2276 | 202.800 |
| D_janelas | pandas rolling [252382 linhas (completa)] | 252382 | 0.572 | 441410 | 1.000 |
| D_janelas | Spark somas acumuladas [252382 linhas (completa)] | 252382 | 4.634 | 54465 | 8.100 |

## t_tempos_execucao

Tempo de parede de cada etapa em uma execução limpa (depende do hardware; ver ambiente_execucao.json).

| passo | descricao | duracao_s | status |
|---|---|---|---|
| bronze | CSV -> Bronze (Parquet) | 24.900 | ok |
| silver_e0 | Silver E0 (10 s, sem regras) | 15.400 | ok |
| silver_e1 | Silver E1 (1 min, qualidade) | 30.900 | ok |
| gold_e0_h24 | Gold E0, horizonte 24 h | 17.300 | ok |
| gold_e1_h24 | Gold E1, horizonte 24 h | 14.200 | ok |
| gold_e2_h24 | Gold E2 (janelas do config.py), horizonte 24 h | 23.900 | ok |
| tests | Testes automatizados | 4.800 | ok |
| nb01 | Notebook 01 (EDA) | 15.300 | ok |
| nb02_h24 | Notebook 02 (prognóstico), horizonte 24 h | 128.300 | ok |
| nb03 | Notebook 03 (supervisionado E0–E4) | 227.500 | ok |
| nb04 | Notebook 04 (sem rótulos) | 106.300 | ok |
| nb05 | Notebook 05 (P2 degradação) | 71.200 | ok |
| custos | Benchmark de custos (P3) | 104.000 | ok |
| assets | Tabelas e figuras do artigo | 3.400 | ok |
| TOTAL | soma das etapas | 787.400 |  |

## t_tempos_modelos

Tempo acumulado de treino (fit) e predição por tipo de modelo, somando todos os notebooks.

| estimador | metodo | chamadas | total_s | mediana_s |
|---|---|---|---|---|
| RandomForestClassifier | fit | 24 | 163.888 | 4.078 |
| IsolationForest | score_samples | 70 | 53.764 | 0.742 |
| LGBMClassifier | fit | 25 | 37.828 | 1.330 |
| LogisticRegression | fit | 24 | 36.651 | 1.290 |
| IsolationForest | fit | 35 | 15.972 | 0.438 |
| LGBMClassifier | predict_proba | 48 | 8.448 | 0.169 |
| RandomForestClassifier | predict_proba | 48 | 5.729 | 0.103 |
| LedoitWolf | mahalanobis | 102 | 4.491 | 0.016 |
| LedoitWolf | fit | 51 | 1.377 | 0.030 |
| LogisticRegression | predict_proba | 48 | 0.219 | 0.003 |
