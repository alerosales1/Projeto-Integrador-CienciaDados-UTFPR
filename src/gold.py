"""Camada Gold (a implementar).

  - rotulagem: cruzar cada leitura com as janelas de data/external/failure_reports.csv
    usando o horizonte config.HORIZON_HOURS
  - features em janelas deslizantes (config.ROLLING_WINDOWS): média, desvio, delta
  - divisão temporal treino/teste (nunca aleatória)
"""
