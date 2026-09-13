# Análise exploratória: impactos socioeconômicos da COVID-19 no Brasil

## Como executar

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python analise.py
```

## Arquivos de dados

- `data/brazil_owid.csv` — Our World in Data (Brasil)
- `data/brazil_mobility.csv` — Google COVID-19 Community Mobility Reports (Brasil)

## Visualizações

Os gráficos são gerados em `figures/`:

1. `figures/01_mobilidade_vs_obitos.png`
2. `figures/02_restricao_vs_trabalho_box.png`
3. `figures/03_comparacao_abril_barras.png`
4. `figures/04_vacinacao_vs_obitos.png`
