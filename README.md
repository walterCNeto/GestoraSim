# GestoraSim

MVP educacional de sistema de gestão de ativos. **Todos os dados são simulados** (seed fixa, sem dependência externa).

## Uso

```bash
pip install -e ".[dev]"
python -m gestorasim.dados     # gera data/sim/*.parquet (seed=42) e imprime um resumo
pytest
```

```python
from gestorasim.dados import gerar_universo
tabelas = gerar_universo(seed=42)   # dict de DataFrames; também grava data/sim/*.parquet
```

Convenções: nomes em português; números em relatórios e prints com 3 casas decimais (`gestorasim.util.fmt3`).

## Fluxo FRONT / MIDDLE / BACK

| Etapa | Função | Pacotes |
|-------|--------|---------|
| FRONT | decisão de investimento e boletagem de ordens | `boletagem` |
| MIDDLE | posição, marcação a mercado, risco e enquadramento ao regulamento | `posicao`, `mtm`, `risco`, `enquadramento` |
| BACK | conciliação, cálculo da cota (passivo/cotistas) e relatórios | `conciliacao`, `cota`, `relatorios` |

`dominio` (modelos) e `dados` (simulação) alimentam todas as etapas.

## Roadmap

- **Bloco 1 (atual):** `dominio` + `dados`.
- **Bloco 2:** `boletagem` e `posicao` — ordens, execução simulada, posição e caixa; processamento dos eventos.
- **Bloco 3:** `mtm` — marcação a mercado (ações, LTN/NTN-B, debêntures, DI) sobre a curva simulada.
- **Bloco 4:** `risco` e `enquadramento` — VaR/volatilidade/DV01 e checagem dos limites do regulamento.
- **Bloco 5:** `conciliacao` e `cota` — conciliação posição × custódia, cota, aplicações/resgates.
- **Bloco 6:** `relatorios` e `app/` — relatórios e interface.

## Dados simulados (`data/sim/`, ignorado pelo git)

20 ações (GBM com correlação sintética PSD), 504 pregões a partir de 2023-01-02 (calendário B3 aproximado),
curva pré Nelson-Siegel com fatores AR(1), CDI e IPCA sintéticos, 4 LTN, 4 NTN-B, 5 debêntures (AAA a A-),
3 futuros de DI, 3 fundos (FIA, Multimercado, RF Crédito Privado), 200 cotistas com fluxo de aplicação/resgate
e eventos (dividendos, cupons, amortização).

Convenção dos PUs de renda fixa: ex-cupom na data de pagamento; os fluxos de `eventos` completam o retorno total.
