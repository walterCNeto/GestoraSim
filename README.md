# GestoraSim

MVP educacional de sistema de gestão de ativos. **Todos os dados são simulados** (seed fixa, sem dependência externa).

## Uso

```bash
pip install -e ".[dev]"
python -m gestorasim.dados     # gera data/sim/*.parquet (seed=42) e imprime um resumo
python -m gestorasim.boletagem # roda um dia de boletagem (ordens, pré-trade, execução, rateio)
python -m gestorasim.mtm       # série de 504 pregões com MTM, eventos e P&L (~5 min)
python -m gestorasim.risco     # risco diário e enquadramento pós-trade (usa a série em disco, se válida)
python -m gestorasim.cota      # passivo (cotistas, cota, taxas, tributos) + conciliação com o administrador (XML)
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

- **Bloco 1:** `dominio` + `dados`.
- **Bloco 2:** `boletagem`, `enquadramento` (pré-trade) e `posicao` mínima — ciclo de vida da ordem com trilha de auditoria, ordem agregada, execução simulada, rateio.
- **Bloco 3:** `posicao` (ledger com liquidação D+2/D+0, eventos), `mtm` (ações, LTN, NTN-B, debêntures, DI), atribuição de P&L e série diária.
- **Bloco 4:** `risco` (fatores, DV01/CS01, VaR paramétrico e histórico, ES, stress, backtest Kupiec/Basileia, liquidez) e `enquadramento/pos_trade` (regras do regulamento + política de risco, alertas SOFT/HARD, desenquadramento ativo/passivo/vencido). Fonte única de preço: `mtm`. Caixa remunerado; dividendos reduzem o preço na data ex.
- **Bloco 5 (atual):** `cota` (cotistas com conversão/pagamento por classe, cota bruta e líquida, taxa de administração e de performance com linha d'água, come-cotas, IR/IOF por lote FIFO), `conciliacao` (arquivo XML do administrador, divergências injetadas, quebras com diagnóstico, cota paralela), backtest limpo/sujo com burn-in e gestão de caixa/reenquadramento no gerador de ordens.
- **Bloco 6:** `relatorios` e `app/` — relatórios e interface.

## Dados simulados (`data/sim/`, ignorado pelo git)

20 ações (GBM com correlação sintética PSD), 504 pregões a partir de 2023-01-02 (calendário B3 aproximado),
curva pré Nelson-Siegel com fatores AR(1), CDI e IPCA sintéticos, 4 LTN, 4 NTN-B, 5 debêntures (AAA a A-),
3 futuros de DI, 3 fundos (FIA, Multimercado, RF Crédito Privado), 200 cotistas com fluxo de aplicação/resgate
e eventos (dividendos, cupons, amortização).

Preços de renda fixa vêm só do `mtm` (PU ex-cupom na data de pagamento; os fluxos de `eventos` completam o retorno total). O bloco 1 não grava mais `precos_renda_fixa`.
