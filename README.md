# GestoraSim

MVP educacional de um sistema de gestão de ativos de ponta a ponta — boletagem, enquadramento, posição, marcação a
mercado, risco, passivo/cota, conciliação e relatórios — com **dados 100% simulados** (seed fixa, sem dependência de
dados externos, CNPJs fictícios). Python, pandas, numpy, pydantic; app em Streamlit.

> Material educacional. Nada aqui é recomendação de investimento nem substitui sistemas e controles reais.

## Como rodar

```bash
pip install -e ".[dev,app]"                  # pacote + pytest + streamlit

python -m gestorasim.orquestrador --de 2023-01-02 --ate 2024-12-30    # ciclo diário completo (~1,5 min) -> data/sim/
python -m gestorasim.orquestrador --data 2023-06-15 --reprocessar      # recalcula 15/06/2023 e as datas seguintes
streamlit run app/Comece_aqui.py                                              # app (lê data/sim; nunca recalcula a série)

pytest                                       # ~2 min
```

Sem `--de/--ate` o orquestrador roda os 504 pregões do calendário (até janeiro de 2025). Outras entradas (cada uma
imprime números com 3 casas decimais):

| Comando | O que faz |
|---|---|
| `python -m gestorasim.dados` | gera o universo simulado (`data/sim/*.parquet`) |
| `python -m gestorasim.boletagem` | um dia de boletagem: pré-trade, execução, rateio |
| `python -m gestorasim.mtm` | série com MTM, eventos e atribuição de P&L |
| `python -m gestorasim.risco` | risco diário e enquadramento pós-trade |
| `python -m gestorasim.cota` | passivo, cota, taxas, tributos e conciliação |
| `python -m gestorasim.orquestrador.diagnostico [--ablacao]` | decomposição do retorno do Multimercado |
| `python -m gestorasim.orquestrador.perfil [--completa]` | perfil de desempenho (10 funções mais caras) |

O app lê, nesta ordem: `GESTORASIM_DADOS` (se definida e não vazia), `data/sim` (se o ciclo estiver gravado) e, por fim,
`data/demo` — um dataset reduzido (2023-01-02 a 2023-08-31, seed 42, ~5 MB; inclui um episódio passivo de enquadramento) versionado no git, gerado com
`python -m gestorasim.orquestrador --de 2023-01-02 --ate 2023-08-31 --destino data/demo`.

### Deploy no Streamlit Community Cloud

Arquivo principal `app/Comece_aqui.py`; `requirements.txt` (`.[app]`) instala o pacote e o Streamlit; o tema e `headless`
ficam em `.streamlit/config.toml`. Sem `data/sim` no repositório, o app usa `data/demo`. Como o disco do Cloud é efêmero
e pode ser somente leitura, a boletagem manual e o workflow de conciliação gravam em pasta temporária quando a pasta de
dados não é gravável (aviso discreto na tela; `GESTORASIM_ESCRITA=<pasta>` força outro destino).

## Fluxo do ciclo diário (D0)

O orquestrador executa, para cada pregão, 13 etapas em ordem. Cada uma grava status (OK/ERRO/PULADA), duração e hash
de entrada em `execucoes_ciclo.parquet`; se uma etapa falha, as seguintes ficam PULADAS.

| # | Etapa | O que acontece |
|---|---|---|
| 1 | abertura | juros do caixa (CDI se positivo; CDI + 2% a.a. se negativo) sobre o saldo projetado do fechamento anterior |
| 2 | mercado | preços oficiais do dia (mtm); ajuste diário dos futuros de DI (liquida em D+1) |
| 3 | eventos | dividendos (direito na data-com), cupons e amortizações (direito no pregão anterior; PU ex-cupom) |
| 4 | passivo | pagamentos já devidos: resgates convertidos antes e taxa de administração do mês anterior |
| 5 | caixa e reenquadramento | gera ordens de gestão de caixa (buffer 3σ do DI + liquidações D..D+2), reinvestimento do excesso e reenquadramento de violações |
| 6 | boletagem | pré-trade (regulamento, caixa mínimo, liquidez), execução simulada (slippage, fills parciais) e rateio; ordens regulares a cada 5 pregões |
| 7 | liquidação | caixa liquidado × a liquidar (ações D+2, títulos D+0, ajuste de DI D+1) |
| 8 | mtm | posições, valor de mercado, fonte de preço e atribuição de P&L (preço, carrego, eventos, DI, custos, juros) |
| 9 | provisões e cota | conversão de aplicações/resgates, taxas (adm e performance), come-cotas, IR, cota bruta e líquida |
| 10 | conciliação | arquivo XML do administrador (com divergências injetadas), quebras com diagnóstico, cota paralela |
| 11 | pós-trade | regras do regulamento e da política de risco na carteira de fechamento; episódios ATIVO/PASSIVO/VENCIDO |
| 12 | risco | VaR/ES, sensibilidades, stress, liquidez e backtest do dia (em % do PL líquido) |
| 13 | relatórios | linha do relatório diário por fundo |

**Reprocessamento idempotente.** `--data X --reprocessar` restaura o estado do checkpoint mais recente anterior a X
(um a cada 63 pregões), reexecuta os dias intermediários sem tocar nas tabelas gravadas e recalcula X e as datas
seguintes. Tudo é determinístico por seed (ordens, injeções de divergência, conciliação), então as datas anteriores
ficam intactas e as seguintes são reproduzidas — com os mesmos hashes de entrada no log.

## Mapa: módulo do GestoraSim → função numa plataforma de gestão real

| Módulo | Área | Função equivalente numa plataforma real |
|---|---|---|
| `dominio` | todas | modelo de dados do cadastro: ativos, fundos (CVM 175), regulamento, cotistas, ordens, eventos |
| `dados` | — | *market data* e cadastros (B3, ANBIMA, Tesouro); aqui, simulados (GBM correlacionado, Nelson-Siegel, IPCA) |
| `boletagem` (ordens, trilha, rateio) | **front** | OMS: ordem agregada por vários fundos, ciclo de vida, auditoria, alocação justa a preço médio |
| `boletagem.execucao` | **front** | EMS/corretora: fills parciais e impacto de mercado (sem protocolo FIX) |
| `boletagem.gestao_caixa` | **front** | mesa de gestão: caixa mínimo, reinvestimento e ordens de reenquadramento |
| `enquadramento.pre_trade` | **middle** | compliance pré-trade: simula a carteira pós-ordem e bloqueia/libera com override justificado |
| `enquadramento.pos_trade` | **middle** | compliance pós-trade: regras do regulamento e da política, alertas SOFT/HARD, prazo de reenquadramento |
| `posicao` | **middle** | PMS / contabilidade de carteira: ledger, posição, caixa disponível × projetado, eventos |
| `mtm` | **middle** | marcação a mercado (curvas ANBIMA, PU, VNA, ajuste de futuros) e atribuição de P&L |
| `risco` | **middle** | motor de risco: fatores, DV01/CS01, VaR/ES, stress, backtest (Kupiec, Basileia), liquidez |
| `cota` | **back** | administração fiduciária: passivo, cotização, taxas, come-cotas/IR, cálculo da cota |
| `conciliacao` | **back** | conciliação gestor × administrador/custodiante, tratamento de quebras e cota paralela |
| `relatorios` | **back** | lâmina mensal e relatórios de gestão |
| `orquestrador` | transversal | agendador do batch de fechamento (EOD), controle de execução, reprocessamento |
| `app` | transversal | portal do gestor/risco/compliance: boletagem manual, painéis, workflow de quebras, auditoria |

## App (Streamlit)

Páginas: **Visão geral** (PL líquido, cota × benchmark, alertas e quebras abertas, status do último ciclo),
**Boletagem** (ordem manual com pré-trade ao vivo, envio ou override com justificativa, execução e rateio; as ordens
ficam em um ledger **separado** com a tag MANUAL e não alteram a série oficial), **Carteira**, **Risco**,
**Enquadramento**, **Passivo e cota**, **Conciliação** (workflow ABERTA → JUSTIFICADA → RESOLVIDA com usuário e
comentário, mais cota paralela), **Trilha de auditoria** e **Lâmina mensal** (HTML para download). As páginas só leem
os parquets (com `st.cache_data`); o app nunca recalcula a série inteira.

## Dados gerados (`data/sim/`, ignorado pelo git)

- **Universo:** 20 ações (GBM com correlação sintética PSD, dividendos que reduzem o preço na data ex), 504 pregões a
  partir de 2023-01-02 (calendário B3 aproximado), curva pré Nelson-Siegel com fatores AR(1), CDI e IPCA sintéticos,
  4 LTN, 4 NTN-B, 5 debêntures (AAA a A-), 3 futuros de DI, 3 fundos (FIA, Multimercado, RF Crédito Privado), 200
  cotistas com fluxo de aplicação/resgate.
- **Ciclo:** `posicoes_diarias`, `mtm_diario`, `pnl_atribuicao`, `pl_diario`, `liquidacao_diaria`, `ledger`,
  `ordens_serie`, `alocacoes_serie`, `auditoria_serie`, `cotistas_mov`, `lotes_cotistas`, `provisoes`, `cota_diaria`,
  `tributos`, `risco_diario`, `sensibilidades`, `stress`, `backtest_var`, `excecoes_var`, `liquidez`,
  `enquadramento_diario`, `alertas`, `episodios`, `quebras_conciliacao`, `injecoes`, `cota_paralela`,
  `relatorio_diario`, `execucoes_ciclo`, `diagnostico_mm`; mais `adm_xml/` (um XML do administrador por fundo e dia).
- **App:** `workflow_quebras`, `ordens_manuais`, `execucoes_manuais`, `alocacoes_manuais`, `auditoria_manual`,
  `ledger_manual`.

Preço único: o `mtm` é a fonte oficial de preços de todos os ativos (ordens, pré-trade, P&L, risco e app leem o mesmo
painel); o bloco 1 não grava mais `precos_renda_fixa`.

## Convenções

- Números em relatórios, prints e no app com **3 casas decimais**; nomes em português.
- PU de renda fixa **ex-cupom** na data de pagamento; os fluxos de `eventos` completam o retorno total.
- Futuro de DI: ajuste diário `(PU_d − PU_{d−1}·(1+CDI_{d−1})^(1/252)) × qtd`, liquidado em D+1; "comprado" = comprado em PU.
- Risco em % do **PL líquido** (`cota_diaria`); regras do regulamento em % do PL dos ativos.
- Backtest do VaR sobre P&L hipotético **limpo** (só fatores de mercado), com o sujo ao lado; burn-in de 60 pregões.

## Desempenho

Série completa de 504 pregões (posição, MTM, passivo e ordens): **~480 s → 43 s** (≈11×). Ciclo completo de 13 etapas
(com conciliação, pós-trade e risco): **~90 s**. Medidas na mesma máquina, seed 42.

| Janela | Antes | Depois |
|---|---|---|
| 48 pregões, sob cProfile | 64,1 s | 7,6 s |
| 124 pregões | 87,4 s | 10,3 s |
| 504 pregões (série) | ~480 s | 43,3 s |

O que mudou: carteira e regras sem pandas no caminho quente (o DataFrame `posicoes` é montado sob demanda); cache dos
resultados "antes da ordem" por estado do ledger; liquidações indexadas por data; dicionários no lugar de `.loc` em
colunas de string Arrow; registros em dicts no engine do passivo; painel de preços em cache (memória e parquet).
`python -m gestorasim.orquestrador.perfil` lista as 10 funções mais caras do pacote.

## Limitações conhecidas

- **Sem FIX, EMS real ou corretoras:** a execução é um modelo (participação máxima de 20% do volume, slippage linear).
- **Sem margem B3, garantias, depósitos ou chamadas de margem:** futuros de DI só têm ajuste diário; o caixa pode
  ficar negativo (paga CDI + 2%) sem financiamento.
- **Tributação simplificada:** come-cotas a 15% para todos os lotes (RF e MM), IR regressivo/15% por lote FIFO, **IOF
  desligado** (a tabela existe e se liga por parâmetro); sem compensação de prejuízo entre fundos, sem IR sobre ganho de
  capital em ações.
- **Preços e curvas sintéticos:** nenhuma série de mercado real; debêntures marcadas por curva + spread por rating
  (passeio aleatório), sem liquidez de mercado secundário; sem opções nem derivativos além do DI.
- **Calendário B3 aproximado** (feriados nacionais, Carnaval, Sexta-feira Santa, Corpus Christi).
- **Taxa de performance pelo método do ativo:** aplicações novas entram à cota líquida e herdam a provisão dos antigos
  (efeito conhecido do método).
- **XML do administrador simplificado:** inspirado no layout de posição de carteira da ANBIMA, mas não é o schema
  oficial; cota, PL e quantidade de cotas do cabeçalho são campos independentes.
- **Conciliação:** as divergências são injetadas com seed; a taxa de detecção de 100% vale para esse conjunto rotulado.
- **Boletagem manual no app:** parte do fechamento oficial da data; a liquidação pendente não é reconstruída (o caixa
  disponível parte do projetado).
- **Cotistas:** o fluxo de aplicações/resgates vem do bloco 1 (valores aleatórios); o capital inicial dos fundos é um lote
  do cotista `CAPITAL` que nunca resgata.
- **Segurança e multiusuário:** o app não tem autenticação; os usuários do workflow e da trilha são texto livre.

## Estrutura

```
gestorasim/  dominio dados boletagem enquadramento posicao mtm risco cota conciliacao relatorios orquestrador
app/         Home.py, pages/ (8 páginas), paginas/ (lógica de cada página), dados.py, formatacao.py
tests/       pytest (unitários, de integração e de fumaça do app)
```


## Interface do app (revisão de UX)

- **Comece aqui** (primeira do menu): roteiro de 7 passos (boletagem com violação e override → trilha → risco D+1 →
  enquadramento ativo × passivo → conciliação → passivo/come-cotas → lâmina), com botões que ajustam fundo e data para o dia
  em que cada fenômeno ocorre no dataset carregado.
- Barra lateral com **fundo e data** persistentes entre páginas (padrão: último pregão); cada página abre com "numa gestora
  real", KPIs com variação e o expander "Como ler esta página" (glossário).
- Toda a formatação numérica passa por `app/formatacao.py` (pt-BR: `R$ 12,345 mi`, `1,234%`, `12,345 bp`, `12.300`, cota com
  8 casas, `dd/mm/aaaa`); as tabelas ordenam pelo número e trazem a unidade no cabeçalho. Cores: verde OK, âmbar SOFT, vermelho
  HARD/violação/quebra aberta, cinza neutro.
- Capturas das telas em `docs/screens/` (`python scripts/capturar_telas.py`; requer `pip install playwright && playwright install chromium`).
