"""Série diária: ledger rodando sobre os pregões, MTM, eventos, ordens periódicas e atribuição de P&L."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from gestorasim.boletagem import Mesa
from gestorasim.boletagem.execucao import gerar_volume
from gestorasim.boletagem.geradores import gerar_ordens_exemplo
from gestorasim.dados import calendario, carregar_fundos, gerar_universo
from gestorasim.dados.universo import DESTINO_PADRAO
from gestorasim.enquadramento import montar_carteira
from gestorasim.mtm.precificacao import Precificador, ajuste_di
from gestorasim.posicao import Ledger, ProcessadorEventos, gerar_carteira_inicial
from gestorasim.util import fmt3

TOLERANCIA_FECHAMENTO = 1e-6
COMPONENTES = ["efeito_preco", "carrego", "eventos", "ajuste_futuros", "custos"]
TABELAS_SAIDA = ("posicoes_diarias", "mtm_diario", "pnl_atribuicao", "pl_diario")


def rodar_serie(seed: int = 42, destino: Path | str | None = None, inicio=None, fim=None,
                passo_ordens: int = 5, com_ordens: bool = True, escrever: bool = True) -> dict[str, pd.DataFrame]:
    """Roda todos os fundos de `inicio` a `fim` (default: o calendário inteiro).

    Ordens do gerador do bloco 2 (rebalanceamento + aleatórias + violadoras sem override) a cada
    `passo_ordens` pregões. Preço de execução = MTM do dia ± slippage; cotistas sem movimentação.
    """
    destino = Path(destino) if destino is not None else DESTINO_PADRAO
    tabelas = gerar_universo(seed=seed, destino=destino)
    fundos = carregar_fundos(destino)
    nomes = {f.cnpj: f.nome for f in fundos}
    prec = Precificador(tabelas, seed)
    meta, datas, estados = prec.meta, prec.datas, prec.estados
    painel = prec.painel()
    fontes = {tk: prec.fonte(tk) for tk in meta.index}
    volume = gerar_volume(meta, datas, seed).pivot(index="data", columns="ticker", values="volume")
    i0 = 0 if inicio is None else int(datas.searchsorted(pd.Timestamp(inicio)))
    i1 = len(datas) - 1 if fim is None else int(datas.searchsorted(pd.Timestamp(fim), side="right")) - 1

    ledger = Ledger()
    gerar_carteira_inicial(ledger, fundos, meta, painel.iloc[i0].dropna(), datas[i0], seed)
    proc = ProcessadorEventos(ledger, tabelas["eventos"], prec.fracoes)
    cnpjs = [f.cnpj for f in fundos]

    pos_rows, mtm_rows, pl_rows, pnl_rows = [], [], [], []
    ordens_estado: Counter = Counter()
    q_prev: dict[str, dict[str, int]] = {}
    pl_prev: dict[str, float] = {}

    for i in range(i0, i1 + 1):
        data, P = datas[i], painel.iloc[i]
        Pv = P.dropna()
        pnl = defaultdict(lambda: dict.fromkeys(COMPONENTES, 0.0))
        if i > i0:
            n0 = len(ledger)
            proc.processar(data, cnpjs)
            Pant = painel.iloc[i - 1]
            for c in cnpjs:  # ajuste diário dos futuros de DI, liquidado em D+1
                for tk, q in q_prev[c].items():
                    if meta.loc[tk, "classe"] == "FUTURO_DI":
                        aj = ajuste_di(P[tk], Pant[tk], estados[i - 1].cdi_aa, q)
                        ledger.creditar(data, c, tk, aj, "AJUSTE_DI_DIA", calendario.somar_du(data, 1))
            if com_ordens and (i - i0) % passo_ordens == 0:
                mesa = Mesa(ledger, fundos, meta, data, Pv, volume.iloc[i].dropna())
                for o in gerar_ordens_exemplo(ledger, fundos, meta, Pv, data, seed, com_override=False, dia=i):
                    ordens_estado[mesa.processar(o).estado.value] += 1
            # ---- atribuição do dia
            for c in cnpjs:
                for tk, q in q_prev[c].items():
                    if meta.loc[tk, "classe"] == "FUTURO_DI":
                        continue
                    pc = Pant[tk] if meta.loc[tk, "classe"] == "ACAO" else prec.preco_carrego(tk, i)
                    pnl[(c, tk)]["carrego"] += q * (pc - Pant[tk])
                    pnl[(c, tk)]["efeito_preco"] += q * (P[tk] - pc)
            for _, _, c, tipo, tk, delta, preco_l, origem, _ in ledger.linhas_desde(n0):
                if tipo == "CAIXA" and origem.startswith("EVT:"):
                    pnl[(c, tk)]["eventos"] += delta
                elif tipo == "CAIXA" and origem == "AJUSTE_DI_DIA":
                    pnl[(c, tk)]["ajuste_futuros"] += delta
                elif tipo == "ATIVO":  # execução vs MTM do dia = custo de slippage (inclui entrada de DI)
                    pnl[(c, tk)]["custos"] += delta * (P[tk] - preco_l)
        # ---- fechamento do dia
        for f in fundos:
            c = f.cnpj
            q = ledger.quantidades(c)
            cart = montar_carteira(q, ledger.caixa_projetado(c), meta, Pv)
            for tk, qt in q.items():
                pos_rows.append((data, c, tk, qt, ledger.fator_vne(c, tk)))
                mtm_rows.append((data, c, tk, qt, float(P[tk]), float(cart.posicoes.loc[tk, "valor"]),
                                 float(cart.posicoes.loc[tk, "exposicao"]), fontes[tk]))
            soma = sum(sum(v.values()) for (cc, _), v in pnl.items() if cc == c)
            var_pl = cart.pl - pl_prev[c] if i > i0 else 0.0
            pl_rows.append((data, c, nomes[c], ledger.caixa_disponivel(c, data), ledger.caixa_projetado(c),
                            float(cart.posicoes["valor"].sum()), cart.pl, var_pl, soma, soma - var_pl))
            pl_prev[c] = cart.pl
        for (c, tk), v in pnl.items():
            pnl_rows.append((data, c, tk, *[v[k] for k in COMPONENTES], sum(v.values())))
        q_prev = {f.cnpj: ledger.quantidades(f.cnpj) for f in fundos}

    saidas = {
        "posicoes_diarias": pd.DataFrame(pos_rows, columns=["data", "fundo_cnpj", "ticker", "quantidade", "fator_vne"]),
        "mtm_diario": pd.DataFrame(mtm_rows, columns=["data", "fundo_cnpj", "ticker", "qtd", "preco",
                                                      "valor_mercado", "exposicao", "fonte_preco"]),
        "pnl_atribuicao": pd.DataFrame(pnl_rows, columns=["data", "fundo_cnpj", "ticker", *COMPONENTES, "total"]),
        "pl_diario": pd.DataFrame(pl_rows, columns=[
            "data", "fundo_cnpj", "fundo", "caixa_disponivel", "caixa_projetado", "valor_mercado", "pl",
            "variacao_pl", "pnl_total", "erro_fechamento"]),
        "ledger": ledger.para_df(),
        "ordens_por_estado": pd.DataFrame(sorted(ordens_estado.items()), columns=["estado", "n"]),
    }
    if escrever:
        for nome in TABELAS_SAIDA:
            saidas[nome].to_parquet(destino / f"{nome}.parquet", index=False)
    return saidas


def serie_pl(fundo: str, inicio=None, fim=None, seed: int = 42, destino: Path | str | None = None,
             resultado: dict[str, pd.DataFrame] | None = None) -> pd.Series:
    """PL diário de um fundo (nome ou CNPJ) de `inicio` a `fim`.

    Ordens agregadas cobrem vários fundos, então a série roda todos juntos; passe `resultado`
    (retorno de `rodar_serie`) para reaproveitar uma simulação já feita.
    """
    res = resultado if resultado is not None else rodar_serie(seed, destino, inicio, fim, escrever=False)
    pl = res["pl_diario"]
    sel = pl[(pl["fundo"] == fundo) | (pl["fundo_cnpj"] == fundo)]
    if sel.empty:
        raise KeyError(f"fundo não encontrado: {fundo}")
    s = sel.set_index("data")["pl"]
    if inicio is not None:
        s = s[s.index >= pd.Timestamp(inicio)]
    if fim is not None:
        s = s[s.index <= pd.Timestamp(fim)]
    return s.rename(sel["fundo"].iloc[0])


def resumo(res: dict[str, pd.DataFrame]) -> str:
    pl = res["pl_diario"]
    linhas = []
    for nome, g in pl.groupby("fundo", sort=False):
        s = g.set_index("data")["pl"]
        r = s.pct_change().dropna()
        pior = r.idxmin()
        linhas.append(f"{nome}: PL inicial={fmt3(s.iloc[0])} final={fmt3(s.iloc[-1])} "
                      f"retorno={fmt3(s.iloc[-1] / s.iloc[0] - 1)} pior dia={pior:%Y-%m-%d} ({fmt3(r.min())})")
    linhas.append(f"maior erro de fechamento do P&L (R$): {fmt3(float(pl['erro_fechamento'].abs().max()))}")
    oe = res["ordens_por_estado"]
    if len(oe):
        linhas.append("ordens: " + ", ".join(f"{e}={n}" for e, n in zip(oe["estado"], oe["n"])))
    return "\n".join(linhas)
