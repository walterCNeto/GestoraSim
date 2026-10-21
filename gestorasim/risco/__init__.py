"""Risco diário por fundo e enquadramento pós-trade sobre a série do bloco 3."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from gestorasim.boletagem.execucao import gerar_volume
from gestorasim.dados import carregar_fundos, gerar_universo
from gestorasim.dados.universo import DESTINO_PADRAO
from gestorasim.enquadramento.pos_trade import (
    POLITICA_RISCO, PRAZO_REENQUADRAMENTO_DU, rodar_pos_trade,
)
from gestorasim.mtm import Precificador, carregar_serie, painel_oficial, rodar_serie
from gestorasim.risco.fatores import construir_fatores
from gestorasim.risco.motor import BURN_IN_BACKTEST, JANELA_HISTORICA, calcular_risco
from gestorasim.risco.sensibilidades import CalculadoraSens
from gestorasim.util import fmt3

TABELAS_RISCO = ("risco_diario", "sensibilidades", "stress", "backtest_var", "excecoes_var", "liquidez",
                 "enquadramento_diario", "alertas")


def executar_risco(seed: int = 42, destino: Path | str | None = None,
                   serie: dict[str, pd.DataFrame] | None = None, janela: int = JANELA_HISTORICA,
                   politica: dict | None = None, prazo_reenquadramento_du: int = PRAZO_REENQUADRAMENTO_DU,
                   burn_in: int = BURN_IN_BACKTEST, escrever: bool = True) -> dict[str, pd.DataFrame]:
    """Calcula risco e pós-trade. Usa a série em disco se ela bate com o universo (senão a executa)."""
    destino = Path(destino) if destino is not None else DESTINO_PADRAO
    tabelas = gerar_universo(seed=seed, destino=destino)
    fundos = carregar_fundos(destino)
    if serie is None:
        serie = carregar_serie(seed, destino) or rodar_serie(seed, destino)
    prec = Precificador(tabelas, seed)
    painel = painel_oficial(tabelas, seed, destino)
    volume = gerar_volume(prec.meta, prec.datas, seed).pivot(index="data", columns="ticker", values="volume")
    fatores = construir_fatores(tabelas, prec, painel)
    calc = CalculadoraSens(prec, fatores, painel)
    res = calcular_risco(serie, tabelas, prec, painel, volume, calc, janela, burn_in)
    pt = rodar_pos_trade(serie, res["risco_diario"], prec, painel, volume, fundos, calc, politica,
                         prazo_reenquadramento_du)
    saidas = {**res, **pt}
    if escrever:
        for nome in (*TABELAS_RISCO, "episodios"):
            saidas[nome].to_parquet(destino / f"{nome}.parquet", index=False)
    return saidas


def resumo(res: dict[str, pd.DataFrame]) -> str:
    r, bt, st, ep = res["risco_diario"], res["backtest_var"], res["stress"], res["episodios"]
    linhas = []
    for c, g in r.groupby("fundo_cnpj", sort=False):
        nome = g["fundo"].iloc[0]
        f = lambda s: f"{fmt3(float(s.mean()))}/{fmt3(float(s.max()))}"  # noqa: E731
        linhas.append(f"{nome}")
        linhas.append(f"  VaR 99% paramétrico (% PL, médio/máx): {f(g['var_param_pct'].dropna())}")
        if g["var_hist_pct"].notna().any():
            linhas.append(f"  VaR 99% histórico   (% PL, médio/máx): {f(g['var_hist_pct'].dropna())}")
            linhas.append(f"  ES 97,5% histórico  (% PL, médio/máx): {f(g['es_hist_pct'].dropna())}")
        for _, b in bt[bt["fundo_cnpj"] == c].iterrows():
            extra = (f" [com burn-in: {b['excecoes_com_burn_in']}/{b['n_obs_com_burn_in']}]"
                     if b["excecoes_com_burn_in"] != b["excecoes"] else "")
            linhas.append(f"  backtest {b['metodo']:<11} {b['base']:<6}: {b['excecoes']}/{b['n_obs']} exceções "
                          f"(esperado {fmt3(b['esperado'])}), Kupiec p={fmt3(b['kupiec_p'])}, zona {b['zona_basileia']}{extra}")
        ex = res["excecoes_var"]
        ex = ex[(ex["fundo_cnpj"] == c) & (ex["metodo"] == "PARAMETRICO") & (ex["base"] == "LIMPO")] if len(ex) else ex
        if nome.endswith("FIA") and len(ex):
            antes = ex[ex["em_burn_in"]]["data"].dt.strftime("%Y-%m-%d").tolist()
            depois = ex[~ex["em_burn_in"]]["data"].dt.strftime("%Y-%m-%d").tolist()
            linhas.append(f"  exceções do FIA (paramétrico, limpo) dentro do burn-in: {len(antes)} {antes}; "
                          f"depois: {len(depois)} {depois}")
        s = st[(st["fundo_cnpj"] == c) & (st["cenario"] != "PIOR_DIA_HIST")]
        pior = s.loc[s["pnl_pct_pl"].idxmin()]
        linhas.append(f"  pior stress: {pior['cenario']} {fmt3(pior['pnl_pct_pl'])} do PL em {pior['data']:%Y-%m-%d}")
        e = ep[ep["fundo_cnpj"] == c] if len(ep) else ep
        n = lambda cond: int(cond.sum()) if len(e) else 0  # noqa: E731
        linhas.append(f"  desenquadramentos (episódios): ativos={n(e['classe'] == 'ATIVO') if len(e) else 0} "
                      f"passivos={n(e['classe'] == 'PASSIVO') if len(e) else 0} "
                      f"vencidos={n(e['vencido']) if len(e) else 0}")
    return "\n".join(linhas)
