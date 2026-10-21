"""Roda um dia de exemplo: carteira inicial em D0, ordens no dia seguinte, saídas em data/sim/."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from gestorasim.boletagem.execucao import gerar_volume
from gestorasim.boletagem.fluxo import Mesa
from gestorasim.boletagem.geradores import gerar_ordens_exemplo
from gestorasim.dados import carregar_fundos, gerar_universo
from gestorasim.dados.universo import DESTINO_PADRAO
from gestorasim.enquadramento import montar_carteira
from gestorasim.mtm.cache import painel_oficial
from gestorasim.posicao import Ledger, gerar_carteira_inicial
from gestorasim.util import fmt3

TABELAS_SAIDA = ("ordens", "execucoes", "alocacoes", "auditoria", "pretrade", "ledger", "volume_diario")


def precos_do_dia(tabelas: dict[str, pd.DataFrame], data, seed: int = 42) -> pd.Series:
    """Preço oficial do dia (mtm), indexado por ticker; ativos sem preço (vencidos) ficam fora."""
    return painel_oficial(tabelas, seed).loc[pd.Timestamp(data)].dropna()


def rodar_dia_exemplo(seed: int = 42, destino: Path | str | None = None,
                      indice_dia: int = 1) -> dict[str, pd.DataFrame]:
    destino = Path(destino) if destino is not None else DESTINO_PADRAO
    tabelas = gerar_universo(seed=seed, destino=destino)  # determinístico: regrava o mesmo universo
    fundos = carregar_fundos(destino)
    meta = tabelas["ativos"].set_index("ticker")
    datas = pd.DatetimeIndex(tabelas["calendario"]["data"])
    d0, dia = datas[0], datas[indice_dia]

    ledger = Ledger()
    gerar_carteira_inicial(ledger, fundos, meta, precos_do_dia(tabelas, d0, seed), d0, seed)

    precos = precos_do_dia(tabelas, dia, seed)
    volume_df = gerar_volume(meta, datas, seed)
    volume = volume_df.query("data == @dia").set_index("ticker")["volume"]

    ordens = gerar_ordens_exemplo(ledger, fundos, meta, precos, dia, seed)
    mesa = Mesa(ledger, fundos, meta, dia, precos, volume)
    pl_antes = {f.cnpj: montar_carteira(ledger.quantidades(f.cnpj), ledger.caixa_projetado(f.cnpj), meta, precos).pl
                for f in fundos}
    for o in ordens:
        mesa.processar(o)
    saidas = mesa.tabelas()
    saidas["volume_diario"] = volume_df
    saidas["pl_fundos"] = pd.DataFrame([{
        "fundo": f.nome, "pl_antes": pl_antes[f.cnpj],
        "pl_depois": montar_carteira(ledger.quantidades(f.cnpj), ledger.caixa_projetado(f.cnpj), meta, precos).pl,
        "caixa_depois": ledger.caixa_projetado(f.cnpj)} for f in fundos])
    for nome in TABELAS_SAIDA:
        saidas[nome].to_parquet(destino / f"{nome}.parquet", index=False)
    return saidas


def resumo(s: dict[str, pd.DataFrame]) -> str:
    o, ex, al, au = s["ordens"], s["execucoes"], s["alocacoes"], s["auditoria"]
    linhas = [f"ordens: {len(o)}  execuções (fills): {len(ex)}  alocações: {len(al)}  eventos de auditoria: {len(au)}",
              "estados finais:"]
    linhas += [f"  {e:<18} {n}" for e, n in o["estado"].value_counts().items()]
    parcial = o[(o["quantidade_executada"] > 0) & (o["quantidade_executada"] < o["quantidade_alvo"])]
    linhas.append(f"execuções parciais: {len(parcial)}")
    if len(ex):
        sinal = ex["lado"].map({"C": 1, "V": -1})
        slip = ((ex["preco"] / ex["fechamento"] - 1) * sinal * 1e4)
        linhas.append(f"slippage médio (bps): {fmt3(float(slip.mean()))}   participação máxima: "
                      f"{fmt3(float(ex.groupby('ordem_id')['quantidade'].sum().div(ex.groupby('ordem_id')['volume_dia'].first()).max()))}")
        linhas.append(f"financeiro executado (R$): {fmt3(float((al['financeiro']).sum()))}")
    bloq = o[o["estado"] == "BLOQUEADA"]
    for _, r in bloq.iterrows():
        motivo = au[(au.ordem_id == r.ordem_id) & (au.para == "BLOQUEADA")]["motivo"].iloc[0]
        linhas.append(f"  BLOQUEADA {r.ordem_id} [{r.origem}] {r.lado} {r.ticker}: {motivo[:110]}")
    ov = au[au.para == "LIBERADA_OVERRIDE"]
    for _, r in ov.iterrows():
        linhas.append(f"  OVERRIDE {r.ordem_id} por {r.usuario}: {r.motivo}")
    linhas.append("PL dos fundos (R$):")
    for _, r in s["pl_fundos"].iterrows():
        linhas.append(f"  {r.fundo:<24} antes={fmt3(r.pl_antes)} depois={fmt3(r.pl_depois)} caixa={fmt3(r.caixa_depois)}")
    return "\n".join(linhas)
