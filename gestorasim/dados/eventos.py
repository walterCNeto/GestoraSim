"""Eventos corporativos/de crédito no calendário: dividendos (ações); cupons e amortizações vêm de renda_fixa."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gestorasim.dados import calendario
from gestorasim.dados.mercado import TICKERS_ACOES

DEFASAGEM_PAGAMENTO_DU = 10


def gerar_dividendos(rng: np.random.Generator, datas: pd.DatetimeIndex,
                     precos: pd.DataFrame) -> tuple[list[dict], pd.DataFrame]:
    """Dois dividendos por ação por ano (yield anual 3%-7% sobre o preço na data com).

    Devolve (eventos, preços ajustados). Na data ex (pregão seguinte à data com) o preço cai pelo
    dividendo por ação: P'_ex = P'_{ex-1} × (P_ex / P_{ex-1}) − D, e os retornos seguintes do caminho
    GBM ficam intactos (fator multiplicativo F ← F − D/P_ex a partir do ex). Com o dividendo
    reinvestido na data ex, o retorno total é o do GBM original.
    """
    T = len(datas)
    bruto = precos.pivot(index="data", columns="ticker", values="preco")
    ajustado = bruto.copy()
    eventos = []
    for tk in TICKERS_ACOES:
        dy = rng.uniform(0.03, 0.07)
        sorteios = []
        for ini in range(0, T, 252):
            fim = min(ini + 252, T)
            meio = ini + (fim - ini) // 2
            for lo, hi in ((ini, meio), (meio, fim)):
                if hi - lo >= 2:
                    sorteios.append(int(rng.integers(lo, min(hi, T - 1))))  # o ex (i+1) precisa existir
        fator = 1.0
        for i in sorted(sorteios):
            valor = dy / 2 * float(ajustado[tk].iloc[i])  # sobre o preço já ajustado
            fator -= valor / float(bruto[tk].iloc[i + 1])
            ajustado.iloc[i + 1:, ajustado.columns.get_loc(tk)] = bruto[tk].iloc[i + 1:] * fator
            eventos.append({
                "tipo": "DIVIDENDO", "ticker": tk, "data_com": datas[i],
                "data_pagamento": calendario.somar_du(datas[i], DEFASAGEM_PAGAMENTO_DU),
                "valor_por_unidade": valor,
            })
    longo = ajustado.rename_axis("data").reset_index().melt(
        id_vars="data", var_name="ticker", value_name="preco").sort_values(["ticker", "data"], ignore_index=True)
    return eventos, longo
