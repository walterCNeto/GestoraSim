"""Eventos corporativos/de crédito no calendário: dividendos (ações); cupons e amortizações vêm de renda_fixa."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gestorasim.dados import calendario
from gestorasim.dados.mercado import TICKERS_ACOES

DEFASAGEM_PAGAMENTO_DU = 10


def gerar_dividendos(rng: np.random.Generator, datas: pd.DatetimeIndex, precos: pd.DataFrame) -> list[dict]:
    """Dois dividendos por ano por ação; yield anual 3%-7% sobre o preço na data com."""
    T = len(datas)
    pivot = precos.pivot(index="data", columns="ticker", values="preco")
    eventos = []
    for tk in TICKERS_ACOES:
        dy = rng.uniform(0.03, 0.07)
        for ini in range(0, T, 252):
            fim = min(ini + 252, T)
            for bloco in (slice(ini, ini + (fim - ini) // 2), slice(ini + (fim - ini) // 2, fim)):
                lo, hi = bloco.start, bloco.stop
                if hi - lo < 2:
                    continue
                i = int(rng.integers(lo, hi))
                pag = calendario.somar_du(datas[i], DEFASAGEM_PAGAMENTO_DU)
                eventos.append({
                    "tipo": "DIVIDENDO", "ticker": tk, "data_com": datas[i], "data_pagamento": pag,
                    "valor_por_unidade": dy / 2 * float(pivot.loc[datas[i], tk]),
                })
    return eventos
