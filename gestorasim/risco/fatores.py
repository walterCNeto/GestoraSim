"""Fatores de risco: retorno das ações, vértices das curvas pré e real, spread de crédito por rating."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gestorasim.dados.mercado import TICKERS_ACOES
from gestorasim.mtm.curvas import RATINGS_DEBENTURES, VERTICES_RISCO
from gestorasim.risco.estatistica import ewma_cov


@dataclass
class Fatores:
    nomes: list[str]
    d_f: np.ndarray  # (T, n): variações diárias (ações: retorno total; curvas e spreads: bp)
    cov: np.ndarray  # (T, n, n): covariância EWMA com informação até t (NaN no começo)

    @property
    def idx(self) -> dict[str, int]:
        return {n: j for j, n in enumerate(self.nomes)}


def nomes_fatores() -> list[str]:
    return ([f"ACAO:{t}" for t in TICKERS_ACOES] + [f"PRE:{v}" for v in VERTICES_RISCO]
            + [f"REAL:{v}" for v in VERTICES_RISCO] + [f"SPREAD:{r}" for r in RATINGS_DEBENTURES])


def construir_fatores(tabelas: dict[str, pd.DataFrame], prec, painel: pd.DataFrame,
                      lam: float = 0.94, n_init: int = 20) -> Fatores:
    datas = prec.datas
    acoes = list(TICKERS_ACOES)
    P = painel[acoes].to_numpy()
    div = np.zeros_like(P)  # dividendo por ação que sai do preço na data ex
    for _, e in tabelas["eventos"].query("tipo == 'DIVIDENDO'").iterrows():
        div[prec.i_de[pd.Timestamp(e["data_com"])] + 1, acoes.index(e["ticker"])] += e["valor_por_unidade"]
    ret = np.zeros_like(P)
    ret[1:] = (P[1:] + div[1:]) / P[:-1] - 1  # retorno total: o dividendo não é "perda" de risco
    pre = (tabelas["curva_pre"].pivot(index="data", columns="prazo_du", values="taxa")
           [list(VERTICES_RISCO)].loc[datas].to_numpy())
    real = np.array([[est.taxa_real(v) for v in VERTICES_RISCO] for est in prec.estados])
    spr = np.array([[est.z_credito[r] for r in RATINGS_DEBENTURES] for est in prec.estados])

    def bp(x):
        d = np.zeros_like(x)
        d[1:] = np.diff(x, axis=0) * 1e4
        return d

    d_f = np.hstack([ret, bp(pre), bp(real), bp(spr)])
    return Fatores(nomes_fatores(), d_f, ewma_cov(d_f, lam, n_init))
