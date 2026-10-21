"""Benchmarks da taxa de performance: CDI (RF, MM) e índice sintético das 20 ações (FIA)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gestorasim.dados.mercado import TICKERS_ACOES
from gestorasim.dominio import ClasseFundo


def indice_cdi(tabelas: dict[str, pd.DataFrame]) -> np.ndarray:
    """Fator de CDI acumulado desde a primeira data (1,0 no início)."""
    return tabelas["cdi"]["fator_acumulado"].to_numpy(dtype=float)


def indice_acoes(tabelas: dict[str, pd.DataFrame], painel: pd.DataFrame, seed: int) -> np.ndarray:
    """Índice de retorno total ponderado por valor de mercado (quantidade de ações sintética × preço do
    pregão anterior), base 1,0. O dividendo entra no retorno da data ex."""
    datas = pd.DatetimeIndex(tabelas["calendario"]["data"])
    P = painel[list(TICKERS_ACOES)].to_numpy()
    pos = {d: i for i, d in enumerate(datas)}
    div = np.zeros_like(P)
    for _, e in tabelas["eventos"].query("tipo == 'DIVIDENDO'").iterrows():
        div[pos[pd.Timestamp(e["data_com"])] + 1, TICKERS_ACOES.index(e["ticker"])] += e["valor_por_unidade"]
    acoes_em_circulacao = np.random.default_rng([seed, 500]).lognormal(np.log(5e8), 0.8, P.shape[1])
    idx = np.ones(len(datas))
    for t in range(1, len(datas)):
        w = acoes_em_circulacao * P[t - 1]
        w = w / w.sum()
        idx[t] = idx[t - 1] * (1 + float(w @ ((P[t] + div[t]) / P[t - 1] - 1)))
    return idx


def benchmark_do_fundo(classe: ClasseFundo, nome_bench: str, cdi: np.ndarray, acoes: np.ndarray) -> np.ndarray:
    return acoes if nome_bench == "ACOES" else cdi
