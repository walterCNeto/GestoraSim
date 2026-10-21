"""Estatísticas de apoio: volatilidade EWMA."""

from __future__ import annotations

import numpy as np

LAMBDA_EWMA = 0.94


def ewma_vol(retornos: np.ndarray, lam: float = LAMBDA_EWMA, n_init: int = 20, padrao: float = 0.001) -> np.ndarray:
    """Desvio-padrão EWMA com a informação até cada t (inclusive). `retornos[0]` é ignorado (NaN).
    Até `n_init` retornos usa o desvio amostral disponível (>= 2 obs) ou `padrao`."""
    r = np.asarray(retornos, dtype=float)
    out = np.full(len(r), padrao)
    var = None
    for t in range(1, len(r)):
        if t < n_init:
            if t >= 2:
                out[t] = float(np.sqrt(np.mean(r[1:t + 1] ** 2)))
            continue
        if var is None:
            var = float(np.mean(r[1:t + 1] ** 2))
        else:
            var = lam * var + (1 - lam) * r[t] ** 2
        out[t] = np.sqrt(var)
    return out
