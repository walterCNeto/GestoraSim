"""Execução simulada: volume diário sintético, slippage proporcional à participação e fills parciais."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gestorasim.dominio import ClasseAtivo
from gestorasim.enquadramento import lote

PARTICIPACAO_MAX = 0.20  # fração do volume diário executável
SLIPPAGE_K = 0.05  # slippage = K × participação (1% na participação máxima)
N_FATIAS = 4  # fills por ordem

VOLUME_BASE = {  # mediana diária (ações: papéis; demais: unidades/contratos)
    ClasseAtivo.ACAO: 3_000_000,
    ClasseAtivo.TITULO_PUBLICO: 60_000,
    ClasseAtivo.DEBENTURE: 20_000,
    ClasseAtivo.FUTURO_DI: 80_000,
}


def gerar_volume(meta: pd.DataFrame, datas: pd.DatetimeIndex, seed: int) -> pd.DataFrame:
    """Volume diário sintético (data, ticker, volume), lognormal por ativo e por dia."""
    rng = np.random.default_rng([seed, 100])
    base = np.array([VOLUME_BASE[ClasseAtivo(c)] for c in meta["classe"]], dtype=float)
    nivel = base * rng.lognormal(0.0, 0.4, len(meta))
    vol = nivel[None, :] * rng.lognormal(0.0, 0.3, (len(datas), len(meta)))
    df = pd.DataFrame(vol.round().astype(np.int64), index=datas, columns=list(meta.index))
    return df.rename_axis("data").reset_index().melt(
        id_vars="data", var_name="ticker", value_name="volume"
    ).sort_values(["data", "ticker"], ignore_index=True)


def simular_fills(quantidade: int, lado: str, fechamento: float, volume: int,
                  classe: str) -> list[tuple[int, float]]:
    """Fills (quantidade, preço). Teto de 20% do volume (múltiplo do lote, ao menos 1 lote);
    execução em até N_FATIAS fatias, cada uma com slippage pela participação acumulada (ponto médio)."""
    lt = lote(classe)
    teto = max(lt, int(PARTICIPACAO_MAX * volume) // lt * lt)
    total = min(quantidade, teto)
    lotes = total // lt
    n = min(N_FATIAS, lotes)
    base, resto = divmod(lotes, n)
    sinal = 1 if lado == "C" else -1
    fills, acumulado = [], 0
    for i in range(n):
        q = (base + (1 if i < resto else 0)) * lt
        part = (acumulado + q / 2) / volume
        fills.append((q, fechamento * (1 + sinal * SLIPPAGE_K * part)))
        acumulado += q
    return fills


def preco_medio(fills: list[tuple[int, float]]) -> float:
    return sum(q * p for q, p in fills) / sum(q for q, _ in fills)
