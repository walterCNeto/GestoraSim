"""Liquidez: dias para zerar a 20% do volume diário e matriz por bucket."""

from __future__ import annotations

import numpy as np

from gestorasim.dominio import PARTICIPACAO_LIQUIDEZ

HORIZONTES_DU = (1, 5, 21, 63)  # buckets D+1, D+5, D+21, D+63; o resto é > D+63


def dias_para_zerar(quantidade: float, volume: float, participacao: float = PARTICIPACAO_LIQUIDEZ) -> float:
    """Dias úteis para zerar a posição negociando `participacao` do volume diário."""
    if volume <= 0:
        return float("inf")
    return abs(quantidade) / (participacao * volume)


def bucket(dias: float) -> str:
    for h in HORIZONTES_DU:
        if dias <= h:
            return f"D+{h}"
    return ">D+63"


def matriz_liquidez(posicoes: list[tuple[float, float]], caixa: float, pl: float) -> dict[str, float]:
    """`posicoes` = [(valor de mercado, dias para zerar)]. Devolve a fração do PL líquida até cada horizonte
    (acumulada, inclui o caixa projetado em D+1) e a fração restante além de D+63."""
    out = {}
    for h in HORIZONTES_DU:
        liquido = caixa + sum(v * min(1.0, h / d) if d > 0 else v for v, d in posicoes)
        out[f"liq_d{h}"] = liquido / pl if pl > 0 else float("nan")
    out["liq_gt63"] = 1.0 - out["liq_d63"] if np.isfinite(out["liq_d63"]) else float("nan")
    return out
