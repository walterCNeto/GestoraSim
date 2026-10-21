"""Vendas que reduzem VaR e stress até a folga da política (usadas nas ordens de reenquadramento)."""

from __future__ import annotations

import math

import numpy as np

from gestorasim.enquadramento.pos_trade import POLITICA_RISCO
from gestorasim.risco.estatistica import Z99
from gestorasim.risco.motor import metricas_politica

FOLGA = 0.05


def contribuicoes_var(calc, fatores, q: dict[str, int], i: int) -> dict[str, float]:
    """VaR paramétrico por componente (Euler): soma das contribuições = VaR da carteira."""
    cov = fatores.cov[i]
    delta, _ = calc.carteira(q, i)
    sigma2 = float(delta @ cov @ delta)
    if np.isnan(cov).any() or sigma2 <= 0:
        return {}
    sigma = math.sqrt(sigma2)
    out = {}
    for tk, qt in q.items():
        d, _ = calc.sens_unitaria(tk, i)
        out[tk] = Z99 * float((qt * d) @ cov @ delta) / sigma
    return out


def contribuicoes_stress(calc, q: dict[str, int], i: int) -> tuple[str, dict[str, float]]:
    """Cenário mais severo da carteira e a perda (positiva) de cada posição nele."""
    total = calc.stress_carteira(q, i)
    pior = min(total, key=total.get)
    return pior, {tk: -qt * calc.stress_unitario(tk, i)[pior] for tk, qt in q.items()}


class ReducaoPolitica:
    def __init__(self, calc, fatores, politica=POLITICA_RISCO, folga: float = FOLGA) -> None:
        self.calc, self.fatores, self.politica, self.folga = calc, fatores, politica, folga

    def vendas(self, cnpj: str, classe_fundo, q: dict[str, int], i: int, pl: float) -> dict[str, int]:
        """Quantidades a vender para levar VaR e stress a (1 − folga) do limite, a partir das maiores contribuições."""
        lim = self.politica[classe_fundo]
        m = metricas_politica(self.calc, self.fatores, q, i, pl)
        vender: dict[str, int] = {}
        pedidos = []
        if m["var_pct"] > lim["var"]:
            pedidos.append((contribuicoes_var(self.calc, self.fatores, q, i), 1 - (1 - self.folga) * lim["var"] / m["var_pct"]))
        if m["stress_pct"] > lim["stress"]:
            pedidos.append((contribuicoes_stress(self.calc, q, i)[1],
                            1 - (1 - self.folga) * lim["stress"] / m["stress_pct"]))
        for comp, frac in pedidos:
            falta = frac * sum(v for v in comp.values() if v > 0)
            for tk, v in sorted(comp.items(), key=lambda kv: -kv[1]):
                if falta <= 0 or v <= 0 or q[tk] <= 0:
                    continue
                f = min(1.0, falta / v)
                vender[tk] = max(vender.get(tk, 0), int(math.ceil(f * q[tk])))
                falta -= f * v
        return vender
