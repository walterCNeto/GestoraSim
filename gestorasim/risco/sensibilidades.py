"""Sensibilidades (delta, gama diagonal) por bump-and-reprice e stress com reavaliação completa."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from gestorasim.dominio import ClasseAtivo
from gestorasim.mtm.curvas import VERTICES_RISCO, peso_vertice

BUMP_BP = 1.0
CENARIOS: dict[str, dict[str, float]] = {
    "PRE_+200": {"dy": 0.02},
    "PRE_-200": {"dy": -0.02},
    "ACOES_-15": {"acoes": -0.15},
    "SPREAD_+100": {"dz": 0.01},
    "IPCA_IMPL_+100": {"dbe": 0.01},
    "COMBINADO": {"acoes": -0.15, "dy": 0.02, "dz": 0.01},
}


def bump_pre(est, k: int, bp: float):
    pesos = np.array([peso_vertice(k, v) for v in est.vert_du])
    return replace(est, taxas=est.taxas + bp * 1e-4 * pesos)


def bump_real(est, k: int, bp: float):
    b = list(est.bump_real or (0.0,) * len(VERTICES_RISCO))
    b[k] += bp * 1e-4
    return replace(est, bump_real=tuple(b))


def bump_spread(est, rating: str, bp: float):
    return replace(est, z_credito={**est.z_credito, rating: est.z_credito[rating] + bp * 1e-4})


def aplicar_cenario(est, c: dict[str, float]):
    return replace(
        est, taxas=est.taxas + c.get("dy", 0.0), inflacao_implicita=est.inflacao_implicita + c.get("dbe", 0.0),
        z_credito={r: z + c.get("dz", 0.0) for r, z in est.z_credito.items()})


def _sensivel(prec, ticker: str) -> tuple[str | None, str | None]:
    """(curva, rating) a que o ativo é sensível: curva em {"PRE", "REAL"}."""
    m = prec.meta.loc[ticker]
    classe = ClasseAtivo(m["classe"])
    if classe == ClasseAtivo.FUTURO_DI:
        return "PRE", None
    if classe == ClasseAtivo.TITULO_PUBLICO:
        return ("REAL" if ticker.startswith("NTNB") else "PRE"), None
    if classe == ClasseAtivo.DEBENTURE:
        return ("REAL" if m["indexador"] == "IPCA" else "PRE"), m["rating"]
    return None, None


class CalculadoraSens:
    """Sensibilidades por unidade de cada ativo/data (em cache) e agregação por carteira."""

    def __init__(self, prec, fatores, painel: pd.DataFrame) -> None:
        self.prec, self.fatores, self.painel = prec, fatores, painel
        self.idx = fatores.idx
        self.n = len(fatores.nomes)
        self._unit: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
        self._stress: dict[tuple[str, int], dict[str, float]] = {}

    def sens_unitaria(self, ticker: str, i: int) -> tuple[np.ndarray, np.ndarray]:
        """(delta, gama) por unidade: ações em R$ por 100% de retorno; RF/DI em R$ por bp (e R$ por bp²)."""
        chave = (ticker, i)
        if chave in self._unit:
            return self._unit[chave]
        delta, gama = np.zeros(self.n), np.zeros(self.n)
        curva, rating = _sensivel(self.prec, ticker)
        if self.prec.meta.loc[ticker, "classe"] == ClasseAtivo.ACAO.value:
            delta[self.idx[f"ACAO:{ticker}"]] = float(self.painel[ticker].iloc[i])
        else:
            data, est = self.prec.datas[i], self.prec.estados[i]

            def preco(e):
                return self.prec.preco_em_estado(ticker, data, e)

            p0 = preco(est)
            bumpar = bump_pre if curva == "PRE" else bump_real
            for k, v in enumerate(VERTICES_RISCO):
                j = self.idx[f"{curva}:{v}"]
                pu, pb = preco(bumpar(est, k, BUMP_BP)), preco(bumpar(est, k, -BUMP_BP))
                delta[j], gama[j] = (pu - pb) / (2 * BUMP_BP), (pu + pb - 2 * p0) / BUMP_BP ** 2
            if rating is not None:
                j = self.idx[f"SPREAD:{rating}"]
                pu, pb = preco(bump_spread(est, rating, BUMP_BP)), preco(bump_spread(est, rating, -BUMP_BP))
                delta[j], gama[j] = (pu - pb) / (2 * BUMP_BP), (pu + pb - 2 * p0) / BUMP_BP ** 2
        self._unit[chave] = (delta, gama)
        return delta, gama

    def carteira(self, q: dict[str, int], i: int) -> tuple[np.ndarray, np.ndarray]:
        delta, gama = np.zeros(self.n), np.zeros(self.n)
        for tk, qt in q.items():
            d, g = self.sens_unitaria(tk, i)
            delta += qt * d
            gama += qt * g
        return delta, gama

    def stress_unitario(self, ticker: str, i: int) -> dict[str, float]:
        """Variação de preço por unidade em cada cenário (reavaliação completa)."""
        chave = (ticker, i)
        if chave in self._stress:
            return self._stress[chave]
        data, est = self.prec.datas[i], self.prec.estados[i]
        classe = self.prec.meta.loc[ticker, "classe"]
        base = float(self.painel[ticker].iloc[i])
        out = {}
        for nome, c in CENARIOS.items():
            if classe == ClasseAtivo.ACAO.value:
                out[nome] = base * c.get("acoes", 0.0)
            elif not {"dy", "dz", "dbe"} & set(c):
                out[nome] = 0.0
            else:
                out[nome] = self.prec.preco_em_estado(ticker, data, aplicar_cenario(est, c)) - base
        self._stress[chave] = out
        return out

    def stress_carteira(self, q: dict[str, int], i: int) -> dict[str, float]:
        """P&L (R$) por cenário. Futuros de DI entram pelo ajuste (ΔPU × quantidade); o caixa não se move."""
        tot = dict.fromkeys(CENARIOS, 0.0)
        for tk, qt in q.items():
            for nome, dp in self.stress_unitario(tk, i).items():
                tot[nome] += qt * dp
        return tot
