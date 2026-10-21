"""Estatística de risco: covariância EWMA, VaR delta-normal e histórico, ES, Kupiec e zonas de Basileia."""

from __future__ import annotations

import math

import numpy as np

Z99 = 2.3263478740408408  # quantil 99% da normal padrão
LAMBDA = 0.94


def ewma_cov(d_f: np.ndarray, lam: float = LAMBDA, n_init: int = 20) -> np.ndarray:
    """Covariância EWMA (RiskMetrics, média zero) com a informação até cada t, shape (T, n, n).

    `d_f[0]` é ignorado (sem retorno). Antes de `n_init` retornos fica NaN; em t = n_init usa o
    segundo momento amostral dos primeiros `n_init` retornos; depois, recursão com λ.
    """
    T, n = d_f.shape
    cov = np.full((T, n, n), np.nan)
    atual = None
    for t in range(1, T):
        if t < n_init:
            continue
        if atual is None:
            janela = d_f[1:t + 1]
            atual = janela.T @ janela / len(janela)
        else:
            atual = lam * atual + (1 - lam) * np.outer(d_f[t], d_f[t])
        cov[t] = atual
    return cov


def var_delta_normal(delta: np.ndarray, cov: np.ndarray, z: float = Z99) -> float:
    """VaR = z × sqrt(δ' Σ δ) (perda positiva, em R$ se δ em R$ por unidade do fator)."""
    return float(z * math.sqrt(max(float(delta @ cov @ delta), 0.0)))


def pnl_delta_gama(delta: np.ndarray, gama: np.ndarray, choques: np.ndarray) -> np.ndarray:
    """P&L aproximado por cenário: δ·x + ½ Σ γ_k x_k² (gama diagonal). `choques` shape (S, n)."""
    return choques @ delta + 0.5 * (choques ** 2) @ gama


def var_es_historico(pnl: np.ndarray, nivel_var: float = 0.99, nivel_es: float = 0.975) -> tuple[float, float]:
    """VaR histórico (quantil da perda) e ES (média das perdas além do quantil `nivel_es`), ambos positivos."""
    var = -float(np.quantile(pnl, 1 - nivel_var))
    corte = np.quantile(pnl, 1 - nivel_es)
    return var, -float(pnl[pnl <= corte].mean())


def kupiec_pof(n: int, x: int, p: float = 0.01) -> tuple[float, float]:
    """Teste de proporção de falhas de Kupiec: (LR, p-valor) com χ²(1). H0: taxa de exceções = p."""
    if n <= 0:
        return float("nan"), float("nan")

    def ll(taxa: float) -> float:
        a = (n - x) * math.log(1 - taxa) if n - x > 0 else 0.0
        b = x * math.log(taxa) if x > 0 else 0.0
        return a + b

    lr = max(-2 * (ll(p) - ll(x / n)), 0.0)
    return lr, math.erfc(math.sqrt(lr / 2))  # sobrevivência da χ²(1)


def binomial_cdf(k: int, n: int, p: float) -> float:
    return sum(math.comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(0, k + 1))


def zona_basileia(n: int, x: int, p: float = 0.01) -> str:
    """Zonas do Comitê de Basileia pela probabilidade binomial acumulada: verde < 95%, amarela < 99,99%.
    Com n = 250 e p = 1% reproduz 0–4 (verde), 5–9 (amarela), 10+ (vermelha)."""
    c = binomial_cdf(x, n, p)
    return "VERDE" if c < 0.95 else "AMARELA" if c < 0.9999 else "VERMELHA"
