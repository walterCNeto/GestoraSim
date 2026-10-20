"""Geradores de mercado: ações (GBM correlacionado), curva pré (Nelson-Siegel), CDI e IPCA."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gestorasim.dados import calendario
from gestorasim.dominio import Ativo, ClasseAtivo

TICKERS_ACOES = [
    "ALFA3", "BETA3", "GAMA4", "DELT3", "EPSI3", "ZETA3", "ETAA4", "TETA3", "IOTA3", "KAPA3",
    "LAMB3", "MIUU4", "NIUU3", "XIII3", "OMIC3", "PIII4", "ROOO3", "SIGM3", "TAUU3", "UPSI4",
]
PRAZOS_CURVA_DU = [21, 63, 126, 252, 504, 756, 1008, 1260, 1512, 2520]
LAMBDA_NS = 0.6  # decaimento Nelson-Siegel (por ano)


def ativos_acoes() -> list[Ativo]:
    return [
        Ativo(ticker=t, classe=ClasseAtivo.ACAO, emissor=f"Empresa {t[:4]} S.A.")
        for t in TICKERS_ACOES
    ]


def gerar_correlacao(rng: np.random.Generator, n: int = 20, n_fatores: int = 3) -> np.ndarray:
    """Correlação sintética via modelo de fatores: B B' + D, normalizada. PSD por construção."""
    B = rng.normal(0.0, 0.5, (n, n_fatores))
    B[:, 0] = rng.uniform(0.5, 1.0, n)  # fator de mercado
    cov = B @ B.T + np.diag(rng.uniform(0.3, 0.8, n))
    d = np.sqrt(np.diag(cov))
    corr = cov / np.outer(d, d)
    return (corr + corr.T) / 2


def gerar_precos_acoes(
    rng: np.random.Generator, datas: pd.DatetimeIndex, corr: np.ndarray
) -> pd.DataFrame:
    """GBM correlacionado (formato largo: data, ticker, preco)."""
    n, T = corr.shape[0], len(datas)
    dt = 1 / 252
    mu = rng.uniform(0.05, 0.15, n)
    sigma = rng.uniform(0.20, 0.45, n)
    p0 = rng.uniform(10.0, 100.0, n)
    L = np.linalg.cholesky(corr)
    z = rng.standard_normal((T - 1, n)) @ L.T
    retornos = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    logp = np.log(p0) + np.vstack([np.zeros(n), np.cumsum(retornos, axis=0)])
    precos = np.exp(logp)
    df = pd.DataFrame(precos, index=datas, columns=TICKERS_ACOES)
    return df.rename_axis("data").reset_index().melt(
        id_vars="data", var_name="ticker", value_name="preco"
    ).sort_values(["ticker", "data"], ignore_index=True)


# ---------------------------------------------------------------- curva pré / CDI / IPCA

def taxa_pre(beta: np.ndarray, du: np.ndarray | float) -> np.ndarray:
    """Taxa pré a.a. (base 252) por Nelson-Siegel. `beta` (...,3) e `du` com broadcast elemento a elemento."""
    tau = np.maximum(np.asarray(du, dtype=float), 1.0) / 252
    x = LAMBDA_NS * tau
    carga1 = (1 - np.exp(-x)) / x
    carga2 = carga1 - np.exp(-x)
    b = np.asarray(beta)
    return b[..., 0] + b[..., 1] * carga1 + b[..., 2] * carga2


def gerar_fatores_ns(rng: np.random.Generator, datas: pd.DatetimeIndex) -> pd.DataFrame:
    """Fatores nível/inclinação/curvatura com dinâmica AR(1) diária."""
    T = len(datas)
    mu = np.array([0.125, -0.020, 0.010])
    phi = np.array([0.995, 0.990, 0.980])
    sigma = np.array([0.0005, 0.0006, 0.0010])
    x = np.empty((T, 3))
    x[0] = mu + np.array([0.0, 0.01, 0.0])  # curva inicial um pouco mais inclinada
    eps = rng.standard_normal((T, 3))
    for t in range(1, T):
        x[t] = mu + phi * (x[t - 1] - mu) + sigma * eps[t]
    x[:, 0] = np.maximum(x[:, 0], 0.04)
    df = pd.DataFrame(x, columns=["beta0", "beta1", "beta2"], index=datas)
    return df.rename_axis("data").reset_index()


def curva_pre_vertices(fatores: pd.DataFrame) -> pd.DataFrame:
    """Curva pré nos vértices padrão (formato largo -> longo: data, prazo_du, taxa)."""
    beta = fatores[["beta0", "beta1", "beta2"]].to_numpy()
    taxas = taxa_pre(beta[:, None, :], np.array(PRAZOS_CURVA_DU)[None, :])
    df = pd.DataFrame(taxas, columns=PRAZOS_CURVA_DU)
    df.insert(0, "data", fatores["data"].to_numpy())
    return df.melt(id_vars="data", var_name="prazo_du", value_name="taxa").astype(
        {"prazo_du": int}
    ).sort_values(["data", "prazo_du"], ignore_index=True)


def gerar_cdi(fatores: pd.DataFrame) -> pd.DataFrame:
    """CDI sintético = ponta curta da curva (1 mês); fator acumulado desde a primeira data."""
    beta = fatores[["beta0", "beta1", "beta2"]].to_numpy()
    cdi = taxa_pre(beta, np.full(len(beta), 21.0))
    fator_dia = (1 + cdi) ** (1 / 252)
    acum = np.concatenate([[1.0], np.cumprod(fator_dia[:-1])])
    return pd.DataFrame({"data": fatores["data"], "cdi_aa": cdi, "fator_acumulado": acum})


def gerar_ipca(rng: np.random.Generator, datas: pd.DatetimeIndex) -> pd.DataFrame:
    """IPCA mensal AR(1); número-índice diário pela interpolação geométrica em dias úteis."""
    meses = datas.to_period("M")
    unicos = meses.unique()
    phi, mu, sigma = 0.6, 0.0040, 0.0025
    var = np.empty(len(unicos))
    var[0] = mu
    for k in range(1, len(unicos)):
        var[k] = mu + phi * (var[k - 1] - mu) + sigma * rng.standard_normal()
    var = np.maximum(var, -0.002)
    mapa = dict(zip(unicos, var))
    var_dia = np.array([mapa[m] for m in meses])
    du_mes = np.array([
        calendario.du(m.start_time.date(), (m + 1).start_time.date()) for m in meses
    ])
    fator = (1 + var_dia) ** (1 / du_mes)
    indice = 1000.0 * np.concatenate([[1.0], np.cumprod(fator[:-1])])
    return pd.DataFrame({"data": datas, "variacao_mensal": var_dia, "indice": indice})
