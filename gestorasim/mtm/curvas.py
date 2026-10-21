"""Estado de mercado por data: curva pré (flat-forward), inflação implícita, spreads de crédito, CDI/IPCA."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from gestorasim.dados import calendario

RATINGS_DEBENTURES = ("AAA", "AA+", "AA-", "A+", "A-")
VOL_SPREAD_DIARIA = {"AAA": 0.00008, "AA+": 0.00010, "AA-": 0.00014, "A+": 0.00020, "A-": 0.00030}
INFLACAO_IMPLICITA_MEDIA = 0.04
VERTICES_RISCO = (21, 63, 126, 252, 504, 1260)  # vértices de risco (du) das curvas pré e real


class CalendarioDU:
    """Contagem de dias úteis O(1) entre dias úteis, por índice em uma grade 2022-2045."""

    def __init__(self) -> None:
        dias = pd.bdate_range("2022-01-03", "2045-12-31")
        dias = dias[np.is_busday(dias.values.astype("datetime64[D]"), holidays=calendario.FERIADOS)]
        self._idx = {d: i for i, d in enumerate(dias)}

    def du(self, d0, d1) -> int:
        """Dias úteis em [d0, d1) para datas que são dias úteis."""
        return self._idx[pd.Timestamp(d1)] - self._idx[pd.Timestamp(d0)]


CAL = CalendarioDU()


def taxa_flat_forward(vert_du: np.ndarray, taxas: np.ndarray, du: float) -> float:
    """Taxa a.a. (base 252) no prazo `du`, interpolando ln(fator de desconto) linearmente em du
    entre os vértices (taxa a termo constante entre vértices). Antes do 1º vértice: taxa constante;
    depois do último: prolonga a taxa a termo do último trecho."""
    if du <= 0:
        du = 1e-9
    ln_df = -np.log1p(taxas) * vert_du / 252
    if du <= vert_du[0]:
        return float(taxas[0])
    if du >= vert_du[-1]:
        inclinacao = (ln_df[-1] - ln_df[-2]) / (vert_du[-1] - vert_du[-2])
        v = ln_df[-1] + inclinacao * (du - vert_du[-1])
    else:
        v = np.interp(du, vert_du, ln_df)
    return float(np.exp(-v * 252 / du) - 1)


def peso_vertice(k: int, du: float) -> float:
    """Peso triangular do vértice de risco k em `du` (soma 1 em toda a curva; constante nas pontas)."""
    v = VERTICES_RISCO
    if k > 0 and du <= v[k - 1]:
        return 0.0
    if k < len(v) - 1 and du >= v[k + 1]:
        return 0.0
    if du <= v[k]:
        return 1.0 if k == 0 else (du - v[k - 1]) / (v[k] - v[k - 1])
    return 1.0 if k == len(v) - 1 else (v[k + 1] - du) / (v[k + 1] - v[k])


@dataclass(frozen=True)
class EstadoMercado:
    data: pd.Timestamp
    vert_du: np.ndarray = field(repr=False)
    taxas: np.ndarray = field(repr=False)
    inflacao_implicita: float
    z_credito: dict  # rating -> variação acumulada do spread de crédito (a.a.)
    cdi_aa: float
    cdi_acum: float  # fator de CDI acumulado desde a 1ª data (realizado)
    ipca_indice: float  # número-índice realizado
    bump_real: tuple | None = None  # choques (decimais) nos vértices de risco da curva real, só p/ sensibilidades

    def taxa_pre(self, du: float) -> float:
        return taxa_flat_forward(self.vert_du, self.taxas, du)

    def taxa_real(self, du: float) -> float:
        """Curva real sintética = pré menos inflação implícita (razão de fatores), mais bump nos vértices."""
        r = (1 + self.taxa_pre(du)) / (1 + self.inflacao_implicita) - 1
        if self.bump_real is not None:
            r += sum(b * peso_vertice(k, du) for k, b in enumerate(self.bump_real) if b)
        return r


def gerar_inflacao_implicita(rng: np.random.Generator, n: int) -> np.ndarray:
    x = np.empty(n)
    x[0] = INFLACAO_IMPLICITA_MEDIA
    e = rng.standard_normal(n)
    for t in range(1, n):
        x[t] = INFLACAO_IMPLICITA_MEDIA + 0.995 * (x[t - 1] - INFLACAO_IMPLICITA_MEDIA) + 0.0004 * e[t]
    return x


def gerar_spreads_credito(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    """Passeio aleatório por rating (proxy de taxa indicativa ANBIMA): variação do spread desde a 1ª data."""
    out = {}
    for r in RATINGS_DEBENTURES:
        z = np.concatenate([[0.0], np.cumsum(VOL_SPREAD_DIARIA[r] * rng.standard_normal(n - 1))])
        out[r] = np.maximum(z, -0.004)
    return out


def construir_estados(tabelas: dict[str, pd.DataFrame], seed: int) -> list[EstadoMercado]:
    datas = pd.DatetimeIndex(tabelas["calendario"]["data"])
    n = len(datas)
    curva = tabelas["curva_pre"].pivot(index="data", columns="prazo_du", values="taxa").loc[datas]
    vert = curva.columns.to_numpy(dtype=float)
    infl = gerar_inflacao_implicita(np.random.default_rng([seed, 400]), n)
    z = gerar_spreads_credito(np.random.default_rng([seed, 401]), n)
    cdi = tabelas["cdi"].set_index("data").loc[datas]
    ipca = tabelas["ipca"].set_index("data").loc[datas]
    return [
        EstadoMercado(
            data=datas[i], vert_du=vert, taxas=curva.iloc[i].to_numpy(), inflacao_implicita=float(infl[i]),
            z_credito={r: float(z[r][i]) for r in z}, cdi_aa=float(cdi["cdi_aa"].iloc[i]),
            cdi_acum=float(cdi["fator_acumulado"].iloc[i]), ipca_indice=float(ipca["indice"].iloc[i]),
        )
        for i in range(n)
    ]
