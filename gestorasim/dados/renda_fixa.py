"""Renda fixa simulada: LTN, NTN-B, debêntures e futuro de DI.

Convenção de preços: PU "ex-cupom" na data de pagamento (o PU cai após o pagamento de juros/amortização),
de modo que os fluxos em `eventos` somados ao PU formam o retorno total, sem dupla contagem.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gestorasim.dados import calendario
from gestorasim.dados.mercado import taxa_pre
from gestorasim.dominio import Ativo, ClasseAtivo, Indexador

CUPOM_NTNB = 0.06
CF_NTNB = (1 + CUPOM_NTNB) ** 0.5 - 1  # cupom semestral por unidade de VNA
VNE = 1000.0  # valor nominal de emissão


@dataclass
class Debenture:
    ticker: str
    emissor: str
    rating: str
    indexador: Indexador
    spread: float  # CDI + spread, ou IPCA + taxa real (a.a.)
    anos: float
    amortizacao_no_cupom: tuple[int, float] | None = None  # (n-ésimo cupom, fração do principal)
    vol_credito: float = 0.0003  # vol diária do spread de mercado


DEBENTURES = [
    Debenture("ALEN11", "Alfa Energia S.A.", "AAA", Indexador.CDI, 0.0080, 4.5, vol_credito=0.00015),
    Debenture("BSAN12", "Beta Saneamento S.A.", "AA+", Indexador.IPCA, 0.0620, 6.5, vol_credito=0.00020),
    Debenture("GROD13", "Gama Rodovias S.A.", "AA-", Indexador.CDI, 0.0160, 3.5, (3, 0.5), 0.00030),
    Debenture("DLOG14", "Delta Logística S.A.", "A+", Indexador.IPCA, 0.0750, 5.0, vol_credito=0.00040),
    Debenture("EVAR15", "Epsilon Varejo S.A.", "A-", Indexador.CDI, 0.0320, 3.0, vol_credito=0.00060),
]


def _ar1(rng: np.random.Generator, n: int, phi: float, sigma: float, mu: float = 0.0) -> np.ndarray:
    x = np.empty(n)
    x[0] = mu
    e = rng.standard_normal(n)
    for t in range(1, n):
        x[t] = mu + phi * (x[t - 1] - mu) + sigma * e[t]
    return x


def _du_ate(datas: pd.DatetimeIndex, alvo) -> np.ndarray:
    return calendario.du(datas.values.astype("datetime64[D]"), calendario._d(alvo))


def _datas_semestrais(inicio: pd.Timestamp, venc: pd.Timestamp, ancora_mes: int | None = None):
    """Datas de pagamento semestrais retroativas a partir do vencimento (ajustadas a dia útil)."""
    datas, k = [], 0
    while True:
        d = venc - pd.DateOffset(months=6 * k)
        if d <= inicio:
            break
        datas.append(calendario.proximo_dia_util(d))
        k += 1
    return sorted(datas)


def _evento(tipo: str, ticker: str, data, valor: float) -> dict:
    return {"tipo": tipo, "ticker": ticker, "data_com": data, "data_pagamento": data,
            "valor_por_unidade": valor}


# ------------------------------------------------------------------------------ títulos públicos

def gerar_ltn(rng, datas, beta) -> tuple[list[Ativo], pd.DataFrame]:
    vencs = [pd.Timestamp(d) for d in ("2025-07-01", "2026-01-01", "2027-01-01", "2029-01-01")]
    ativos, linhas = [], []
    for v in vencs:
        v = calendario.proximo_dia_util(v)
        tk = f"LTN{v:%y%m%d}"
        ativos.append(Ativo(ticker=tk, classe=ClasseAtivo.TITULO_PUBLICO, emissor="Tesouro Nacional (sim)",
                            vencimento=v.date(), cupom=0.0, indexador=Indexador.PRE))
        du_rest = _du_ate(datas, v)
        prem = rng.uniform(-0.001, 0.002) + _ar1(rng, len(datas), 0.98, 0.0002)
        y = taxa_pre(beta, du_rest) + prem
        pu = 1000.0 / (1 + y) ** (du_rest / 252)
        m = du_rest > 0
        linhas.append(pd.DataFrame({"data": datas[m], "ticker": tk, "taxa": y[m], "pu": pu[m]}))
    return ativos, pd.concat(linhas, ignore_index=True)


def gerar_ntnb(rng, datas, beta, ipca) -> tuple[list[Ativo], pd.DataFrame, list[dict]]:
    vencs = ["2026-08-15", "2028-08-15", "2030-08-15", "2035-05-15"]
    be = _ar1(rng, len(datas), 0.995, 0.0004, mu=0.040)  # inflação implícita
    vna = ipca["indice"].to_numpy() / ipca["indice"].iloc[0] * VNE
    ativos, linhas, eventos = [], [], []
    for v in vencs:
        v = calendario.proximo_dia_util(pd.Timestamp(v))
        tk = f"NTNB{v:%y%m%d}"
        ativos.append(Ativo(ticker=tk, classe=ClasseAtivo.TITULO_PUBLICO, emissor="Tesouro Nacional (sim)",
                            vencimento=v.date(), cupom=CUPOM_NTNB, indexador=Indexador.IPCA))
        du_rest = _du_ate(datas, v)
        nominal = taxa_pre(beta, du_rest) + rng.uniform(0.0, 0.003)
        r = (1 + nominal) / (1 + be) - 1
        pag = _datas_semestrais(datas[0] - pd.DateOffset(years=1), v)
        soma = np.zeros(len(datas))
        for d in pag:
            du_k = _du_ate(datas, d)
            fluxo = CF_NTNB + (1.0 if d == pag[-1] else 0.0)
            soma += np.where(du_k > 0, fluxo / (1 + r) ** (np.maximum(du_k, 0) / 252), 0.0)
            i = datas.searchsorted(d)
            if i < len(datas) and datas[i] == d:
                eventos.append(_evento("CUPOM", tk, d, CF_NTNB * vna[i]))
        m = du_rest > 0
        linhas.append(pd.DataFrame({"data": datas[m], "ticker": tk, "taxa": r[m], "pu": (vna * soma)[m]}))
    return ativos, pd.concat(linhas, ignore_index=True), eventos


# ------------------------------------------------------------------------------ debêntures

def gerar_debentures(rng, datas, cdi, ipca) -> tuple[list[Ativo], pd.DataFrame, list[dict]]:
    T = len(datas)
    cdi_acum = cdi["fator_acumulado"].to_numpy()
    ipca_rel = ipca["indice"].to_numpy() / ipca["indice"].iloc[0]
    ativos, linhas, eventos = [], [], []
    for deb in DEBENTURES:
        venc = calendario.proximo_dia_util(datas[0] + pd.DateOffset(months=int(deb.anos * 12)))
        ativos.append(Ativo(ticker=deb.ticker, classe=ClasseAtivo.DEBENTURE, emissor=deb.emissor,
                            rating=deb.rating, vencimento=venc.date(), cupom=deb.spread,
                            indexador=deb.indexador))
        idx_cupom = []
        for d in _datas_semestrais(datas[0], venc):
            i = datas.searchsorted(d)
            if i < T and datas[i] == d:
                idx_cupom.append(i)
        du_rest = _du_ate(datas, venc)
        z = _ar1(rng, T, 0.995, deb.vol_credito)
        z = z - z[0]  # emitida ao par
        z = np.maximum(z, -0.8 * deb.spread)

        def valor(i, L, resto):
            idx = cdi_acum[i] / cdi_acum[L] if deb.indexador == Indexador.CDI else ipca_rel[i]
            return VNE * resto * idx * (1 + deb.spread) ** ((i - L) / 252)

        def principal(i, resto):
            idx = 1.0 if deb.indexador == Indexador.CDI else ipca_rel[i]
            return VNE * resto * idx

        resto, L, pu = 1.0, 0, np.empty(T)
        for i in range(T):
            if i in idx_cupom and i > 0:
                n_cupom = idx_cupom.index(i) + 1
                eventos.append(_evento("CUPOM", deb.ticker, datas[i], valor(i, L, resto) - principal(i, resto)))
                if deb.amortizacao_no_cupom and n_cupom == deb.amortizacao_no_cupom[0]:
                    frac = deb.amortizacao_no_cupom[1]
                    eventos.append(_evento("AMORTIZACAO", deb.ticker, datas[i], principal(i, resto) * frac))
                    resto *= 1 - frac
                L = i
            curva = valor(i, L, resto)
            pu[i] = curva * ((1 + deb.spread) / (1 + deb.spread + z[i])) ** (du_rest[i] / 252)
        m = du_rest > 0
        linhas.append(pd.DataFrame({"data": datas[m], "ticker": deb.ticker,
                                    "taxa": (deb.spread + z)[m], "pu": pu[m]}))
    return ativos, pd.concat(linhas, ignore_index=True), eventos


# ------------------------------------------------------------------------------ futuro de DI

CODIGOS_MES = "FGHJKMNQUVXZ"


def gerar_futuros_di(rng, datas, beta) -> tuple[list[Ativo], pd.DataFrame]:
    ativos, linhas = [], []
    ano0 = datas[0].year
    for k in (3, 4, 5):  # vértices janeiro/ano0+3, +4, +5
        v = calendario.proximo_dia_util(pd.Timestamp(ano0 + k, 1, 1))
        tk = f"DI1{CODIGOS_MES[v.month - 1]}{v:%y}"
        ativos.append(Ativo(ticker=tk, classe=ClasseAtivo.FUTURO_DI, emissor="B3 (sim)",
                            vencimento=v.date(), cupom=0.0, indexador=Indexador.PRE))
        du_rest = _du_ate(datas, v)
        y = taxa_pre(beta, du_rest) + _ar1(rng, len(datas), 0.97, 0.0003)
        pu = 100_000.0 / (1 + y) ** (du_rest / 252)
        linhas.append(pd.DataFrame({"data": datas, "ticker": tk, "taxa": y, "pu": pu}))
    return ativos, pd.concat(linhas, ignore_index=True)
