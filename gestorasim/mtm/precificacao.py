"""Marcação a mercado por classe: funções puras por data + `Precificador` que monta o painel.

Convenção (igual ao bloco 1): PU ex-cupom — na data de pagamento o PU já exclui o fluxo do dia,
exceto o resgate no vencimento (NTN-B no vencimento = VNA; LTN no vencimento = 1000).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from gestorasim.dados import calendario
from gestorasim.dados.renda_fixa import CF_NTNB, VNE
from gestorasim.dominio import ClasseAtivo, Indexador
from gestorasim.mtm.curvas import CAL, EstadoMercado, construir_estados

PU_DI_VENCIMENTO = 100_000.0
FONTES = {
    ClasseAtivo.ACAO: "FECHAMENTO",
    ClasseAtivo.TITULO_PUBLICO: "CURVA_PRE_FLAT_FORWARD",
    ClasseAtivo.DEBENTURE: "CURVA_REF+SPREAD_RATING",
    ClasseAtivo.FUTURO_DI: "CURVA_PRE_DI",
}
FONTE_NTNB = "VNA_IPCA+CURVA_REAL"


# ----------------------------------------------------------------------------- funções puras

def pu_ltn(est: EstadoMercado, du: int) -> float:
    """PU = 1000/(1+taxa_pre)^(du/252), taxa interpolada flat-forward. Vencimento (du=0) = 1000."""
    if du <= 0:
        return VNE
    return VNE / (1 + est.taxa_pre(du)) ** (du / 252)


def vna(ipca_indice: float, ipca_indice_0: float) -> float:
    """VNA = 1000 corrigido pelo IPCA sintético (índice diário já é pro rata geométrico em du no mês)."""
    return VNE * ipca_indice / ipca_indice_0


def datas_semestrais(inicio: pd.Timestamp, venc: pd.Timestamp) -> list[pd.Timestamp]:
    """Datas de pagamento semestrais, retroativas a partir do vencimento, > inicio, ajustadas a dia útil."""
    datas, k = [], 0
    while True:
        d = calendario.proximo_dia_util(venc - pd.DateOffset(months=6 * k))
        if d <= inicio:
            break
        datas.append(d)
        k += 1
    return sorted(datas)


def pu_ntnb(est: EstadoMercado, data: pd.Timestamp, venc: pd.Timestamp, cupons: list[pd.Timestamp],
            ipca_indice_0: float) -> float:
    """VNA × Σ fluxos (cupom 6% a.a. semestral + principal) descontados pela curva real."""
    v = vna(est.ipca_indice, ipca_indice_0)
    if data == venc:
        return v
    if data > venc:
        return float("nan")
    soma = 0.0
    for d in (c for c in cupons if c > data):
        du = CAL.du(data, d)
        fluxo = CF_NTNB + (1.0 if d == venc else 0.0)
        soma += fluxo / (1 + est.taxa_real(du)) ** (du / 252)
    return v * soma


@dataclass(frozen=True)
class DescDebenture:
    ticker: str
    indexador: Indexador
    spread: float
    rating: str
    emissao: pd.Timestamp
    venc: pd.Timestamp
    cupons: tuple  # todas as datas de pagamento (inclui as já passadas e o vencimento)
    amortizacoes: dict = field(default_factory=dict)  # data -> fração do VNE atual


def pu_debenture(d: DescDebenture, data: pd.Timestamp, est: EstadoMercado, cdi_acum_em,
                 ipca_indice_0: float, ref0: float) -> float:
    """Fluxos futuros descontados na curva de referência + spread de crédito do rating.

    CDI+: coupons projetados pela curva pré (forward do CDI) mais o CDI já realizado no período
    corrente; desconto = curva pré × (1 + spread_contrato + Δspread do rating).
    IPCA+: fluxos reais (cupom composto + amortização) descontados por uma taxa real plana =
    spread_contrato + Δspread do rating + variação da curva real no prazo do papel desde a emissão.
    Em t = emissão (Δ = 0) o PU é o par; o PU é ex-cupom/ex-amortização na data de pagamento.
    """
    if data >= d.venc:
        return float("nan")
    L = max([c for c in d.cupons if c <= data], default=d.emissao)
    futuros = [c for c in d.cupons if c > data]
    resto = float(np.prod([1 - f for dt, f in d.amortizacoes.items() if dt <= data]))
    z = est.z_credito[d.rating]
    pv, anterior, df0_ant = 0.0, L, 1.0
    if d.indexador == Indexador.IPCA:
        y_tot = d.spread + z + (est.taxa_real(CAL.du(data, d.venc)) - ref0)
    for j, T in enumerate(futuros):
        n = CAL.du(anterior, T)
        du_T = CAL.du(data, T)
        ultimo = j == len(futuros) - 1
        pago = resto * (1.0 if ultimo else d.amortizacoes.get(T, 0.0))
        if d.indexador == Indexador.CDI:
            df0 = (1 + est.taxa_pre(du_T)) ** (-du_T / 252)
            fwd = est.cdi_acum / cdi_acum_em(L) / df0 if j == 0 else df0_ant / df0
            cupom = resto * (fwd * (1 + d.spread) ** (n / 252) - 1)
            pv += VNE * (cupom + pago) * df0 * (1 + d.spread + z) ** (-du_T / 252)
            df0_ant = df0
        else:
            cupom = resto * ((1 + d.spread) ** (n / 252) - 1)
            pv += (cupom + pago) / (1 + y_tot) ** (du_T / 252)
        resto -= pago
        anterior = T
    if d.indexador == Indexador.IPCA:
        pv *= vna(est.ipca_indice, ipca_indice_0)
    return pv


def pu_di(est: EstadoMercado, du: int) -> float:
    """PU = 100000/(1+taxa)^(du/252)."""
    if du < 0:
        return float("nan")
    if du == 0:
        return PU_DI_VENCIMENTO
    return PU_DI_VENCIMENTO / (1 + est.taxa_pre(du)) ** (du / 252)


def fracoes_amortizacao(eventos: pd.DataFrame, meta: pd.DataFrame, ipca: pd.DataFrame) -> dict:
    """Fração do VNE amortizada em cada evento AMORTIZACAO: valor / principal atualizado na data."""
    ipca_rel = (ipca.set_index("data")["indice"] / ipca["indice"].iloc[0])
    out = {}
    for tk, g in eventos[eventos["tipo"] == "AMORTIZACAO"].sort_values("data_com").groupby("ticker"):
        resto = 1.0
        for _, ev in g.iterrows():
            idx = ipca_rel[ev["data_com"]] if meta.loc[tk, "indexador"] == "IPCA" else 1.0
            fr = float(ev["valor_por_unidade"]) / (VNE * resto * idx)
            out[(tk, pd.Timestamp(ev["data_com"]))] = fr
            resto *= 1 - fr
    return out


def ajuste_di(pu_hoje: float, pu_ontem: float, cdi_ontem_aa: float, quantidade: int,
              multiplicador: float = 1.0) -> float:
    """Ajuste diário = (PU_d − PU_{d−1} × (1+CDI_{d−1})^(1/252)) × qtd × multiplicador.
    Quantidade negativa (vendido) inverte o sinal. Liquidado em caixa em D+1."""
    return (pu_hoje - pu_ontem * (1 + cdi_ontem_aa) ** (1 / 252)) * quantidade * multiplicador


# ----------------------------------------------------------------------------- precificador

class Precificador:
    """Preços de todos os ativos em todas as datas; também o preço de carrego (curva do dia anterior)."""

    def __init__(self, tabelas: dict[str, pd.DataFrame], seed: int) -> None:
        self.meta = tabelas["ativos"].set_index("ticker")
        self.datas = pd.DatetimeIndex(tabelas["calendario"]["data"])
        self.estados = construir_estados(tabelas, seed)
        self.i_de = {d: i for i, d in enumerate(self.datas)}
        self.ipca0 = float(tabelas["ipca"]["indice"].iloc[0])
        self._cdi_acum = tabelas["cdi"].set_index("data")["fator_acumulado"]
        self._acoes = tabelas["precos_acoes"].pivot(index="data", columns="ticker", values="preco")
        self.fracoes = fracoes_amortizacao(tabelas["eventos"], self.meta, tabelas["ipca"])
        self.cupons_ntnb: dict[str, list] = {}
        self.deb: dict[str, DescDebenture] = {}
        self.ref0: dict[str, float] = {}
        for tk, m in self.meta.iterrows():
            venc = pd.Timestamp(m["vencimento"]) if pd.notna(m["vencimento"]) else None
            if tk.startswith("NTNB"):
                self.cupons_ntnb[tk] = datas_semestrais(self.datas[0] - pd.DateOffset(years=1), venc)
            elif m["classe"] == "DEBENTURE":
                am = {pd.Timestamp(dt): self.fracoes[(tk, pd.Timestamp(dt))]
                      for (t2, dt) in self.fracoes if t2 == tk}
                self.deb[tk] = DescDebenture(
                    tk, Indexador(m["indexador"]), float(m["cupom"]), m["rating"], self.datas[0], venc,
                    tuple(datas_semestrais(self.datas[0], venc)), am)
                self.ref0[tk] = self.estados[0].taxa_real(CAL.du(self.datas[0], venc))

    def cdi_acum_em(self, data) -> float:
        return float(self._cdi_acum[pd.Timestamp(data)])

    def preco_em_estado(self, ticker: str, data: pd.Timestamp, est: EstadoMercado) -> float:
        """Preço do ativo na `data` usando `est` (permite separar efeito de carrego e de mercado)."""
        m = self.meta.loc[ticker]
        classe = ClasseAtivo(m["classe"])
        if classe == ClasseAtivo.ACAO:
            return float(self._acoes.loc[data, ticker])
        venc = pd.Timestamp(m["vencimento"])
        if classe == ClasseAtivo.FUTURO_DI:
            return pu_di(est, CAL.du(data, venc)) if data <= venc else float("nan")
        if classe == ClasseAtivo.DEBENTURE:
            return pu_debenture(self.deb[ticker], data, est, self.cdi_acum_em, self.ipca0, self.ref0[ticker])
        if ticker.startswith("NTNB"):
            return pu_ntnb(est, data, venc, self.cupons_ntnb[ticker], self.ipca0)
        return pu_ltn(est, CAL.du(data, venc)) if data <= venc else float("nan")

    def fonte(self, ticker: str) -> str:
        if ticker.startswith("NTNB"):
            return FONTE_NTNB
        return FONTES[ClasseAtivo(self.meta.loc[ticker, "classe"])]

    def preco(self, ticker: str, i: int) -> float:
        return self.preco_em_estado(ticker, self.datas[i], self.estados[i])

    def preco_carrego(self, ticker: str, i: int) -> float:
        """Preço em i com curva/spread/inflação de i−1 e indexadores (CDI, IPCA) realizados de i."""
        ant, hoje = self.estados[i - 1], self.estados[i]
        est = replace(ant, data=hoje.data, cdi_acum=hoje.cdi_acum, ipca_indice=hoje.ipca_indice,
                      cdi_aa=hoje.cdi_aa)
        return self.preco_em_estado(ticker, self.datas[i], est)

    def painel(self) -> pd.DataFrame:
        """Preços (linhas = datas, colunas = tickers); NaN depois do vencimento."""
        dados = {tk: [self.preco(tk, i) for i in range(len(self.datas))] for tk in self.meta.index}
        return pd.DataFrame(dados, index=self.datas)
