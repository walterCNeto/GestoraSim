"""Trilha de auditoria das ordens: filtro por ordem, fundo, data e tipo de evento."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app import dados
from app.formatacao import tres_casas


def render() -> None:
    dados.exigir("auditoria_serie", "ordens_serie", "fundos")
    aud, ordens = dados.tabela("auditoria_serie"), dados.tabela("ordens_serie")
    nomes = dados.fundos()
    st.caption("Trilha append-only: cada transição de estado (e eventos como EXCLUIDO_PRE_TRADE) com timestamp lógico, "
               "usuário, de→para e motivo (inclui o critério de rateio).")
    if aud.empty:
        st.info("Sem eventos de auditoria.")
        st.stop()
    if "fundos" not in ordens:  # dados gravados antes do bloco 6
        ordens = ordens.assign(fundos="")
    aud = aud.merge(ordens[["data_serie", "ordem_id", "origem", "fundos", "ticker", "lado"]].drop_duplicates(),
                    on=["data_serie", "ordem_id"], how="left")

    c1, c2, c3, c4 = st.columns(4)
    ordem = c1.text_input("Ordem (ex.: ORD0007)", key="tr_ordem").strip()
    fundo = c2.selectbox("Fundo", ["(todos)", *nomes.values()], key="tr_fundo")
    tipo = c3.selectbox("Tipo de evento", ["(todos)", *sorted(aud["tipo"].unique())], key="tr_tipo")
    origem = c4.selectbox("Origem da ordem", ["(todas)", *sorted(aud["origem"].dropna().str.split(":").str[0].unique())], key="tr_origem")
    d1, d2 = st.columns(2)
    datas = sorted(aud["data_serie"].unique())
    ini = d1.selectbox("De", datas, index=0, key="tr_ini", format_func=lambda d: f"{pd.Timestamp(d):%Y-%m-%d}")
    fim = d2.selectbox("Até", datas, index=len(datas) - 1, key="tr_fim", format_func=lambda d: f"{pd.Timestamp(d):%Y-%m-%d}")

    f = aud[(aud["data_serie"] >= ini) & (aud["data_serie"] <= fim)]
    if ordem:
        f = f[f["ordem_id"] == ordem]
    if fundo != "(todos)":
        cnpj = next(c for c, n in nomes.items() if n == fundo)
        f = f[f["fundos"].fillna("").str.contains(cnpj, regex=False)]
    if tipo != "(todos)":
        f = f[f["tipo"] == tipo]
    if origem != "(todas)":
        f = f[f["origem"].fillna("").str.startswith(origem)]
    st.metric("Eventos", len(f))
    ordem_cols = ["data_serie", "seq"] if "seq" in f else ["data_serie"]
    cols = ["data_serie", "timestamp", "ordem_id", "origem", "ticker", "lado", "usuario", "tipo", "de", "para", "motivo"]
    st.dataframe(tres_casas(f.sort_values(ordem_cols)[cols]), hide_index=True, height=480)
    st.subheader("Eventos por tipo e destino")
    st.dataframe(f.groupby(["tipo", "para"]).size().rename("n").reset_index(), hide_index=True)
