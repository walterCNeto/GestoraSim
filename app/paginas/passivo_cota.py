"""Passivo e cota: movimentações dos cotistas, cota bruta/líquida, linha d'água, taxas, come-cotas e IR."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app import dados
from app.formatacao import tres_casas


def render() -> None:
    dados.exigir("cota_diaria", "cotistas_mov", "provisoes", "fundos")
    cota, mov, prov = dados.tabela("cota_diaria"), dados.tabela("cotistas_mov"), dados.tabela("provisoes")
    trib = dados.tabela_ou_vazia("tributos")
    cnpj = dados.escolher_fundo()
    c = cota[cota["fundo_cnpj"] == cnpj].sort_values("data").set_index("data")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("PL líquido (R$)", f"{c['pl_liquido'].iloc[-1]:.3f}")
    m2.metric("Cotas", f"{c['cotas'].iloc[-1]:.3f}")
    m3.metric("Cota bruta", f"{c['cota_bruta'].iloc[-1]:.3f}")
    m4.metric("Cota líquida", f"{c['cota_liquida'].iloc[-1]:.3f}")
    st.caption("Cota bruta = antes da provisão de performance. PL líquido = ativos − taxas a pagar − resgates a pagar; "
               "cota = PL líquido / cotas.")

    st.subheader("Cota bruta, líquida e linha d'água")
    st.line_chart(c[["cota_bruta", "cota_liquida", "linha_dagua"]])

    st.subheader("Taxas")
    p = prov[prov["fundo_cnpj"] == cnpj].sort_values("data").set_index("data")
    t1, t2, t3 = st.columns(3)
    t1.metric("Administração provisionada (R$)", f"{p['adm_dia'].sum():.3f}")
    t2.metric("Administração paga (R$)", f"{p['adm_pago'].sum():.3f}")
    t3.metric("Performance paga (R$)", f"{p['perf_pago'].sum():.3f}")
    st.area_chart(p[["adm_a_pagar", "perf_saldo", "resgates_a_pagar"]])

    st.subheader("Tributos retidos")
    if trib.empty:
        st.info("Sem tributos no período.")
    else:
        t = trib[trib["fundo_cnpj"] == cnpj]
        resumo = t.groupby("tipo")["valor"].agg(["sum", "count"]).rename(columns={"sum": "valor (R$)", "count": "lançamentos"})
        st.dataframe(tres_casas(resumo.reset_index()), hide_index=True)
        com_cotas = t[t["tipo"] == "COME_COTAS"].groupby("data")["valor"].sum()
        if len(com_cotas):
            st.bar_chart(com_cotas)

    st.subheader("Movimentações dos cotistas")
    m = mov[mov["fundo_cnpj"] == cnpj]
    f1, f2 = st.columns(2)
    tipo = f1.selectbox("Tipo", ["(todos)", "APLICACAO", "RESGATE"], key="pas_tipo")
    status = f2.selectbox("Status", ["(todos)", *sorted(m["status"].unique())], key="pas_status")
    if tipo != "(todos)":
        m = m[m["tipo"] == tipo]
    if status != "(todos)":
        m = m[m["status"] == status]
    st.dataframe(tres_casas(m.sort_values("data_solicitacao")), hide_index=True, height=300)
    lotes = dados.tabela_ou_vazia("lotes_cotistas")
    if len(lotes):
        with st.expander("Lotes de cotas (FIFO) na última data"):
            st.dataframe(tres_casas(lotes[lotes["fundo_cnpj"] == cnpj].sort_values("data_aplicacao")), hide_index=True)
