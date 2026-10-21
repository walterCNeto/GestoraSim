"""Enquadramento: episódios ATIVO/PASSIVO, prazos, vencidos e a ordem principal causadora."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app import dados
from app.formatacao import tres_casas


def render() -> None:
    dados.exigir("enquadramento_diario", "episodios", "fundos")
    ep, diario = dados.tabela("episodios"), dados.tabela("enquadramento_diario")
    nomes = dados.fundos()
    st.caption("ATIVO: causado pelas ordens do dia (a carteira sem elas estaria enquadrada). PASSIVO: mercado ou eventos; "
               "conta prazo de reenquadramento em dias úteis e vira VENCIDO se estourar.")
    if ep.empty:
        st.success("Nenhum episódio de desenquadramento no período.")
    else:
        ep = ep.assign(fundo=ep["fundo_cnpj"].map(nomes),
                       situacao=lambda d: d["fim"].isna().map({True: "EM ABERTO", False: "REENQUADRADO"}))
        a, p, v, ab = st.columns(4)
        a.metric("ATIVOS", int((ep["classe"] == "ATIVO").sum()))
        p.metric("PASSIVOS", int((ep["classe"] == "PASSIVO").sum()))
        v.metric("VENCIDOS", int(ep["vencido"].sum()))
        ab.metric("Em aberto", int(ep["fim"].isna().sum()))
        f1, f2, f3 = st.columns(3)
        fundo = f1.selectbox("Fundo", ["(todos)", *nomes.values()], key="enq_fundo")
        classe = f2.selectbox("Classe", ["(todas)", "ATIVO", "PASSIVO"], key="enq_classe")
        so_venc = f3.checkbox("Só vencidos", key="enq_venc")
        f = ep
        if fundo != "(todos)":
            f = f[f["fundo"] == fundo]
        if classe != "(todas)":
            f = f[f["classe"] == classe]
        if so_venc:
            f = f[f["vencido"]]
        cols = ["fundo", "regra", "detalhe", "classe", "inicio", "fim", "situacao", "du_max", "vencido",
                "ordem_principal", "n_ordens", "ordens_causa"]
        st.dataframe(tres_casas(f[cols].sort_values("inicio", ascending=False)), hide_index=True)
        st.caption("`ordens_causa`: ordens do dia ranqueadas pela contribuição à métrica da regra (leave-one-out); "
                   "`ordem_principal` é a de maior contribuição.")

    st.subheader("Utilização dos limites no tempo")
    c1, c2 = st.columns(2)
    cnpj = next(c for c, n in nomes.items() if n == c1.selectbox("Fundo", list(nomes.values()), key="enq_fundo2"))
    regras = sorted(diario.loc[diario["fundo_cnpj"] == cnpj, "regra"].unique())
    regra = c2.selectbox("Regra", regras, key="enq_regra")
    d = diario[(diario["fundo_cnpj"] == cnpj) & (diario["regra"] == regra)]
    if d["detalhe"].nunique() > 1:
        det = st.selectbox("Detalhe", sorted(d["detalhe"].unique()), key="enq_detalhe")
        d = d[d["detalhe"] == det]
    d = d.sort_values("data").set_index("data")
    st.line_chart(d[["valor", "limite"]])
    st.dataframe(tres_casas(d.reset_index()[["data", "regra", "detalhe", "valor", "limite", "folga", "status", "alerta",
                                             "classificacao", "du_desde_inicio", "vencido"]].tail(20)), hide_index=True)
