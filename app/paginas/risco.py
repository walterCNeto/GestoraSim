"""Risco: VaR/ES no tempo, sensibilidades por vértice, stress e backtest (P&L limpo vs sujo)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app import dados
from app.formatacao import tres_casas


def render() -> None:
    dados.exigir("risco_diario", "sensibilidades", "stress", "backtest_var", "fundos")
    risco, sens, stress = dados.tabela("risco_diario"), dados.tabela("sensibilidades"), dados.tabela("stress")
    bt, exc = dados.tabela("backtest_var"), dados.tabela_ou_vazia("excecoes_var")
    cnpj = dados.escolher_fundo()
    r = risco[risco["fundo_cnpj"] == cnpj].sort_values("data").set_index("data")
    st.caption("Percentuais sobre o PL líquido (cota_diaria). VaR 99% 1 dia; ES 97,5% histórico (janela de 252 pregões).")

    st.subheader("VaR e ES no tempo (fração do PL líquido)")
    st.line_chart(r[["var_param_pct", "var_hist_pct", "es_hist_pct"]])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("VaR paramétrico médio", f"{r['var_param_pct'].mean():.3f}")
    c2.metric("VaR histórico médio", f"{r['var_hist_pct'].mean():.3f}" if r["var_hist_pct"].notna().any() else "n/d")
    c3.metric("DV01 total (R$/bp)", f"{r['dv01_total'].iloc[-1]:.3f}")
    c4.metric("Duration (anos do PL)", f"{r['duration_pl'].iloc[-1]:.3f}")

    st.subheader("Sensibilidades por vértice")
    datas = sorted(sens["data"].unique())
    data = dados.escolher_data(datas, "Data", "risco_data")
    s = sens[(sens["fundo_cnpj"] == cnpj) & (sens["data"] == data)].copy()
    curvas = s[s["fator"].str.startswith(("PRE:", "REAL:", "SPREAD:"))]
    if curvas.empty:
        st.info("Sem sensibilidade a juros/crédito nesta data.")
    else:
        curvas = curvas.assign(tipo=curvas["fator"].str.split(":").str[0], vertice=curvas["fator"].str.split(":").str[1])
        st.bar_chart(curvas.pivot_table(index="vertice", columns="tipo", values="delta", aggfunc="sum"))
        st.caption("DV01 (pré e real) e CS01 (spread por rating), em R$ por bp, bump de 1 bp com reavaliação.")
    st.dataframe(tres_casas(s[s["fator"].str.startswith("ACAO:")][["fator", "delta"]].rename(columns={"delta": "delta (R$ por 100% de retorno)"})),
                 hide_index=True, height=180)

    st.subheader("Stress com reavaliação completa")
    st.dataframe(tres_casas(stress[(stress["fundo_cnpj"] == cnpj) & (stress["data"] == data)][["cenario", "pnl", "pnl_pct_pl"]]),
                 hide_index=True)

    st.subheader("Backtest do VaR 99%")
    base = st.radio("Base do P&L hipotético", ["LIMPO", "SUJO"], horizontal=True, key="risco_base",
                    help="LIMPO: só efeito dos fatores de mercado. SUJO: inclui carrego e eventos de renda fixa.")
    metodo = st.radio("Método", ["PARAMETRICO", "HISTORICO"], horizontal=True, key="risco_metodo")
    col_pnl = "pnl_hipotetico_limpo" if base == "LIMPO" else "pnl_hipotetico"
    col_var = "var_param_ant" if metodo == "PARAMETRICO" else "var_hist_ant"
    g = r[[col_pnl, col_var, "em_burn_in"]].dropna(subset=[col_pnl, col_var]).copy()
    g["menos_var"] = -g[col_var]
    g["excecao"] = g[col_pnl].where(g[col_pnl] < g["menos_var"])
    st.line_chart(g[[col_pnl, "menos_var", "excecao"]])
    st.caption("Pontos em 'excecao' = P&L hipotético abaixo de −VaR do dia anterior. Burn-in: "
               f"{int(r['em_burn_in'].sum())} pregões iniciais fora do teste.")
    sel = bt[(bt["fundo_cnpj"] == cnpj) & (bt["base"] == base) & (bt["metodo"] == metodo)]
    st.dataframe(tres_casas(sel.drop(columns=["fundo_cnpj"])), hide_index=True)
    if len(exc):
        e = exc[(exc["fundo_cnpj"] == cnpj) & (exc["base"] == base) & (exc["metodo"] == metodo)]
        st.markdown(f"**Exceções ({len(e)})**")
        st.dataframe(tres_casas(e[["data", "pnl", "var_dia_anterior", "em_burn_in"]]), hide_index=True)
