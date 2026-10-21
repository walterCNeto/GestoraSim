"""Carteira: posição por fundo e data, valor de mercado, fonte de preço e liquidez em dias."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app import dados
from app.formatacao import tres_casas


def render() -> None:
    dados.exigir("mtm_diario", "pl_diario", "ativos", "fundos")
    mtm, pl, ativos = dados.tabela("mtm_diario"), dados.tabela("pl_diario"), dados.tabela("ativos").set_index("ticker")
    c1, c2 = st.columns(2)
    with c1:
        cnpj = dados.escolher_fundo()
    with c2:
        data = dados.escolher_data(mtm["data"].unique(), chave="cart_data")

    p = mtm[(mtm["fundo_cnpj"] == cnpj) & (mtm["data"] == data)].copy()
    linha = pl[(pl["fundo_cnpj"] == cnpj) & (pl["data"] == data)].iloc[0]
    liq = dados.tabela_ou_vazia("liquidez")
    if len(liq):
        l = liq[(liq["fundo_cnpj"] == cnpj) & (liq["data"] == data)].set_index("ticker")[["dias_para_zerar", "bucket"]]
        p = p.join(l, on="ticker")
    p["classe"] = p["ticker"].map(ativos["classe"])
    p["emissor"] = p["ticker"].map(ativos["emissor"])
    p["rating"] = p["ticker"].map(ativos["rating"])
    pl_ativos = float(linha["pl"])
    p["valor_pct_pl"] = p["valor_mercado"] / pl_ativos
    p["exposicao_pct_pl"] = p["exposicao"] / pl_ativos
    p = p.sort_values("exposicao", ascending=False)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("PL dos ativos (R$)", f"{pl_ativos:.3f}")
    m2.metric("Valor de mercado (R$)", f"{linha['valor_mercado']:.3f}")
    m3.metric("Caixa projetado (R$)", f"{linha['caixa_projetado']:.3f}")
    m4.metric("Caixa disponível (R$)", f"{linha['caixa_disponivel']:.3f}")

    colunas = ["ticker", "classe", "emissor", "rating", "qtd", "preco", "valor_mercado", "valor_pct_pl", "exposicao",
               "exposicao_pct_pl", "fonte_preco"] + (["dias_para_zerar", "bucket"] if "dias_para_zerar" in p else [])
    st.dataframe(tres_casas(p[colunas]), hide_index=True)
    st.caption("Futuros de DI têm valor de mercado 0 (só ajuste diário); a exposição é o nocional. "
               "Liquidez: dias para zerar a 20% do volume do dia.")

    esq, dir_ = st.columns(2)
    with esq:
        st.subheader("Composição por classe (exposição / PL)")
        st.bar_chart(p.groupby("classe")["exposicao_pct_pl"].sum())
    with dir_:
        st.subheader("Evolução do valor de mercado por classe")
        h = mtm[mtm["fundo_cnpj"] == cnpj].assign(classe=lambda d: d["ticker"].map(ativos["classe"]))
        st.area_chart(h.pivot_table(index="data", columns="classe", values="valor_mercado", aggfunc="sum").fillna(0.0))
