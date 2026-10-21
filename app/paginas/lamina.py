"""Lâmina mensal por fundo (HTML para download)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app import dados
from gestorasim.relatorios.lamina import gerar_lamina


def render() -> None:
    dados.exigir("cota_diaria", "mtm_diario", "ativos", "fundos")
    cota = dados.tabela("cota_diaria")
    res = {"cota_diaria": cota, "mtm_diario": dados.tabela("mtm_diario"), "ativos": dados.tabela("ativos"),
           "risco_diario": dados.tabela_ou_vazia("risco_diario")}
    classes = dict(zip(dados.tabela("fundos")["cnpj"], dados.tabela("fundos")["classe"]))
    c1, c2 = st.columns(2)
    with c1:
        cnpj = dados.escolher_fundo()
    meses = sorted(cota["data"].dt.to_period("M").astype(str).unique())
    mes = c2.selectbox("Mês", meses, index=len(meses) - 1, key="lam_mes")
    html = gerar_lamina(res, cnpj, mes, classes.get(cnpj, ""))
    st.download_button("Baixar lâmina (HTML)", data=html.encode("utf-8"), file_name=f"lamina_{mes}_{cnpj[7:10]}.html",
                       mime="text/html")
    if hasattr(st, "iframe"):  # HTML gerado por nós a partir dos dados do ciclo (texto escapado)
        st.iframe(html, height=1100)
    else:
        import streamlit.components.v1 as components

        components.html(html, height=1100, scrolling=True)
