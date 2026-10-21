"""Lâmina do fundo na data escolhida (HTML para download)."""

from __future__ import annotations

import streamlit as st

from app import dados
from app.lamina_html import gerar_lamina_html

TITULO = "Lâmina mensal"
REAL = ("é o material que o gestor envia a cotistas e distribuidores todo mês: rentabilidade contra o benchmark, "
        "composição, risco e maiores posições em uma página.")
GLOSSARIO = ("PL líquido", "VaR", "ES", "Stress")


def render() -> None:
    dados.exigir("cota_diaria", "mtm_diario", "ativos", "fundos")
    data, cnpj = dados.data_global(), dados.fundo_global()
    cut = lambda t: dados.ate_data(t, "data", data) if len(t) else t  # noqa: E731
    res = {"cota_diaria": cut(dados.tabela("cota_diaria")), "mtm_diario": cut(dados.tabela("mtm_diario")),
           "ativos": dados.tabela("ativos"), "risco_diario": cut(dados.tabela_ou_vazia("risco_diario")),
           "enquadramento_diario": cut(dados.tabela_ou_vazia("enquadramento_diario"))}
    classes = dict(zip(dados.tabela("fundos")["cnpj"], dados.tabela("fundos")["classe"]))
    html = gerar_lamina_html(res, cnpj, data, classes.get(cnpj, ""))
    st.caption("A lâmina usa o fundo e a data da barra lateral (fechamento do mês até essa data).")
    st.download_button("Baixar lâmina (HTML)", data=html.encode("utf-8"), file_name=f"lamina_{data:%Y-%m-%d}_{cnpj[7:10]}.html",
                       mime="text/html")
    if hasattr(st, "iframe"):  # HTML gerado por nós a partir dos dados do ciclo (texto escapado)
        st.iframe(html, height=1250)
    else:
        import streamlit.components.v1 as components

        components.html(html, height=1250, scrolling=True)
