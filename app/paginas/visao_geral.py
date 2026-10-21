"""Visão geral: PL líquido, cota vs benchmark, alertas e quebras abertas, status do último ciclo."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app import dados
from app.formatacao import tres_casas
from gestorasim.conciliacao.workflow import WorkflowQuebras
from gestorasim.util import fmt3


def render() -> None:
    dados.exigir("cota_diaria", "fundos")
    cota = dados.tabela("cota_diaria")
    ultima = cota["data"].max()
    st.caption(f"Dados simulados · pasta `{dados.pasta()}`{' (dataset demo reduzido)' if dados.modo_demo() else ''}"
               f" · último pregão {ultima:%Y-%m-%d}")

    cols = st.columns(len(dados.fundos()))
    for col, (cnpj, nome) in zip(cols, dados.fundos().items()):
        c = cota[cota["fundo_cnpj"] == cnpj].sort_values("data")
        ret = c["cota_liquida"].iloc[-1] / c["cota_liquida"].iloc[0] - 1
        bench = c["benchmark"].iloc[-1] / c["benchmark"].iloc[0] - 1
        with col:
            st.subheader(nome)
            st.metric("PL líquido (R$)", fmt3(c["pl_liquido"].iloc[-1]))
            st.metric("Cota líquida", fmt3(c["cota_liquida"].iloc[-1]), delta=fmt3(ret))
            st.metric("Benchmark no período", fmt3(bench), delta=fmt3(ret - bench), help="Diferença: cota líquida − benchmark")

    st.subheader("Cota líquida vs benchmark (base 1,000 no início)")
    base = cota.sort_values("data").assign(
        cota=lambda d: d["cota_liquida"] / d.groupby("fundo_cnpj")["cota_liquida"].transform("first"),
        bench=lambda d: d["benchmark"] / d.groupby("fundo_cnpj")["benchmark"].transform("first"))
    graf = pd.concat([base.pivot(index="data", columns="fundo", values="cota").add_suffix(" — cota"),
                      base.pivot(index="data", columns="fundo", values="bench").add_suffix(" — benchmark")], axis=1)
    st.line_chart(graf)

    esq, dir_ = st.columns(2)
    with esq:
        st.subheader("Alertas abertos (último pregão)")
        al = dados.tabela_ou_vazia("alertas")
        if al.empty:
            st.info("Sem alertas.")
        else:
            hoje = al[al["data"] == al["data"].max()]
            resumo = hoje.groupby(["fundo_cnpj", "nivel"]).size().unstack(fill_value=0).rename(index=dados.fundos())
            st.dataframe(resumo)
            st.dataframe(tres_casas(hoje.sort_values(["nivel", "utilizacao"], ascending=[True, False])
                                    [["fundo_cnpj", "regra", "detalhe", "nivel", "utilizacao", "valor", "limite"]].head(15)),
                         hide_index=True)
    with dir_:
        st.subheader("Quebras de conciliação abertas")
        q = dados.tabela_ou_vazia("quebras_conciliacao")
        if q.empty:
            st.info("Sem quebras.")
        else:
            q = WorkflowQuebras(dados.pasta_gravavel()).quebras_com_status(q)
            abertas = q[q["status_workflow"] != "RESOLVIDA"]
            st.metric("Abertas ou justificadas", len(abertas), delta=f"{len(q) - len(abertas)} resolvidas")
            st.dataframe(abertas.groupby(["tipo", "status_workflow"]).size().unstack(fill_value=0))

    st.subheader("Último ciclo")
    log = dados.tabela_ou_vazia("execucoes_ciclo")
    if log.empty:
        st.info("Sem log de execução (rode o orquestrador).")
    else:
        d = log[log["data"] == log["data"].max()]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Pregão", f"{d['data'].iloc[0]:%Y-%m-%d}")
        c2.metric("Etapas OK", int((d["status"] == "OK").sum()))
        c3.metric("Puladas / erros", f"{int((d['status'] == 'PULADA').sum())} / {int((d['status'] == 'ERRO').sum())}")
        c4.metric("Duração (s)", fmt3(float(d["duracao_s"].sum())))
        st.dataframe(tres_casas(d[["ordem", "etapa", "status", "duracao_s", "hash_entrada", "mensagem"]]), hide_index=True)
        meta = dados.meta_ciclo()
        if meta:
            st.caption(f"Ciclo gravado em {meta.get('gravado_em', '?')} · {meta.get('primeiro_pregao')} a {meta.get('ultimo_pregao')}")
