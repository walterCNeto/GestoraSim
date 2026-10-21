"""Conciliação: quebras com diagnóstico, workflow ABERTA → JUSTIFICADA → RESOLVIDA e cota paralela."""

from __future__ import annotations

import streamlit as st

from app import dados
from app.formatacao import tres_casas
from gestorasim.conciliacao.workflow import TransicaoInvalida, WorkflowQuebras


def render() -> None:
    dados.exigir("quebras_conciliacao", "cota_paralela", "fundos")
    q = dados.tabela("quebras_conciliacao")
    wf = WorkflowQuebras(dados.pasta_gravavel())
    dados.aviso_escrita()
    nomes = dados.fundos()
    st.caption("O arquivo diário do administrador é um XML sintético (`adm_xml/`) com divergências injetadas; cada quebra "
               "traz um diagnóstico automático provável. Status do workflow: ABERTA → JUSTIFICADA → RESOLVIDA.")

    if q.empty:
        st.success("Nenhuma quebra de conciliação.")
    else:
        q = wf.quebras_com_status(q).assign(fundo=lambda d: d["fundo_cnpj"].map(nomes))
        c1, c2, c3 = st.columns(3)
        status = c1.selectbox("Status do workflow", ["(todos)", "ABERTA", "JUSTIFICADA", "RESOLVIDA"], key="conc_status")
        tipo = c2.selectbox("Tipo", ["(todos)", *sorted(q["tipo"].unique())], key="conc_tipo")
        fundo = c3.selectbox("Fundo", ["(todos)", *nomes.values()], key="conc_fundo")
        f = q
        if status != "(todos)":
            f = f[f["status_workflow"] == status]
        if tipo != "(todos)":
            f = f[f["tipo"] == tipo]
        if fundo != "(todos)":
            f = f[f["fundo"] == fundo]
        m1, m2, m3 = st.columns(3)
        m1.metric("Quebras", len(q))
        m2.metric("Abertas", int((q["status_workflow"] == "ABERTA").sum()))
        m3.metric("Resolvidas", int((q["status_workflow"] == "RESOLVIDA").sum()))
        st.dataframe(tres_casas(f[["id_quebra", "data", "fundo", "tipo", "item", "valor_gestor", "valor_adm", "diferenca",
                                   "status_workflow", "diagnostico"]]), hide_index=True, height=320)

        st.subheader("Tratar quebra")
        ident = st.selectbox("Quebra", list(f["id_quebra"]) or list(q["id_quebra"]), key="conc_id")
        atual = wf.status_atual([ident]).iloc[0]
        linha = q[q["id_quebra"] == ident].iloc[0]
        st.write(f"**{ident}** · {linha['tipo']} · {linha['item']} · status **{atual}**")
        st.info(f"Diagnóstico: {linha['diagnostico']}")
        usuario = st.text_input("Usuário", key="conc_usuario")
        comentario = st.text_area("Comentário", key="conc_comentario")
        b1, b2, b3 = st.columns(3)
        for botao, para in ((b1, "JUSTIFICADA"), (b2, "RESOLVIDA"), (b3, "ABERTA")):
            if botao.button({"JUSTIFICADA": "Justificar", "RESOLVIDA": "Resolver", "ABERTA": "Reabrir"}[para], key=f"conc_{para}"):
                try:
                    wf.transicionar(ident, para, usuario, comentario, q["id_quebra"])
                    st.success(f"{ident}: {atual} → {para}")
                    st.rerun()
                except TransicaoInvalida as e:
                    st.error(str(e))
        h = wf.historico()
        if len(h):
            with st.expander("Histórico do workflow"):
                st.dataframe(h[h["id_quebra"] == ident] if (h["id_quebra"] == ident).any() else h, hide_index=True)

    st.subheader("Cota paralela (gestor × administrador)")
    cp = dados.tabela("cota_paralela")
    cnpj = dados.escolher_fundo(chave="conc_fundo_cota")
    d = cp[cp["fundo_cnpj"] == cnpj].sort_values("data").set_index("data")
    st.line_chart(d[["diferenca_bp"]])
    st.caption("Diferença em bp = cota do administrador / cota do gestor − 1. Picos são as divergências de cota injetadas.")
    st.dataframe(tres_casas(d.reset_index().sort_values("diferenca_bp", key=abs, ascending=False).head(10)), hide_index=True)
