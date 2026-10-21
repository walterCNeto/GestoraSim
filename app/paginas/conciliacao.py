"""Conciliação: contadores por status, quebras com diagnóstico e workflow ABERTA → JUSTIFICADA → RESOLVIDA, cota paralela."""

from __future__ import annotations

import streamlit as st

from app import dados, graficos
from app import formatacao as F
from app.layout import kpis, selo
from gestorasim.conciliacao.workflow import TransicaoInvalida, WorkflowQuebras

TITULO = "Conciliação"
REAL = ("é o fechamento diário com o administrador: o gestor compara sua carteira e sua cota com as do administrador, "
        "investiga cada divergência (quebra), justifica e só então a resolve.")
GLOSSARIO = ("Quebra", "Cota paralela")
FORMATOS_VALOR = {"PRECO": "num", "COTA": "num", "CAIXA": "mil", "PROVISAO_ADM": "num", "ATIVO_FALTANTE": "num"}


def render() -> None:
    dados.exigir("quebras_conciliacao", "cota_paralela", "fundos")
    data = dados.data_global()
    q = dados.tabela("quebras_conciliacao")
    wf = WorkflowQuebras(dados.pasta_gravavel())
    dados.aviso_escrita()
    nomes = dados.fundos()
    st.caption("O arquivo diário do administrador é um XML sintético (`adm_xml/`) com divergências injetadas; cada quebra "
               "traz um diagnóstico automático provável.")

    if q.empty:
        st.success("Nenhuma quebra de conciliação.")
    else:
        q = wf.quebras_com_status(q).assign(fundo=lambda d: d["fundo_cnpj"].map(nomes))
        q = q[q["data"] <= data]
        cont = q["status_workflow"].value_counts()
        k = st.columns(4)
        k[0].metric("Quebras até a data", F.qtd(len(q)))
        for col, s in zip(k[1:], ("ABERTA", "JUSTIFICADA", "RESOLVIDA")):
            col.markdown(f"{selo(s)}", unsafe_allow_html=True)
            col.metric({"ABERTA": "Abertas", "JUSTIFICADA": "Justificadas", "RESOLVIDA": "Resolvidas"}[s], F.qtd(cont.get(s, 0)))
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
        F.exibir(f, {"id_quebra": ("Quebra", "txt"), "data": ("Data", "data"), "fundo": ("Fundo", "txt"), "tipo": ("Tipo", "txt"),
                     "item": ("Item", "txt"), "valor_gestor": ("Valor do gestor", "num"), "valor_adm": ("Valor do administrador", "num"),
                     "diferenca": ("Diferença", "num"), "status_workflow": ("Status", "txt"), "diagnostico": ("Diagnóstico", "txt")},
                 badges=("status_workflow",), altura=320, vazio="Nenhuma quebra com estes filtros.")

        st.subheader("Tratar quebra")
        opcoes = list(f["id_quebra"]) or list(q["id_quebra"])
        if opcoes:
            ident = st.selectbox("Quebra", opcoes, key="conc_id")
            atual = wf.status_atual([ident]).iloc[0]
            linha = q[q["id_quebra"] == ident].iloc[0]
            st.markdown(f"**{ident}** · {linha['tipo']} · {linha['item']} · {F.data(linha['data'])} · status {selo(atual)}",
                        unsafe_allow_html=True)
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
                    F.exibir(h[h["id_quebra"] == ident] if (h["id_quebra"] == ident).any() else h,
                             {c: (c, "txt") for c in h.columns}, badges=("para",))
        else:
            st.info("Nenhuma quebra até esta data.")

    st.subheader("Cota paralela (gestor × administrador)")
    cp = dados.tabela("cota_paralela")
    cnpj = dados.fundo_global()
    d = cp[(cp["fundo_cnpj"] == cnpj) & (cp["data"] <= data)].sort_values("data")
    graficos.mostrar(graficos.linhas([{"nome": "Diferença (bp)", "x": d["data"], "y": d["diferenca_bp"], "cor": F.AZUL, "fmt": F.bp}],
                                     fmt=F.bp, titulo_y="bp", altura=280, tickfmt=".3f"), "conc_cota")
    st.caption("Diferença em bp = cota do administrador / cota do gestor − 1. Picos são as divergências de cota injetadas.")
    F.exibir(d.sort_values("diferenca_bp", key=abs, ascending=False).head(10),
             {"data": ("Data", "data"), "cota_gestor": ("Cota do gestor", "cota"), "cota_adm": ("Cota do administrador", "cota"),
              "diferenca_bp": ("Diferença", "bp")})
