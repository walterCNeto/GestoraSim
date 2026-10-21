"""Boletagem manual: pré-trade ao vivo, envio, override com justificativa, execução e rateio.

As ordens manuais vão para um ledger separado (`*_manuais.parquet`, tag MANUAL) e não alteram a série oficial.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app import dados
from app.formatacao import tres_casas
from gestorasim.boletagem.manual import BoletagemManual


@st.cache_resource(show_spinner="Carregando posições, preços e volumes…")
def _servico(pasta: str, mtime: float) -> BoletagemManual:
    return BoletagemManual(pasta)


def render() -> None:
    dados.exigir("posicoes_diarias", "pl_diario", "cota_diaria", "mtm_painel", "ativos", "fundos", "calendario")
    mtime = (dados.pasta() / "pl_diario.parquet").stat().st_mtime
    bm = _servico(str(dados.pasta()), mtime)
    st.info("Ordens manuais ficam em um ledger **separado** (tag MANUAL): a série oficial não é alterada. "
            "O estado de partida é o fechamento oficial da data escolhida.")

    c1, c2, c3 = st.columns(3)
    data = dados.escolher_data(list(bm.datas), "Data da ordem", "bol_data")
    ticker = c1.selectbox("Ativo", bm.ativos_negociaveis(data), key="bol_ticker")
    lado = c2.radio("Lado", ["C", "V"], horizontal=True, format_func=lambda x: "Compra" if x == "C" else "Venda", key="bol_lado")
    lote = bm.lote(ticker)
    c3.metric("Preço de fechamento (mtm)", f"{bm.preco(data, ticker):.3f}")
    st.caption(f"Lote padrão: {lote}. Quantidade por fundo (0 = não participa):")
    qtds = {}
    cols = st.columns(len(bm.fundos))
    for col, f in zip(cols, bm.fundos):
        qtds[f.cnpj] = col.number_input(f.nome, min_value=0, step=lote, value=0, key=f"bol_q_{f.cnpj}")
    if not any(qtds.values()):
        st.stop()

    st.subheader("Pré-trade ao vivo")
    try:
        pre = bm.pre_trade(data, ticker, lado, qtds)
    except ValueError as e:
        st.error(str(e))
        st.stop()
    violadas = pre[pre["status"] == "VIOLADO"]
    st.dataframe(tres_casas(pre[["fundo", "regra", "detalhe", "valor_pos_ordem", "limite", "folga", "status"]]),
                 hide_index=True, height=260)
    if violadas.empty:
        st.success("Nenhuma regra violada: a ordem passa no pré-trade.")
    else:
        fundos_viol = ", ".join(sorted(set(violadas["fundo"])))
        st.error(f"{len(violadas)} violação(ões) — fundos: {fundos_viol}. Fundos que violam são excluídos da ordem; "
                 "se todos violam, a ordem fica BLOQUEADA (use override com justificativa).")

    st.subheader("Envio")
    usuario = st.text_input("Usuário", value="gestor.manual", key="bol_usuario")
    just = st.text_area("Justificativa do override (obrigatória para liberar ordem bloqueada)", key="bol_just")
    b1, b2 = st.columns(2)
    resultado = None
    if b1.button("Enviar ordem", type="primary"):
        resultado = bm.enviar(data, ticker, lado, qtds, usuario)
    if b2.button("Enviar com override"):
        if len(just.strip()) < 10:
            st.error("Override exige justificativa (mínimo 10 caracteres) e usuário.")
        else:
            resultado = bm.enviar(data, ticker, lado, qtds, usuario, (usuario, just))
    if resultado is not None:
        estado = resultado["estado"]
        (st.success if estado == "ALOCADA" else st.warning)(f"Estado final: {estado}")
        for titulo, chave in (("Trilha de auditoria", "auditoria"), ("Execução (fills)", "execucoes"),
                              ("Rateio entre fundos", "alocacoes")):
            if len(resultado[chave]):
                st.markdown(f"**{titulo}**")
                st.dataframe(tres_casas(resultado[chave]), hide_index=True)

    st.subheader("Ordens manuais gravadas")
    om = dados.tabela_ou_vazia("ordens_manuais")
    st.dataframe(tres_casas(om.sort_values("data", ascending=False)) if len(om) else pd.DataFrame({"": ["(nenhuma)"]}),
                 hide_index=True)
