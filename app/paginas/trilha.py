"""Trilha de auditoria: linha do tempo de uma ordem (estados como etapas) e busca em toda a trilha."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app import dados
from app import formatacao as F
from app.layout import kpis, stepper

TITULO = "Trilha de auditoria"
REAL = ("é a memória regulatória da mesa: para qualquer ordem, quem fez o quê e quando, por que foi bloqueada ou liberada "
        "e como foi rateada. É o que o auditor e o compliance pedem em uma inspeção.")
GLOSSARIO = ("Pré-trade", "Override", "Rateio")
SPEC = {"data_serie": ("Data", "data"), "seq": ("Seq", "qtd"), "timestamp": ("Timestamp", "txt"), "ordem_id": ("Ordem", "txt"),
        "origem": ("Origem", "txt"), "ticker": ("Ativo", "txt"), "lado": ("Lado", "txt"), "usuario": ("Usuário", "txt"),
        "tipo": ("Tipo", "txt"), "de": ("De", "txt"), "para": ("Para", "txt"), "motivo": ("Motivo", "txt")}


def render() -> None:
    dados.exigir("auditoria_serie", "ordens_serie", "fundos")
    aud, ordens = dados.tabela("auditoria_serie"), dados.tabela("ordens_serie")
    nomes = dados.fundos()
    data = dados.data_global()
    st.caption("Trilha append-only: cada transição de estado (e eventos como EXCLUIDO_PRE_TRADE) com timestamp lógico, "
               "usuário, de→para e motivo (inclui o critério de rateio).")
    if aud.empty:
        st.info("Sem eventos de auditoria.")
        st.stop()
    if "fundos" not in ordens:  # dados gravados antes do bloco 6
        ordens = ordens.assign(fundos="")
    aud = aud.merge(ordens[["data_serie", "ordem_id", "origem", "fundos", "ticker", "lado", "estado"]].drop_duplicates(["data_serie", "ordem_id"]),
                    on=["data_serie", "ordem_id"], how="left")

    st.subheader(f"Linha do tempo de uma ordem — {F.data(data)}")
    do_dia = ordens[ordens["data_serie"] == data].sort_values("ordem_id")
    if do_dia.empty:
        st.info("Não há ordens nesta data (escolha outro pregão na barra lateral).")
    else:
        do_dia = do_dia.assign(rotulo=lambda d: [f"{o} · {t} ({'compra' if lado == 'C' else 'venda'}) · {e} · {org}" for o, t, lado, e, org
                                                 in zip(d["ordem_id"], d["ticker"], d["lado"], d["estado"], d["origem"])])
        bloq = do_dia[do_dia["estado"] == "BLOQUEADA"]
        padrao = 0 if bloq.empty else list(do_dia["ordem_id"]).index(bloq["ordem_id"].iloc[0])
        escolha = st.selectbox("Ordem do dia", list(do_dia["rotulo"]), index=padrao, key="tr_sel")
        o = do_dia[do_dia["rotulo"] == escolha].iloc[0]
        ev = aud[(aud["data_serie"] == data) & (aud["ordem_id"] == o["ordem_id"])].sort_values("seq" if "seq" in aud else "timestamp")
        kpis([("Estado final", str(o["estado"]), None, "off"), ("Quantidade alvo", F.qtd(o["quantidade_alvo"]), None, "off"),
              ("Quantidade executada", F.qtd(o["quantidade_executada"]), None, "off"),
              ("Fundos (excluídos no pré-trade)", f"{F.qtd(o['n_fundos'])} ({F.qtd(o['n_fundos_excluidos'])})", None, "off")])
        trans = ev[ev["tipo"] == "TRANSICAO"]
        if len(trans):
            stepper([(r.para, f"{r.usuario} · {r.motivo}") for r in trans.itertuples()])
        extra = ev[ev["tipo"] != "TRANSICAO"]
        if len(extra):
            st.markdown("**Outros eventos**")
            F.exibir(extra, SPEC, badges=("tipo", "para"))
        with st.expander("Eventos da ordem (tabela)"):
            F.exibir(ev, SPEC, badges=("para",))

    st.subheader("Busca em toda a trilha")
    c1, c2, c3, c4 = st.columns(4)
    ordem = c1.text_input("Ordem (ex.: ORD0007)", key="tr_ordem").strip()
    fundo = c2.selectbox("Fundo", ["(todos)", *nomes.values()], key="tr_fundo")
    tipo = c3.selectbox("Tipo de evento", ["(todos)", *sorted(aud["tipo"].unique())], key="tr_tipo")
    origem = c4.selectbox("Origem da ordem", ["(todas)", *sorted(aud["origem"].dropna().str.split(":").str[0].unique())], key="tr_origem")
    d1, d2 = st.columns(2)
    datas = sorted(aud["data_serie"].unique())
    ini = d1.selectbox("De", datas, index=0, key="tr_ini", format_func=F.data)
    fim = d2.selectbox("Até", datas, index=len(datas) - 1, key="tr_fim", format_func=F.data)

    f = aud[(aud["data_serie"] >= ini) & (aud["data_serie"] <= fim)]
    if ordem:
        f = f[f["ordem_id"] == ordem]
    if fundo != "(todos)":
        cnpj = next(c for c, n in nomes.items() if n == fundo)
        f = f[f["fundos"].fillna("").str.contains(cnpj, regex=False)]
    if tipo != "(todos)":
        f = f[f["tipo"] == tipo]
    if origem != "(todas)":
        f = f[f["origem"].fillna("").str.startswith(origem)]
    st.metric("Eventos", F.qtd(len(f)))
    ordem_cols = ["data_serie", "seq"] if "seq" in f else ["data_serie"]
    F.exibir(f.sort_values(ordem_cols), SPEC, badges=("para",), altura=420)
    st.subheader("Eventos por tipo e destino")
    resumo = f.groupby(["tipo", "para"]).size().rename("n").reset_index()
    F.exibir(resumo, {"tipo": ("Tipo", "txt"), "para": ("Destino", "txt"), "n": ("Eventos", "qtd")}, badges=("para",))
