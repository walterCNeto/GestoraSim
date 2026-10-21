"""Enquadramento: semáforo regra × fundo no dia, linha do tempo dos episódios (ATIVO/PASSIVO/VENCIDO) e uso dos limites."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import dados, graficos
from app import formatacao as F
from app.layout import kpis

TITULO = "Enquadramento"
REAL = ("é o compliance do gestor: confere todo dia se cada fundo respeita o regulamento (limites por classe, emissor, "
        "rating, liquidez, alavancagem, VaR) e separa o desenquadramento causado por ordem do causado por mercado.")
GLOSSARIO = ("Desenquadramento ativo", "Desenquadramento passivo", "VaR", "Stress")


def _semaforo(d: pd.DataFrame, nomes: dict[str, str]) -> None:
    """Tabela regra × fundo: pior utilização do dia, colorida pelo pior status."""
    d = d.assign(fundo=d["fundo_cnpj"].map(nomes))
    ordem = {"VIOLADO": 2, "SOFT": 1}
    d = d.assign(grau=[ordem.get("VIOLADO" if s == "VIOLADO" else a, 0) for s, a in zip(d["status"], d["alerta"])])
    linhas, cores = {}, {}
    for (regra, fundo), g in d.groupby(["regra", "fundo"]):
        pior = g.sort_values(["grau", "utilizacao"], ascending=False).iloc[0]
        txt = F.pct(pior["utilizacao"], 1) if pd.notna(pior["utilizacao"]) else "OK"
        if pior["grau"] == 2:
            txt = f"VIOLADO · {txt}"
        linhas.setdefault(regra, {})[fundo] = txt
        cores.setdefault(regra, {})[fundo] = {2: "VIOLADO", 1: "SOFT", 0: "OK"}[int(pior["grau"])]
    txt = pd.DataFrame(linhas).T.reindex(columns=list(nomes.values())).fillna("—")
    cor = pd.DataFrame(cores).T.reindex(index=txt.index, columns=txt.columns)
    sty = txt.style.apply(lambda _: cor.map(lambda s: "" if pd.isna(s) else
                                            f"background-color:{F.FUNDO_CLARO[F.cor_status(s)]};color:{F.cor_status(s)};font-weight:600"),
                          axis=None).format(na_rep="—")
    st.dataframe(sty, height=38 + 35 * len(txt))


def _timeline(ep: pd.DataFrame, data: pd.Timestamp, nomes: dict[str, str]) -> None:
    fig = go.Figure()
    um_dia = pd.Timedelta(days=1)
    ep = ep.assign(estado=lambda d: ["VENCIDO" if v else c for v, c in zip(d["vencido"], d["classe"])])
    for estado, cor in (("PASSIVO", F.AMBAR), ("ATIVO", F.VERMELHO), ("VENCIDO", "#7b1010")):
        g = ep[ep["estado"] == estado]
        if g.empty:
            continue
        fim = g["fim"].fillna(data).where(g["fim"].fillna(data) > g["inicio"], g["inicio"] + um_dia)
        fig.add_trace(go.Bar(
            y=[f"{nomes[c].split()[-1]} · {r}" for c, r in zip(g["fundo_cnpj"], g["regra"])], base=g["inicio"], x=(fim - g["inicio"]),
            orientation="h", name=estado, marker_color=cor, width=0.5,
            customdata=[f"{estado} · {F.data(i)} a {F.data(f) if pd.notna(f) else 'em aberto'} · {F.qtd(du)} du · ordem principal: {o or '—'}"
                        for i, f, du, o in zip(g["inicio"], g["fim"], g["du_max"], g["ordem_principal"])],
            hovertemplate="%{y}<br>%{customdata}<extra></extra>"))
    fig.update_layout(height=max(200, 60 + 38 * ep.assign(k=ep["fundo_cnpj"] + ep["regra"])["k"].nunique()), barmode="overlay",
                      margin=dict(l=10, r=10, t=30, b=10), template="plotly_white", separators=",.",
                      legend=dict(orientation="h", y=1.15), hovermode="closest")
    fig.update_xaxes(type="date", tickformat="%d/%m", range=[ep["inicio"].min() - pd.Timedelta(days=10), data + pd.Timedelta(days=5)])
    fig.update_yaxes(autorange="reversed")
    graficos.mostrar(fig, "enq_timeline")


def render() -> None:
    dados.exigir("enquadramento_diario", "episodios", "fundos")
    ep, diario = dados.tabela("episodios"), dados.tabela("enquadramento_diario")
    nomes = dados.fundos()
    data, cnpj = dados.data_global(), dados.fundo_global()
    hoje = diario[diario["data"] == data]
    if hoje.empty:
        st.info("Sem enquadramento calculado nesta data.")
        st.stop()
    ep = ep[ep["inicio"] <= data] if len(ep) else ep
    aberto = ep[ep["fim"].isna() | (ep["fim"] > data)] if len(ep) else ep

    kpis([("Regras violadas hoje", F.qtd((hoje["status"] == "VIOLADO").sum()), None, "off"),
          ("Alertas SOFT hoje", F.qtd((hoje["alerta"] == "SOFT").sum()), None, "off", "Uso do limite acima do patamar de alerta."),
          ("Episódios ATIVOS (até hoje)", F.qtd((ep["classe"] == "ATIVO").sum()) if len(ep) else "0", None, "off"),
          ("Episódios PASSIVOS (até hoje)", F.qtd((ep["classe"] == "PASSIVO").sum()) if len(ep) else "0", None, "off"),
          ("Em aberto / vencidos", f"{F.qtd(len(aberto))} / {F.qtd(ep['vencido'].sum()) if len(ep) else 0}", None, "off")])

    st.subheader(f"Semáforo por regra × fundo — {F.data(data)}")
    _semaforo(hoje, nomes)
    st.caption("Verde: dentro do limite. Âmbar: alerta SOFT (uso alto do limite). Vermelho: violado. "
               "Percentual = uso do limite (pior detalhe da regra).")

    st.subheader("Episódios de desenquadramento no tempo")
    if ep.empty:
        st.success("Nenhum episódio de desenquadramento até esta data.")
    else:
        _timeline(ep, data, nomes)
        ep = ep.assign(fundo=ep["fundo_cnpj"].map(nomes),
                       situacao=lambda d: ["EM ABERTO" if pd.isna(f) else "REENQUADRADO" for f in d["fim"]],
                       estado=lambda d: ["VENCIDO" if v else c for v, c in zip(d["vencido"], d["classe"])])
        F.exibir(ep.sort_values("inicio", ascending=False),
                 {"fundo": ("Fundo", "txt"), "regra": ("Regra", "txt"), "detalhe": ("Detalhe", "txt"), "estado": ("Classe", "txt"),
                  "inicio": ("Início", "data"), "fim": ("Fim", "data"), "situacao": ("Situação", "txt"), "du_max": ("Dias úteis", "qtd"),
                  "ordem_principal": ("Ordem principal", "txt"), "n_ordens": ("Ordens do dia", "qtd")},
                 badges=("estado", "situacao"))
        st.caption("ATIVO: causado pelas ordens do dia (a carteira sem elas estaria enquadrada); a **ordem principal** é a de maior "
                   "contribuição à métrica da regra (leave-one-out). PASSIVO: mercado ou eventos; conta prazo e vira VENCIDO se estourar.")

    st.subheader(f"Uso dos limites no tempo — {nomes[cnpj]}")
    regras = sorted(diario.loc[diario["fundo_cnpj"] == cnpj, "regra"].unique())
    c1, c2 = st.columns(2)
    regra = c1.selectbox("Regra", regras, index=regras.index("politica_var") if "politica_var" in regras else 0, key="enq_regra")
    d = diario[(diario["fundo_cnpj"] == cnpj) & (diario["regra"] == regra) & (diario["data"] <= data)]
    if d["detalhe"].nunique() > 1:
        det = c2.selectbox("Detalhe", sorted(d["detalhe"].unique()), key="enq_detalhe")
        d = d[d["detalhe"] == det]
    d = d.sort_values("data")
    if d.empty:
        st.info("Sem série para esta regra.")
    else:
        pc = d["limite"].abs().max() <= 1.5  # regras em fração do PL -> percentual
        fmt = F.pct if pc else F.brl_mi
        graficos.mostrar(graficos.linhas([
            {"nome": "Valor", "x": d["data"], "y": d["valor"], "cor": F.AZUL},
            {"nome": "Limite", "x": d["data"], "y": d["limite"], "cor": F.VERMELHO, "dash": "dash"}], fmt=fmt,
            tickfmt=".1%" if pc else None), "enq_uso")
        d = d.assign(uso=lambda x: x["utilizacao"])
        F.exibir(d.tail(15).sort_values("data", ascending=False),
                 {"data": ("Data", "data"), "valor": ("Valor", "pct" if pc else "num"), "limite": ("Limite", "pct" if pc else "num"),
                  "folga": ("Folga", "pct" if pc else "num"), "uso": ("Uso do limite", "pct"), "status": ("Status", "txt"),
                  "alerta": ("Alerta", "txt"), "classificacao": ("Classe", "txt"), "du_desde_inicio": ("Dias úteis", "qtd")},
                 badges=("status", "alerta", "classificacao"))
