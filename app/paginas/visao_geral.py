"""Visão geral: um card por fundo (PL, rentabilidade vs benchmark, VaR vs limite, alertas e quebras) e a cota base 100."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app import dados, graficos
from app import formatacao as F
from app.layout import selo, selo_md
from gestorasim.conciliacao.workflow import WorkflowQuebras
from gestorasim.relatorios.lamina import periodos

TITULO = "Visão geral"
REAL = ("é o painel da manhã do gestor e do diretor: quanto cada fundo vale, como rendeu contra o benchmark, "
        "quanto risco está usando do limite e o que está pendente (alertas, quebras com o administrador).")
GLOSSARIO = ("PL líquido", "VaR", "Linha d'água", "Quebra")


def _limite_var(enq: pd.DataFrame, cnpj: str, data: pd.Timestamp) -> float | None:
    e = enq[(enq["fundo_cnpj"] == cnpj) & (enq["regra"] == "politica_var") & (enq["data"] <= data)]
    return None if e.empty else float(e.sort_values("data")["limite"].iloc[-1])


def render() -> None:
    dados.exigir("cota_diaria", "fundos")
    data = dados.data_global()
    cota = dados.ate_data(dados.tabela("cota_diaria"), "data", data)
    risco = dados.ate_data(dados.tabela_ou_vazia("risco_diario"), "data", data) if dados.existe("risco_diario") else pd.DataFrame()
    enq = dados.tabela_ou_vazia("enquadramento_diario")
    alertas = dados.tabela_ou_vazia("alertas")
    q = dados.tabela_ou_vazia("quebras_conciliacao")
    if len(q):
        q = WorkflowQuebras(dados.pasta_gravavel()).quebras_com_status(q)
        q = q[q["data"] <= data]
    nomes = dados.fundos()
    st.caption(f"Dados simulados · posição de **{F.data(data)}**"
               f"{' · dataset demo reduzido' if dados.modo_demo() else ''} · pasta `{dados.pasta()}`")

    cols = st.columns(len(nomes))
    for col, (cnpj, nome) in zip(cols, nomes.items()):
        c = cota[cota["fundo_cnpj"] == cnpj].sort_values("data")
        if c.empty:
            continue
        per = periodos(c, f"{data:%Y-%m}")
        pl, pl_ant = c["pl_liquido"].iloc[-1], c["pl_liquido"].iloc[-2] if len(c) > 1 else None
        r = risco[risco["fundo_cnpj"] == cnpj].sort_values("data") if len(risco) else pd.DataFrame()
        var = r["var_param_pct"].iloc[-1] if len(r) else None
        var_ant = r["var_param_pct"].iloc[-2] if len(r) > 1 else None
        lim = _limite_var(enq, cnpj, data) if len(enq) else None
        hoje = alertas[(alertas["fundo_cnpj"] == cnpj) & (alertas["data"] == data)] if len(alertas) else alertas
        n_soft = int((hoje["nivel"] == "SOFT").sum()) if len(hoje) else 0
        n_hard = int((enq[(enq["fundo_cnpj"] == cnpj) & (enq["data"] == data)]["status"] == "VIOLADO").sum()) if len(enq) else 0
        ab = q[(q["fundo_cnpj"] == cnpj) & (q["status_workflow"] != "RESOLVIDA")] if len(q) else q
        with col, st.container(border=True):
            st.markdown(f"#### {nome}")
            st.metric("PL líquido", F.brl_mi(pl), delta=None if pl_ant is None else f"{F.sinal(F.brl_mi, pl - pl_ant)} vs dia anterior",
                      help="Patrimônio líquido após taxas e resgates a pagar.")
            m, a = per["mês"], per["ano"]
            st.metric("Rentabilidade no mês", F.pct(m[0]), delta=None if m[1] is None else f"{F.sinal(F.pct, m[0] - m[1])} vs benchmark",
                      help="Cota líquida de taxas. Delta: diferença para o benchmark no mesmo período.")
            st.metric("Rentabilidade no ano", F.pct(a[0]), delta=None if a[1] is None else f"{F.sinal(F.pct, a[0] - a[1])} vs benchmark")
            if var is not None and pd.notna(var):
                uso = None if not lim else var / lim
                st.metric("VaR 99% 1d (% do PL)", F.pct(var),
                          delta=None if var_ant is None or pd.isna(var_ant) else F.sinal(F.pct, var - var_ant), delta_color="inverse",
                          help="Em % do PL líquido. Delta: variação sobre o dia anterior.")
                if lim:
                    st.progress(min(uso, 1.0), text=f"{F.pct(uso, 1)} do limite de {F.pct(lim)}")
            else:
                st.metric("VaR 99% 1d (% do PL)", "n/d", help="Período de burn-in: o backtest ainda não tem janela suficiente.")
            st.markdown(f"Alertas: {selo('HARD') if n_hard else selo('OK')} {n_hard} HARD · "
                        f"{selo('SOFT') if n_soft else selo('OK')} {n_soft} SOFT<br>"
                        f"Quebras abertas: {selo('ABERTA') if len(ab) else selo('OK')} {len(ab)}", unsafe_allow_html=True)

    st.subheader("Cota líquida vs benchmark (base 100 no início do período)")
    gcols = st.columns(len(nomes))
    for gcol, (cnpj, nome) in zip(gcols, nomes.items()):
        c = cota[cota["fundo_cnpj"] == cnpj].sort_values("data")
        if c.empty:
            continue
        with gcol:
            st.markdown(f"**{nome}**")
            fig = graficos.linhas([
                {"nome": "Cota líquida", "x": c["data"], "y": graficos.serie_base100(c, "cota_liquida"), "cor": F.AZUL},
                {"nome": "Benchmark", "x": c["data"], "y": graficos.serie_base100(c, "benchmark"), "cor": F.CINZA, "dash": "dot"}],
                altura=260)
            graficos.mostrar(fig, f"vg_cota_{cnpj}")

    esq, dir_ = st.columns(2)
    with esq:
        st.subheader("Alertas do dia")
        if alertas.empty:
            st.info("Sem alertas.")
        else:
            hoje = alertas[alertas["data"] == data].assign(fundo=lambda d: d["fundo_cnpj"].map(nomes))
            F.exibir(hoje.sort_values("utilizacao", ascending=False).head(15),
                     {"fundo": ("Fundo", "txt"), "regra": ("Regra", "txt"), "detalhe": ("Detalhe", "txt"), "nivel": ("Nível", "txt"),
                      "utilizacao": ("Uso do limite", "pct"), "valor": ("Valor", "num"), "limite": ("Limite", "num")},
                     badges=("nivel",), vazio="Nenhum alerta nesta data.")
    with dir_:
        st.subheader("Quebras de conciliação")
        if q.empty:
            st.info("Sem quebras até esta data.")
        else:
            res = q.groupby(["tipo", "status_workflow"]).size().unstack(fill_value=0)
            res.columns.name = None
            res = res.reset_index()
            for s in ("ABERTA", "JUSTIFICADA", "RESOLVIDA"):
                if s not in res:
                    res[s] = 0
            F.exibir(res, {"tipo": ("Tipo", "txt"), "ABERTA": ("Abertas", "qtd"), "JUSTIFICADA": ("Justificadas", "qtd"),
                           "RESOLVIDA": ("Resolvidas", "qtd")})

    log = dados.tabela_ou_vazia("execucoes_ciclo")
    with st.expander("Último ciclo do orquestrador"):
        if log.empty:
            st.info("Sem log de execução (rode o orquestrador).")
        else:
            d = log[log["data"] == min(data, log["data"].max())]
            c1, c2, c3 = st.columns(3)
            c1.metric("Etapas OK", F.qtd((d["status"] == "OK").sum()))
            c2.metric("Puladas / erros", f"{F.qtd((d['status'] == 'PULADA').sum())} / {F.qtd((d['status'] == 'ERRO').sum())}")
            c3.metric("Duração (s)", F.num(float(d["duracao_s"].sum())))
            F.exibir(d, {"ordem": ("Ordem", "qtd"), "etapa": ("Etapa", "txt"), "status": ("Status", "txt"),
                         "duracao_s": ("Duração (s)", "num"), "hash_entrada": ("Hash da entrada", "txt"), "mensagem": ("Mensagem", "txt")},
                     badges=("status",))
            meta = dados.meta_ciclo()
            if meta:
                st.caption(f"Ciclo gravado em {meta.get('gravado_em', '?')} · {meta.get('primeiro_pregao')} a {meta.get('ultimo_pregao')}")
