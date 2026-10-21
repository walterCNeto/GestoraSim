"""Passivo e cota: cota bruta, líquida e linha d'água; aplicações/resgates; taxas; come-cotas e IR."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app import dados, graficos
from app import formatacao as F
from app.layout import kpis

TITULO = "Passivo e cota"
REAL = ("é o lado do cotista: aplicações e resgates convertidos em cotas, cálculo da cota líquida (taxa de administração, "
        "performance com linha d'água) e o imposto retido, inclusive o come-cotas semestral.")
GLOSSARIO = ("PL líquido", "Linha d'água", "Come-cotas")


def render() -> None:
    dados.exigir("cota_diaria", "cotistas_mov", "provisoes", "fundos")
    cnpj, data = dados.fundo_global(), dados.data_global()
    cota, mov, prov = dados.tabela("cota_diaria"), dados.tabela("cotistas_mov"), dados.tabela("provisoes")
    trib = dados.tabela_ou_vazia("tributos")
    c = cota[(cota["fundo_cnpj"] == cnpj) & (cota["data"] <= data)].sort_values("data").set_index("data")
    u, a = c.iloc[-1], (c.iloc[-2] if len(c) > 1 else None)
    d = lambda col, fn: None if a is None else F.sinal(fn, u[col] - a[col])  # noqa: E731
    kpis([("PL líquido", F.brl_mi(u["pl_liquido"]), d("pl_liquido", F.brl_mi)),
          ("Cotas (qtd)", F.qtd(u["cotas"]), d("cotas", F.qtd)),
          ("Cota bruta", F.cota(u["cota_bruta"]), d("cota_bruta", F.cota), "normal", "Antes da provisão de performance."),
          ("Cota líquida", F.cota(u["cota_liquida"]), d("cota_liquida", F.cota)),
          ("Linha d'água", F.cota(u["linha_dagua"]), None, "off", "Maior cota líquida já paga ao cotista (high-water mark).")],
         por_linha=3)
    st.caption("Cota bruta = antes da provisão de performance. PL líquido = ativos − taxas a pagar − resgates a pagar; "
               "cota = PL líquido / cotas.")

    st.subheader("Cota bruta, líquida e linha d'água")
    x = c.index
    graficos.mostrar(graficos.linhas([
        {"nome": "Cota bruta", "x": x, "y": c["cota_bruta"], "cor": "#0f8b8d"},
        {"nome": "Cota líquida", "x": x, "y": c["cota_liquida"], "cor": F.AZUL},
        {"nome": "Linha d'água", "x": x, "y": c["linha_dagua"], "cor": F.AMBAR, "dash": "dash"}], fmt=F.cota,
        tickfmt=".4f"), "pas_cota")

    st.subheader("Aplicações e resgates (R$ mi)")
    mv = c[["aplicacoes", "resgates_liquidos"]].copy()
    mv = mv[(mv["aplicacoes"].abs() > 0) | (mv["resgates_liquidos"].abs() > 0)]
    if mv.empty:
        st.info("Sem movimentações até esta data.")
    else:
        fm = lambda v: f"R$ {F.num(v)} mi"  # noqa: E731
        graficos.mostrar(graficos.barras_empilhadas(
            mv.index, [{"nome": "Aplicações", "y": mv["aplicacoes"].abs() / 1e6, "cor": F.VERDE},
                       {"nome": "Resgates", "y": -mv["resgates_liquidos"].abs() / 1e6, "cor": F.VERMELHO}],
            fmt=fm, titulo_y="R$ mi"), "pas_mov")

    st.subheader("Taxas e provisões")
    p = prov[(prov["fundo_cnpj"] == cnpj) & (prov["data"] <= data)].sort_values("data").set_index("data")
    kpis([("Administração provisionada", F.brl_mi(p["adm_dia"].sum())), ("Administração paga", F.brl_mi(p["adm_pago"].sum())),
          ("Performance paga", F.brl_mi(p["perf_pago"].sum()))])
    graficos.mostrar(graficos.areas(p.index, [{"nome": "Administração a pagar", "y": p["adm_a_pagar"] / 1e6},
                                              {"nome": "Performance (saldo)", "y": p["perf_saldo"] / 1e6},
                                              {"nome": "Resgates a pagar", "y": p["resgates_a_pagar"] / 1e6}],
                                     fmt=lambda v: f"R$ {F.num(v)} mi", titulo_y="R$ mi", altura=260), "pas_prov")

    st.subheader("Come-cotas e IR")
    if trib.empty:
        st.info("Sem tributos no período.")
    else:
        t = trib[(trib["fundo_cnpj"] == cnpj) & (trib["data"] <= data)]
        if t.empty:
            st.info("Sem tributos até esta data.")
        else:
            resumo = t.groupby("tipo").agg(valor=("valor", "sum"), lancamentos=("valor", "size"),
                                           primeira=("data", "min"), ultima=("data", "max")).reset_index()
            F.exibir(resumo, {"tipo": ("Tributo", "txt"), "valor": ("Total retido", "mil"), "lancamentos": ("Lançamentos", "qtd"),
                              "primeira": ("Primeira data", "data"), "ultima": ("Última data", "data")})
            cc = t[t["tipo"] == "COME_COTAS"]
            if len(cc):
                st.markdown(f"**Come-cotas em {', '.join(F.data(x) for x in sorted(cc['data'].unique()))}** — "
                            f"{F.qtd(len(cc))} lotes, total {F.brl_mi(cc['valor'].sum())}; cotas canceladas: "
                            f"{F.num(cc['cotas_canceladas'].sum())}.")
            with st.expander("Lançamentos por lote/cotista"):
                F.exibir(t.sort_values(["data", "tipo"], ascending=[False, True]),
                         {"data": ("Data", "data"), "tipo": ("Tributo", "txt"), "cotista": ("Cotista", "txt"), "lote_id": ("Lote", "txt"),
                          "base": ("Base de cálculo", "num"), "aliquota": ("Alíquota", "pct"), "valor": ("Valor retido", "num"),
                          "cotas_canceladas": ("Cotas canceladas", "num")}, altura=300)

    st.subheader("Movimentações dos cotistas")
    m = mov[mov["fundo_cnpj"] == cnpj]
    m = m[m["data_solicitacao"] <= data]
    f1, f2 = st.columns(2)
    tipo = f1.selectbox("Tipo", ["(todos)", "APLICACAO", "RESGATE"], key="pas_tipo")
    status = f2.selectbox("Status", ["(todos)", *sorted(m["status"].unique())], key="pas_status")
    if tipo != "(todos)":
        m = m[m["tipo"] == tipo]
    if status != "(todos)":
        m = m[m["status"] == status]
    F.exibir(m.sort_values("data_solicitacao", ascending=False),
             {"id_mov": ("Mov.", "txt"), "id_cotista": ("Cotista", "txt"), "tipo": ("Tipo", "txt"), "data_solicitacao": ("Solicitação", "data"),
              "data_conversao": ("Conversão", "data"), "data_pagamento": ("Pagamento", "data"), "valor_solicitado": ("Valor solicitado", "mil"),
              "cota_conversao": ("Cota de conversão", "cota"), "cotas": ("Cotas", "num"), "valor_bruto": ("Valor bruto", "mil"),
              "ir": ("IR", "mil"), "valor_liquido": ("Valor líquido", "mil"), "status": ("Status", "txt")},
             badges=("status",), altura=300)
    lotes = dados.tabela_ou_vazia("lotes_cotistas")
    if len(lotes):
        with st.expander("Lotes de cotas (FIFO) na última data"):
            F.exibir(lotes[lotes["fundo_cnpj"] == cnpj].sort_values("data_aplicacao"),
                     {"lote_id": ("Lote", "txt"), "cotista": ("Cotista", "txt"), "data_aplicacao": ("Aplicação", "data"), "cotas": ("Cotas", "num"),
                      "cota_aplicacao": ("Cota de aplicação", "cota"), "capital": ("Capital", "mil"), "come_cotas_pago": ("Come-cotas pago", "mil"),
                      "valor_atual": ("Valor atual", "mil")})
