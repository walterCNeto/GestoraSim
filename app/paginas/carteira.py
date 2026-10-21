"""Carteira: posições por classe com subtotal e % do PL, treemap classe/emissor e evolução por classe."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import dados, graficos
from app import formatacao as F
from app.layout import kpis

TITULO = "Carteira"
REAL = ("é o 'o que eu tenho hoje' do gestor: posição, marcação a mercado, concentração por classe e emissor e quanto "
        "tempo levaria para vender cada posição (liquidez).")
GLOSSARIO = ("PL líquido", "Dias para zerar")
ROTULO_CLASSE = {"ACAO": "Ações", "TITULO_PUBLICO": "Títulos públicos", "DEBENTURE": "Debêntures", "FUTURO_DI": "Futuros de DI"}


def _treemap(p: pd.DataFrame, pl: float) -> go.Figure:
    p = p[p["valor_mercado"] > 0]
    ids, rot, par, val, hov, cores = [], [], [], [], [], []
    paleta = dict(zip(sorted(p["classe"].unique()), graficos.PALETA))
    for cl, g in p.groupby("classe"):
        ids.append(cl), rot.append(ROTULO_CLASSE.get(cl, cl)), par.append(""), val.append(g["valor_mercado"].sum())
        hov.append(f"{F.brl_mi(g['valor_mercado'].sum())} · {F.pct(g['valor_mercado'].sum() / pl)} do PL"), cores.append(paleta[cl])
        for em, ge in g.groupby("emissor"):
            ide = f"{cl}|{em}"
            ids.append(ide), rot.append(em), par.append(cl), val.append(ge["valor_mercado"].sum())
            hov.append(f"{F.brl_mi(ge['valor_mercado'].sum())} · {F.pct(ge['valor_mercado'].sum() / pl)} do PL"), cores.append(paleta[cl])
            for r in ge.itertuples():
                ids.append(f"{ide}|{r.ticker}"), rot.append(r.ticker), par.append(ide), val.append(r.valor_mercado)
                hov.append(f"{F.brl_mi(r.valor_mercado)} · {F.pct(r.valor_mercado / pl)} do PL"), cores.append(paleta[cl])
    fig = go.Figure(go.Treemap(ids=ids, labels=rot, parents=par, values=val, branchvalues="total", customdata=hov,
                               marker=dict(colors=cores), hovertemplate="%{label}<br>%{customdata}<extra></extra>",
                               textinfo="label"))
    fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0), separators=",.")
    return fig


def render() -> None:
    dados.exigir("mtm_diario", "pl_diario", "ativos", "fundos")
    mtm, pl, ativos = dados.tabela("mtm_diario"), dados.tabela("pl_diario"), dados.tabela("ativos").set_index("ticker")
    cnpj, data = dados.fundo_global(), dados.escolher_data(mtm["data"].unique())
    plf = pl[pl["fundo_cnpj"] == cnpj].sort_values("data")
    linha = plf[plf["data"] == data].iloc[0]
    ant = plf[plf["data"] < data].iloc[-1] if (plf["data"] < data).any() else None

    p = mtm[(mtm["fundo_cnpj"] == cnpj) & (mtm["data"] == data)].copy()
    liq = dados.tabela_ou_vazia("liquidez")
    if len(liq):
        p = p.join(liq[(liq["fundo_cnpj"] == cnpj) & (liq["data"] == data)].set_index("ticker")[["dias_para_zerar"]], on="ticker")
    else:
        p["dias_para_zerar"] = float("nan")
    for c in ("classe", "emissor", "rating"):
        p[c] = p["ticker"].map(ativos[c])
    pl_ativos = float(linha["pl"])
    p["pct_pl"] = p["valor_mercado"] / pl_ativos
    p["pct_exp"] = p["exposicao"] / pl_ativos
    p["unid"] = p["classe"].map({"TITULO_PUBLICO": "PU", "DEBENTURE": "PU", "FUTURO_DI": "PU", "ACAO": "R$"})

    d = lambda col: None if ant is None else F.sinal(F.brl_mi, float(linha[col]) - float(ant[col]))  # noqa: E731
    kpis([("PL dos ativos", F.brl_mi(pl_ativos), d("pl"), "normal", "Ativos + caixa, antes de taxas e resgates a pagar."),
          ("Valor de mercado", F.brl_mi(linha["valor_mercado"]), d("valor_mercado")),
          ("Caixa projetado", F.brl_mi(linha["caixa_projetado"]), d("caixa_projetado"), "normal",
           "Caixa disponível mais o que liquida em D+1/D+2."),
          ("Caixa disponível", F.brl_mi(linha["caixa_disponivel"]), d("caixa_disponivel"))])

    st.subheader(f"Posições em {F.data(data)}")
    spec = {"ticker": ("Ativo", "txt"), "emissor": ("Emissor", "txt"), "rating": ("Rating", "txt"), "qtd": ("Qtd", "qtd"),
            "preco": ("Preço / PU (R$)", "num"), "valor_mercado": ("Valor", "mi"), "pct_pl": ("% do PL", "pct"),
            "exposicao": ("Exposição", "mi"), "dias_para_zerar": ("Dias p/ zerar", "num")}
    for cl in [c for c in ROTULO_CLASSE if c in set(p["classe"])]:
        g = p[p["classe"] == cl].sort_values("exposicao", ascending=False)
        st.markdown(f"**{ROTULO_CLASSE[cl]}** — subtotal {F.brl_mi(g['valor_mercado'].sum())} · "
                    f"{F.pct(g['valor_mercado'].sum() / pl_ativos)} do PL · exposição {F.pct(g['exposicao'].sum() / pl_ativos)}")
        F.exibir(g, {k: v for k, v in spec.items() if k != "rating" or g["rating"].notna().any()})
    st.caption("Futuros de DI têm valor de mercado 0 (só ajuste diário); a exposição é o nocional. "
               "Dias para zerar: venda a 20% do volume diário.")

    esq, dir_ = st.columns(2)
    with esq:
        st.subheader("Mapa da carteira (valor de mercado)")
        if (p["valor_mercado"] > 0).any():
            graficos.mostrar(_treemap(p, pl_ativos), "cart_treemap")
        else:
            st.info("Sem posições com valor de mercado positivo.")
    with dir_:
        st.subheader("Evolução do valor de mercado por classe (R$ mi)")
        h = mtm[(mtm["fundo_cnpj"] == cnpj) & (mtm["data"] <= data)].assign(classe=lambda x: x["ticker"].map(ativos["classe"]))
        pv = h.pivot_table(index="data", columns="classe", values="valor_mercado", aggfunc="sum").fillna(0.0) / 1e6
        graficos.mostrar(graficos.areas(pv.index, [{"nome": ROTULO_CLASSE.get(c, c), "y": pv[c]} for c in pv.columns],
                                        fmt=lambda v: f"R$ {F.num(v)} mi", altura=380), "cart_area")
