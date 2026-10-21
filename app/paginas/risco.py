"""Risco: VaR/ES no tempo com o limite da política, DV01 por vértice, stress e backtest (P&L limpo vs sujo)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app import dados, graficos
from app import formatacao as F
from app.layout import kpis

TITULO = "Risco"
REAL = ("é a área de risco do gestor: mede quanto o fundo pode perder, compara com o limite da política, estressa a "
        "carteira e valida o modelo comparando o VaR de ontem com o resultado de hoje (backtest).")
GLOSSARIO = ("VaR", "ES", "DV01", "CS01", "Stress", "Backtest")


def _limite(enq: pd.DataFrame, cnpj: str, regra: str, data) -> float | None:
    e = enq[(enq["fundo_cnpj"] == cnpj) & (enq["regra"] == regra) & (enq["data"] <= data)]
    return None if e.empty else float(e.sort_values("data")["limite"].iloc[-1])


def render() -> None:
    dados.exigir("risco_diario", "sensibilidades", "stress", "backtest_var", "fundos")
    cnpj, data = dados.fundo_global(), dados.data_global()
    risco, sens, stress = dados.tabela("risco_diario"), dados.tabela("sensibilidades"), dados.tabela("stress")
    bt, exc = dados.tabela("backtest_var"), dados.tabela_ou_vazia("excecoes_var")
    enq = dados.tabela_ou_vazia("enquadramento_diario")
    r = risco[(risco["fundo_cnpj"] == cnpj) & (risco["data"] <= data)].sort_values("data").set_index("data")
    if r.empty:
        st.info("Sem risco calculado até esta data.")
        st.stop()
    lim_var, lim_stress = _limite(enq, cnpj, "politica_var", data), _limite(enq, cnpj, "politica_stress", data)
    st.caption("Percentuais sobre o PL líquido. VaR 99% 1 dia; ES 97,5% histórico (janela de 252 pregões).")

    u, a = r.iloc[-1], (r.iloc[-2] if len(r) > 1 else None)
    dlt = lambda col, fn, inv=False: None if a is None or pd.isna(u[col]) or pd.isna(a[col]) else F.sinal(fn, u[col] - a[col])  # noqa: E731
    kpis([("VaR paramétrico (% PL)", F.pct(u["var_param_pct"]), dlt("var_param_pct", F.pct), "inverse",
           f"Limite da política: {F.pct(lim_var)}" if lim_var else None),
          ("ES histórico (% PL)", F.pct(u["es_hist_pct"]), dlt("es_hist_pct", F.pct), "inverse"),
          ("DV01 total (R$ mil/bp)", F.mil(u["dv01_total"]), dlt("dv01_total", F.mil), "off",
           "Negativo: perde valor se os juros sobem."),
          ("Duration (anos do PL)", F.num(u["duration_pl"]), dlt("duration_pl", F.num), "off"),
          ("Pior stress (% PL)", F.pct(u["stress_pior_pct"]), dlt("stress_pior_pct", F.pct), "normal",
           f"Limite: {F.pct(-lim_stress)}" if lim_stress else None)])

    st.subheader("VaR e ES no tempo (% do PL líquido)")
    x = r.index
    hl = [(lim_var, "limite VaR (política)", F.VERMELHO)] if lim_var else []
    graficos.mostrar(graficos.linhas([
        {"nome": "VaR paramétrico", "x": x, "y": r["var_param_pct"], "cor": F.AZUL, "fmt": F.pct},
        {"nome": "VaR histórico", "x": x, "y": r["var_hist_pct"], "cor": "#0f8b8d", "fmt": F.pct},
        {"nome": "ES histórico", "x": x, "y": r["es_hist_pct"], "cor": "#7a4fb5", "dash": "dot", "fmt": F.pct}],
        fmt=F.pct, tickfmt=".1%", hlinhas=hl), "risco_var")

    esq, dir_ = st.columns(2)
    with esq:
        st.subheader("DV01 e CS01 por vértice (R$ mil/bp)")
        s = sens[(sens["fundo_cnpj"] == cnpj) & (sens["data"] == data)].copy()
        curvas = s[s["fator"].str.startswith(("PRE:", "REAL:", "SPREAD:"))]
        if curvas.empty:
            st.info("Sem sensibilidade a juros/crédito nesta data.")
        else:
            curvas = curvas.assign(tipo=curvas["fator"].str.split(":").str[0], vertice=curvas["fator"].str.split(":").str[1])
            pv = curvas.pivot_table(index="vertice", columns="tipo", values="delta", aggfunc="sum") / 1e3
            ratings = ["AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-"]
            ordem = sorted(pv.index, key=lambda v: (0, int(v), "") if str(v).isdigit() else (1, ratings.index(v) if v in ratings else 99, v))
            pv = pv.loc[ordem]
            rot = {"PRE": "DV01 pré", "REAL": "DV01 real (IPCA)", "SPREAD": "CS01 spread"}
            graficos.mostrar(graficos.barras_empilhadas(
                list(pv.index), [{"nome": rot.get(c, c), "y": pv[c]} for c in pv.columns],
                fmt=lambda v: f"R$ {F.num(v)} mil/bp", titulo_y="R$ mil por bp"), "risco_dv01")
            st.caption("Bump de 1 bp com reavaliação completa. Eixo: vértice da curva (ou rating, no CS01).")
    with dir_:
        st.subheader("Stress com reavaliação completa (% do PL)")
        stf = stress[(stress["fundo_cnpj"] == cnpj) & (stress["data"] == data)].dropna(subset=["pnl_pct_pl"]).sort_values("pnl_pct_pl")
        if stf.empty:
            st.info("Sem stress nesta data.")
        else:
            cores = [F.VERMELHO if (lim_stress and v <= -lim_stress) else F.AMBAR if (lim_stress and v <= -0.85 * lim_stress) else F.AZUL
                     for v in stf["pnl_pct_pl"]]
            graficos.mostrar(graficos.barras_h(stf["cenario"], stf["pnl_pct_pl"], fmt=F.pct, cores=cores,
                                               limite=(-lim_stress, f"limite {F.pct(-lim_stress)}") if lim_stress else None,
                                               tickfmt=".0%"), "risco_stress")
            F.exibir(stf, {"cenario": ("Cenário", "txt"), "pnl": ("P&L", "mi"), "pnl_pct_pl": ("P&L / PL", "pct")})

    ac = s[s["fator"].str.startswith("ACAO:")][["fator", "delta"]]
    if len(ac):
        with st.expander("Delta por ação (R$ mi por 100% de retorno)"):
            F.exibir(ac, {"fator": ("Fator", "txt"), "delta": ("Delta", "mi")})

    st.subheader("Backtest do VaR 99%")
    c1, c2 = st.columns(2)
    base = c1.radio("Base do P&L hipotético", ["LIMPO", "SUJO"], horizontal=True, key="risco_base",
                    help="LIMPO: só efeito dos fatores de mercado. SUJO: inclui carrego e eventos de renda fixa.")
    metodo = c2.radio("Método", ["PARAMETRICO", "HISTORICO"], horizontal=True, key="risco_metodo")
    col_pnl = "pnl_hipotetico_limpo" if base == "LIMPO" else "pnl_hipotetico"
    col_var = "var_param_ant" if metodo == "PARAMETRICO" else "var_hist_ant"
    g = r[[col_pnl, col_var, "em_burn_in"]].dropna(subset=[col_pnl, col_var]).copy()
    if g.empty:
        st.info("Sem janela suficiente para o backtest até esta data (burn-in).")
    else:
        g["menos_var"] = -g[col_var]
        ex = g[g[col_pnl] < g["menos_var"]]
        graficos.mostrar(graficos.linhas(
            [{"nome": "P&L hipotético (R$ mi)", "x": g.index, "y": g[col_pnl] / 1e6, "cor": F.AZUL, "fmt": F.num},
             {"nome": "−VaR do dia anterior (R$ mi)", "x": g.index, "y": g["menos_var"] / 1e6, "cor": F.VERMELHO, "dash": "dash", "fmt": F.num}],
            fmt=F.num, titulo_y="R$ mi",
            pontos=[{"nome": "Exceção", "x": ex.index, "y": ex[col_pnl] / 1e6, "cor": F.VERMELHO}]), "risco_bt")
        st.caption(f"Marcas × = exceções (P&L abaixo de −VaR do dia anterior): {F.qtd(len(ex))} no período mostrado. "
                   f"Burn-in: {F.qtd(r['em_burn_in'].sum())} pregões iniciais fora do teste.")
    sel = bt[(bt["fundo_cnpj"] == cnpj) & (bt["base"] == base) & (bt["metodo"] == metodo)]
    F.exibir(sel, {"n_obs": ("Observações", "qtd"), "excecoes": ("Exceções", "qtd"), "esperado": ("Esperadas", "num"),
                   "taxa": ("Taxa de exceção", "pct"), "kupiec_lr": ("Kupiec LR", "num"), "kupiec_p": ("Kupiec p-valor", "num"),
                   "zona_basileia": ("Zona Basileia", "txt"), "excecoes_basileia": ("Exceções (250d)", "qtd")}, badges=("zona_basileia",))
    if len(exc):
        e = exc[(exc["fundo_cnpj"] == cnpj) & (exc["base"] == base) & (exc["metodo"] == metodo) & (exc["data"] <= data)]
        st.markdown(f"**Exceções ({F.qtd(len(e))})**")
        F.exibir(e, {"data": ("Data", "data"), "pnl": ("P&L", "mi"), "var_dia_anterior": ("VaR do dia anterior", "mi"),
                     "em_burn_in": ("Em burn-in", "txt")}, vazio="Nenhuma exceção.")
