"""Motor de risco diário por fundo sobre a carteira de fechamento."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gestorasim.enquadramento.regras import liquidez_d1
from gestorasim.enquadramento import montar_carteira
from gestorasim.mtm.curvas import VERTICES_RISCO
from gestorasim.risco.estatistica import (
    Z99, kupiec_pof, pnl_delta_gama, var_delta_normal, var_es_historico, zona_basileia,
)
from gestorasim.risco.liquidez import bucket, dias_para_zerar, matriz_liquidez
from gestorasim.risco.sensibilidades import CENARIOS, CalculadoraSens

JANELA_HISTORICA = 252
JANELA_BASILEIA = 250
CENARIOS_REAVALIACAO = tuple(CENARIOS)


def pnl_hipotetico(serie: dict[str, pd.DataFrame], tabelas: dict[str, pd.DataFrame], prec) -> pd.Series:
    """P&L do dia sobre a carteira do fechamento anterior (exclui trades, juros do caixa e custos).

    = preço + carrego + ajuste de futuros + eventos de renda fixa (mesmo dia do ex-cupom)
      + dividendo por ação creditado na data ex (acumulado, não na data de pagamento).
    """
    p = serie["pnl_atribuicao"].merge(prec.meta[["classe"]], left_on="ticker", right_index=True, how="left")
    rf = p["classe"] != "ACAO"
    p["hip"] = p["efeito_preco"] + p["carrego"] + p["ajuste_futuros"] + p["eventos"].where(rf & p["classe"].notna(), 0.0)
    hip = p.groupby(["data", "fundo_cnpj"])["hip"].sum()
    pos = serie["posicoes_diarias"]
    q = pos.set_index(["data", "fundo_cnpj", "ticker"])["quantidade"]
    acc = {}
    for _, e in tabelas["eventos"].query("tipo == 'DIVIDENDO'").iterrows():
        i_com = prec.i_de[pd.Timestamp(e["data_com"])]
        d_com, d_ex = prec.datas[i_com], prec.datas[i_com + 1]
        for c in pos["fundo_cnpj"].unique():
            qt = q.get((d_com, c, e["ticker"]), 0)
            if qt:
                acc[(d_ex, c)] = acc.get((d_ex, c), 0.0) + qt * e["valor_por_unidade"]
    div = pd.Series(acc, dtype=float)
    if len(div):
        div.index = pd.MultiIndex.from_tuples(div.index, names=["data", "fundo_cnpj"])
    return hip.add(div, fill_value=0.0).rename("pnl_hipotetico")


def metricas_politica(calc: CalculadoraSens, fatores, q: dict[str, int], i: int, pl: float) -> dict[str, float]:
    """VaR paramétrico 99% 1d e pior cenário de stress, em fração do PL (para a política de risco)."""
    if pl <= 0:
        return {"var_pct": float("nan"), "stress_pct": float("nan")}
    delta, _ = calc.carteira(q, i)
    cov = fatores.cov[i]
    var = var_delta_normal(delta, cov) if not np.isnan(cov).any() else float("nan")
    stress = calc.stress_carteira(q, i)
    return {"var_pct": var / pl, "stress_pct": -min(min(stress.values()), 0.0) / pl}


def calcular_risco(serie: dict[str, pd.DataFrame], tabelas: dict[str, pd.DataFrame], prec, painel: pd.DataFrame,
                   volume: pd.DataFrame, calc: CalculadoraSens, janela: int = JANELA_HISTORICA) -> dict[str, pd.DataFrame]:
    """Tabelas risco_diario, sensibilidades, stress, backtest_var, liquidez para todos os fundos e datas."""
    fatores = calc.fatores
    pos = serie["posicoes_diarias"]
    Q = {k: dict(zip(g["ticker"], g["quantidade"])) for k, g in pos.groupby(["data", "fundo_cnpj"])}
    pl = serie["pl_diario"].set_index(["data", "fundo_cnpj"])
    hip = pnl_hipotetico(serie, tabelas, prec)
    idx = fatores.idx
    pre_idx = [idx[f"PRE:{v}"] for v in VERTICES_RISCO]
    real_idx = [idx[f"REAL:{v}"] for v in VERTICES_RISCO]
    meta = prec.meta

    r_rows, s_rows, st_rows, l_rows = [], [], [], []
    anterior: dict[str, dict] = {}
    for (data, c), row in pl.sort_index().iterrows():
        i = prec.i_de[data]
        q = Q.get((data, c), {})
        PL = float(row["pl"])
        delta, gama = calc.carteira(q, i)
        cov = fatores.cov[i]
        var_p = var_delta_normal(delta, cov) if not np.isnan(cov).any() else np.nan
        var_h = es_h = pior_hist = np.nan
        if i >= janela:
            choques = fatores.d_f[i - janela + 1:i + 1]
            pnl_s = pnl_delta_gama(delta, gama, choques)
            var_h, es_h = var_es_historico(pnl_s)
            pior_hist = float(pnl_s.min())
        stress = calc.stress_carteira(q, i)
        for nome, v in stress.items():
            st_rows.append((data, c, nome, v, v / PL))
        st_rows.append((data, c, "PIOR_DIA_HIST", pior_hist, pior_hist / PL if np.isfinite(pior_hist) else np.nan))
        for j in np.flatnonzero((delta != 0) | (gama != 0)):
            s_rows.append((data, c, fatores.nomes[j], float(delta[j]), float(gama[j])))

        Pv, vol = painel.iloc[i].dropna(), volume.iloc[i]
        cart_ant = montar_carteira(q, float(row["caixa_projetado"]), meta, Pv)
        cart = montar_carteira(q, float(row["caixa_projetado"]), meta, Pv, vol)
        lista = []
        for tk, qt in q.items():
            if meta.loc[tk, "classe"] == "FUTURO_DI":
                continue
            dias = dias_para_zerar(qt, float(vol[tk]))
            valor = float(cart.posicoes.loc[tk, "valor"])
            lista.append((valor, dias))
            l_rows.append((data, c, tk, valor, dias, bucket(dias)))
        mat = matriz_liquidez(lista, float(row["caixa_projetado"]), PL)

        dv01 = float(delta[pre_idx].sum() + delta[real_idx].sum())
        ant = anterior.get(c, {})
        pnl_h = float(hip.get((data, c), 0.0)) if ant else np.nan
        exc_p = bool(pnl_h < -ant["var_p"]) if ant and np.isfinite(ant["var_p"]) else None
        exc_h = bool(pnl_h < -ant["var_h"]) if ant and np.isfinite(ant["var_h"]) else None
        r_rows.append({
            "data": data, "fundo_cnpj": c, "fundo": row["fundo"], "pl": PL,
            "var_param_99": var_p, "var_param_pct": var_p / PL, "var_hist_99": var_h, "var_hist_pct": var_h / PL,
            "es_hist_975": es_h, "es_hist_pct": es_h / PL, "dv01_total": dv01, "duration_pl": -dv01 / (PL * 1e-4),
            "stress_pior": min(min(stress.values()), 0.0),
            "stress_pior_pct": min(min(stress.values()), 0.0) / PL,
            "pior_dia_hist": pior_hist, "liquidez_d1": liquidez_d1(cart), "liquidez_d1_antes": liquidez_d1(cart_ant, corrigida=False),
            **mat, "pnl_hipotetico": pnl_h, "excecao_param": exc_p, "excecao_hist": exc_h,
        })
        anterior[c] = {"var_p": var_p, "var_h": var_h}

    risco = pd.DataFrame(r_rows)
    return {
        "risco_diario": risco,
        "sensibilidades": pd.DataFrame(s_rows, columns=["data", "fundo_cnpj", "fator", "delta", "gama"]),
        "stress": pd.DataFrame(st_rows, columns=["data", "fundo_cnpj", "cenario", "pnl", "pnl_pct_pl"]),
        "liquidez": pd.DataFrame(l_rows, columns=["data", "fundo_cnpj", "ticker", "valor_mercado", "dias_para_zerar", "bucket"]),
        "backtest_var": backtest_var(risco),
    }


def backtest_var(risco: pd.DataFrame, p: float = 0.01) -> pd.DataFrame:
    """Exceções do VaR 99% (P&L hipotético < −VaR do dia anterior), Kupiec (todas as observações) e
    zona de Basileia (últimas 250 observações) por fundo e método."""
    linhas = []
    for c, g in risco.groupby("fundo_cnpj", sort=False):
        for metodo, col in (("PARAMETRICO", "excecao_param"), ("HISTORICO", "excecao_hist")):
            obs = g[col].dropna().astype(bool)
            n, x = len(obs), int(obs.sum())
            lr, pv = kupiec_pof(n, x, p)
            ultimas = obs.iloc[-JANELA_BASILEIA:]
            nz, xz = len(ultimas), int(ultimas.sum())
            linhas.append({"fundo_cnpj": c, "fundo": g["fundo"].iloc[0], "metodo": metodo, "n_obs": n,
                           "excecoes": x, "esperado": n * p, "taxa": x / n if n else np.nan, "kupiec_lr": lr,
                           "kupiec_p": pv, "n_obs_basileia": nz, "excecoes_basileia": xz,
                           "zona_basileia": zona_basileia(nz, xz, p) if nz else ""})
    return pd.DataFrame(linhas)
