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
from gestorasim.enquadramento.regras import classe_de

JANELA_HISTORICA = 252
JANELA_BASILEIA = 250
BURN_IN_BACKTEST = 60  # pregões iniciais da série fora do backtest (EWMA ainda aquecendo)
CENARIOS_REAVALIACAO = tuple(CENARIOS)


def pnl_hipotetico(serie: dict[str, pd.DataFrame], tabelas: dict[str, pd.DataFrame], prec) -> pd.DataFrame:
    """P&L do dia sobre a carteira do fechamento anterior (sem trades, custos nem juros do caixa).

    - `pnl_hipotetico_limpo`: só o efeito dos fatores de mercado = efeito preço + ajuste de futuros
      + dividendo por ação na data ex (devolve a queda de preço do ex, que não é movimento de fator).
    - `pnl_hipotetico` (sujo): o limpo + carrego + eventos de renda fixa (cupom/amortização).
    """
    p = serie["pnl_atribuicao"].merge(prec.meta[["classe"]], left_on="ticker", right_index=True, how="left")
    rf = p["classe"].notna() & (p["classe"] != "ACAO")
    p["limpo"] = p["efeito_preco"] + p["ajuste_futuros"]
    p["sujo"] = p["limpo"] + p["carrego"] + p["eventos"].where(rf, 0.0)
    base = p.groupby(["data", "fundo_cnpj"])[["limpo", "sujo"]].sum()
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
    out = base.copy()
    if len(div):
        ajuste = div.reindex(out.index).fillna(0.0)
        out["limpo"] += ajuste
        out["sujo"] += ajuste
    return out.rename(columns={"limpo": "pnl_hipotetico_limpo", "sujo": "pnl_hipotetico"})


def metricas_politica(calc: CalculadoraSens, fatores, q: dict[str, int], i: int, pl: float) -> dict[str, float]:
    """VaR paramétrico 99% 1d e pior cenário de stress, em fração do PL (para a política de risco)."""
    if pl <= 0:
        return {"var_pct": float("nan"), "stress_pct": float("nan")}
    delta, _ = calc.carteira(q, i)
    cov = fatores.cov[i]
    var = var_delta_normal(delta, cov) if not np.isnan(cov).any() else float("nan")
    stress = calc.stress_carteira(q, i)
    return {"var_pct": var / pl, "stress_pct": -min(min(stress.values()), 0.0) / pl}


class MotorRiscoDiario:
    """Risco de um fundo em um pregão, sobre a carteira de fechamento. Guarda só o VaR do dia anterior
    (para o backtest); os percentuais são sobre o PL LÍQUIDO (cota_diaria)."""

    def __init__(self, prec, painel: pd.DataFrame, volume: pd.DataFrame, calc: CalculadoraSens,
                 janela: int = JANELA_HISTORICA, burn_in: int = BURN_IN_BACKTEST) -> None:
        self.prec, self.painel, self.volume, self.calc = prec, painel, volume, calc
        self.janela, self.burn_in = janela, burn_in
        self.meta, self.fatores = prec.meta, calc.fatores
        idx = self.fatores.idx
        self.pre_idx = [idx[f"PRE:{v}"] for v in VERTICES_RISCO]
        self.real_idx = [idx[f"REAL:{v}"] for v in VERTICES_RISCO]
        self.anterior: dict[str, dict] = {}  # estado: VaR do dia anterior por fundo
        self.i_primeira: int | None = None  # estado: 1º pregão da série (define o burn-in)

    def dia(self, i: int, data: pd.Timestamp, cnpj: str, nome: str, q: dict[str, int], pl_liquido: float,
            caixa_projetado: float, hip_limpo: float | None, hip_sujo: float | None) -> dict:
        """Devolve {'risco': linha, 'sens': [...], 'stress': [...], 'liq': [...]} do fundo no pregão `i`."""
        calc, fatores, meta, c, PL = self.calc, self.fatores, self.meta, cnpj, float(pl_liquido)
        if self.i_primeira is None:
            self.i_primeira = i
        delta, gama = calc.carteira(q, i)
        cov = fatores.cov[i]
        var_p = var_delta_normal(delta, cov) if not np.isnan(cov).any() else np.nan
        var_h = es_h = pior_hist = np.nan
        if i >= self.janela:
            pnl_s = pnl_delta_gama(delta, gama, fatores.d_f[i - self.janela + 1:i + 1])
            var_h, es_h = var_es_historico(pnl_s)
            pior_hist = float(pnl_s.min())
        stress = calc.stress_carteira(q, i)
        st_rows = [(data, c, n, v, v / PL) for n, v in stress.items()]
        st_rows.append((data, c, "PIOR_DIA_HIST", pior_hist, pior_hist / PL if np.isfinite(pior_hist) else np.nan))
        s_rows = [(data, c, fatores.nomes[j], float(delta[j]), float(gama[j]))
                  for j in np.flatnonzero((delta != 0) | (gama != 0))]

        Pv, vol = self.painel.iloc[i].dropna(), self.volume.iloc[i]
        cart_ant = montar_carteira(q, caixa_projetado, meta, Pv)
        cart = montar_carteira(q, caixa_projetado, meta, Pv, vol)
        lista, l_rows = [], []
        for tk, qt in q.items():
            if classe_de(meta, tk) == "FUTURO_DI":
                continue
            dias = dias_para_zerar(qt, float(vol[tk]))
            valor = cart.linha(tk).valor
            lista.append((valor, dias))
            l_rows.append((data, c, tk, valor, dias, bucket(dias)))
        mat = matriz_liquidez(lista, caixa_projetado, PL)

        dv01 = float(delta[self.pre_idx].sum() + delta[self.real_idx].sum())
        ant = self.anterior.get(c, {})
        pnl_s = float(hip_sujo) if ant and hip_sujo is not None else np.nan
        pnl_l = float(hip_limpo) if ant and hip_limpo is not None else np.nan
        pode_p, pode_h = bool(ant) and np.isfinite(ant["var_p"]), bool(ant) and np.isfinite(ant["var_h"])
        exc = {
            "excecao_param": bool(pnl_l < -ant["var_p"]) if pode_p else None,
            "excecao_hist": bool(pnl_l < -ant["var_h"]) if pode_h else None,
            "excecao_param_sujo": bool(pnl_s < -ant["var_p"]) if pode_p else None,
            "excecao_hist_sujo": bool(pnl_s < -ant["var_h"]) if pode_h else None,
        }
        pior_stress = min(min(stress.values()), 0.0)
        linha = {
            "data": data, "fundo_cnpj": c, "fundo": nome, "pl": PL,
            "var_param_99": var_p, "var_param_pct": var_p / PL, "var_hist_99": var_h, "var_hist_pct": var_h / PL,
            "es_hist_975": es_h, "es_hist_pct": es_h / PL, "dv01_total": dv01, "duration_pl": -dv01 / (PL * 1e-4),
            "stress_pior": pior_stress, "stress_pior_pct": pior_stress / PL, "pior_dia_hist": pior_hist,
            "liquidez_d1": liquidez_d1(cart, True, PL), "liquidez_d1_antes": liquidez_d1(cart_ant, False, PL),
            **mat, "pnl_hipotetico_limpo": pnl_l, "pnl_hipotetico": pnl_s, **exc,
            "var_param_ant": ant.get("var_p", np.nan), "var_hist_ant": ant.get("var_h", np.nan),
            "em_burn_in": (i - self.i_primeira) < self.burn_in,
        }
        self.anterior[c] = {"var_p": var_p, "var_h": var_h}
        return {"risco": linha, "sens": s_rows, "stress": st_rows, "liq": l_rows}


def montar_tabelas_risco(linhas: list[dict], sens: list, stress: list, liq: list) -> dict[str, pd.DataFrame]:
    risco = pd.DataFrame(linhas)
    return {
        "risco_diario": risco,
        "sensibilidades": pd.DataFrame(sens, columns=["data", "fundo_cnpj", "fator", "delta", "gama"]),
        "stress": pd.DataFrame(stress, columns=["data", "fundo_cnpj", "cenario", "pnl", "pnl_pct_pl"]),
        "liquidez": pd.DataFrame(liq, columns=["data", "fundo_cnpj", "ticker", "valor_mercado", "dias_para_zerar", "bucket"]),
        "backtest_var": backtest_var(risco),
        "excecoes_var": excecoes_var(risco),
    }


def calcular_risco(serie: dict[str, pd.DataFrame], tabelas: dict[str, pd.DataFrame], prec, painel: pd.DataFrame,
                   volume: pd.DataFrame, calc: CalculadoraSens, janela: int = JANELA_HISTORICA,
                   burn_in: int = BURN_IN_BACKTEST) -> dict[str, pd.DataFrame]:
    """Tabelas risco_diario, sensibilidades, stress, backtest_var, liquidez para todos os fundos e datas
    (versão em lote: percorre a série inteira com o mesmo `MotorRiscoDiario` do orquestrador)."""
    pos = serie["posicoes_diarias"]
    Q = {k: dict(zip(g["ticker"], g["quantidade"])) for k, g in pos.groupby(["data", "fundo_cnpj"])}
    pl = serie["pl_diario"].set_index(["data", "fundo_cnpj"])
    pl_liq = (serie["cota_diaria"].set_index(["data", "fundo_cnpj"])["pl_liquido"]
              if "cota_diaria" in serie and len(serie["cota_diaria"]) else pl["pl"])
    hip = pnl_hipotetico(serie, tabelas, prec)
    motor = MotorRiscoDiario(prec, painel, volume, calc, janela, burn_in)
    linhas, s_rows, st_rows, l_rows = [], [], [], []
    for (data, c), row in pl.sort_index().iterrows():
        h = hip.loc[(data, c)] if (data, c) in hip.index else None
        r = motor.dia(prec.i_de[data], data, c, row["fundo"], Q.get((data, c), {}), float(pl_liq.loc[(data, c)]),
                      float(row["caixa_projetado"]), None if h is None else h["pnl_hipotetico_limpo"],
                      None if h is None else h["pnl_hipotetico"])
        linhas.append(r["risco"])
        s_rows += r["sens"]
        st_rows += r["stress"]
        l_rows += r["liq"]
    return montar_tabelas_risco(linhas, s_rows, st_rows, l_rows)


BASES_BACKTEST = (("LIMPO", "", "pnl_hipotetico_limpo"), ("SUJO", "_sujo", "pnl_hipotetico"))
METODOS_BACKTEST = (("PARAMETRICO", "param", "var_param_ant"), ("HISTORICO", "hist", "var_hist_ant"))


def excecoes_var(risco: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por exceção (P&L hipotético < −VaR do dia anterior), com a marca de burn-in."""
    linhas = []
    for base, suf, col_pnl in BASES_BACKTEST:
        for metodo, tag, col_var in METODOS_BACKTEST:
            g = risco[risco[f"excecao_{tag}{suf}"].fillna(False).astype(bool)]
            for _, r in g.iterrows():
                linhas.append({"data": r["data"], "fundo_cnpj": r["fundo_cnpj"], "fundo": r["fundo"], "metodo": metodo,
                               "base": base, "pnl": r[col_pnl], "var_dia_anterior": r[col_var],
                               "em_burn_in": bool(r["em_burn_in"])})
    return pd.DataFrame(linhas, columns=["data", "fundo_cnpj", "fundo", "metodo", "base", "pnl", "var_dia_anterior",
                                         "em_burn_in"])


def backtest_var(risco: pd.DataFrame, p: float = 0.01) -> pd.DataFrame:
    """Exceções do VaR 99% por fundo, método e base de P&L (LIMPO: só fatores; SUJO: com carrego e eventos).

    O burn-in (`em_burn_in`) fica fora de n_obs/excecoes; `excecoes_com_burn_in` mostra o que ele escondia.
    Kupiec sobre todas as observações restantes; zona de Basileia sobre as últimas 250.
    """
    linhas = []
    for c, g in risco.groupby("fundo_cnpj", sort=False):
        for base, suf, _ in BASES_BACKTEST:
            for metodo, tag, _ in METODOS_BACKTEST:
                col = f"excecao_{tag}{suf}"
                todas = g[col].dropna().astype(bool)
                obs = g.loc[~g["em_burn_in"].astype(bool), col].dropna().astype(bool)
                n, x = len(obs), int(obs.sum())
                lr, pv = kupiec_pof(n, x, p)
                ultimas = obs.iloc[-JANELA_BASILEIA:]
                nz, xz = len(ultimas), int(ultimas.sum())
                linhas.append({"fundo_cnpj": c, "fundo": g["fundo"].iloc[0], "metodo": metodo, "base": base,
                               "n_obs": n, "excecoes": x, "esperado": n * p, "taxa": x / n if n else np.nan,
                               "kupiec_lr": lr, "kupiec_p": pv, "n_obs_basileia": nz, "excecoes_basileia": xz,
                               "zona_basileia": zona_basileia(nz, xz, p) if nz else "",
                               "n_obs_com_burn_in": len(todas), "excecoes_com_burn_in": int(todas.sum())})
    return pd.DataFrame(linhas)
