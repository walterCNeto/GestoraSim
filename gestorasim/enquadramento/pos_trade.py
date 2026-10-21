"""Enquadramento pós-trade: regras do regulamento + política de risco na carteira de fechamento de cada pregão.

Classificação dos desenquadramentos (por episódio, na data em que começam):
- ATIVO: a carteira sem as ordens do dia (mercado, eventos e juros apenas) estaria enquadrada na regra;
  a causa são as ordens executadas no dia (cruzadas com a trilha de auditoria do bloco 2).
- PASSIVO: a carteira sem as ordens do dia já estaria desenquadrada (mercado ou eventos).
  Conta prazo de reenquadramento em dias úteis; passado o prazo, VENCIDO.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from gestorasim.dominio import ClasseFundo
from gestorasim.enquadramento.regras import (
    OK, VIOLADO, ResultadoRegra, _res, montar_carteira, todas_as_regras,
)

PRAZO_REENQUADRAMENTO_DU = 10
LIMIAR_SOFT = 0.80
_ORDEM = re.compile(r"^ORD\d+$")

# Política de risco por fundo: VaR 99% 1d (paramétrico) e pior cenário de stress, em fração do PL.
POLITICA_RISCO: dict[ClasseFundo, dict[str, float]] = {
    ClasseFundo.FIA: {"var": 0.035, "stress": 0.170},
    ClasseFundo.MULTIMERCADO: {"var": 0.015, "stress": 0.140},
    ClasseFundo.RENDA_FIXA: {"var": 0.007, "stress": 0.085},
}


def regras_politica(var_pct: float, stress_pct: float, limites: dict[str, float]) -> list[ResultadoRegra]:
    out = []
    if np.isfinite(var_pct):
        out.append(_res("politica_var", "VaR 99% 1d / PL", var_pct, limites["var"]))
    if np.isfinite(stress_pct):
        out.append(_res("politica_stress", "pior stress / PL", stress_pct, limites["stress"]))
    return out


def chave_episodio(regra: str, detalhe: str) -> tuple[str, str]:
    """O emissor mais concentrado muda de nome ao longo do tempo: é o mesmo episódio de limite_emissor."""
    return (regra, "") if regra == "limite_emissor" else (regra, detalhe)


def utilizacao(r: ResultadoRegra) -> float:
    """Consumo do limite: valor/limite (máximos) ou limite/valor (mínimos). NaN se o limite é 0."""
    if r.limite == 0:
        return float("nan")
    if r.sentido == "max":
        return r.valor_pos_ordem / r.limite
    return r.limite / r.valor_pos_ordem if r.valor_pos_ordem > 0 else float("inf")


def nivel_alerta(r: ResultadoRegra, soft: float = LIMIAR_SOFT) -> str:
    if r.status == VIOLADO:
        return "HARD"
    u = utilizacao(r)
    return "SOFT" if np.isfinite(u) and u >= soft - 1e-9 else ""


@dataclass
class Episodio:
    fundo_cnpj: str
    regra: str
    detalhe: str
    inicio: pd.Timestamp
    i_inicio: int
    classe: str  # ATIVO | PASSIVO
    ordens_causa: str
    fim: pd.Timestamp | None = None
    du_max: int = 0
    vencido: bool = False


def _ordens_do_dia(serie: dict[str, pd.DataFrame]) -> dict[tuple[pd.Timestamp, str], str]:
    """(data, fundo) -> ordens executadas no dia que alocaram ao fundo, marcando as liberadas por override."""
    al, au = serie["alocacoes_serie"], serie["auditoria_serie"]
    if al.empty:
        return {}
    ov = set(map(tuple, au.loc[au["para"] == "LIBERADA_OVERRIDE", ["data_serie", "ordem_id"]].to_numpy()))
    out: dict[tuple, list[str]] = {}
    for _, a in al[al["quantidade_alocada"] > 0].iterrows():
        marca = "(OVERRIDE)" if (a["data_serie"], a["ordem_id"]) in ov else ""
        out.setdefault((a["data_serie"], a["fundo_cnpj"]), []).append(a["ordem_id"] + marca)
    return {k: ",".join(sorted(set(v))) for k, v in out.items()}


def rodar_pos_trade(serie: dict[str, pd.DataFrame], risco: pd.DataFrame, prec, painel: pd.DataFrame,
                    volume: pd.DataFrame, fundos, calc, politica: dict | None = None,
                    prazo_reenquadramento_du: int = PRAZO_REENQUADRAMENTO_DU,
                    soft: float = LIMIAR_SOFT) -> dict[str, pd.DataFrame]:
    """Devolve enquadramento_diario (uma linha por regra/fundo/data), alertas e episodios."""
    politica = politica or POLITICA_RISCO
    meta, fatores = prec.meta, calc.fatores
    por_cnpj = {f.cnpj: f for f in fundos}
    pos = serie["posicoes_diarias"]
    Q = {k: dict(zip(g["ticker"], g["quantidade"])) for k, g in pos.groupby(["data", "fundo_cnpj"])}
    pl = serie["pl_diario"].set_index(["data", "fundo_cnpj"])
    led = serie["ledger"]
    cx = led[(led["tipo"] == "CAIXA") & led["origem"].str.match(_ORDEM)]
    cx = cx.assign(liq0=cx["delta"].where(cx["data_liquidacao"] <= cx["data"], 0.0))
    caixa_trades = {k: (r["delta"], r["liq0"])
                    for k, r in cx.groupby(["data", "fundo_cnpj"])[["delta", "liq0"]].sum().iterrows()}
    ordens_dia = _ordens_do_dia(serie)
    risco_idx = risco.set_index(["data", "fundo_cnpj"])

    abertos: dict[tuple, Episodio] = {}
    todos: list[Episodio] = []
    linhas, alertas = [], []
    for (data, c), row in pl.sort_index().iterrows():
        f, i = por_cnpj[c], prec.i_de[data]
        q = Q.get((data, c), {})
        Pv, vol = painel.iloc[i].dropna(), volume.iloc[i]
        rk = risco_idx.loc[(data, c)]
        cart = montar_carteira(q, float(row["caixa_projetado"]), meta, Pv, vol, float(row["caixa_disponivel"]))
        lim = politica[f.classe]
        resultados = todas_as_regras(cart, f.regulamento) + regras_politica(
            float(rk["var_param_pct"]), float(-rk["stress_pior_pct"]), lim)
        cf = None  # carteira sem as ordens do dia, calculada só se algum episódio começar hoje
        violadas_hoje = set()
        for r in resultados:
            chave = (c, *chave_episodio(r.regra, r.detalhe))
            ep = abertos.get(chave)
            if r.status == VIOLADO:
                violadas_hoje.add(chave)
                if ep is None:
                    classe = "PASSIVO"
                    causa = ""
                    if (data, c) in ordens_dia and i > 0:
                        if cf is None:
                            cf = _contrafactual(serie, calc, fatores, meta, painel, volume, Q, pl, caixa_trades,
                                                prec, f, c, data, i, row, lim)
                        violou_sem_ordens = any(
                            x.status == VIOLADO and chave_episodio(x.regra, x.detalhe) == chave[1:] for x in cf)
                        classe = "PASSIVO" if violou_sem_ordens else "ATIVO"
                        causa = ordens_dia[(data, c)] if classe == "ATIVO" else ""
                    ep = Episodio(c, r.regra, r.detalhe, data, i, classe, causa)
                    abertos[chave] = ep
                    todos.append(ep)
                du = i - ep.i_inicio
                ep.du_max = max(ep.du_max, du)
                ep.vencido = ep.classe == "PASSIVO" and ep.du_max > prazo_reenquadramento_du
            u = utilizacao(r)
            nivel = nivel_alerta(r, soft)
            ep = abertos.get(chave) if r.status == VIOLADO else None
            linhas.append({
                "data": data, "fundo_cnpj": c, "regra": r.regra, "detalhe": r.detalhe,
                "valor": r.valor_pos_ordem, "limite": r.limite, "folga": r.folga, "status": r.status,
                "utilizacao": u, "alerta": nivel,
                "classificacao": ep.classe if ep else "", "inicio_episodio": ep.inicio if ep else pd.NaT,
                "du_desde_inicio": (i - ep.i_inicio) if ep else 0,
                "prazo_du": prazo_reenquadramento_du if ep and ep.classe == "PASSIVO" else 0,
                "vencido": bool(ep.vencido) if ep else False, "ordens_causa": ep.ordens_causa if ep else "",
            })
            if nivel:
                alertas.append({"data": data, "fundo_cnpj": c, "regra": r.regra, "detalhe": r.detalhe,
                                "nivel": nivel, "utilizacao": u, "valor": r.valor_pos_ordem, "limite": r.limite})
        for chave in [k for k in abertos if k[0] == c and k not in violadas_hoje]:
            abertos.pop(chave).fim = data  # reenquadrado

    episodios = pd.DataFrame([vars(e) for e in todos])
    return {"enquadramento_diario": pd.DataFrame(linhas), "alertas": pd.DataFrame(alertas), "episodios": episodios}


def _contrafactual(serie, calc, fatores, meta, painel, volume, Q, pl, caixa_trades, prec, f, c, data, i,
                   row, lim) -> list[ResultadoRegra]:
    """Regras da carteira que existiria sem as ordens do dia: posições e caixa do fechamento anterior
    + mercado e eventos de hoje."""
    from gestorasim.risco.motor import metricas_politica

    ant = prec.datas[i - 1]
    q_ant = Q.get((ant, c), {})
    tot, liq0 = caixa_trades.get((data, c), (0.0, 0.0))
    caixa_proj = float(row["caixa_projetado"]) - tot
    caixa_disp = float(row["caixa_disponivel"]) - liq0
    Pv, vol = painel.iloc[i].dropna(), volume.iloc[i]
    cart = montar_carteira(q_ant, caixa_proj, meta, Pv, vol, caixa_disp)
    res = todas_as_regras(cart, f.regulamento)
    m = metricas_politica(calc, fatores, q_ant, i, cart.pl)
    return res + regras_politica(m["var_pct"], m["stress_pct"], lim)
