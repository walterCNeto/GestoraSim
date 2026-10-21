"""Enquadramento pós-trade: regras do regulamento + política de risco na carteira de fechamento de cada pregão.

Classificação dos desenquadramentos (por episódio, na data em que começam):
- ATIVO: a carteira sem as ordens do dia (mercado, eventos e juros apenas) estaria enquadrada na regra;
  a causa são as ordens executadas no dia (cruzadas com a trilha de auditoria do bloco 2).
- PASSIVO: a carteira sem as ordens do dia já estaria desenquadrada (mercado ou eventos).
  Conta prazo de reenquadramento em dias úteis; passado o prazo, VENCIDO.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, fields

import numpy as np
import pandas as pd

from gestorasim.dominio import PRAZO_LIQUIDACAO_DU, ClasseAtivo, ClasseFundo
from gestorasim.enquadramento.regras import (
    OK, VIOLADO, ResultadoRegra, _res, classe_de, montar_carteira, todas_as_regras,
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
    ordem_principal: str = ""
    n_ordens: int = 0
    du_max: int = 0
    vencido: bool = False


def _ordens_do_dia(serie: dict[str, pd.DataFrame]) -> dict[tuple[pd.Timestamp, str], list[dict]]:
    """(data, fundo) -> alocações executadas no dia (ordem, ativo, lado, quantidade, financeiro, override)."""
    al, au = serie["alocacoes_serie"], serie["auditoria_serie"]
    if al.empty:
        return {}
    ov = set(map(tuple, au.loc[au["para"] == "LIBERADA_OVERRIDE", ["data_serie", "ordem_id"]].to_numpy()))
    out: dict[tuple, list[dict]] = {}
    for _, a in al[al["quantidade_alocada"] > 0].iterrows():
        out.setdefault((a["data_serie"], a["fundo_cnpj"]), []).append({
            "ordem_id": a["ordem_id"], "ticker": a["ticker"], "lado": a["lado"], "quantidade": int(a["quantidade_alocada"]),
            "financeiro": float(a["financeiro"]), "override": (a["data_serie"], a["ordem_id"]) in ov})
    return out


def _valor_da_regra(resultados: list[ResultadoRegra], chave: tuple[str, str]) -> tuple[float, str] | None:
    for x in resultados:
        if chave_episodio(x.regra, x.detalhe) == chave:
            return x.valor_pos_ordem, x.sentido
    return None


def ranquear_ordens(alocacoes: list[dict], regra: ResultadoRegra, avaliar_sem) -> tuple[str, str, int]:
    """Ordena as ordens do dia pela contribuição à métrica da regra: valor com todas as ordens menos o valor sem a
    ordem (positivo = piora; em regras de mínimo, o sinal é invertido). `avaliar_sem(alocacao)` devolve a lista de
    resultados da carteira sem aquela ordem. Devolve (ordens_causa, ordem_principal, n_ordens)."""
    chave = chave_episodio(regra.regra, regra.detalhe)
    por_ordem: dict[str, float] = {}
    marca: dict[str, bool] = {}
    for a in alocacoes:
        sem = _valor_da_regra(avaliar_sem(a), chave)
        valor_sem = sem[0] if sem else 0.0
        contrib = (regra.valor_pos_ordem - valor_sem) * (1 if regra.sentido == "max" else -1)
        por_ordem[a["ordem_id"]] = por_ordem.get(a["ordem_id"], 0.0) + contrib
        marca[a["ordem_id"]] = marca.get(a["ordem_id"], False) or a["override"]
    ranque = sorted(por_ordem.items(), key=lambda kv: (-kv[1], kv[0]))
    texto = ",".join(f"{o}{'(OVERRIDE)' if marca[o] else ''}({c:+.4f})" for o, c in ranque)
    return texto, ranque[0][0], len(ranque)


class PosTradeDiario:
    """Enquadramento pós-trade pregão a pregão. Estado: episódios de desenquadramento abertos e já encerrados."""

    def __init__(self, prec, painel: pd.DataFrame, volume: pd.DataFrame, fundos, calc, politica: dict | None = None,
                 prazo_reenquadramento_du: int = PRAZO_REENQUADRAMENTO_DU, soft: float = LIMIAR_SOFT) -> None:
        self.prec, self.painel, self.volume, self.calc = prec, painel, volume, calc
        self.politica = politica or POLITICA_RISCO
        self.prazo, self.soft = prazo_reenquadramento_du, soft
        self.meta, self.fatores = prec.meta, calc.fatores
        self.fundos = {f.cnpj: f for f in fundos}
        self.abertos: dict[tuple, Episodio] = {}  # estado
        self.todos: list[Episodio] = []  # estado

    def dia(self, i: int, data: pd.Timestamp, c: str, caixa_projetado: float, caixa_disponivel: float,
            q: dict[str, int], q_ant: dict[str, int], var_pct: float, stress_pct: float, passivos: float,
            alocs: list[dict], caixa_trade: tuple[float, float]) -> tuple[list[dict], list[dict]]:
        """`var_pct`/`stress_pct`: política de risco em fração do PL líquido; `passivos`: taxas, performance e
        resgates a pagar (PL líquido = PL dos ativos − passivos); `alocs`: alocações do dia ao fundo;
        `caixa_trade`: (caixa total das ordens do dia, parte que liquida no próprio dia)."""
        f, meta, fatores, calc = self.fundos[c], self.meta, self.fatores, self.calc
        row = {"caixa_projetado": caixa_projetado, "caixa_disponivel": caixa_disponivel}
        Pv, vol = self.painel.iloc[i].dropna(), self.volume.iloc[i]
        cart = montar_carteira(q, caixa_projetado, meta, Pv, vol, caixa_disponivel)
        lim = self.politica[f.classe]
        resultados = todas_as_regras(cart, f.regulamento) + regras_politica(var_pct, stress_pct, lim)
        cf = None  # carteira sem as ordens do dia, calculada só se algum episódio começar hoje
        violadas_hoje, linhas, alertas = set(), [], []
        for r in resultados:
            chave = (c, *chave_episodio(r.regra, r.detalhe))
            ep = self.abertos.get(chave)
            if r.status == VIOLADO:
                violadas_hoje.add(chave)
                if ep is None:
                    classe, causa, principal, n_ord = "PASSIVO", "", "", 0
                    if alocs and i > 0:
                        if cf is None:
                            cf = self._contrafactual(f, i, q_ant, caixa_trade, row, passivos, lim, Pv, vol)
                        violou_sem_ordens = any(
                            x.status == VIOLADO and chave_episodio(x.regra, x.detalhe) == chave[1:] for x in cf)
                        classe = "PASSIVO" if violou_sem_ordens else "ATIVO"
                        if classe == "ATIVO":
                            causa, principal, n_ord = ranquear_ordens(
                                alocs, r, lambda a: self._regras_sem_ordem(a, Pv, vol, q, row, f, lim, i, passivos))
                    ep = Episodio(c, r.regra, r.detalhe, data, i, classe, causa, None, principal, n_ord)
                    self.abertos[chave] = ep
                    self.todos.append(ep)
                ep.du_max = max(ep.du_max, i - ep.i_inicio)
                ep.vencido = ep.classe == "PASSIVO" and ep.du_max > self.prazo
            u = utilizacao(r)
            nivel = nivel_alerta(r, self.soft)
            ep = self.abertos.get(chave) if r.status == VIOLADO else None
            linhas.append({
                "data": data, "fundo_cnpj": c, "regra": r.regra, "detalhe": r.detalhe,
                "valor": r.valor_pos_ordem, "limite": r.limite, "folga": r.folga, "status": r.status,
                "utilizacao": u, "alerta": nivel,
                "classificacao": ep.classe if ep else "", "inicio_episodio": ep.inicio if ep else pd.NaT,
                "du_desde_inicio": (i - ep.i_inicio) if ep else 0,
                "prazo_du": self.prazo if ep and ep.classe == "PASSIVO" else 0,
                "vencido": bool(ep.vencido) if ep else False, "ordens_causa": ep.ordens_causa if ep else "",
                "ordem_principal": ep.ordem_principal if ep else "", "n_ordens": ep.n_ordens if ep else 0,
            })
            if nivel:
                alertas.append({"data": data, "fundo_cnpj": c, "regra": r.regra, "detalhe": r.detalhe,
                                "nivel": nivel, "utilizacao": u, "valor": r.valor_pos_ordem, "limite": r.limite})
        for chave in [k for k in self.abertos if k[0] == c and k not in violadas_hoje]:
            self.abertos.pop(chave).fim = data  # reenquadrado
        return linhas, alertas

    def episodios(self) -> pd.DataFrame:
        return pd.DataFrame([vars(e) for e in self.todos], columns=[f.name for f in fields(Episodio)])

    def _contrafactual(self, f, i, q_ant, caixa_trade, row, passivos, lim, Pv, vol) -> list[ResultadoRegra]:
        """Regras da carteira que existiria sem as ordens do dia: posições e caixa do fechamento anterior
        + mercado e eventos de hoje."""
        from gestorasim.risco.motor import metricas_politica

        tot, liq0 = caixa_trade
        cart = montar_carteira(q_ant, row["caixa_projetado"] - tot, self.meta, Pv, vol, row["caixa_disponivel"] - liq0)
        m = metricas_politica(self.calc, self.fatores, q_ant, i, cart.pl - passivos)
        return todas_as_regras(cart, f.regulamento) + regras_politica(m["var_pct"], m["stress_pct"], lim)

    def _regras_sem_ordem(self, a: dict, Pv, vol, q, row, fundo, lim, i, passivos) -> list[ResultadoRegra]:
        """Regras da carteira de fechamento retirando a alocação `a` (posição, caixa projetado e disponível)."""
        from gestorasim.risco.motor import metricas_politica

        meta = self.meta
        sinal = 1 if a["lado"] == "C" else -1
        q2 = dict(q)
        q2[a["ticker"]] = q2.get(a["ticker"], 0) - sinal * a["quantidade"]
        if q2[a["ticker"]] == 0:
            del q2[a["ticker"]]
        di = classe_de(meta, a["ticker"]) == "FUTURO_DI"
        fluxo = 0.0 if di else -sinal * a["financeiro"]  # caixa que a ordem moveu
        liq0 = PRAZO_LIQUIDACAO_DU[ClasseAtivo(classe_de(meta, a["ticker"]))] == 0
        cart = montar_carteira(q2, row["caixa_projetado"] - fluxo, meta, Pv, vol,
                               row["caixa_disponivel"] - (fluxo if liq0 else 0.0))
        m = metricas_politica(self.calc, self.fatores, q2, i, cart.pl - passivos)
        return todas_as_regras(cart, fundo.regulamento) + regras_politica(m["var_pct"], m["stress_pct"], lim)


def rodar_pos_trade(serie: dict[str, pd.DataFrame], risco: pd.DataFrame, prec, painel: pd.DataFrame,
                    volume: pd.DataFrame, fundos, calc, politica: dict | None = None,
                    prazo_reenquadramento_du: int = PRAZO_REENQUADRAMENTO_DU,
                    soft: float = LIMIAR_SOFT) -> dict[str, pd.DataFrame]:
    """Devolve enquadramento_diario (uma linha por regra/fundo/data), alertas e episodios
    (versão em lote: percorre a série com o mesmo `PosTradeDiario` do orquestrador)."""
    pos = serie["posicoes_diarias"]
    Q = {k: dict(zip(g["ticker"], g["quantidade"])) for k, g in pos.groupby(["data", "fundo_cnpj"])}
    pl = serie["pl_diario"].set_index(["data", "fundo_cnpj"])
    cota = serie.get("cota_diaria")
    passivos = {}
    if cota is not None and len(cota):
        passivos = (cota["adm_a_pagar"] + cota["perf_provisao"] + cota["resgates_a_pagar"]).groupby(
            [cota["data"], cota["fundo_cnpj"]]).first().to_dict()
    led = serie["ledger"]
    cx = led[(led["tipo"] == "CAIXA") & led["origem"].str.match(_ORDEM)]
    cx = cx.assign(liq0=cx["delta"].where(cx["data_liquidacao"] <= cx["data"], 0.0))
    caixa_trades = {k: (r["delta"], r["liq0"])
                    for k, r in cx.groupby(["data", "fundo_cnpj"])[["delta", "liq0"]].sum().iterrows()}
    ordens_dia = _ordens_do_dia(serie)
    risco_idx = risco.set_index(["data", "fundo_cnpj"])
    motor = PosTradeDiario(prec, painel, volume, fundos, calc, politica, prazo_reenquadramento_du, soft)
    linhas, alertas = [], []
    for (data, c), row in pl.sort_index().iterrows():
        i = prec.i_de[data]
        rk = risco_idx.loc[(data, c)]
        ant = prec.datas[i - 1] if i > 0 else None
        l, a = motor.dia(i, data, c, float(row["caixa_projetado"]), float(row["caixa_disponivel"]),
                         Q.get((data, c), {}), Q.get((ant, c), {}), float(rk["var_param_pct"]),
                         float(-rk["stress_pior_pct"]), float(passivos.get((data, c), 0.0)),
                         ordens_dia.get((data, c), []), caixa_trades.get((data, c), (0.0, 0.0)))
        linhas += l
        alertas += a
    return {"enquadramento_diario": pd.DataFrame(linhas), "alertas": pd.DataFrame(alertas),
            "episodios": motor.episodios()}
