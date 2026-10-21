"""Ordens de gestão de caixa e de reenquadramento (geradas antes das ordens regulares do dia).

- Gestão de caixa (origem GESTAO_CAIXA nas vendas e REINVESTIMENTO nas compras): o buffer mínimo de cada fundo é o choque de ajuste de DI de 3σ mais as liquidações e saídas do
  passivo previstas em D..D+2 (o perfil de caixa do pré-trade). Se o caixa mínimo do perfil não cobre o choque,
  vende o ativo mais líquido (menos dias para zerar a 20% do volume; de preferência o que liquida a tempo) para
  recompor. Caixa acima do alvo estratégico do fundo (+ tolerância) é reinvestido nas posições existentes, pro rata.
- Reenquadramento: toda violação de máximo do regulamento (emissor, classe, rating, alavancagem) ou da política de
  risco (VaR, stress) no início do pregão — portanto não causada pelas ordens do dia — gera ordens de venda (tag
  REENQUADRAMENTO) que levam a métrica até 95% do limite (folga de 5%). Passam pelo pré-trade como as demais.
"""

from __future__ import annotations

import math

import pandas as pd

from gestorasim.boletagem.geradores import _agrupar, _Seq
from gestorasim.boletagem.modelos import OrdemAgregada
from gestorasim.dominio import PARTICIPACAO_LIQUIDEZ, PRAZO_LIQUIDACAO_DU, ClasseAtivo
from gestorasim.enquadramento import lote, montar_carteira
from gestorasim.enquadramento.pre_trade import caixa_minimo_projetado, choque_ajuste_di
from gestorasim.enquadramento.regras import VIOLADO, todas_as_regras
from gestorasim.enquadramento.regras import classe_de

K_SIGMA_BUFFER = 3.0
TOLERANCIA_EXCESSO = 0.015  # fração do PL acima do alvo de caixa que dispara reinvestimento
FOLGA_REENQUADRAMENTO = 0.05  # reduz até (1 − 5%) do limite
REGRAS_DE_MAXIMO = ("limite_emissor", "limite_classe", "limite_rating", "alavancagem")


def _qtd_vender(valor: float, preco: float, classe: str, posicao: int, teto_volume: float | None = None) -> int:
    lt = lote(classe)
    q = int(math.ceil(valor / preco / lt)) * lt
    q = min(q, posicao)
    if teto_volume is not None:
        q = min(q, max(lt, int(PARTICIPACAO_LIQUIDEZ * teto_volume) // lt * lt))
    return max(q, 0)


def ordens_gestao_caixa(mesa, caixa_alvo: dict[str, float], novo_id: _Seq, k_sigma: float = K_SIGMA_BUFFER,
                        tolerancia_excesso: float = TOLERANCIA_EXCESSO) -> list[OrdemAgregada]:
    meta, P, vol, led = mesa.meta, mesa.precos, mesa.volume, mesa.ledger
    deltas: list[tuple[str, str, int]] = []
    for c in mesa.fundos:
        q = led.quantidades(c)
        pl = montar_carteira(q, led.caixa_projetado(c), meta, P).pl
        perfil = mesa.perfil_caixa(c)
        choque = k_sigma * choque_ajuste_di(q, meta, P, mesa.sigma_di)
        folga = caixa_minimo_projetado(perfil, choque)
        if folga < 0:
            falta = -folga * 1.02
            k_viol = next(k for k, v in enumerate(perfil) if v - (choque if k >= 1 else 0.0) < 0)
            candidatos = [(abs(qt) / (PARTICIPACAO_LIQUIDEZ * max(float(vol.get(tk, 0.0)), 1.0)), tk, qt)
                          for tk, qt in q.items() if qt > 0 and classe_de(meta, tk) != ClasseAtivo.FUTURO_DI.value
                          and tk in P.index]
            a_tempo = [x for x in candidatos
                       if PRAZO_LIQUIDACAO_DU[ClasseAtivo(classe_de(meta, x[1]))] <= k_viol]
            if a_tempo or candidatos:
                _, tk, qt = min(a_tempo or candidatos)
                n = _qtd_vender(falta, float(P[tk]), classe_de(meta, tk), qt, float(vol.get(tk, 0.0)))
                if n > 0:
                    deltas.append((c, tk, -n))
        else:
            excesso = folga - (caixa_alvo[c] + tolerancia_excesso) * pl
            if excesso > 0:
                pos = {tk: qt * float(P[tk]) for tk, qt in q.items() if qt > 0 and tk in P.index
                       and classe_de(meta, tk) != ClasseAtivo.FUTURO_DI.value}
                total = sum(pos.values())
                for tk, v in pos.items():
                    n = int(excesso * v / total // (float(P[tk]) * lote(classe_de(meta, tk)))) * lote(classe_de(meta, tk))
                    if n > 0:
                        deltas.append((c, tk, n))
    vendas = [x for x in deltas if x[2] < 0]  # recompõe o caixa
    compras = [x for x in deltas if x[2] > 0]  # reinveste o excesso de caixa
    return (_agrupar(vendas, meta, mesa.data.date(), novo_id, "GESTAO_CAIXA", "gestor.caixa")
            + _agrupar(compras, meta, mesa.data.date(), novo_id, "REINVESTIMENTO", "gestor.caixa"))


def _vender_ate(cart, ordenadas: list[str], reduzir: float) -> dict[str, float]:
    """Distribui a redução de exposição (R$) pelos tickers na ordem dada, até esgotá-la."""
    out = {}
    for tk in ordenadas:
        if reduzir <= 0:
            break
        v = min(reduzir, float(cart.posicoes.loc[tk, "exposicao"]))
        out[tk] = v
        reduzir -= v
    return out


def ordens_reenquadramento(mesa, novo_id: _Seq, indice_dia: int, politica=None,
                           persistencia: dict | None = None, dias_min: int = 1) -> list[OrdemAgregada]:
    """Vendas que corrigem violações de máximo existentes no início do pregão. `politica`: objeto com
    `vendas(cnpj, classe_fundo, quantidades, indice_dia, pl) -> {ticker: quantidade a vender}` (VaR e stress), com
    o PL líquido. Histerese: com `persistencia` (contadores de pregões consecutivos em violação, guardados pelo
    chamador) e `dias_min` > 1, só age em violações que duram `dias_min` pregões seguidos."""
    meta, P, vol, led = mesa.meta, mesa.precos, mesa.volume, mesa.ledger
    deltas: list[tuple[str, str, int]] = []
    rotulos: dict[tuple[str, str], str] = {}
    for c, f in mesa.fundos.items():
        q = led.quantidades(c)
        cart = montar_carteira(q, led.caixa_projetado(c), meta, P, vol, led.caixa_disponivel(c, mesa.data))
        pl, p = cart.pl, cart.posicoes
        vender: dict[str, float] = {}
        rotulo = {}
        violadas = [r for r in todas_as_regras(cart, f.regulamento) if r.status == VIOLADO and r.regra in REGRAS_DE_MAXIMO]
        chaves = {r: (c, r.regra, "" if r.regra == "limite_emissor" else r.detalhe) for r in violadas}
        pol = (politica.vendas(c, f.classe, q, indice_dia, pl - mesa.passivos.get(c, 0.0))
               if politica is not None else {})
        agir = set(chaves.values()) | ({(c, "politica", "")} if any(v > 0 for v in pol.values()) else set())
        if persistencia is not None:  # histerese: contadores de violação consecutiva
            for k in [k for k in persistencia if k[0] == c and k not in agir]:
                del persistencia[k]
            for k in agir:
                persistencia[k] = persistencia.get(k, 0) + 1
            agir = {k for k in agir if persistencia[k] >= dias_min}
        for r in violadas:
            if chaves[r] not in agir:
                continue
            alvo_valor = (1 - FOLGA_REENQUADRAMENTO) * r.limite * pl
            if r.regra == "limite_emissor":
                grupo = p[(p["emissor"] == r.detalhe) & ~p["classe"].isin([ClasseAtivo.TITULO_PUBLICO, ClasseAtivo.FUTURO_DI])]
                atual = float(grupo["valor"].sum())
            elif r.regra == "limite_classe":
                grupo, atual = p[p["classe"] == r.detalhe], float(p.loc[p["classe"] == r.detalhe, "exposicao"].sum())
            elif r.regra == "limite_rating":
                grupo = p[(p["classe"] == ClasseAtivo.DEBENTURE) & (p["rating"] == r.detalhe)]
                atual = float(grupo["exposicao"].sum())
            else:  # alavancagem: zera primeiro o nocional de DI, depois o resto
                grupo = p.assign(_ordem=(p["classe"] != ClasseAtivo.FUTURO_DI).astype(int))
                grupo = grupo.sort_values(["_ordem", "exposicao"], ascending=[True, False])
                atual = float(p["exposicao"].sum())
            reduzir = atual - alvo_valor
            ordenados = list(grupo.index) if r.regra == "alavancagem" else list(
                grupo.sort_values("exposicao", ascending=False).index)
            for tk, v in _vender_ate(cart, [t for t in ordenados if p.loc[t, "quantidade"] > 0], reduzir).items():
                vender[tk] = max(vender.get(tk, 0.0), v)
                rotulo.setdefault(tk, r.regra)
        if (c, "politica", "") in agir:
            for tk, qt in pol.items():
                if qt > 0:
                    vender[tk] = max(vender.get(tk, 0.0), qt * float(P[tk]))
                    rotulo.setdefault(tk, "politica")
        for tk, valor in vender.items():
            n = _qtd_vender(valor, float(P[tk]), classe_de(meta, tk), int(q[tk]))
            if n > 0:
                deltas.append((c, tk, -n))
                rotulos[(c, tk)] = rotulo[tk]
    ordens = _agrupar(deltas, meta, mesa.data.date(), novo_id, "REENQUADRAMENTO", "gestor.reenquadramento")
    for o in ordens:  # tag da regra na origem, para a trilha
        regra = next(rotulos[(c, o.ticker)] for c in o.alvos if (c, o.ticker) in rotulos)
        o.origem = f"REENQUADRAMENTO:{regra}"
    return ordens
