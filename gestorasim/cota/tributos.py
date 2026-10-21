"""Lotes de cotas e tributação simplificada: come-cotas, IR no resgate (FIFO) e IOF."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# IOF regressivo sobre o rendimento, dias corridos 1..29 (0 a partir do 30º dia).
IOF_TABELA = [0.96, 0.93, 0.90, 0.86, 0.83, 0.80, 0.76, 0.73, 0.70, 0.66, 0.63, 0.60, 0.56, 0.53, 0.50,
              0.46, 0.43, 0.40, 0.36, 0.33, 0.30, 0.26, 0.23, 0.20, 0.16, 0.13, 0.10, 0.06, 0.03]


def aliquota_ir_regressiva(dias_corridos: int) -> float:
    """Tabela regressiva: até 180d 22,5%; 181-360d 20%; 361-720d 17,5%; acima de 720d 15%."""
    if dias_corridos <= 180:
        return 0.225
    if dias_corridos <= 360:
        return 0.20
    if dias_corridos <= 720:
        return 0.175
    return 0.15


def aliquota_iof(dias_corridos: int) -> float:
    if dias_corridos < 1:
        return IOF_TABELA[0]
    return IOF_TABELA[dias_corridos - 1] if dias_corridos <= len(IOF_TABELA) else 0.0


@dataclass
class Lote:
    """Lote de cotas de um cotista (FIFO por data de conversão)."""

    id: str
    cotista: str
    data: pd.Timestamp
    cotas: float
    cota_aplicacao: float
    capital: float  # R$ aplicado correspondente às cotas remanescentes
    base_cc: float  # cota-base do próximo come-cotas (cota da aplicação ou do último come-cotas tributado)
    cc_pago: float = 0.0  # come-cotas já pago (R$), a abater do IR no resgate


def come_cotas(lote: Lote, cota: float, aliquota: float) -> tuple[float, float]:
    """Aplica o come-cotas ao lote: cancela cotas no valor do imposto, sem alterar a cota.
    Perda (cota abaixo da base) não gera imposto e a base é mantida (compensação). Devolve (imposto, cotas canceladas)."""
    ganho = (cota - lote.base_cc) * lote.cotas
    if ganho <= 0:
        return 0.0, 0.0
    imposto = aliquota * ganho
    canceladas = imposto / cota
    lote.cotas -= canceladas
    lote.cc_pago += imposto
    lote.base_cc = cota
    return imposto, canceladas


def resgatar_fifo(lotes: list[Lote], cotas_resgate: float, cota: float, data: pd.Timestamp, ir: str,
                  aliquota_fia: float = 0.15, iof_ligado: bool = False) -> tuple[list[dict], float, float]:
    """Consome `cotas_resgate` dos lotes do mais antigo ao mais novo e calcula IR (e IOF) por lote.

    Rendimento da parcela = valor atual + come-cotas já pago − capital; IR = alíquota (regressiva pelos dias
    do lote, ou 15% fixo no FIA) × (rendimento − IOF) − come-cotas já pago, nunca negativo.
    Devolve (partes por lote, IR total, IOF total). Remove os lotes zerados da lista.
    """
    partes, ir_total, iof_total = [], 0.0, 0.0
    restante = cotas_resgate
    for lote in list(lotes):
        if restante <= 1e-12:
            break
        consumidas = min(lote.cotas, restante)
        phi = consumidas / lote.cotas
        ganho = phi * (lote.cotas * cota + lote.cc_pago - lote.capital)
        ganho_pos = max(ganho, 0.0)
        dias = (data - lote.data).days
        iof = aliquota_iof(dias) * ganho_pos if iof_ligado else 0.0
        aliq = aliquota_fia if ir == "FIXA15" else aliquota_ir_regressiva(dias)
        imposto = max(0.0, aliq * (ganho_pos - iof) - phi * lote.cc_pago)
        partes.append({"lote_id": lote.id, "cotista": lote.cotista, "cotas": consumidas, "base": ganho,
                       "aliquota": aliq, "ir": imposto, "iof": iof, "dias": dias})
        ir_total += imposto
        iof_total += iof
        lote.capital -= phi * lote.capital
        lote.cc_pago -= phi * lote.cc_pago
        lote.cotas -= consumidas
        restante -= consumidas
        if lote.cotas <= 1e-12:
            lotes.remove(lote)
    return partes, ir_total, iof_total
