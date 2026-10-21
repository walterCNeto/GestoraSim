"""Prazos de liquidação financeira por classe (dias úteis após a data da operação)."""

from __future__ import annotations

import pandas as pd

from gestorasim.dados import calendario
from gestorasim.dominio import ClasseAtivo

PRAZO_LIQUIDACAO_DU = {
    ClasseAtivo.ACAO: 2,
    ClasseAtivo.TITULO_PUBLICO: 0,
    ClasseAtivo.DEBENTURE: 0,  # tratada como título de renda fixa (D+0)
    ClasseAtivo.FUTURO_DI: 1,  # ajuste diário liquidado em D+1
}


def data_liquidacao(data, classe: ClasseAtivo | str) -> pd.Timestamp:
    n = PRAZO_LIQUIDACAO_DU[ClasseAtivo(classe)]
    return pd.Timestamp(data) if n == 0 else calendario.somar_du(data, n)
