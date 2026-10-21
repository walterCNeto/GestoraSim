"""Prazos de liquidação financeira por classe (dias úteis após a data da operação)."""

from __future__ import annotations

import pandas as pd

from gestorasim.dados import calendario
from gestorasim.dominio import PRAZO_LIQUIDACAO_DU, ClasseAtivo

def data_liquidacao(data, classe: ClasseAtivo | str) -> pd.Timestamp:
    n = PRAZO_LIQUIDACAO_DU[ClasseAtivo(classe)]
    return pd.Timestamp(data) if n == 0 else calendario.somar_du(data, n)
