"""Processamento de eventos (dividendo, cupom, amortização) no ledger.

Direito ao evento:
- DIVIDENDO: quantidade ao fim da data-com; o crédito ocorre na data de pagamento.
- CUPOM / AMORTIZACAO (renda fixa, PU ex-cupom na data de pagamento): quantidade ao fim do pregão
  anterior à data de pagamento (quem compra no dia do pagamento já compra ex).
"""

from __future__ import annotations

import pandas as pd

from gestorasim.dados import calendario
from gestorasim.posicao.ledger import Ledger


class ProcessadorEventos:
    def __init__(self, ledger: Ledger, eventos: pd.DataFrame,
                 fracoes_amortizacao: dict[tuple[str, pd.Timestamp], float] | None = None) -> None:
        self.ledger = ledger
        self.eventos = eventos.sort_values(["data_pagamento", "ticker", "tipo"], ignore_index=True)
        self.fracoes = fracoes_amortizacao or {}
        self._feitos: set[int] = set()

    def data_de_direito(self, ev) -> pd.Timestamp:
        if ev["tipo"] == "DIVIDENDO":
            return pd.Timestamp(ev["data_com"])
        return calendario.somar_du(ev["data_com"], -1)

    def processar(self, data, cnpjs: list[str]) -> list[dict]:
        """Credita em caixa os eventos com pagamento em `data`. Devolve os créditos feitos."""
        data = pd.Timestamp(data)
        creditos = []
        dia = self.eventos[self.eventos["data_pagamento"] == data]
        for idx, ev in dia.iterrows():
            if idx in self._feitos:
                continue
            self._feitos.add(idx)
            ref = self.data_de_direito(ev)
            origem = f"EVT:{ev['tipo']}:{ev['ticker']}:{pd.Timestamp(ev['data_com']):%Y%m%d}"
            for cnpj in cnpjs:
                q = self.ledger.quantidade_em(cnpj, ev["ticker"], ref)
                if q <= 0:
                    continue
                valor = q * float(ev["valor_por_unidade"])
                self.ledger.creditar(data, cnpj, ev["ticker"], valor, origem)
                if ev["tipo"] == "AMORTIZACAO":
                    fr = self.fracoes[(ev["ticker"], pd.Timestamp(ev["data_com"]))]
                    self.ledger.reduzir_vne(data, cnpj, ev["ticker"], fr, origem)
                creditos.append({"fundo_cnpj": cnpj, "ticker": ev["ticker"], "tipo": ev["tipo"],
                                 "quantidade": q, "valor": valor})
        return creditos
