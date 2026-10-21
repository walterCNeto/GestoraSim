"""Remuneração do caixa: saldo positivo rende CDI; saldo negativo (descoberto) paga CDI + spread."""

SPREAD_CAIXA_NEGATIVO = 0.02  # a.a., parâmetro


def juros_caixa(saldo: float, cdi_aa: float, spread_negativo: float = SPREAD_CAIXA_NEGATIVO) -> float:
    """Juros de 1 dia útil (base 252) sobre o saldo do fechamento anterior."""
    taxa = cdi_aa if saldo >= 0 else cdi_aa + spread_negativo
    return saldo * ((1 + taxa) ** (1 / 252) - 1)
