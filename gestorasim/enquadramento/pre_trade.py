"""Pré-trade: simula a carteira pós-ordem e checa o regulamento + caixa."""

from __future__ import annotations

import pandas as pd

from gestorasim.dominio import ClasseAtivo, LimitesRegulamento
from gestorasim.enquadramento.regras import (
    EPS, VIOLADO, Carteira, ResultadoRegra, montar_carteira, regra_caixa, todas_as_regras,
)

DESENQUADRADO_PREEXISTENTE = "DESENQUADRADO_PREEXISTENTE"
BUFFER_SLIPPAGE = 0.01  # reserva de caixa p/ compras = slippage máximo da execução simulada


def lote(classe: ClasseAtivo | str) -> int:
    return 100 if ClasseAtivo(classe) == ClasseAtivo.ACAO else 1


def simular_ordem(quantidades: dict[str, int], caixa: float, *, ticker: str, classe: str,
                  lado: str, quantidade: int, preco: float,
                  buffer: float = BUFFER_SLIPPAGE) -> tuple[dict[str, int], float]:
    """Carteira pós-ordem. Compras reservam caixa com `buffer` de slippage; futuros não movem caixa."""
    sinal = 1 if lado == "C" else -1
    novas = dict(quantidades)
    novas[ticker] = novas.get(ticker, 0) + sinal * quantidade
    if ClasseAtivo(classe) == ClasseAtivo.FUTURO_DI:
        return novas, caixa
    px = preco * (1 + buffer) if sinal > 0 else preco
    return novas, caixa - sinal * quantidade * px


def avaliar_pre_trade(quantidades: dict[str, int], caixa: float, reg: LimitesRegulamento,
                      meta: pd.DataFrame, precos: pd.Series, *, ticker: str, lado: str,
                      quantidade: int) -> list[ResultadoRegra]:
    """Resultados por regra da carteira pós-ordem.

    Uma regra só é VIOLADO se a ordem a viola e a agrava; desenquadramento já existente que a
    ordem não piora é reportado como DESENQUADRADO_PREEXISTENTE (não bloqueia).
    """
    cart_vazia = montar_carteira({}, 0.0, meta, precos).posicoes
    antes = {(r.regra, r.detalhe): r
             for r in todas_as_regras(montar_carteira(quantidades, caixa, meta, precos), reg)}
    args = dict(ticker=ticker, classe=meta.loc[ticker, "classe"], lado=lado, quantidade=quantidade,
                preco=float(precos[ticker]))
    q2, c2 = simular_ordem(quantidades, caixa, buffer=0.0, **args)  # PL e percentuais sem a reserva
    _, c_reservado = simular_ordem(quantidades, caixa, **args)  # caixa suficiente com reserva de slippage
    depois = [r if r.regra != "caixa_suficiente" else regra_caixa(Carteira(c_reservado, cart_vazia))[0]
              for r in todas_as_regras(montar_carteira(q2, c2, meta, precos), reg)]
    saida = []
    for r in depois:
        if r.status == VIOLADO:
            pre = antes.get((r.regra, r.detalhe))
            folga_pre = pre.folga if pre else (r.limite if r.sentido == "max" else -r.limite)
            if r.folga >= folga_pre - EPS:  # ordem não agravou
                r = ResultadoRegra(**{**r.asdict(), "status": DESENQUADRADO_PREEXISTENTE})
        saida.append(r)
    return saida


def violacoes(resultados: list[ResultadoRegra]) -> list[ResultadoRegra]:
    return [r for r in resultados if r.status == VIOLADO]
