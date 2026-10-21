"""Pré-trade: simula a carteira pós-ordem e checa o regulamento, o caixa e o caixa mínimo em D..D+2."""

from __future__ import annotations

import pandas as pd

from gestorasim.dominio import PRAZO_LIQUIDACAO_DU, ClasseAtivo, LimitesRegulamento
from gestorasim.enquadramento.regras import (
    EPS, VIOLADO, Carteira, ResultadoRegra, _res, montar_carteira, regra_caixa, todas_as_regras,
)

DESENQUADRADO_PREEXISTENTE = "DESENQUADRADO_PREEXISTENTE"
BUFFER_SLIPPAGE = 0.01  # reserva de caixa p/ compras = slippage máximo da execução simulada
HORIZONTE_CAIXA_DU = 2  # perfil de caixa em D, D+1, D+2


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


def choque_ajuste_di(quantidades: dict[str, int], meta: pd.DataFrame, precos: pd.Series,
                     sigma_di: dict[str, float] | None) -> float:
    """Perda de caixa de 1 desvio-padrão do ajuste diário dos futuros de DI: Σ |q| × PU × σ(retorno do PU)."""
    if not sigma_di:
        return 0.0
    return sum(abs(q) * float(precos[tk]) * sigma_di.get(tk, 0.0) for tk, q in quantidades.items()
               if q and meta.loc[tk, "classe"] == ClasseAtivo.FUTURO_DI.value)


def caixa_minimo_projetado(perfil: list[float], choque: float) -> float:
    """Menor saldo liquidado em D..D+2; o choque do ajuste de DI (liquidado em D+1 e D+2) entra a partir de D+1."""
    return min(c - (choque if k >= 1 else 0.0) for k, c in enumerate(perfil))


def regra_caixa_minimo(perfil: list[float], choque: float) -> ResultadoRegra:
    return _res("caixa_minimo_projetado", "D..D+2 c/ choque DI", caixa_minimo_projetado(perfil, choque), 0.0, "min")


def avaliar_pre_trade(quantidades: dict[str, int], caixa: float, reg: LimitesRegulamento,
                      meta: pd.DataFrame, precos: pd.Series, *, ticker: str, lado: str,
                      quantidade: int, volume: pd.Series | None = None,
                      perfil_caixa: list[float] | None = None,
                      sigma_di: dict[str, float] | None = None) -> list[ResultadoRegra]:
    """Resultados por regra da carteira pós-ordem.

    `volume` ativa a liquidez D+1 corrigida (ações limitadas a 20% do volume). `perfil_caixa` =
    saldos liquidados esperados em D, D+1, D+2 antes da ordem (liquidações pendentes incluídas);
    com ele, checa o caixa mínimo projetado com choque de 1 desvio-padrão do ajuste de DI.

    Uma regra só é VIOLADO se a ordem a viola e a agrava; desenquadramento já existente que a
    ordem não piora é reportado como DESENQUADRADO_PREEXISTENTE (não bloqueia).
    """
    classe = meta.loc[ticker, "classe"]
    cart_vazia = montar_carteira({}, 0.0, meta, precos).posicoes
    pre = todas_as_regras(montar_carteira(quantidades, caixa, meta, precos, volume), reg)
    args = dict(ticker=ticker, classe=classe, lado=lado, quantidade=quantidade, preco=float(precos[ticker]))
    q2, c2 = simular_ordem(quantidades, caixa, buffer=0.0, **args)  # PL e percentuais sem a reserva
    _, c_reservado = simular_ordem(quantidades, caixa, **args)  # caixa suficiente com reserva de slippage
    depois = [r if r.regra != "caixa_suficiente" else regra_caixa(Carteira(c_reservado, cart_vazia))[0]
              for r in todas_as_regras(montar_carteira(q2, c2, meta, precos, volume), reg)]
    if perfil_caixa is not None:
        _, c_res_q = simular_ordem(quantidades, 0.0, **args)  # efeito de caixa da ordem (com reserva)
        efeito = c_res_q
        k_liq = min(PRAZO_LIQUIDACAO_DU[ClasseAtivo(classe)], HORIZONTE_CAIXA_DU)
        perfil_pos = [c + (efeito if k >= k_liq else 0.0) for k, c in enumerate(perfil_caixa)]
        pre.append(regra_caixa_minimo(perfil_caixa, choque_ajuste_di(quantidades, meta, precos, sigma_di)))
        depois.append(regra_caixa_minimo(perfil_pos, choque_ajuste_di(q2, meta, precos, sigma_di)))
    antes = {(r.regra, r.detalhe): r for r in pre}
    saida = []
    for r in depois:
        if r.status == VIOLADO:
            p = antes.get((r.regra, r.detalhe))
            folga_pre = p.folga if p else (r.limite if r.sentido == "max" else -r.limite)
            if r.folga >= folga_pre - EPS:  # ordem não agravou
                r = ResultadoRegra(**{**r.asdict(), "status": DESENQUADRADO_PREEXISTENTE})
        saida.append(r)
    return saida


def violacoes(resultados: list[ResultadoRegra]) -> list[ResultadoRegra]:
    return [r for r in resultados if r.status == VIOLADO]
