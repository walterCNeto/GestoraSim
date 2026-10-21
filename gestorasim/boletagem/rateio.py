"""Rateio justo de uma execução agregada entre fundos."""

from __future__ import annotations

CRITERIO = ("rateio proporcional ao alvo; preço médio único; lote={lote}; "
            "resíduo de arredondamento ao maior alvo ({fundo})")


def ratear(alvos: dict[str, int], executado: int, lote: int) -> tuple[dict[str, int], str]:
    """Quantidade por fundo ∝ alvo, arredondada para baixo ao lote; resíduo ao fundo de maior alvo
    (empate: o primeiro na ordem de `alvos`). Aritmética inteira para evitar erro de ponto flutuante."""
    soma = sum(alvos.values())
    if executado > soma:
        raise ValueError("executado maior que a soma dos alvos")
    aloc = {f: (executado * a // soma) // lote * lote for f, a in alvos.items()}
    maior = max(alvos, key=lambda f: alvos[f])  # max devolve o primeiro em caso de empate
    aloc[maior] += executado - sum(aloc.values())
    return aloc, CRITERIO.format(lote=lote, fundo=maior)
