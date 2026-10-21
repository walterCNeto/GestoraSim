"""Rateio justo de uma execução agregada entre fundos."""

from __future__ import annotations

CRITERIO = ("rateio proporcional ao alvo; preço médio único; lote={lote}; resíduo por maior resto "
            "com teto no alvo de cada fundo; cancelado={cancelado}")


def ratear(alvos: dict[str, int], executado: int, lote: int) -> tuple[dict[str, int], str, int]:
    """Rateia `executado` entre os fundos: quantidade ∝ alvo, múltipla do lote e nunca acima do alvo.

    1. piso de cada fundo = ⌊executado × alvo / Σalvo⌋ arredondado para baixo ao lote;
    2. o resíduo (em lotes) vai, um lote por vez, ao fundo de maior resto fracionário que ainda caiba
       no teto (alvo arredondado para baixo ao lote); empate: maior alvo, depois ordem de `alvos`;
    3. o que não couber nos tetos é cancelado (só ocorre se os alvos não forem múltiplos do lote).
    Aritmética inteira (restos comparados como numeradores sobre Σalvo).
    Devolve (alocação por fundo, critério para a trilha, quantidade cancelada).
    """
    soma = sum(alvos.values())
    if executado > soma:
        raise ValueError("executado maior que a soma dos alvos")
    teto = {f: a // lote * lote for f, a in alvos.items()}
    aloc = {f: min((executado * a // soma) // lote * lote, teto[f]) for f, a in alvos.items()}
    resto = {f: executado * a - aloc[f] * soma for f, a in alvos.items()}
    residuo = executado - sum(aloc.values())
    while residuo >= lote:
        cabem = [f for f in alvos if aloc[f] + lote <= teto[f]]
        if not cabem:
            break
        f = max(cabem, key=lambda f: (resto[f], alvos[f]))  # max devolve o primeiro em empate total
        aloc[f] += lote
        resto[f] -= lote * soma
        residuo -= lote
    return aloc, CRITERIO.format(lote=lote, cancelado=residuo), residuo
