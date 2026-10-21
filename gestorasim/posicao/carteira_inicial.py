"""Carteira inicial D0 por fundo, gerada do universo com seed e verificada contra o regulamento."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gestorasim.dominio import ClasseAtivo, ClasseFundo, Fundo
from gestorasim.enquadramento import lote, montar_carteira, todas_as_regras
from gestorasim.enquadramento.regras import VIOLADO
from gestorasim.posicao.ledger import Ledger, aplicar_execucao

PL_INICIAL = 100_000_000.0


def _ruido(rng, n: int, amp: float) -> np.ndarray:
    return rng.uniform(1 - amp, 1 + amp, n)


def _grupo(rng, tickers: list[str], total: float, amp: float = 0.2) -> dict[str, float]:
    w = _ruido(rng, len(tickers), amp)
    return dict(zip(tickers, w / w.sum() * total))


def pesos_template(classe: ClasseFundo, meta: pd.DataFrame, rng: np.random.Generator) -> dict[str, float]:
    """Pesos (fração do PL; nocional para DI) por classe de fundo. O caixa é o resíduo."""
    def da_classe(c: ClasseAtivo) -> list[str]:
        return list(meta.index[meta["classe"] == c.value])

    def com_rating(*ratings: str) -> list[str]:
        return [t for r in ratings for t in meta.index[(meta["classe"] == "DEBENTURE") & (meta["rating"] == r)]]

    acoes, tps, dis = da_classe(ClasseAtivo.ACAO), da_classe(ClasseAtivo.TITULO_PUBLICO), da_classe(ClasseAtivo.FUTURO_DI)
    ltn = [t for t in tps if t.startswith("LTN")]
    ntnb = [t for t in tps if t.startswith("NTNB")]
    pick = lambda lst, n: [lst[i] for i in sorted(rng.choice(len(lst), n, replace=False))]

    if classe == ClasseFundo.FIA:
        return {**_grupo(rng, pick(acoes, 10), 0.85), **_grupo(rng, pick(ltn, 1), 0.10)}
    if classe == ClasseFundo.MULTIMERCADO:
        return {**_grupo(rng, pick(acoes, 6), 0.25),
                **_grupo(rng, pick(ltn, 1) + pick(ntnb, 1), 0.40),
                **_grupo(rng, com_rating("AAA", "AA+", "AA-"), 0.15, 0.1),
                **_grupo(rng, pick(dis, 1), 0.40)}
    if classe == ClasseFundo.RENDA_FIXA:
        base = {"AAA": 0.18, "AA+": 0.14, "AA-": 0.11, "A+": 0.07, "A-": 0.05}
        deb = {t: base[meta.loc[t, "rating"]] * f
               for t, f in zip(com_rating(*base), _ruido(rng, 5, 0.05))}
        s = sum(deb.values())
        deb = {t: v / s * 0.55 for t, v in deb.items()}
        return {**deb, **_grupo(rng, pick(ltn, 1) + pick(ntnb, 1), 0.20), **_grupo(rng, pick(dis, 1), 0.15)}
    raise ValueError(classe)


def quantidade_para_valor(valor: float, preco: float, classe: str) -> int:
    lt = lote(classe)
    return int(valor // (preco * lt)) * lt


def gerar_carteira_inicial(ledger: Ledger, fundos: list[Fundo], meta: pd.DataFrame, precos: pd.Series,
                           data, seed: int, pl: float = PL_INICIAL) -> None:
    """Integraliza `pl` e compra a carteira-template de cada fundo ao fechamento de `data`."""
    for k, f in enumerate(fundos):
        rng = np.random.default_rng([seed, 200 + k])
        ledger.integralizar(data, f.cnpj, pl)
        for tk, w in pesos_template(f.classe, meta, rng).items():
            q = quantidade_para_valor(w * pl, float(precos[tk]), meta.loc[tk, "classe"])
            if q > 0:
                aplicar_execucao(ledger, data=data, fundo_cnpj=f.cnpj, ticker=tk,
                                 classe=meta.loc[tk, "classe"], lado="C", quantidade=q,
                                 preco=float(precos[tk]), origem="CARTEIRA_INICIAL")
        cart = montar_carteira(ledger.quantidades(f.cnpj), ledger.caixa_projetado(f.cnpj), meta, precos)
        ruins = [r for r in todas_as_regras(cart, f.regulamento) if r.status == VIOLADO]
        if ruins:
            raise RuntimeError(f"carteira inicial de {f.nome} viola o regulamento: {ruins}")
