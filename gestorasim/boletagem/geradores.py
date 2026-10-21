"""Gerador de ordens de exemplo: rebalanceamento, ordens aleatórias e ordens violadoras de propósito."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gestorasim.boletagem.modelos import OrdemAgregada
from gestorasim.dominio import ClasseAtivo, ClasseFundo, Fundo
from gestorasim.enquadramento import lote, montar_carteira
from gestorasim.posicao import Ledger

TILT = 0.25  # amplitude do tilt da carteira-alvo em cada posição
MIN_TRADE_PL = 0.0005  # ignora ajustes menores que 0,05% do PL


class _Seq:
    def __init__(self) -> None:
        self.n = 0

    def __call__(self) -> str:
        self.n += 1
        return f"ORD{self.n:04d}"


def _qtd(valor: float, preco: float, classe: str) -> int:
    lt = lote(classe)
    return int(valor // (preco * lt)) * lt


def _agrupar(deltas: list[tuple[str, str, int]], meta, data, novo_id, origem, usuario) -> list[OrdemAgregada]:
    """(cnpj, ticker, delta) -> ordens agregadas por (ticker, lado)."""
    blocos: dict[tuple[str, str], dict[str, int]] = {}
    for cnpj, tk, d in deltas:
        blocos.setdefault((tk, "C" if d > 0 else "V"), {})[cnpj] = abs(d)
    return [
        OrdemAgregada(id=novo_id(), data=data, ticker=tk, classe=ClasseAtivo(meta.loc[tk, "classe"]),
                      lado=lado, alvos=alvos, origem=origem, usuario=usuario)
        for (tk, lado), alvos in sorted(blocos.items())
    ]


def ordens_rebalanceamento(ledger: Ledger, fundos: list[Fundo], meta, precos: pd.Series, data,
                           rng: np.random.Generator, novo_id) -> list[OrdemAgregada]:
    """Carteira-alvo = carteira atual com tilt em cada posição, preservando o total de cada classe."""
    deltas = []
    for f in fundos:
        q = ledger.quantidades(f.cnpj)
        cart = montar_carteira(q, ledger.caixa(f.cnpj), meta, precos)
        pl = cart.pl
        for classe, grupo in cart.posicoes.groupby("classe"):
            expo = grupo["exposicao"]
            fator = rng.uniform(1 - TILT, 1 + TILT, len(grupo))
            novo = expo * fator
            novo = novo / novo.sum() * expo.sum()
            for tk, valor_alvo in novo.items():
                sinal = 1 if q[tk] > 0 else -1
                q_alvo = sinal * _qtd(valor_alvo, float(precos[tk]), str(classe))
                delta = q_alvo - q[tk]
                if abs(delta) >= lote(classe) and abs(delta) * float(precos[tk]) >= MIN_TRADE_PL * pl:
                    deltas.append((f.cnpj, tk, int(delta)))
    return _agrupar(deltas, meta, data, novo_id, "REBALANCEAMENTO", "gestor.rebalanceamento")


def ordens_aleatorias(ledger: Ledger, fundos: list[Fundo], meta, precos: pd.Series, data,
                      rng: np.random.Generator, novo_id, n: int = 6) -> list[OrdemAgregada]:
    tickers = list(meta.index[meta.index.isin(precos.dropna().index)])
    ordens = []
    for _ in range(n):
        tk = tickers[rng.integers(len(tickers))]
        classe = meta.loc[tk, "classe"]
        lado = "C" if rng.random() < 0.6 else "V"
        escolhidos = rng.choice(len(fundos), size=int(rng.integers(1, 4)), replace=False)
        alvos = {}
        for i in escolhidos:
            f = fundos[i]
            pl = montar_carteira(ledger.quantidades(f.cnpj), ledger.caixa(f.cnpj), meta, precos).pl
            q = max(lote(classe), _qtd(rng.uniform(0.005, 0.02) * pl, float(precos[tk]), classe))
            if lado == "V":
                q = min(q, max(ledger.quantidades(f.cnpj).get(tk, 0), 0))
            if q > 0:
                alvos[f.cnpj] = int(q)
        if not alvos:
            lado, alvos = "C", {fundos[escolhidos[0]].cnpj: int(lote(classe))}
        ordens.append(OrdemAgregada(id=novo_id(), data=data, ticker=tk, classe=ClasseAtivo(classe),
                                    lado=lado, alvos=alvos, origem="ALEATORIA", usuario="gestor.discricionario"))
    return ordens


def ordens_violadoras(ledger: Ledger, fundos: list[Fundo], meta, precos: pd.Series, data,
                      novo_id) -> list[OrdemAgregada]:
    """Ordens que violam o regulamento de propósito (a de rating pede override, para demonstrá-lo)."""
    por_classe = {f.classe: f for f in fundos}
    fia, mm, rf = (por_classe[ClasseFundo.FIA], por_classe[ClasseFundo.MULTIMERCADO],
                   por_classe[ClasseFundo.RENDA_FIXA])

    def pl(f):
        return montar_carteira(ledger.quantidades(f.cnpj), ledger.caixa(f.cnpj), meta, precos).pl

    def qtd(f, tk, fracao):
        return max(lote(meta.loc[tk, "classe"]), _qtd(fracao * pl(f), float(precos[tk]), meta.loc[tk, "classe"]))

    def ordem(f, tk, fracao, rotulo, **kw):
        return OrdemAgregada(id=novo_id(), data=data, ticker=tk, classe=ClasseAtivo(meta.loc[tk, "classe"]),
                             lado="C", alvos={f.cnpj: int(qtd(f, tk, fracao))},
                             origem=f"VIOLADORA:{rotulo}", usuario="gestor.discricionario", **kw)

    maior_acao = max((t for t in ledger.quantidades(fia.cnpj) if meta.loc[t, "classe"] == "ACAO"),
                     key=lambda t: ledger.quantidades(fia.cnpj)[t] * float(precos[t]))
    di = meta.index[meta["classe"] == "FUTURO_DI"][0]
    ntnb = meta.index[meta.index.str.startswith("NTNB")][0]
    return [
        ordem(fia, "DLOG14", 0.05, "classe"),           # debênture num FIA
        ordem(fia, maior_acao, 0.10, "emissor"),         # concentração em um emissor
        ordem(mm, di, 1.00, "alavancagem"),              # nocional de DI = 100% do PL
        ordem(mm, ntnb, 1.20, "caixa"),                  # compra acima do caixa
        ordem(rf, "EVAR15", 0.08, "rating",
              pedido_override=("diretor.risco", "Excecao aprovada em comite de risco 2026-10")),
    ]


def gerar_ordens_exemplo(ledger: Ledger, fundos: list[Fundo], meta, precos: pd.Series, data,
                         seed: int, n_aleatorias: int = 6) -> list[OrdemAgregada]:
    """Vendas primeiro (liberam caixa), depois compras; dentro de cada lado, ordem de criação."""
    rng = np.random.default_rng([seed, 300])
    novo_id = _Seq()
    ordens = (ordens_rebalanceamento(ledger, fundos, meta, precos, data, rng, novo_id)
              + ordens_aleatorias(ledger, fundos, meta, precos, data, rng, novo_id, n_aleatorias)
              + ordens_violadoras(ledger, fundos, meta, precos, data, novo_id))
    ordens = sorted(ordens, key=lambda o: (o.lado.value != "V", o.id))
    for i, o in enumerate(ordens, 1):  # ids sequenciais na ordem de processamento
        o.id = f"ORD{i:04d}"
    return ordens
