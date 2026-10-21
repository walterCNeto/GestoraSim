"""Ledger append-only de quantidades e caixa por fundo."""

from __future__ import annotations

import pandas as pd

from gestorasim.dominio import ClasseAtivo

CAIXA = "CAIXA"
COLUNAS = ["seq", "data", "fundo_cnpj", "tipo", "ticker", "delta", "preco", "origem"]


class Ledger:
    """Só admite novos lançamentos; saldos são derivados (cache incremental)."""

    def __init__(self) -> None:
        self._linhas: list[tuple] = []
        self._qtd: dict[str, dict[str, int]] = {}
        self._caixa: dict[str, float] = {}

    def __len__(self) -> int:
        return len(self._linhas)

    def _lancar(self, data, fundo_cnpj: str, tipo: str, ticker: str, delta: float,
                preco: float | None, origem: str) -> None:
        self._linhas.append((len(self._linhas) + 1, pd.Timestamp(data), fundo_cnpj, tipo, ticker,
                             delta, preco, origem))
        if tipo == "CAIXA":
            self._caixa[fundo_cnpj] = self._caixa.get(fundo_cnpj, 0.0) + delta
        else:
            q = self._qtd.setdefault(fundo_cnpj, {})
            q[ticker] = q.get(ticker, 0) + int(delta)

    def integralizar(self, data, fundo_cnpj: str, valor: float, origem: str = "INTEGRALIZACAO") -> None:
        self._lancar(data, fundo_cnpj, "CAIXA", CAIXA, valor, None, origem)

    def quantidades(self, fundo_cnpj: str) -> dict[str, int]:
        return {t: q for t, q in self._qtd.get(fundo_cnpj, {}).items() if q != 0}

    def caixa(self, fundo_cnpj: str) -> float:
        return self._caixa.get(fundo_cnpj, 0.0)

    def para_df(self) -> pd.DataFrame:
        return pd.DataFrame(self._linhas, columns=COLUNAS)


def aplicar_execucao(ledger: Ledger, *, data, fundo_cnpj: str, ticker: str, classe: str, lado: str,
                     quantidade: int, preco: float, origem: str) -> None:
    """Lança a quantidade e o caixa de uma execução alocada ao fundo.

    Futuro de DI não move caixa (sem ajuste diário/margem neste bloco).
    """
    if quantidade <= 0 or preco <= 0:
        raise ValueError("quantidade e preço devem ser positivos")
    sinal = 1 if lado == "C" else -1
    ledger._lancar(data, fundo_cnpj, "ATIVO", ticker, sinal * quantidade, preco, origem)
    if ClasseAtivo(classe) != ClasseAtivo.FUTURO_DI:
        ledger._lancar(data, fundo_cnpj, "CAIXA", CAIXA, -sinal * quantidade * preco, preco, origem)
