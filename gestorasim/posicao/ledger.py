"""Ledger append-only de quantidades, caixa (com data de liquidação) e fator de VNE por fundo."""

from __future__ import annotations

from bisect import bisect_right

import pandas as pd

from gestorasim.dominio import ClasseAtivo
from gestorasim.posicao.liquidacao import data_liquidacao

CAIXA = "CAIXA"
COLUNAS = ["seq", "data", "fundo_cnpj", "tipo", "ticker", "delta", "preco", "origem", "data_liquidacao"]
_JANELA_PENDENTES = pd.Timedelta(days=10)  # liquidação máxima (D+2) com folga p/ feriados


class Ledger:
    """Só admite novos lançamentos, em ordem cronológica; saldos são derivados.

    Tipos de linha: ATIVO (delta = quantidade), CAIXA (delta = R$; `ticker` = ativo de origem ou
    "CAIXA"; vale em `data_liquidacao`) e VNE (delta = fração amortizada do valor nominal).
    """

    def __init__(self) -> None:
        self._linhas: list[tuple] = []
        self._ultima = pd.Timestamp.min
        self._qtd: dict[str, dict[str, int]] = {}
        self._hist: dict[tuple[str, str], tuple[list, list]] = {}  # (cnpj,ticker) -> (datas, qtd acumulada)
        self._caixa: dict[str, float] = {}
        self._vne: dict[tuple[str, str], float] = {}

    def __len__(self) -> int:
        return len(self._linhas)

    def _lancar(self, data, fundo_cnpj: str, tipo: str, ticker: str, delta: float, preco: float | None,
                origem: str, data_liq=None) -> None:
        data = pd.Timestamp(data)
        if data < self._ultima:
            raise ValueError(f"lançamento fora de ordem cronológica: {data} < {self._ultima}")
        self._ultima = data
        liq = pd.Timestamp(data_liq) if data_liq is not None else data
        self._linhas.append((len(self._linhas) + 1, data, fundo_cnpj, tipo, ticker, delta, preco, origem, liq))
        if tipo == "CAIXA":
            self._caixa[fundo_cnpj] = self._caixa.get(fundo_cnpj, 0.0) + delta
        elif tipo == "ATIVO":
            q = self._qtd.setdefault(fundo_cnpj, {})
            q[ticker] = q.get(ticker, 0) + int(delta)
            datas, acum = self._hist.setdefault((fundo_cnpj, ticker), ([], []))
            datas.append(data)
            acum.append(q[ticker])
        elif tipo == "VNE":
            self._vne[(fundo_cnpj, ticker)] = self._vne.get((fundo_cnpj, ticker), 1.0) * (1 - delta)

    # ------------------------------------------------------------------ lançamentos
    def integralizar(self, data, fundo_cnpj: str, valor: float, origem: str = "INTEGRALIZACAO") -> None:
        self._lancar(data, fundo_cnpj, "CAIXA", CAIXA, valor, None, origem)

    def creditar(self, data, fundo_cnpj: str, ticker: str, valor: float, origem: str, data_liq=None) -> None:
        """Movimento de caixa avulso (evento, ajuste de futuro). `data_liq` default = `data` (D+0)."""
        self._lancar(data, fundo_cnpj, "CAIXA", ticker, valor, None, origem, data_liq)

    def reduzir_vne(self, data, fundo_cnpj: str, ticker: str, fracao: float, origem: str) -> None:
        if not 0 < fracao <= 1:
            raise ValueError("fração de amortização deve estar em (0, 1]")
        self._lancar(data, fundo_cnpj, "VNE", ticker, fracao, None, origem)

    # ------------------------------------------------------------------ consultas
    def quantidades(self, fundo_cnpj: str) -> dict[str, int]:
        return {t: q for t, q in self._qtd.get(fundo_cnpj, {}).items() if q != 0}

    def quantidade_em(self, fundo_cnpj: str, ticker: str, data) -> int:
        """Quantidade ao fim do dia `data` (inclui operações do próprio dia)."""
        datas, acum = self._hist.get((fundo_cnpj, ticker), ([], []))
        i = bisect_right(datas, pd.Timestamp(data))
        return acum[i - 1] if i else 0

    def fator_vne(self, fundo_cnpj: str, ticker: str) -> float:
        return self._vne.get((fundo_cnpj, ticker), 1.0)

    def caixa_projetado(self, fundo_cnpj: str) -> float:
        """Caixa incluindo tudo que está a liquidar (a receber e a pagar)."""
        return self._caixa.get(fundo_cnpj, 0.0)

    caixa = caixa_projetado  # compatibilidade: o saldo "padrão" é o projetado

    def caixa_disponivel(self, fundo_cnpj: str, data) -> float:
        """Caixa já liquidado até `data` (inclusive): projetado menos os lançamentos que liquidam depois."""
        data = pd.Timestamp(data)
        a_liquidar = 0.0
        for lin in reversed(self._linhas):
            if lin[1] < data - _JANELA_PENDENTES:
                break
            if lin[2] == fundo_cnpj and lin[3] == "CAIXA" and lin[8] > data:
                a_liquidar += lin[5]
        return self.caixa_projetado(fundo_cnpj) - a_liquidar

    def linhas_desde(self, n: int) -> list[tuple]:
        """Lançamentos de índice >= n (em ordem)."""
        return self._linhas[n:]

    def para_df(self) -> pd.DataFrame:
        return pd.DataFrame(self._linhas, columns=COLUNAS)


def aplicar_execucao(ledger: Ledger, *, data, fundo_cnpj: str, ticker: str, classe: str, lado: str,
                     quantidade: int, preco: float, origem: str,
                     preco_referencia: float | None = None) -> None:
    """Lança a quantidade e o caixa de uma execução alocada ao fundo.

    O caixa de ações/títulos liquida conforme `liquidacao`. Futuro de DI não move caixa pela
    quantidade; se `preco_referencia` (PU de marcação do dia) vier, lança o ajuste do dia da
    entrada/saída, (PU_ref − preço) × quantidade com sinal, liquidado em D+1.
    """
    if quantidade <= 0 or preco <= 0:
        raise ValueError("quantidade e preço devem ser positivos")
    sinal = 1 if lado == "C" else -1
    classe = ClasseAtivo(classe)
    ledger._lancar(data, fundo_cnpj, "ATIVO", ticker, sinal * quantidade, preco, origem)
    liq = data_liquidacao(data, classe)
    if classe != ClasseAtivo.FUTURO_DI:
        ledger._lancar(data, fundo_cnpj, "CAIXA", ticker, -sinal * quantidade * preco, preco, origem, liq)
    elif preco_referencia is not None:
        aj = (preco_referencia - preco) * sinal * quantidade
        ledger._lancar(data, fundo_cnpj, "CAIXA", ticker, aj, preco, f"AJUSTE_DI_ENTRADA:{origem}", liq)
