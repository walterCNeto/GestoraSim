"""Painel oficial de preços (mtm) com cache: memória por processo + parquet em disco.

O mtm é a fonte única de preço do sistema; calcular o painel (36 ativos × 504 datas) leva alguns
segundos, então ele é guardado com uma impressão digital das entradas (seed + tabelas de mercado).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from gestorasim.mtm.precificacao import Precificador

_ENTRADAS = ("calendario", "ativos", "precos_acoes", "curva_pre", "cdi", "ipca", "eventos")
_MEMO: dict[str, pd.DataFrame] = {}


def impressao_digital(tabelas: dict[str, pd.DataFrame], seed: int) -> str:
    h = hashlib.sha1(f"seed={seed}".encode())
    for nome in _ENTRADAS:
        h.update(pd.util.hash_pandas_object(tabelas[nome].astype(str), index=False).values.tobytes())
    return h.hexdigest()


def painel_oficial(tabelas: dict[str, pd.DataFrame], seed: int, destino: Path | str | None = None) -> pd.DataFrame:
    """Preços (linhas = datas, colunas = tickers). Reaproveita memória ou parquet quando as entradas coincidem."""
    chave = impressao_digital(tabelas, seed)
    arq = meta = None
    if destino is not None:
        arq, meta = Path(destino) / "mtm_painel.parquet", Path(destino) / "mtm_painel.json"
    em_disco = arq is not None and arq.exists() and meta.exists() and json.loads(meta.read_text())["chave"] == chave
    if chave in _MEMO:
        if arq is not None and not em_disco:  # memória quente, mas o destino ainda não tem o cache
            _MEMO[chave].to_parquet(arq)
            meta.write_text(json.dumps({"chave": chave}))
        return _MEMO[chave]
    if destino is not None:
        if em_disco:
            painel = pd.read_parquet(arq)
            painel.index = pd.DatetimeIndex(painel.index, name="data")
            _MEMO[chave] = painel
            return painel
    painel = Precificador(tabelas, seed).painel()
    _MEMO[chave] = painel
    if arq is not None:
        painel.to_parquet(arq)
        meta.write_text(json.dumps({"chave": chave}))
    return painel
