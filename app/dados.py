"""Acesso aos dados do app: parquets em `data/sim` (ou na pasta indicada por GESTORASIM_DADOS), com cache."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from gestorasim.dados.universo import DESTINO_PADRAO

AJUDA = ("Rode `python -m gestorasim.orquestrador --de 2023-01-02 --ate 2024-12-30` para gerar os dados "
         "(ou use o dataset demo em data/demo).")


DEMO = DESTINO_PADRAO.parent / "demo"
_SENTINELA = "cota_diaria.parquet"


def pasta() -> Path:
    """GESTORASIM_DADOS (se definida e não vazia); senão data/sim, se tiver o ciclo gravado; senão data/demo."""
    env = os.environ.get("GESTORASIM_DADOS", "").strip()
    if env:
        return Path(env)
    return DESTINO_PADRAO if (DESTINO_PADRAO / _SENTINELA).exists() else DEMO


def modo_demo() -> bool:
    return pasta().resolve() == DEMO.resolve()


def _gravavel(destino: Path) -> bool:
    try:
        with tempfile.NamedTemporaryFile(dir=destino):
            return True
    except OSError:
        return False


def pasta_gravavel() -> Path:
    """Onde o app grava ordens manuais e workflow: a pasta de dados, se for gravável (e GESTORASIM_ESCRITA não
    mandar outra); senão uma pasta temporária (no Streamlit Cloud o disco é efêmero e somente leitura)."""
    env = os.environ.get("GESTORASIM_ESCRITA", "").strip()
    if env:
        alvo = Path(env)
        alvo.mkdir(parents=True, exist_ok=True)
        return alvo
    destino = pasta()
    if _gravavel(destino):
        return destino
    tmp = Path(tempfile.gettempdir()) / "gestorasim_app"
    tmp.mkdir(parents=True, exist_ok=True)
    return tmp


def aviso_escrita() -> None:
    """Aviso discreto quando as gravações do app vão para uma pasta temporária."""
    if pasta_gravavel().resolve() != pasta().resolve():
        st.caption("ℹ️ Pasta de dados somente leitura: ordens manuais e workflow ficam em disco temporário "
                   "e se perdem quando a sessão do servidor reinicia.")


@st.cache_data(show_spinner=False)
def _ler(caminho: str, mtime: float) -> pd.DataFrame:
    return pd.read_parquet(caminho)


def existe(nome: str) -> bool:
    return (pasta() / f"{nome}.parquet").exists()


def tabela(nome: str) -> pd.DataFrame:
    """Tabela gravada; o cache é invalidado quando o arquivo muda (mtime)."""
    arq = pasta() / f"{nome}.parquet"
    return _ler(str(arq), arq.stat().st_mtime)


def tabela_ou_vazia(nome: str) -> pd.DataFrame:
    return tabela(nome) if existe(nome) else pd.DataFrame()


def tabela_gravavel(nome: str) -> pd.DataFrame:
    """Tabela escrita pelo app (manuais, workflow), lida da pasta gravável; vazia se ainda não existe."""
    arq = pasta_gravavel() / f"{nome}.parquet"
    return pd.read_parquet(arq) if arq.exists() else pd.DataFrame()


def meta_ciclo() -> dict:
    arq = pasta() / "ciclo_meta.json"
    return json.loads(arq.read_text()) if arq.exists() else {}


def configurar(titulo: str) -> None:
    st.set_page_config(page_title=f"GestoraSim — {titulo}", page_icon="📈", layout="wide")
    st.title(titulo)


def exigir(*nomes: str) -> None:
    """Interrompe a página, com instrução, se faltar alguma tabela."""
    faltam = [n for n in nomes if not existe(n)]
    if faltam:
        st.warning(f"Faltam dados em `{pasta()}`: {', '.join(faltam)}. {AJUDA}")
        st.stop()


def fundos() -> dict[str, str]:
    """CNPJ -> nome dos fundos."""
    f = tabela("fundos")
    return dict(zip(f["cnpj"], f["nome"]))


def escolher_fundo(rotulo: str = "Fundo", chave: str = "fundo") -> str:
    fs = fundos()
    nome = st.selectbox(rotulo, list(fs.values()), key=chave)
    return next(c for c, n in fs.items() if n == nome)


def escolher_data(datas: list, rotulo: str = "Data", chave: str = "data", padrao_ultima: bool = True):
    datas = sorted(pd.Timestamp(d) for d in datas)
    return st.selectbox(rotulo, datas, index=len(datas) - 1 if padrao_ultima else 0, key=chave,
                        format_func=lambda d: f"{d:%Y-%m-%d}")
