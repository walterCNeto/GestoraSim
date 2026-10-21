"""Acesso aos dados do app: parquets em `data/sim` (ou na pasta indicada por GESTORASIM_DADOS), com cache."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from app import formatacao as fmt
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


def datas_pregao() -> list[pd.Timestamp]:
    """Pregões do ciclo gravado (da cota diária)."""
    return sorted(pd.Timestamp(d) for d in tabela("cota_diaria")["data"].unique()) if existe("cota_diaria") else []


def barra_lateral() -> None:
    """Filtros globais (fundo e data) na barra lateral; persistem entre páginas via st.session_state."""
    if not (existe("cota_diaria") and existe("fundos")):
        return
    datas, nomes = datas_pregao(), list(fundos().values())
    ss = st.session_state
    for k in ("gs_fundo", "gs_data"):  # mantém o valor quando o widget some numa página e volta em outra
        if k in ss:
            ss[k] = ss[k]
    if ss.get("gs_fundo") not in nomes:
        ss["gs_fundo"] = nomes[0]
    if ss.get("gs_data") not in datas:
        ss["gs_data"] = datas[-1]
    with st.sidebar:
        st.markdown("### Filtros")
        st.selectbox("Fundo", nomes, key="gs_fundo")
        st.selectbox("Data", datas, key="gs_data", format_func=lambda d: fmt.data(d))
        st.caption("Valem para todas as páginas. Padrão: último pregão.")
        if modo_demo():
            st.caption(f"Dataset demo reduzido ({fmt.data(datas[0])} a {fmt.data(datas[-1])}).")


def data_global() -> pd.Timestamp:
    """Data escolhida na barra lateral (último pregão, se ainda não há escolha)."""
    d = st.session_state.get("gs_data")
    if d is not None:
        return pd.Timestamp(d)
    datas = datas_pregao()
    return datas[-1] if datas else pd.Timestamp.today().normalize()


def fundo_global() -> str:
    """CNPJ do fundo escolhido na barra lateral (o primeiro, se ainda não há escolha)."""
    fs = fundos()
    nome = st.session_state.get("gs_fundo")
    return next((c for c, n in fs.items() if n == nome), next(iter(fs)))


def ate_data(df: pd.DataFrame, coluna: str = "data", data=None) -> pd.DataFrame:
    """Linhas até a data global (inclusive)."""
    return df[df[coluna] <= (data_global() if data is None else pd.Timestamp(data))]


def configurar(titulo: str, real: str = "", glossario: tuple[str, ...] = ()) -> None:
    """Cabeçalho padrão: config da página, barra lateral, título, 'o que esta tela faz numa gestora real' e glossário."""
    from app import layout

    st.set_page_config(page_title=f"GestoraSim — {titulo}", page_icon="📈", layout="wide")
    barra_lateral()
    layout.cabecalho(titulo, real, glossario)


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
    """CNPJ do fundo da barra lateral (os argumentos são mantidos por compatibilidade)."""
    return fundo_global()


def escolher_data(datas: list | None = None, rotulo: str = "Data", chave: str = "data", padrao_ultima: bool = True):
    """Data da barra lateral; se `datas` for dado e não a contiver, a última data disponível até ela."""
    d = data_global()
    if datas is None:
        return d
    ds = sorted(pd.Timestamp(x) for x in datas)
    ate = [x for x in ds if x <= d]
    return ate[-1] if ate else ds[0]
