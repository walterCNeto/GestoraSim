"""Formatação pt-BR: toda a exibição numérica do app passa por aqui.

Padrão: vírgula decimal, ponto de milhar, três casas (cota: oito). Valores nulos viram "—".
Os formatadores recebem o valor "cru" (R$, fração, bp) e só mudam a apresentação: as tabelas continuam
ordenando pelo número, não pelo texto.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import pandas as pd

ND = "—"

VERDE, AMBAR, VERMELHO, CINZA, AZUL = "#1b8a3b", "#c77c02", "#c62828", "#6b7480", "#1f5fa8"
FUNDO_CLARO = {VERDE: "#dff3e4", AMBAR: "#fdecc8", VERMELHO: "#fbdada", CINZA: "#e8ebef"}


def _nulo(v) -> bool:
    if v is None or v is pd.NaT:
        return True
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _pt(txt: str) -> str:
    """'1,234.500' -> '1.234,500'."""
    return txt.replace(",", "\0").replace(".", ",").replace("\0", ".")


def num(v, casas: int = 3) -> str:
    """Número genérico com `casas` decimais, milhar com ponto e decimal com vírgula."""
    if _nulo(v):
        return ND
    v = float(v)
    if math.isinf(v):
        return "∞" if v > 0 else "-∞"
    if round(v, casas) == 0:
        v = 0.0  # evita "-0,000"
    return _pt(f"{v:,.{casas}f}")


def brl_mi(v) -> str:
    """R$ 12,345 mi; abaixo de R$ 1 mi, R$ 345,678 mil."""
    if _nulo(v):
        return ND
    v = float(v)
    a = abs(v)
    sinal = "-" if round(a / 1e3, 3) != 0 and v < 0 else ""
    if a >= 1e6 or round(a / 1e3, 3) >= 1000:
        return f"{sinal}R$ {num(a / 1e6)} mi"
    return f"{sinal}R$ {num(a / 1e3)} mil"


def mi(v) -> str:
    """Valor em R$ exibido em milhões, sem prefixo (a unidade vai no cabeçalho)."""
    return ND if _nulo(v) else num(float(v) / 1e6)


def mil(v) -> str:
    """Valor em R$ exibido em milhares, sem prefixo."""
    return ND if _nulo(v) else num(float(v) / 1e3)


def pct(v, casas: int = 3) -> str:
    """Fração -> percentual: 0.01234 -> '1,234%'."""
    return ND if _nulo(v) else f"{num(float(v) * 100, casas)}%"


def bp(v) -> str:
    """Valor já em pontos-base: 12.3456 -> '12,346 bp'."""
    return ND if _nulo(v) else f"{num(v)} bp"


def qtd(v) -> str:
    """Quantidade inteira com ponto de milhar: 12300 -> '12.300'."""
    if _nulo(v):
        return ND
    n = int(round(float(v)))
    return f"{n:,}".replace(",", ".")


def cota(v) -> str:
    """Valor da cota com oito casas."""
    return num(v, 8)


def data(d) -> str:
    """dd/mm/aaaa."""
    return ND if _nulo(d) else pd.Timestamp(d).strftime("%d/%m/%Y")


def data_curta(d) -> str:
    """dd/mm."""
    return ND if _nulo(d) else pd.Timestamp(d).strftime("%d/%m")


def sinal(fn: Callable, v) -> str:
    """Formata com sinal explícito (+/-) para deltas."""
    if _nulo(v):
        return ND
    txt = fn(v)
    return txt if txt.startswith("-") or float(v) == 0 else f"+{txt}"


# ------------------------------------------------------------------------------------------ status e cores
_STATUS_COR = {
    VERDE: {"OK", "RESOLVIDA", "ALOCADA", "EXECUTADA", "PRE_TRADE_OK", "REENQUADRADO", "ENVIADA", "ENQUADRADO", "VERDE"},
    AMBAR: {"SOFT", "JUSTIFICADA", "PASSIVO", "LIBERADA_OVERRIDE", "EXECUTADA_PARCIAL", "EXCLUIDO_PRE_TRADE", "ALERTA",
            "PARCIAL", "AMARELA", "AMARELO"},
    VERMELHO: {"HARD", "VIOLADO", "ABERTA", "BLOQUEADA", "VENCIDO", "ATIVO", "EM ABERTO", "QUEBRA", "VERMELHA", "VERMELHO"},
}


def cor_status(txt) -> str:
    """Cor semântica: verde OK, âmbar SOFT/alerta, vermelho HARD/violação/quebra aberta, cinza neutro."""
    s = str(txt).upper().strip()
    for cor, grupo in _STATUS_COR.items():
        if s in grupo:
            return cor
    return CINZA


def badge_html(txt) -> str:
    """Selo colorido em HTML (usar com unsafe_allow_html; o texto é escapado)."""
    from html import escape

    cor = cor_status(txt)
    return (f"<span style='background:{FUNDO_CLARO[cor]};color:{cor};border:1px solid {cor};border-radius:10px;"
            f"padding:1px 8px;font-size:.82em;font-weight:600'>{escape(str(txt))}</span>")


def _estilo_badge(v) -> str:
    if _nulo(v) or str(v) == "":
        return ""
    cor = cor_status(v)
    return f"background-color:{FUNDO_CLARO[cor]};color:{cor};font-weight:600"


# ------------------------------------------------------------------------------------------------ tabelas
_KINDS: dict[str, tuple[Callable | None, str]] = {
    "mi": (mi, " (R$ mi)"), "mil": (mil, " (R$ mil)"), "pct": (pct, " (%)"), "bp": (bp, " (bp)"),
    "qtd": (qtd, ""), "cota": (cota, ""), "num": (num, ""), "data": (data, ""), "txt": (None, ""),
}


def tabela(df: pd.DataFrame, spec: dict[str, tuple[str, str]], badges: tuple[str, ...] = ()):
    """Styler em português: `spec` = {coluna: (rótulo, tipo)}; o tipo (mi, mil, pct, bp, qtd, cota, num, data, txt)
    escolhe o formatador e acrescenta a unidade ao cabeçalho. `badges`: colunas (nomes originais) de status coloridas.
    O índice não é exibido (use hide_index=True) e a ordenação continua numérica."""
    cols = [c for c in spec if c in df.columns]
    rotulos = {c: spec[c][0] + _KINDS[spec[c][1]][1] for c in cols}
    base = df[cols].rename(columns=rotulos).reset_index(drop=True)
    fmts = {rotulos[c]: _KINDS[spec[c][1]][0] for c in cols if _KINDS[spec[c][1]][0] is not None}
    st_ = base.style.format(fmts, na_rep=ND)
    alvo = [rotulos[c] for c in badges if c in rotulos]
    if alvo:
        st_ = st_.map(_estilo_badge, subset=alvo)
    return st_


def exibir(df: pd.DataFrame, spec: dict[str, tuple[str, str]], badges: tuple[str, ...] = (), altura: int | None = None,
           vazio: str = "Nada a mostrar.") -> None:
    import streamlit as st

    if df is None or len(df) == 0:
        st.info(vazio)
        return
    extra = {"height": altura if altura is not None else min(38 + 35 * len(df), 720)}
    st.dataframe(tabela(df, spec, badges), hide_index=True, **extra)


# retrocompatibilidade (blocos anteriores)
def tres_casas(df: pd.DataFrame):
    """Styler com colunas float em três casas decimais (pt-BR)."""
    fmts = {c: num for c in df.columns if pd.api.types.is_float_dtype(df[c])}
    fmts.update({c: data for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])})
    return df.style.format(fmts, na_rep=ND)
