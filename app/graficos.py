"""Gráficos Plotly do app: hover formatado pelas funções de `formatacao`, datas dd/mm no eixo, cores semânticas."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import formatacao as F

PALETA = [F.AZUL, "#7a4fb5", "#0f8b8d", "#d9731e", "#8a6d3b", "#c2185b"]


def _base(fig: go.Figure, altura: int, titulo_y: str | None = None, tickfmt: str | None = None) -> go.Figure:
    fig.update_layout(
        height=altura, margin=dict(l=10, r=10, t=30, b=10), separators=",.", template="plotly_white",
        legend=dict(orientation="h", y=1.12, x=0), hovermode="x unified",
        font=dict(family="Segoe UI, Arial, sans-serif", size=12), paper_bgcolor="#ffffff", plot_bgcolor="#ffffff")
    fig.update_xaxes(tickformat="%d/%m", hoverformat="%d/%m/%Y", showgrid=False)
    fig.update_yaxes(title_text=titulo_y, tickformat=tickfmt, gridcolor="#e6eaef", zeroline=True, zerolinecolor="#b8c0c9")
    return fig


def _hover(valores, fmt: Callable) -> list[str]:
    return [fmt(v) for v in valores]


def linhas(series: Sequence[dict], fmt: Callable = F.num, altura: int = 340, titulo_y: str | None = None,
           tickfmt: str | None = None, hlinhas: Sequence[tuple[float, str, str]] = (),
           pontos: Sequence[dict] = ()) -> go.Figure:
    """`series`: dicts {nome, x, y, cor?, dash?, largura?, fmt?}. `hlinhas`: (valor, rótulo, cor) tracejadas.
    `pontos`: marcadores {nome, x, y, cor} (ex.: exceções do backtest)."""
    fig = go.Figure()
    for i, s in enumerate(series):
        f = s.get("fmt", fmt)
        fig.add_trace(go.Scatter(
            x=s["x"], y=s["y"], name=s["nome"], mode="lines", connectgaps=False,
            line=dict(color=s.get("cor", PALETA[i % len(PALETA)]), dash=s.get("dash"), width=s.get("largura", 2)),
            customdata=_hover(s["y"], f), hovertemplate="%{customdata}<extra>" + s["nome"] + "</extra>"))
    for p in pontos:
        fig.add_trace(go.Scatter(
            x=p["x"], y=p["y"], name=p["nome"], mode="markers", marker=dict(color=p.get("cor", F.VERMELHO), size=9,
                                                                            symbol="x", line=dict(width=1)),
            customdata=_hover(p["y"], p.get("fmt", fmt)), hovertemplate="%{customdata}<extra>" + p["nome"] + "</extra>"))
    for valor, rotulo, cor in hlinhas:
        fig.add_hline(y=valor, line=dict(color=cor, dash="dash", width=1.5), annotation_text=f"{rotulo}: {fmt(valor)}",
                      annotation_position="top left", annotation_font_color=cor)
    return _base(fig, altura, titulo_y, tickfmt)


def barras(x, y, fmt: Callable = F.num, cores=None, altura: int = 300, titulo_y: str | None = None,
           tickfmt: str | None = None, nome: str = "") -> go.Figure:
    """Barras verticais; `cores`: cor única ou lista (ex.: verde/vermelho conforme o sinal)."""
    fig = go.Figure(go.Bar(x=x, y=y, name=nome, marker_color=cores or F.AZUL, customdata=_hover(y, fmt),
                           hovertemplate="%{customdata}<extra>" + nome + "</extra>"))
    return _base(fig, altura, titulo_y, tickfmt)


def barras_empilhadas(x, series: Sequence[dict], fmt: Callable = F.num, altura: int = 300, titulo_y: str | None = None,
                      tickfmt: str | None = None) -> go.Figure:
    fig = go.Figure()
    for i, s in enumerate(series):
        fig.add_trace(go.Bar(x=x, y=s["y"], name=s["nome"], marker_color=s.get("cor", PALETA[i % len(PALETA)]),
                             customdata=_hover(s["y"], fmt), hovertemplate="%{customdata}<extra>" + s["nome"] + "</extra>"))
    fig.update_layout(barmode="relative")
    return _base(fig, altura, titulo_y, tickfmt)


def areas(x, series: Sequence[dict], fmt: Callable = F.num, altura: int = 300, titulo_y: str | None = None,
          tickfmt: str | None = None) -> go.Figure:
    fig = go.Figure()
    for i, s in enumerate(series):
        fig.add_trace(go.Scatter(x=x, y=s["y"], name=s["nome"], stackgroup="a", mode="lines", line=dict(width=0.5),
                                 fillcolor=None, customdata=_hover(s["y"], fmt),
                                 hovertemplate="%{customdata}<extra>" + s["nome"] + "</extra>",
                                 marker_color=s.get("cor", PALETA[i % len(PALETA)])))
    return _base(fig, altura, titulo_y, tickfmt)


def barras_h(rotulos: Sequence[str], valores: Sequence[float], fmt: Callable = F.pct, cores=None, altura: int = 320,
             limite: tuple[float, str] | None = None, tickfmt: str | None = None) -> go.Figure:
    """Barras horizontais com uma linha vertical de limite opcional (valor, rótulo)."""
    fig = go.Figure(go.Bar(y=list(rotulos), x=list(valores), orientation="h", marker_color=cores or F.AZUL,
                           customdata=_hover(valores, fmt), hovertemplate="%{y}: %{customdata}<extra></extra>"))
    if limite is not None:
        fig.add_vline(x=limite[0], line=dict(color=F.VERMELHO, dash="dash", width=1.5), annotation_text=limite[1],
                      annotation_font_color=F.VERMELHO, annotation_position="top")
    _base(fig, altura, tickfmt=None)
    fig.update_layout(hovermode="closest", showlegend=False)
    fig.update_xaxes(tickformat=tickfmt, hoverformat=None, showgrid=True, gridcolor="#e6eaef")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return fig


def mostrar(fig: go.Figure, chave: str | None = None) -> None:
    st.plotly_chart(fig, key=chave, config={"displayModeBar": False})


def serie_base100(df: pd.DataFrame, col: str) -> pd.Series:
    return 100 * df[col] / df[col].iloc[0]
