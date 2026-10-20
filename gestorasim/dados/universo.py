"""Orquestra a geração do universo simulado e grava data/sim/*.parquet."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from gestorasim.dados import calendario, eventos, fundos, mercado, renda_fixa
from gestorasim.dominio import Fundo
from gestorasim.util import fmt3

N_DIAS = 504
DATA_INICIAL = "2023-01-02"
DESTINO_PADRAO = Path(__file__).resolve().parents[2] / "data" / "sim"


def _rng(seed: int, componente: int) -> np.random.Generator:
    return np.random.default_rng([seed, componente])


def _ativos_para_df(ativos) -> pd.DataFrame:
    df = pd.DataFrame([a.model_dump(mode="json") for a in ativos])
    df["vencimento"] = pd.to_datetime(df["vencimento"])
    return df


def gerar_universo(seed: int = 42, destino: Path | str | None = None) -> dict[str, pd.DataFrame]:
    """Gera todo o universo (determinístico por `seed`), grava parquets em `destino` e devolve as tabelas."""
    destino = Path(destino) if destino is not None else DESTINO_PADRAO
    destino.mkdir(parents=True, exist_ok=True)

    datas = calendario.calendario_b3(DATA_INICIAL, N_DIAS)

    corr = mercado.gerar_correlacao(_rng(seed, 0))
    precos_acoes = mercado.gerar_precos_acoes(_rng(seed, 1), datas, corr)
    fatores = mercado.gerar_fatores_ns(_rng(seed, 2), datas)
    beta = fatores[["beta0", "beta1", "beta2"]].to_numpy()
    cdi = mercado.gerar_cdi(fatores)
    ipca = mercado.gerar_ipca(_rng(seed, 3), datas)

    a_ltn, p_ltn = renda_fixa.gerar_ltn(_rng(seed, 4), datas, beta)
    a_ntnb, p_ntnb, ev_ntnb = renda_fixa.gerar_ntnb(_rng(seed, 5), datas, beta, ipca)
    a_deb, p_deb, ev_deb = renda_fixa.gerar_debentures(_rng(seed, 6), datas, cdi, ipca)
    a_di, p_di = renda_fixa.gerar_futuros_di(_rng(seed, 7), datas, beta)

    lista_fundos: list[Fundo] = fundos.gerar_fundos(datas[0])
    cotistas = fundos.gerar_cotistas(_rng(seed, 8), lista_fundos)
    fluxo = fundos.gerar_fluxo_cotistas(_rng(seed, 9), cotistas, datas)

    ev_div = eventos.gerar_dividendos(_rng(seed, 10), datas, precos_acoes)
    df_eventos = pd.DataFrame(ev_div + ev_ntnb + ev_deb).sort_values(
        ["data_pagamento", "ticker", "tipo"], ignore_index=True
    )

    tabelas = {
        "calendario": pd.DataFrame({"data": datas}),
        "ativos": _ativos_para_df(
            mercado.ativos_acoes() + a_ltn + a_ntnb + a_deb + a_di
        ),
        "precos_acoes": precos_acoes,
        "correlacao_acoes": pd.DataFrame(corr, index=mercado.TICKERS_ACOES,
                                         columns=mercado.TICKERS_ACOES).rename_axis("ticker").reset_index(),
        "fatores_nelson_siegel": fatores,
        "curva_pre": mercado.curva_pre_vertices(fatores),
        "cdi": cdi,
        "ipca": ipca,
        "precos_renda_fixa": pd.concat([p_ltn, p_ntnb, p_deb, p_di], ignore_index=True)
        .sort_values(["ticker", "data"], ignore_index=True),
        "fundos": pd.DataFrame([
            {**f.model_dump(mode="json", exclude={"regulamento"}),
             "data_inicio": pd.Timestamp(f.data_inicio),
             "regulamento": json.dumps(f.regulamento.model_dump(mode="json"))}
            for f in lista_fundos
        ]),
        "cotistas": pd.DataFrame([c.model_dump(mode="json") for c in cotistas]),
        "fluxo_cotistas": fluxo,
        "eventos": df_eventos,
    }
    for nome, df in tabelas.items():
        df.to_parquet(destino / f"{nome}.parquet", index=False)
    return tabelas


def carregar_fundos(destino: Path | str | None = None) -> list[Fundo]:
    """Reconstrói os modelos `Fundo` a partir de fundos.parquet."""
    destino = Path(destino) if destino is not None else DESTINO_PADRAO
    df = pd.read_parquet(destino / "fundos.parquet")
    return [
        Fundo(**{**r, "data_inicio": pd.Timestamp(r["data_inicio"]).date(),
                 "regulamento": json.loads(r["regulamento"])})
        for r in df.to_dict("records")
    ]


def resumo(tabelas: dict[str, pd.DataFrame]) -> str:
    """Resumo textual do universo, números com 3 casas decimais."""
    pa = tabelas["precos_acoes"].pivot(index="data", columns="ticker", values="preco")
    ret = np.log(pa).diff().dropna()
    linhas = [f"{nome}: {len(df)} linhas" for nome, df in tabelas.items()]
    linhas.append(f"vol anualizada média das ações: {fmt3(float(ret.std().mean() * np.sqrt(252)))}")
    linhas.append(f"taxa pré 1 ano (inicial): {fmt3(float(tabelas['curva_pre'].query('prazo_du == 252')['taxa'].iloc[0]))}")
    linhas.append(f"aplicações líquidas dos cotistas (R$): {fmt3(float(tabelas['fluxo_cotistas'].assign(v=lambda d: d.valor.where(d.tipo == 'APLICACAO', -d.valor)).v.sum()))}")
    return "\n".join(linhas)
