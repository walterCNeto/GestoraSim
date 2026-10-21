"""Diagnóstico de retorno de um fundo: decompõe o retorno composto da cota líquida em componentes.

Retorno diário da cota = (P&L dos ativos − taxas) / PL líquido da véspera. O retorno composto é a soma, dia a dia,
da contribuição de cada componente (comp_t / PL_{t−1}) × fator de crescimento acumulado até a véspera, o que fecha
exatamente com a cota (sem resíduo). Os custos de execução (slippage) são abertos pela origem da ordem.

  python -m gestorasim.orquestrador.diagnostico            # decomposição do Multimercado a partir de data/sim
  python -m gestorasim.orquestrador.diagnostico --ablacao   # + variantes (sem passivo, sem gestão de caixa, ...)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from gestorasim.dados.universo import DESTINO_PADRAO
from gestorasim.util import fmt3

ORIGENS = ("REGULAR", "ALEATORIA", "VIOLADORA", "REENQUADRAMENTO", "GESTAO_CAIXA", "REINVESTIMENTO")
_MAPA_ORIGEM = {"REBALANCEAMENTO": "REGULAR"}
_ORDEM = re.compile(r"^ORD\d+$")
RETORNO_BLOCO3_RELATADO = 0.078  # Multimercado no relatório do bloco 3 (referência histórica; o código mudou desde então)


def origem_da_ordem(origem: str) -> str:
    base = origem.split(":")[0]
    return _MAPA_ORIGEM.get(base, base)


def custos_por_origem(res: dict[str, pd.DataFrame], painel: pd.DataFrame, cnpj: str) -> pd.DataFrame:
    """Custo de execução (slippage) por pregão e origem da ordem, em R$ (negativo = custo)."""
    led, ordens = res["ledger"], res["ordens_serie"]
    f = led[(led["fundo_cnpj"] == cnpj) & (led["tipo"] == "ATIVO") & led["origem"].str.match(_ORDEM)].copy()
    if f.empty or ordens.empty:
        return pd.DataFrame(columns=["data", "origem_ordem", "custo"])
    mapa = ordens[["data_serie", "ordem_id", "origem"]].drop_duplicates().rename(
        columns={"data_serie": "data", "ordem_id": "origem", "origem": "origem_ordem"})
    f = f.merge(mapa, on=["data", "origem"], how="left")
    f["origem_ordem"] = f["origem_ordem"].fillna("REGULAR").map(origem_da_ordem)
    ref = painel.stack().rename("preco_mtm").reset_index().rename(columns={"level_0": "data", "level_1": "ticker"})
    ref.columns = ["data", "ticker", "preco_mtm"]
    f = f.merge(ref, on=["data", "ticker"], how="left")
    f["custo"] = f["delta"] * (f["preco_mtm"] - f["preco"])
    return f.groupby(["data", "origem_ordem"], as_index=False)["custo"].sum()


def decompor_retorno(res: dict[str, pd.DataFrame], painel: pd.DataFrame, cnpj: str) -> pd.DataFrame:
    """Contribuição de cada componente ao retorno composto da cota líquida do fundo."""
    cota = res["cota_diaria"].query("fundo_cnpj == @cnpj").sort_values("data").reset_index(drop=True)
    pnl = res["pnl_atribuicao"].query("fundo_cnpj == @cnpj")
    pl_ant = cota["pl_liquido"].shift(1)
    r = ((cota["pnl_ativos"] - cota["taxas"]) / pl_ant).fillna(0.0)
    cresc = (1 + r).cumprod().shift(1).fillna(1.0).to_numpy()
    base = pl_ant.fillna(1.0).to_numpy()
    datas = cota["data"]

    def contrib(serie_dia: pd.Series | np.ndarray) -> tuple[float, float]:
        v = pd.Series(serie_dia).reindex(datas).fillna(0.0).to_numpy() if isinstance(serie_dia, pd.Series) else serie_dia
        return float((v / base * cresc).sum()), float(np.sum(v))

    ativos = res["ativos"] if "ativos" in res else None
    classe = pnl["ticker"].map(ativos.set_index("ticker")["classe"]) if ativos is not None else pd.Series("", index=pnl.index)
    acao = (classe == "ACAO").to_numpy()
    por_dia = lambda col, mask=None: (pnl[col] if mask is None else pnl[col].where(mask, 0.0)).groupby(pnl["data"]).sum()  # noqa: E731
    linhas = [
        ("efeito preço — ações", *contrib(por_dia("efeito_preco", acao))),
        ("efeito preço — renda fixa e DI", *contrib(por_dia("efeito_preco", ~acao))),
        ("carrego (accrual de renda fixa, líquido de pagamentos)", *contrib(por_dia("carrego"))),
        ("eventos (dividendos, cupons, amortizações)", *contrib(por_dia("eventos"))),
        ("ajustes de DI", *contrib(por_dia("ajuste_futuros"))),
        ("juros do caixa", *contrib(por_dia("juros_caixa"))),
    ]
    cp = custos_por_origem(res, painel, cnpj)
    for o in ORIGENS:
        serie = cp[cp["origem_ordem"] == o].set_index("data")["custo"] if len(cp) else pd.Series(dtype=float)
        linhas.append((f"custos de execução — {o}", *contrib(serie)))
    linhas.append(("taxa de administração", *contrib(-cota.set_index("data")["taxa_adm"])))
    linhas.append(("taxa de performance", *contrib(-cota.set_index("data")["taxa_perf"])))
    df = pd.DataFrame(linhas, columns=["componente", "contribuicao_retorno", "valor_brl"])
    total = float(cota["cota_liquida"].iloc[-1] / cota["cota_liquida"].iloc[0] - 1)
    df.loc[len(df)] = ["retorno da cota líquida (soma)", float(df["contribuicao_retorno"].sum()), float(df["valor_brl"].sum())]
    df.loc[len(df)] = ["retorno da cota líquida (cota final / inicial − 1)", total, np.nan]
    df["contribuicao_retorno"] = df["contribuicao_retorno"].astype(float)
    return df


def dominancia_custos(df: pd.DataFrame) -> dict[str, float]:
    """Peso dos custos de reenquadramento + gestão de caixa + reinvestimento no custo total de execução."""
    c = df[df["componente"].str.startswith("custos de execução")].set_index("componente")["contribuicao_retorno"]
    total = float(c.sum())
    gestao = float(sum(c[f"custos de execução — {o}"] for o in ("REENQUADRAMENTO", "GESTAO_CAIXA", "REINVESTIMENTO")))
    return {"custo_total": total, "custo_reenq_gestao": gestao, "peso": gestao / total if total else 0.0}


def diagnostico_mm(destino: Path | str | None = None, res: dict | None = None, escrever: bool = True) -> pd.DataFrame:
    """Decomposição do retorno do Multimercado a partir das tabelas gravadas (ou de `res`)."""
    from gestorasim.dados import gerar_universo
    from gestorasim.mtm.cache import painel_oficial

    destino = Path(destino) if destino is not None else DESTINO_PADRAO
    if res is None:
        from gestorasim.orquestrador import ConfigCiclo, Orquestrador
        res = Orquestrador(ConfigCiclo(destino=destino)).carregar()
    tabelas = gerar_universo(destino=destino)
    res = {**res, "ativos": tabelas["ativos"]}
    painel = painel_oficial(tabelas, 42, destino)
    fundos = pd.read_parquet(destino / "fundos.parquet")
    cnpj = fundos.loc[fundos["classe"] == "MULTIMERCADO", "cnpj"].iloc[0]
    df = decompor_retorno(res, painel, cnpj)
    if escrever:
        df.to_parquet(destino / "diagnostico_mm.parquet", index=False)
    return df


def imprimir(df: pd.DataFrame) -> str:
    linhas = [f"{'componente':<58}{'contribuição ao retorno':>26}{'R$':>20}"]
    for _, r in df.iterrows():
        valor = "" if pd.isna(r["valor_brl"]) else fmt3(float(r["valor_brl"]))
        linhas.append(f"{r['componente']:<58}{fmt3(float(r['contribuicao_retorno'])):>26}{valor:>20}")
    dom = dominancia_custos(df)
    linhas.append(f"custos de execução: total {fmt3(dom['custo_total'])}; reenquadramento + gestão de caixa + "
                  f"reinvestimento {fmt3(dom['custo_reenq_gestao'])} ({fmt3(dom['peso'])} do custo total)")
    linhas.append(f"referência do bloco 3 (relatada, Multimercado): {fmt3(RETORNO_BLOCO3_RELATADO)}")
    return "\n".join(linhas)


def ablacao(destino_base: Path | str, seed: int = 42, de=None, ate=None) -> pd.DataFrame:
    """Retorno do Multimercado em variantes da simulação (só etapas de posição/cota), para isolar o efeito de cada
    mudança desde o bloco 3. A variante 'estilo bloco 3' desliga passivo, juros do caixa, gestão de caixa e reenquadramento."""
    import tempfile

    from gestorasim.orquestrador.config import ConfigCiclo
    from gestorasim.orquestrador.simulacao import Simulacao

    variantes = {
        "completo (bloco 5/6)": {},
        "sem gestão de caixa e reinvestimento": {"com_gestao_caixa": False},
        "sem reenquadramento": {"com_reenquadramento": False, "com_politica_reenquadramento": False},
        "sem juros do caixa": {"com_juros_caixa": False},
        "sem passivo (sem cotistas, taxas e tributos)": {"com_passivo": False},
        "estilo bloco 3 (sem passivo, juros, gestão e reenquadramento)": {
            "com_passivo": False, "com_juros_caixa": False, "com_gestao_caixa": False,
            "com_reenquadramento": False, "com_politica_reenquadramento": False},
    }
    linhas = []
    for nome, kw in variantes.items():
        cfg = ConfigCiclo(seed=seed, destino=tempfile.mkdtemp(), com_conciliacao=False, com_pos_trade=False,
                          com_risco=False, ckpt_cada=0, **kw)
        sim = Simulacao(cfg, de, ate)
        for i in range(sim.i0, sim.i1 + 1):
            sim.executar_dia(i, registrar=False)
        mm = next(f.cnpj for f in sim.fundos if f.classe.value == "MULTIMERCADO")
        t = sim.tabelas_estado()
        if len(t["cota_diaria"]):
            c = t["cota_diaria"].query("fundo_cnpj == @mm")
            ret = float(c["cota_liquida"].iloc[-1] / c["cota_liquida"].iloc[0] - 1)
        else:
            pl = sim.tabelas_dias()["pl_diario"].query("fundo_cnpj == @mm")["pl"]
            ret = float(pl.iloc[-1] / pl.iloc[0] - 1)
        linhas.append({"variante": nome, "retorno_multimercado": ret})
    return pd.DataFrame(linhas)


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m gestorasim.orquestrador.diagnostico")
    ap.add_argument("--destino")
    ap.add_argument("--ablacao", action="store_true", help="roda variantes da simulação (alguns minutos)")
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    df = diagnostico_mm(a.destino)
    print(imprimir(df))
    if a.ablacao:
        ab = ablacao(a.destino or DESTINO_PADRAO)
        print("\nvariantes da simulação (retorno da cota/PL do Multimercado):")
        for _, r in ab.iterrows():
            print(f"  {r['variante']:<66}{fmt3(r['retorno_multimercado']):>10}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
