"""Orquestração do bloco 5: série com passivo + conciliação com o administrador, e resumo por fundo."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from gestorasim.conciliacao import executar_conciliacao
from gestorasim.dados import gerar_universo
from gestorasim.dados.universo import DESTINO_PADRAO
from gestorasim.mtm import carregar_serie, rodar_serie
from gestorasim.util import fmt3

TABELAS_COTA = ("cotistas_mov", "lotes_cotistas", "provisoes", "cota_diaria", "tributos")
TABELAS_CONCILIACAO = ("quebras_conciliacao", "cota_paralela", "injecoes")


def executar_cota(seed: int = 42, destino: Path | str | None = None, serie: dict[str, pd.DataFrame] | None = None,
                  injetar: bool = True, n_por_tipo: int = 4, escrever: bool = True) -> dict[str, pd.DataFrame]:
    """Usa a série em disco se válida (senão a executa), concilia com o arquivo XML do administrador e grava as tabelas."""
    destino = Path(destino) if destino is not None else DESTINO_PADRAO
    tabelas = gerar_universo(seed=seed, destino=destino)
    if serie is None:
        serie = carregar_serie(seed, destino) or rodar_serie(seed, destino)
    meta = tabelas["ativos"].set_index("ticker")
    conc = executar_conciliacao(serie, meta, seed, destino, injetar, n_por_tipo)
    saidas = {**{n: serie[n] for n in TABELAS_COTA}, **conc, "cdi": tabelas["cdi"]}
    if escrever:
        for nome in TABELAS_CONCILIACAO:
            saidas[nome].to_parquet(destino / f"{nome}.parquet", index=False)
    return saidas


def resumo(res: dict[str, pd.DataFrame]) -> str:
    cota, prov, trib = res["cota_diaria"], res["provisoes"], res["tributos"]
    q, inj, falsos = res["quebras_conciliacao"], res["injecoes"], res["falsos_positivos"]
    linhas = []
    for c, g in cota.groupby("fundo_cnpj", sort=False):
        g = g.sort_values("data")
        rent = g["cota_liquida"].iloc[-1] / g["cota_liquida"].iloc[0] - 1
        bruta = g["cota_bruta"].iloc[-1] / g["cota_bruta"].iloc[0] - 1
        bench = g["benchmark"].iloc[-1] / g["benchmark"].iloc[0] - 1
        p = prov[prov["fundo_cnpj"] == c]
        t = trib[trib["fundo_cnpj"] == c] if len(trib) else trib
        soma = (lambda tipo: float(t.loc[t["tipo"] == tipo, "valor"].sum())) if len(t) else (lambda tipo: 0.0)
        linhas += [
            g["fundo"].iloc[0],
            f"  cota líquida: {fmt3(g['cota_liquida'].iloc[-1])} | rentabilidade líquida {fmt3(rent)} "
            f"(bruta da performance {fmt3(bruta)}) vs benchmark {fmt3(bench)}",
            f"  taxas cobradas (R$): administração {fmt3(float(p['adm_pago'].sum()))}, "
            f"performance {fmt3(float(p['perf_pago'].sum()))} | provisões em aberto: adm {fmt3(float(g['adm_a_pagar'].iloc[-1]))}, "
            f"performance {fmt3(float(g['perf_provisao'].iloc[-1]))}",
            f"  tributos retidos (R$): come-cotas {fmt3(soma('COME_COTAS'))}, IR no resgate {fmt3(soma('IR_RESGATE'))}, "
            f"IOF {fmt3(soma('IOF_RESGATE'))}",
            f"  PL líquido final (R$): {fmt3(g['pl_liquido'].iloc[-1])} | cotas: {fmt3(g['cotas'].iloc[-1])} "
            f"| resgates a pagar (R$): {fmt3(g['resgates_a_pagar'].iloc[-1])}",
        ]
        qf = q[q["fundo_cnpj"] == c]
        por_tipo = qf["tipo"].value_counts().to_dict()
        linhas.append("  quebras de conciliação: " + (", ".join(f"{k}={v}" for k, v in sorted(por_tipo.items())) or "nenhuma"))
    n, d = len(inj), int(inj["detectada"].sum()) if len(inj) else 0
    taxa = d / n if n else float("nan")
    linhas.append(f"divergências injetadas: {n} | detectadas: {d} | taxa de detecção: {fmt3(taxa)} | "
                  f"falsos positivos: {len(falsos)}")
    cp = res["cota_paralela"]
    if len(cp):
        linhas.append(f"cota paralela: diferença máxima {fmt3(float(cp['diferenca_bp'].abs().max()))} bp; "
                      f"{int((cp['diferenca_bp'].abs() > 1e-6).sum())} dias com cota divergente")
    return "\n".join(linhas)
