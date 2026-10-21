"""Perfil de desempenho da série (cProfile): as funções mais caras do pacote e o tempo total.

  python -m gestorasim.orquestrador.perfil [--fim 2023-03-15] [--top 10] [--completa]

Referências medidas ANTES das otimizações do bloco 6 (mesma máquina, mesma seed):
  48 pregões (2023-01-02 a 2023-03-15) sob cProfile: 64,1 s;  124 pregões (até 2023-06-30) sem profiler: 87,4 s;
  série completa de 504 pregões: ~480 s.
Gargalos de então: montar_carteira/todas_as_regras em pandas (~60% do pré-trade), `.loc` em colunas de string Arrow,
varredura de lançamentos em Ledger.caixa_disponivel, DataFrame de movimentações no engine do passivo.
"""

from __future__ import annotations

import cProfile
import pstats
import sys
import tempfile
import time

import pandas as pd

from gestorasim.mtm import rodar_serie
from gestorasim.util import fmt3

ANTES = {"48 pregões, sob cProfile (s)": 64.1, "124 pregões (s)": 87.4, "504 pregões (s)": 480.0}


def perfilar(fim="2023-03-15", top: int = 10, seed: int = 42, destino: str | None = None) -> pd.DataFrame:
    """Roda `rodar_serie` até `fim` sob cProfile e devolve as `top` funções do pacote com maior tempo acumulado."""
    destino = destino or tempfile.mkdtemp()
    rodar_serie(seed, destino, fim="2023-01-10", escrever=False)  # aquece os caches (painel, universo)
    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    rodar_serie(seed, destino, fim=fim, escrever=False)
    pr.disable()
    total = time.perf_counter() - t0
    st = pstats.Stats(pr)
    linhas = []
    for (arq, linha, nome), (cc, nc, tt, ct, _) in st.stats.items():
        if "gestorasim" not in arq.replace("\\", "/") or nome in ("rodar_serie", "executar_dia", "<module>", "_executar"):
            continue
        modulo = arq.replace("\\", "/").split("gestorasim/")[-1]
        linhas.append({"funcao": f"{modulo}:{linha}({nome})", "chamadas": nc, "tempo_proprio_s": tt,
                       "tempo_acumulado_s": ct})
    df = pd.DataFrame(linhas).sort_values("tempo_acumulado_s", ascending=False).head(top).reset_index(drop=True)
    df.attrs["total_s"] = total
    return df


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m gestorasim.orquestrador.perfil")
    ap.add_argument("--fim", default="2023-03-15")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--completa", action="store_true", help="mede também a série completa (sem profiler)")
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    df = perfilar(a.fim, a.top)
    print(f"série até {a.fim} sob cProfile: {fmt3(df.attrs['total_s'])} s  (antes: {fmt3(ANTES['48 pregões, sob cProfile (s)'])} s)")
    print(f"{'função':<72}{'chamadas':>10}{'próprio (s)':>13}{'acumulado (s)':>15}")
    for _, r in df.iterrows():
        print(f"{r['funcao']:<72}{r['chamadas']:>10}{fmt3(r['tempo_proprio_s']):>13}{fmt3(r['tempo_acumulado_s']):>15}")
    if a.completa:
        t0 = time.perf_counter()
        rodar_serie(42, tempfile.mkdtemp(), escrever=False)
        print(f"série completa de 504 pregões: {fmt3(time.perf_counter() - t0)} s  (antes: ~{fmt3(ANTES['504 pregões (s)'])} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
