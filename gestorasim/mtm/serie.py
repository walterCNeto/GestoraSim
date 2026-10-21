"""Série diária do gestor (ledger, MTM, eventos, ordens, passivo e P&L), sem conciliação nem risco.

A lógica do pregão vive em `orquestrador.simulacao.Simulacao` (etapas nomeadas); aqui ficam a função de conveniência
`rodar_serie`, que roda só as etapas de posição/cota, a leitura da série gravada e os resumos.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from gestorasim.cota.parametros import ParametrosPassivo
from gestorasim.dados import gerar_universo
from gestorasim.dados.universo import DESTINO_PADRAO
from gestorasim.mtm.cache import impressao_digital
from gestorasim.posicao.caixa import SPREAD_CAIXA_NEGATIVO
from gestorasim.util import fmt3

TOLERANCIA_FECHAMENTO = 1e-6
COMPONENTES = ["efeito_preco", "carrego", "eventos", "ajuste_futuros", "custos", "juros_caixa"]
TABELAS_SAIDA = ("posicoes_diarias", "mtm_diario", "pnl_atribuicao", "pl_diario", "ledger",
                 "ordens_serie", "alocacoes_serie", "auditoria_serie",
                 "cotistas_mov", "lotes_cotistas", "provisoes", "cota_diaria", "tributos")


def rodar_serie(seed: int = 42, destino: Path | str | None = None, inicio=None, fim=None,
                passo_ordens: int = 5, com_ordens: bool = True, escrever: bool = True,
                spread_caixa_negativo: float = SPREAD_CAIXA_NEGATIVO, com_passivo: bool = True,
                params_passivo: ParametrosPassivo | None = None,
                com_politica_reenquadramento: bool = True, **extra) -> dict[str, pd.DataFrame]:
    """Roda todos os fundos de `inicio` a `fim` (default: o calendário inteiro) com as etapas de posição, MTM e cota.

    O caixa projetado do fechamento anterior rende CDI (saldo >= 0) ou paga CDI + `spread_caixa_negativo`.
    Ordens do gerador (rebalanceamento + aleatórias + violadoras sem override) a cada `passo_ordens` pregões, mais
    gestão de caixa e reenquadramento diários. `extra`: demais campos de `ConfigCiclo` (ex.: histerese).
    """
    from gestorasim.orquestrador.config import ConfigCiclo
    from gestorasim.orquestrador.simulacao import Simulacao

    destino = Path(destino) if destino is not None else DESTINO_PADRAO
    cfg = ConfigCiclo(seed=seed, destino=destino, passo_ordens=passo_ordens, com_ordens=com_ordens,
                      spread_caixa_negativo=spread_caixa_negativo, com_passivo=com_passivo,
                      params_passivo=params_passivo, com_politica_reenquadramento=com_politica_reenquadramento,
                      com_conciliacao=False, com_pos_trade=False, com_risco=False, ckpt_cada=0, **extra)
    sim = Simulacao(cfg, inicio, fim)
    for i in range(sim.i0, sim.i1 + 1):
        sim.executar_dia(i, registrar=False)
    todas = {**sim.tabelas_dias(), **sim.tabelas_estado()}
    saidas = {n: todas[n] for n in TABELAS_SAIDA}
    saidas["ordens_por_estado"] = pd.DataFrame(sorted(sim.ordens_estado.items()), columns=["estado", "n"])
    if escrever:
        for nome in TABELAS_SAIDA:
            saidas[nome].to_parquet(destino / f"{nome}.parquet", index=False)
        (destino / "serie_meta.json").write_text(json.dumps(_meta(sim.tabelas, seed, inicio, fim, passo_ordens,
                                                                com_ordens, spread_caixa_negativo)))
    return saidas


def _meta(tabelas, seed, inicio, fim, passo_ordens, com_ordens, spread) -> dict:
    return {"chave": impressao_digital(tabelas, seed), "seed": seed, "inicio": str(inicio), "fim": str(fim),
            "passo_ordens": passo_ordens, "com_ordens": com_ordens, "spread_caixa_negativo": spread}


def carregar_serie(seed: int = 42, destino: Path | str | None = None, inicio=None, fim=None, passo_ordens: int = 5,
                   com_ordens: bool = True, spread_caixa_negativo: float = SPREAD_CAIXA_NEGATIVO,
                   ) -> dict[str, pd.DataFrame] | None:
    """Série gravada em disco se foi gerada com os mesmos parâmetros e universo; senão None (não recalcula)."""
    destino = Path(destino) if destino is not None else DESTINO_PADRAO
    meta = destino / "serie_meta.json"
    if not meta.exists() or any(not (destino / f"{n}.parquet").exists() for n in TABELAS_SAIDA):
        return None
    tabelas = gerar_universo(seed=seed, destino=destino)
    if json.loads(meta.read_text()) != _meta(tabelas, seed, inicio, fim, passo_ordens, com_ordens,
                                             spread_caixa_negativo):
        return None
    return {n: pd.read_parquet(destino / f"{n}.parquet") for n in TABELAS_SAIDA}


def serie_pl(fundo: str, inicio=None, fim=None, seed: int = 42, destino: Path | str | None = None,
             resultado: dict[str, pd.DataFrame] | None = None) -> pd.Series:
    """PL diário de um fundo (nome ou CNPJ) de `inicio` a `fim`.

    Ordens agregadas cobrem vários fundos, então a série roda todos juntos; passe `resultado`
    (retorno de `rodar_serie`) para reaproveitar uma simulação já feita.
    """
    res = resultado if resultado is not None else rodar_serie(seed, destino, inicio, fim, escrever=False)
    pl = res["pl_diario"]
    sel = pl[(pl["fundo"] == fundo) | (pl["fundo_cnpj"] == fundo)]
    if sel.empty:
        raise KeyError(f"fundo não encontrado: {fundo}")
    s = sel.set_index("data")["pl"]
    if inicio is not None:
        s = s[s.index >= pd.Timestamp(inicio)]
    if fim is not None:
        s = s[s.index <= pd.Timestamp(fim)]
    return s.rename(sel["fundo"].iloc[0])


def resumo(res: dict[str, pd.DataFrame]) -> str:
    pl = res["pl_diario"]
    linhas = []
    for nome, g in pl.groupby("fundo", sort=False):
        s = g.set_index("data")["pl"]
        r = s.pct_change().dropna()
        pior = r.idxmin()
        linhas.append(f"{nome}: PL inicial={fmt3(s.iloc[0])} final={fmt3(s.iloc[-1])} "
                      f"retorno={fmt3(s.iloc[-1] / s.iloc[0] - 1)} pior dia={pior:%Y-%m-%d} ({fmt3(r.min())})")
    linhas.append(f"maior erro de fechamento do P&L (R$): {fmt3(float(pl['erro_fechamento'].abs().max()))}")
    oe = res["ordens_por_estado"]
    if len(oe):
        linhas.append("ordens: " + ", ".join(f"{e}={n}" for e, n in zip(oe["estado"], oe["n"])))
    return "\n".join(linhas)
