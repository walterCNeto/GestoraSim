"""Orquestrador: roda o ciclo diário no calendário, grava as tabelas e o log de execução, e reprocessa uma data.

Reprocessar a data X recalcula X e as seguintes e deixa as anteriores intactas: o estado de X−1 vem do checkpoint
mais recente (ou do início) e os dias intermediários são reexecutados sem tocar nas tabelas gravadas. Como todo o
ciclo é determinístico por seed (ordens, injeções de divergência e conciliação incluídas), o resultado é idempotente.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from gestorasim.dados.universo import DESTINO_PADRAO
from gestorasim.mtm.cache import impressao_digital
from gestorasim.orquestrador.config import ConfigCiclo
from gestorasim.orquestrador.simulacao import Simulacao
from gestorasim.risco.motor import backtest_var, excecoes_var

# tabelas por pregão -> coluna de data (as demais vêm do estado acumulado ou são derivadas)
TABELAS_POR_DIA = {
    "posicoes_diarias": "data", "mtm_diario": "data", "pnl_atribuicao": "data", "pl_diario": "data",
    "liquidacao_diaria": "data", "risco_diario": "data", "sensibilidades": "data", "stress": "data",
    "liquidez": "data", "enquadramento_diario": "data", "alertas": "data", "quebras_conciliacao": "data",
    "injecoes": "data", "cota_paralela": "data", "relatorio_diario": "data", "ordens_serie": "data_serie",
    "alocacoes_serie": "data_serie", "auditoria_serie": "data_serie",
}
TABELAS_ESTADO = ("ledger", "cotistas_mov", "lotes_cotistas", "provisoes", "cota_diaria", "tributos", "episodios")
TABELAS_DERIVADAS = ("backtest_var", "excecoes_var")
COLUNAS_LOG = ["data", "etapa", "ordem", "status", "duracao_s", "hash_entrada", "mensagem"]


class Orquestrador:
    def __init__(self, cfg: ConfigCiclo | None = None) -> None:
        self.cfg = cfg or ConfigCiclo()
        self.destino = Path(self.cfg.destino) if self.cfg.destino is not None else DESTINO_PADRAO
        self.pasta_ckpt = self.destino / "orquestrador"

    # ------------------------------------------------------------------ execução
    def executar(self, de=None, ate=None) -> dict[str, pd.DataFrame]:
        """Roda o ciclo de `de` a `ate` (default: calendário inteiro) e grava as tabelas em `destino`."""
        sim = Simulacao(self.cfg, de, ate)
        self._rodar(sim, sim.i0, sim.i1)
        res = self._montar(sim, antigo=None, corte=None)
        self._gravar(res, sim, de, ate)
        return res

    def reprocessar(self, data) -> dict[str, pd.DataFrame]:
        """Recalcula `data` e as datas seguintes; as anteriores ficam como estão. Idempotente."""
        meta = self._ler_meta()
        sim = Simulacao(self.cfg, meta["de"], meta["ate"])
        corte = pd.Timestamp(data)
        if corte not in set(sim.datas[sim.i0:sim.i1 + 1]):
            raise ValueError(f"{corte:%Y-%m-%d} não é um pregão do ciclo gravado")
        ix = int(sim.datas.get_loc(corte))
        antigo = self.carregar()
        ckpts = sorted(int(p.stem.split("_")[1]) for p in self.pasta_ckpt.glob("ckpt_*.pkl")
                       if sim.i0 <= int(p.stem.split("_")[1]) < ix)
        inicio_replay = sim.i0
        if ckpts:  # estado do fechamento de ckpts[-1], reexecutando só os dias até a véspera
            sim.carregar_checkpoint(self.pasta_ckpt / f"ckpt_{ckpts[-1]:04d}.pkl")
            inicio_replay = ckpts[-1] + 1
        for i in range(inicio_replay, ix):
            sim.executar_dia(i, registrar=False)
        sim.dias.clear()  # só as saídas de `data` em diante entram nas tabelas
        self._rodar(sim, ix, sim.i1)
        res = self._montar(sim, antigo=antigo, corte=corte)
        self._gravar(res, sim, meta["de"], meta["ate"])
        return res

    def _rodar(self, sim: Simulacao, i_ini: int, i_fim: int) -> None:
        for i in range(i_ini, i_fim + 1):
            sim.executar_dia(i)
            if self.cfg.ckpt_cada and (i - sim.i0) % self.cfg.ckpt_cada == 0:
                sim.salvar_checkpoint(i, self.pasta_ckpt)

    # ------------------------------------------------------------------ montagem e gravação
    def _montar(self, sim: Simulacao, antigo: dict | None, corte: pd.Timestamp | None) -> dict[str, pd.DataFrame]:
        novas = sim.tabelas_dias()
        res: dict[str, pd.DataFrame] = {}
        for nome, col in TABELAS_POR_DIA.items():
            df = novas[nome]
            if antigo is not None and nome in antigo and len(antigo[nome]):
                ant = antigo[nome]
                df = pd.concat([ant[ant[col] < corte], df], ignore_index=True) if len(df) else ant[ant[col] < corte]
            res[nome] = df.reset_index(drop=True)
        res.update(sim.tabelas_estado())
        q = res["quebras_conciliacao"].drop(columns=["id_quebra"], errors="ignore").reset_index(drop=True)
        q.insert(0, "id_quebra", [f"Q{n:05d}" for n in range(1, len(q) + 1)])  # ids estáveis: mesma ordem, mesmos ids
        res["quebras_conciliacao"] = q
        if len(res["risco_diario"]):
            res["backtest_var"], res["excecoes_var"] = backtest_var(res["risco_diario"]), excecoes_var(res["risco_diario"])
        log = sim.log()
        if antigo is not None and "execucoes_ciclo" in antigo:
            ant = antigo["execucoes_ciclo"]
            log = pd.concat([ant[ant["data"] < corte], log], ignore_index=True)
        res["execucoes_ciclo"] = log.reset_index(drop=True)
        return res

    def _gravar(self, res: dict[str, pd.DataFrame], sim: Simulacao, de, ate) -> None:
        self.destino.mkdir(parents=True, exist_ok=True)
        for nome, df in res.items():
            df.to_parquet(self.destino / f"{nome}.parquet", index=False)
        meta = {"chave": impressao_digital(sim.tabelas, self.cfg.seed), "config": self.cfg.assinatura(),
                "de": str(de) if de is not None else None, "ate": str(ate) if ate is not None else None,
                "primeiro_pregao": f"{sim.datas[sim.i0]:%Y-%m-%d}", "ultimo_pregao": f"{sim.datas[sim.i1]:%Y-%m-%d}",
                "gravado_em": time.strftime("%Y-%m-%dT%H:%M:%S")}
        (self.destino / "ciclo_meta.json").write_text(json.dumps(meta, indent=2))

    def _ler_meta(self) -> dict:
        arq = self.destino / "ciclo_meta.json"
        if not arq.exists():
            raise FileNotFoundError("nenhum ciclo gravado: rode o orquestrador antes de reprocessar")
        meta = json.loads(arq.read_text())
        if meta["config"] != self.cfg.assinatura():
            raise ValueError("a configuração mudou desde o ciclo gravado: reexecute o ciclo inteiro")
        return meta

    def carregar(self) -> dict[str, pd.DataFrame]:
        """Todas as tabelas gravadas pelo orquestrador."""
        nomes = [*TABELAS_POR_DIA, *TABELAS_ESTADO, *TABELAS_DERIVADAS, "execucoes_ciclo"]
        return {n: pd.read_parquet(self.destino / f"{n}.parquet") for n in nomes if (self.destino / f"{n}.parquet").exists()}


def resumo_ciclo(res: dict[str, pd.DataFrame]) -> str:
    log = res["execucoes_ciclo"]
    por = log.groupby("etapa", sort=False).agg(ok=("status", lambda s: int((s == "OK").sum())),
                                              puladas=("status", lambda s: int((s == "PULADA").sum())),
                                              erros=("status", lambda s: int((s == "ERRO").sum())),
                                              tempo_s=("duracao_s", "sum"))
    linhas = [f"pregões: {log['data'].nunique()} | etapas OK={int((log['status'] == 'OK').sum())} "
              f"PULADA={int((log['status'] == 'PULADA').sum())} ERRO={int((log['status'] == 'ERRO').sum())}",
              f"{'etapa':<24}{'ok':>6}{'puladas':>9}{'erros':>7}{'tempo (s)':>12}"]
    for etapa, r in por.iterrows():
        linhas.append(f"{etapa:<24}{r['ok']:>6}{r['puladas']:>9}{r['erros']:>7}{r['tempo_s']:>12.3f}")
    linhas.append(f"{'total':<24}{'':>22}{por['tempo_s'].sum():>12.3f}")
    return "\n".join(linhas)
