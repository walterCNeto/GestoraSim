"""Workflow das quebras de conciliação: ABERTA → JUSTIFICADA → RESOLVIDA (log append-only em parquet próprio).

Uma quebra só pode ser RESOLVIDA depois de JUSTIFICADA; toda mudança de status exige usuário e comentário.
Uma quebra JUSTIFICADA pode ser reaberta (volta a ABERTA); RESOLVIDA é final.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

STATUS = ("ABERTA", "JUSTIFICADA", "RESOLVIDA")
TRANSICOES = {"ABERTA": {"JUSTIFICADA"}, "JUSTIFICADA": {"RESOLVIDA", "ABERTA"}, "RESOLVIDA": set()}
MIN_COMENTARIO = 5  # caracteres
COLUNAS = ["seq", "id_quebra", "status", "usuario", "comentario", "timestamp"]


class TransicaoInvalida(ValueError):
    pass


class WorkflowQuebras:
    def __init__(self, destino: Path | str) -> None:
        self.arq = Path(destino) / "workflow_quebras.parquet"

    def historico(self) -> pd.DataFrame:
        if not self.arq.exists():
            return pd.DataFrame(columns=COLUNAS)
        return pd.read_parquet(self.arq)

    def status_atual(self, ids: Iterable[str] | None = None) -> pd.Series:
        """Status vigente de cada quebra (ABERTA se nunca houve registro)."""
        h = self.historico()
        atual = h.sort_values("seq").groupby("id_quebra")["status"].last() if len(h) else pd.Series(dtype=object)
        if ids is not None:
            atual = atual.reindex(list(ids)).fillna("ABERTA")
        return atual.rename("status_workflow")

    def transicionar(self, id_quebra: str, para: str, usuario: str, comentario: str,
                     existentes: Iterable[str] | None = None) -> None:
        if para not in STATUS:
            raise TransicaoInvalida(f"status desconhecido: {para}")
        if existentes is not None and id_quebra not in set(existentes):
            raise TransicaoInvalida(f"quebra inexistente: {id_quebra}")
        if not usuario or not usuario.strip():
            raise TransicaoInvalida("usuário obrigatório")
        if not comentario or len(comentario.strip()) < MIN_COMENTARIO:
            raise TransicaoInvalida(f"comentário obrigatório (mínimo {MIN_COMENTARIO} caracteres)")
        atual = self.status_atual([id_quebra]).iloc[0]
        if para not in TRANSICOES[atual]:
            raise TransicaoInvalida(
                f"{id_quebra}: {atual} → {para} não permitida"
                + (" (resolver exige justificar antes)" if para == "RESOLVIDA" else ""))
        h = self.historico()
        nova = pd.DataFrame([{"seq": len(h) + 1, "id_quebra": id_quebra, "status": para, "usuario": usuario.strip(),
                              "comentario": comentario.strip(), "timestamp": pd.Timestamp.now()}], columns=COLUNAS)
        self.arq.parent.mkdir(parents=True, exist_ok=True)
        pd.concat([h, nova], ignore_index=True).to_parquet(self.arq, index=False)

    def quebras_com_status(self, quebras: pd.DataFrame) -> pd.DataFrame:
        """Tabela de quebras com o status vigente do workflow (a coluna `status` do motor fica como origem)."""
        if quebras.empty:
            return quebras.assign(status_workflow=pd.Series(dtype=object))
        return quebras.assign(status_workflow=self.status_atual(quebras["id_quebra"]).to_numpy())
