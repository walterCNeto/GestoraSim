"""Trilha de auditoria append-only das transições de estado das ordens."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from gestorasim.boletagem.modelos import TRANSICOES, EstadoOrdem, OrdemAgregada

MIN_JUSTIFICATIVA = 10  # caracteres


class TransicaoInvalida(Exception):
    pass


class Trilha:
    """Único ponto que altera `ordem.estado`. Relógio lógico determinístico (1 s por evento)."""

    def __init__(self, inicio: datetime) -> None:
        self._inicio = inicio
        self._eventos: list[tuple] = []

    def __len__(self) -> int:
        return len(self._eventos)

    def transicionar(self, ordem: OrdemAgregada, para: EstadoOrdem, usuario: str, motivo: str) -> None:
        de = ordem.estado
        if para not in TRANSICOES[de]:
            raise TransicaoInvalida(f"{ordem.id}: {de} -> {para} não permitida")
        if not usuario or not usuario.strip():
            raise ValueError("usuário obrigatório na trilha")
        if not motivo or not motivo.strip():
            raise ValueError("motivo obrigatório na trilha")
        if para == EstadoOrdem.LIBERADA_OVERRIDE and len(motivo.strip()) < MIN_JUSTIFICATIVA:
            raise ValueError(f"override exige justificativa de ao menos {MIN_JUSTIFICATIVA} caracteres")
        self._append(ordem, "TRANSICAO", usuario, de.value if de else None, para.value, motivo)
        ordem.estado = para

    def evento(self, ordem: OrdemAgregada, tipo: str, usuario: str, motivo: str) -> None:
        """Evento que não muda o estado (ex.: EXCLUIDO_PRE_TRADE); `de` = `para` = estado atual."""
        if ordem.estado is None:
            raise TransicaoInvalida(f"{ordem.id}: evento {tipo} antes da criação")
        if not motivo.strip() or not usuario.strip():
            raise ValueError("usuário e motivo obrigatórios na trilha")
        self._append(ordem, tipo, usuario, ordem.estado.value, ordem.estado.value, motivo)

    def _append(self, ordem, tipo, usuario, de, para, motivo) -> None:
        seq = len(self._eventos) + 1
        self._eventos.append((seq, self._inicio + timedelta(seconds=seq), ordem.id, usuario, tipo,
                              de, para, motivo))

    def para_df(self) -> pd.DataFrame:
        return pd.DataFrame(self._eventos, columns=["seq", "timestamp", "ordem_id", "usuario", "tipo",
                                                    "de", "para", "motivo"])
