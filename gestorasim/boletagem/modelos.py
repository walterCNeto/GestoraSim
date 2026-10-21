"""Ordem agregada (block order) e máquina de estados do ciclo de vida."""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from gestorasim.dominio import ClasseAtivo, LadoOrdem


class EstadoOrdem(StrEnum):
    CRIADA = "CRIADA"
    PRE_TRADE_OK = "PRE_TRADE_OK"
    BLOQUEADA = "BLOQUEADA"
    LIBERADA_OVERRIDE = "LIBERADA_OVERRIDE"
    ENVIADA = "ENVIADA"
    EXECUTADA_PARCIAL = "EXECUTADA_PARCIAL"
    EXECUTADA = "EXECUTADA"
    ALOCADA = "ALOCADA"


E = EstadoOrdem
# None = ordem ainda inexistente na trilha.
TRANSICOES: dict[EstadoOrdem | None, frozenset[EstadoOrdem]] = {
    None: frozenset({E.CRIADA}),
    E.CRIADA: frozenset({E.PRE_TRADE_OK, E.BLOQUEADA}),
    E.PRE_TRADE_OK: frozenset({E.ENVIADA}),
    E.BLOQUEADA: frozenset({E.LIBERADA_OVERRIDE}),
    E.LIBERADA_OVERRIDE: frozenset({E.ENVIADA}),
    E.ENVIADA: frozenset({E.EXECUTADA_PARCIAL, E.EXECUTADA}),
    E.EXECUTADA_PARCIAL: frozenset({E.ALOCADA}),  # o saldo não executado é cancelado no fim do dia
    E.EXECUTADA: frozenset({E.ALOCADA}),
    E.ALOCADA: frozenset(),
}


class OrdemAgregada(BaseModel):
    """Uma ordem em um ativo cobrindo N fundos, com quantidade-alvo por fundo (CNPJ -> quantidade)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    data: date
    ticker: str
    classe: ClasseAtivo
    lado: LadoOrdem
    alvos: dict[str, int]  # vigentes; perdem os fundos excluídos no pré-trade
    alvos_originais: dict[str, int] = Field(default_factory=dict)
    origem: str = "MANUAL"  # REBALANCEAMENTO | ALEATORIA | VIOLADORA | MANUAL
    usuario: str = "gestor"
    pedido_override: tuple[str, str] | None = None  # (usuário, justificativa), aplicado se bloqueada
    estado: EstadoOrdem | None = None  # só a Trilha altera
    quantidade_executada: int = 0
    quantidade_cancelada: int = 0  # excedente de rateio que não coube nos tetos dos fundos
    preco_medio: float | None = None
    fills: list[tuple[int, float]] = Field(default_factory=list)  # (quantidade, preço)

    @model_validator(mode="after")
    def _alvos(self) -> OrdemAgregada:
        if not self.alvos or any(q <= 0 for q in self.alvos.values()):
            raise ValueError("a ordem agregada exige quantidades-alvo positivas por fundo")
        if not self.alvos_originais:
            self.alvos_originais = dict(self.alvos)
        return self

    @property
    def quantidade_alvo_total(self) -> int:
        return sum(self.alvos.values())
