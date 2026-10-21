"""Parâmetros do passivo por classe de fundo (todos substituíveis)."""

from __future__ import annotations

from dataclasses import dataclass, field

from gestorasim.dominio import ClasseFundo


@dataclass(frozen=True)
class RegrasFundo:
    taxa_adm: float  # a.a., base 252
    taxa_perf: float  # fração do que exceder o benchmark
    benchmark: str  # "CDI" | "ACOES"
    conv_aplicacao_du: int  # dias úteis até a conversão da aplicação
    conv_resgate_tipo: str  # "du" | "corridos"
    conv_resgate_n: int
    pag_resgate_du: int  # dias úteis entre a conversão e o pagamento
    come_cotas: bool
    ir: str  # "REGRESSIVA" | "FIXA15"


REGRAS_PADRAO: dict[ClasseFundo, RegrasFundo] = {
    ClasseFundo.RENDA_FIXA: RegrasFundo(0.008, 0.20, "CDI", 0, "du", 0, 0, True, "REGRESSIVA"),
    ClasseFundo.MULTIMERCADO: RegrasFundo(0.015, 0.20, "CDI", 1, "corridos", 30, 1, True, "REGRESSIVA"),
    ClasseFundo.FIA: RegrasFundo(0.020, 0.20, "ACOES", 1, "du", 1, 2, False, "FIXA15"),
}


@dataclass(frozen=True)
class ParametrosPassivo:
    regras: dict[ClasseFundo, RegrasFundo] = field(default_factory=lambda: dict(REGRAS_PADRAO))
    cota_inicial: float = 1.0
    aliquota_come_cotas: float = 0.15  # alíquota de longo prazo, aplicada a todos os lotes
    meses_come_cotas: tuple[int, ...] = (5, 11)  # último pregão de maio e novembro
    meses_performance: tuple[int, ...] = (6, 12)  # último pregão de junho e dezembro
    aliquota_fia: float = 0.15
    iof_ligado: bool = False
    cotista_capital_inicial: str = "CAPITAL"
