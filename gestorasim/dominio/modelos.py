"""Modelos de domínio (pydantic) da gestora simulada."""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Escala de rating em ordem decrescente de qualidade de crédito.
RATINGS: tuple[str, ...] = (
    "AAA", "AA+", "AA", "AA-", "A+", "A", "A-",
    "BBB+", "BBB", "BBB-", "BB+", "BB", "BB-", "B+", "B", "B-",
)


class ClasseAtivo(StrEnum):
    ACAO = "ACAO"
    TITULO_PUBLICO = "TITULO_PUBLICO"
    DEBENTURE = "DEBENTURE"
    FUTURO_DI = "FUTURO_DI"


class Indexador(StrEnum):
    NENHUM = "NENHUM"
    PRE = "PRE"
    IPCA = "IPCA"
    CDI = "CDI"


class ClasseFundo(StrEnum):
    """Categorias da Resolução CVM 175 usadas no MVP."""

    FIA = "FIA"
    MULTIMERCADO = "MULTIMERCADO"
    RENDA_FIXA = "RENDA_FIXA"


class TipoPessoa(StrEnum):
    PF = "PF"
    PJ = "PJ"


class LadoOrdem(StrEnum):
    COMPRA = "C"
    VENDA = "V"


class TipoEvento(StrEnum):
    DIVIDENDO = "DIVIDENDO"
    CUPOM = "CUPOM"
    AMORTIZACAO = "AMORTIZACAO"


def digitos_cnpj(base12: str) -> str:
    """Calcula os 2 dígitos verificadores de um CNPJ a partir dos 12 primeiros dígitos."""
    digitos = [int(c) for c in base12]
    for _ in range(2):
        pesos = list(range(len(digitos) - 7, 1, -1)) + list(range(9, 1, -1))
        pesos = pesos[: len(digitos)]
        soma = sum(d * p for d, p in zip(digitos, pesos))
        resto = soma % 11
        digitos.append(0 if resto < 2 else 11 - resto)
    return "".join(str(d) for d in digitos[12:])


def cnpj_valido(cnpj: str) -> bool:
    """Valida formato XX.XXX.XXX/XXXX-XX e dígitos verificadores."""
    apenas = "".join(c for c in cnpj if c.isdigit())
    if len(apenas) != 14 or len(set(apenas)) == 1:
        return False
    return digitos_cnpj(apenas[:12]) == apenas[12:]


def formatar_cnpj(base12: str) -> str:
    dv = digitos_cnpj(base12)
    n = base12 + dv
    return f"{n[:2]}.{n[2:5]}.{n[5:8]}/{n[8:12]}-{n[12:]}"


class _Modelo(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Ativo(_Modelo):
    ticker: str
    classe: ClasseAtivo
    emissor: str
    rating: str | None = None
    vencimento: date | None = None
    cupom: float | None = None  # a.a.; spread para debêntures, taxa real para NTN-B
    indexador: Indexador = Indexador.NENHUM

    @field_validator("rating")
    @classmethod
    def _rating_conhecido(cls, v: str | None) -> str | None:
        if v is not None and v not in RATINGS:
            raise ValueError(f"rating desconhecido: {v}")
        return v

    @model_validator(mode="after")
    def _coerencia(self) -> Ativo:
        if self.classe == ClasseAtivo.ACAO:
            if self.vencimento is not None:
                raise ValueError("ação não tem vencimento")
        elif self.vencimento is None:
            raise ValueError(f"{self.classe} exige vencimento")
        if self.classe == ClasseAtivo.DEBENTURE:
            if self.rating is None or self.cupom is None:
                raise ValueError("debênture exige rating e cupom")
            if self.indexador not in (Indexador.CDI, Indexador.IPCA):
                raise ValueError("debênture é CDI+ ou IPCA+")
        return self


class LimitesRegulamento(_Modelo):
    """Limites do regulamento. Percentuais em fração do patrimônio líquido (0.10 = 10%)."""

    limite_emissor: float = Field(gt=0, le=1)
    limite_classe: dict[ClasseAtivo, float]
    minimo_classe: dict[ClasseAtivo, float] = Field(default_factory=dict)
    limite_rating: dict[str, float] = Field(default_factory=dict)
    liquidez_minima: float = Field(ge=0, le=1)  # fração do PL liquidável em D+1
    alavancagem_maxima: float = Field(ge=1)  # exposição bruta / PL

    @model_validator(mode="after")
    def _coerencia(self) -> LimitesRegulamento:
        for classe, v in {**self.limite_classe, **self.minimo_classe}.items():
            if not 0 <= v <= 1:
                raise ValueError(f"limite fora de [0,1] para {classe}: {v}")
        for classe, minimo in self.minimo_classe.items():
            if minimo > self.limite_classe.get(classe, 0.0):
                raise ValueError(f"mínimo > máximo para {classe}")
        if sum(self.minimo_classe.values()) > 1 + 1e-12:
            raise ValueError("soma dos mínimos por classe excede 100%")
        if sum(self.limite_classe.values()) < 1 - 1e-12:
            raise ValueError("limites por classe não permitem alocar 100% do PL")
        # Rating pior não pode ter limite maior que rating melhor.
        presentes = sorted(self.limite_rating, key=RATINGS.index)
        for r in presentes:
            if r not in RATINGS or not 0 <= self.limite_rating[r] <= 1:
                raise ValueError(f"limite por rating inválido: {r}")
        for melhor, pior in zip(presentes, presentes[1:]):
            if self.limite_rating[pior] > self.limite_rating[melhor]:
                raise ValueError(f"limite de {pior} maior que o de {melhor}")
        return self


class Fundo(_Modelo):
    cnpj: str  # fictício, com dígitos verificadores válidos
    nome: str
    classe: ClasseFundo
    regulamento: LimitesRegulamento
    data_inicio: date
    cota_inicial: float = Field(default=1.0, gt=0)

    @field_validator("cnpj")
    @classmethod
    def _cnpj(cls, v: str) -> str:
        if not cnpj_valido(v):
            raise ValueError(f"CNPJ inválido: {v}")
        return v

    @model_validator(mode="after")
    def _classe_vs_regulamento(self) -> Fundo:
        reg = self.regulamento
        if self.classe == ClasseFundo.FIA:
            if reg.minimo_classe.get(ClasseAtivo.ACAO, 0.0) < 0.67:
                raise ValueError("FIA exige mínimo de 67% em ações")
            if reg.alavancagem_maxima > 1:
                raise ValueError("FIA não pode ser alavancado")
        if self.classe == ClasseFundo.RENDA_FIXA and reg.limite_classe.get(ClasseAtivo.ACAO, 0.0) > 0:
            raise ValueError("fundo de renda fixa não pode ter ações")
        return self


class Cotista(_Modelo):
    id: str
    nome: str
    tipo_pessoa: TipoPessoa
    fundo_cnpj: str


class Ordem(_Modelo):
    id: str
    data: date
    fundo_cnpj: str
    ticker: str
    lado: LadoOrdem
    quantidade: float = Field(gt=0)
    preco: float = Field(gt=0)


class Posicao(_Modelo):
    data: date
    fundo_cnpj: str
    ticker: str
    quantidade: float  # negativa = vendido
    preco_medio: float = Field(gt=0)


class Evento(_Modelo):
    tipo: TipoEvento
    ticker: str
    data_com: date
    data_pagamento: date
    valor_por_unidade: float = Field(gt=0)

    @model_validator(mode="after")
    def _datas(self) -> Evento:
        if self.data_pagamento < self.data_com:
            raise ValueError("pagamento antes da data com")
        return self
