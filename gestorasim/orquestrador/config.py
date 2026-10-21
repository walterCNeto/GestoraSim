"""Configuração do ciclo diário."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from gestorasim.cota.parametros import ParametrosPassivo
from gestorasim.posicao.caixa import SPREAD_CAIXA_NEGATIVO

ETAPAS = ("abertura", "mercado", "eventos", "passivo", "caixa_reenquadramento", "boletagem", "liquidacao",
          "mtm", "provisoes_cota", "conciliacao", "pos_trade", "risco", "relatorios")


@dataclass
class ConfigCiclo:
    seed: int = 42
    destino: Path | str | None = None
    # ordens
    passo_ordens: int = 5  # ordens regulares (rebalanceamento, aleatórias, violadoras) a cada n pregões
    com_ordens: bool = True
    com_gestao_caixa: bool = True
    com_reenquadramento: bool = True
    com_politica_reenquadramento: bool = True
    histerese_reenquadramento_du: int = 1  # só reenquadra violação que persiste n pregões (1 = imediato)
    tolerancia_reinvestimento: float = 0.015  # excesso de caixa (fração do PL, além do alvo) que dispara reinvestimento
    # caixa e passivo
    com_juros_caixa: bool = True
    spread_caixa_negativo: float = SPREAD_CAIXA_NEGATIVO
    com_passivo: bool = True
    params_passivo: ParametrosPassivo | None = None
    # conciliação, enquadramento, risco
    com_conciliacao: bool = True
    taxa_injecao: float = 0.02
    escrever_xml: bool = True
    com_pos_trade: bool = True
    com_risco: bool = True
    janela_hist: int = 252
    burn_in: int = 60
    prazo_reenquadramento_du: int = 10
    # execução
    ckpt_cada: int = 63  # pregões entre checkpoints de estado (acelera o reprocessamento)

    def assinatura(self) -> str:
        """Texto estável com os parâmetros que mudam os resultados (entra na impressão digital do ciclo)."""
        campos = {k: v for k, v in vars(self).items() if k not in ("destino", "ckpt_cada", "escrever_xml")}
        campos["params_passivo"] = repr(self.params_passivo)
        return repr(sorted(campos.items()))
