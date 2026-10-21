"""Passivo do fundo: cotistas, cota, taxas, tributos (e, via `executar_cota`, a conciliação com o administrador)."""

from gestorasim.cota.engine import PassivoFundo, agendar
from gestorasim.cota.parametros import REGRAS_PADRAO, ParametrosPassivo, RegrasFundo
from gestorasim.cota.tributos import Lote, aliquota_ir_regressiva, come_cotas, resgatar_fifo

__all__ = ["Lote", "ParametrosPassivo", "PassivoFundo", "REGRAS_PADRAO", "RegrasFundo", "agendar",
           "aliquota_ir_regressiva", "come_cotas", "executar_cota", "resgatar_fifo"]


def executar_cota(*args, **kwargs):
    """Orquestra série + conciliação (importação tardia para evitar ciclos com mtm/risco)."""
    from gestorasim.cota.relatorio import executar_cota as _executar
    return _executar(*args, **kwargs)
