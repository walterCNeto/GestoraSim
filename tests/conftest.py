"""Fixtures compartilhadas: a série curta (≈75 pregões) é cara, então roda uma vez por sessão."""

import pytest

from gestorasim.mtm import rodar_serie
from gestorasim.risco import executar_risco

SEED = 42
FIM_SERIE_CURTA = "2023-04-14"
JANELA_CURTA = 30  # janela histórica reduzida só para os testes (padrão: 252)


@pytest.fixture(scope="session")
def destino_serie(tmp_path_factory):
    return tmp_path_factory.mktemp("serie_sessao")


@pytest.fixture(scope="session")
def serie_curta(destino_serie):
    return rodar_serie(SEED, destino_serie, fim=FIM_SERIE_CURTA)


@pytest.fixture(scope="session")
def risco_curto(serie_curta, destino_serie):
    return executar_risco(SEED, destino_serie, serie=serie_curta, janela=JANELA_CURTA)


@pytest.fixture(scope="session")
def ciclo_curto(tmp_path_factory):
    """Ciclo completo (13 etapas) de 30 pregões: (orquestrador, tabelas, pasta)."""
    from gestorasim.orquestrador import ConfigCiclo, Orquestrador

    destino = tmp_path_factory.mktemp("ciclo_sessao")
    orq = Orquestrador(ConfigCiclo(seed=SEED, destino=destino, janela_hist=15, burn_in=5, ckpt_cada=10, taxa_injecao=0.15))
    return orq, orq.executar("2023-01-02", "2023-02-10"), destino
