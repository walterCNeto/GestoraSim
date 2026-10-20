"""Fundos (3 regulamentos distintos), cotistas e fluxo de aplicações/resgates."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gestorasim.dominio import (
    ClasseAtivo, ClasseFundo, Cotista, Fundo, LimitesRegulamento, TipoPessoa, formatar_cnpj,
)

A, TP, DEB, DI = (ClasseAtivo.ACAO, ClasseAtivo.TITULO_PUBLICO, ClasseAtivo.DEBENTURE,
                  ClasseAtivo.FUTURO_DI)


def gerar_fundos(data_inicio) -> list[Fundo]:
    inicio = pd.Timestamp(data_inicio).date()
    return [
        Fundo(
            cnpj=formatar_cnpj("990000010001"), nome="Sim Ações FIA", classe=ClasseFundo.FIA,
            data_inicio=inicio,
            regulamento=LimitesRegulamento(
                limite_emissor=0.15,
                limite_classe={A: 1.0, TP: 0.33, DEB: 0.0, DI: 0.0},
                minimo_classe={A: 0.67},
                liquidez_minima=0.05, alavancagem_maxima=1.0,
            ),
        ),
        Fundo(
            cnpj=formatar_cnpj("990000020001"), nome="Sim Macro Multimercado",
            classe=ClasseFundo.MULTIMERCADO, data_inicio=inicio,
            regulamento=LimitesRegulamento(
                limite_emissor=0.10,
                limite_classe={A: 0.40, TP: 1.0, DEB: 0.30, DI: 1.0},
                limite_rating={"AAA": 0.30, "AA+": 0.20, "AA-": 0.15, "A+": 0.10, "A-": 0.05},
                liquidez_minima=0.10, alavancagem_maxima=2.0,
            ),
        ),
        Fundo(
            cnpj=formatar_cnpj("990000030001"), nome="Sim Crédito Privado RF",
            classe=ClasseFundo.RENDA_FIXA, data_inicio=inicio,
            regulamento=LimitesRegulamento(
                limite_emissor=0.20,
                limite_classe={A: 0.0, TP: 1.0, DEB: 1.0, DI: 0.50},
                minimo_classe={DEB: 0.50},
                limite_rating={"AAA": 1.0, "AA+": 0.60, "AA-": 0.40, "A+": 0.25, "A-": 0.10},
                liquidez_minima=0.05, alavancagem_maxima=1.0,
            ),
        ),
    ]


def gerar_cotistas(rng: np.random.Generator, fundos: list[Fundo], n: int = 200) -> list[Cotista]:
    escolha = rng.choice(len(fundos), size=n, p=[0.4, 0.3, 0.3])
    tipo = rng.random(n) < 0.15  # 15% PJ
    return [
        Cotista(id=f"C{i + 1:03d}", nome=f"Cotista {i + 1:03d}",
                tipo_pessoa=TipoPessoa.PJ if tipo[i] else TipoPessoa.PF,
                fundo_cnpj=fundos[escolha[i]].cnpj)
        for i in range(n)
    ]


def gerar_fluxo_cotistas(
    rng: np.random.Generator, cotistas: list[Cotista], datas: pd.DatetimeIndex
) -> pd.DataFrame:
    """Aplicação inicial nos primeiros 80 pregões; depois aplicações/resgates aleatórios.

    Resgates nunca excedem o saldo nominal aplicado do cotista (valores em R$).
    """
    T = len(datas)
    linhas = []
    for c in cotistas:
        escala = 400_000 if c.tipo_pessoa == TipoPessoa.PJ else 40_000
        saldo = 0.0
        t0 = int(rng.integers(0, 80))
        sorteio = rng.random(T)
        tipo_mov = rng.random(T)
        tamanho = rng.lognormal(np.log(escala), 0.8, T)
        fracao = rng.uniform(0.1, 1.0, T)
        for t in range(t0, T):
            if t == t0 or sorteio[t] < 0.012:
                if t != t0 and saldo > 0 and tipo_mov[t] < 0.4:
                    valor = saldo if fracao[t] > 0.9 else saldo * fracao[t]
                    saldo -= valor
                    tipo = "RESGATE"
                else:
                    valor = tamanho[t]
                    saldo += valor
                    tipo = "APLICACAO"
                linhas.append((c.id, c.fundo_cnpj, datas[t], tipo, valor))
    return pd.DataFrame(linhas, columns=["id_cotista", "fundo_cnpj", "data", "tipo", "valor"])
