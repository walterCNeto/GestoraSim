import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from gestorasim.dados import calendario, carregar_fundos, gerar_universo
from gestorasim.dados.universo import N_DIAS
from gestorasim.dominio import (
    Ativo, ClasseAtivo, ClasseFundo, Fundo, LimitesRegulamento, RATINGS, cnpj_valido,
)


@pytest.fixture(scope="module")
def destino(tmp_path_factory):
    return tmp_path_factory.mktemp("sim")


@pytest.fixture(scope="module")
def u(destino):
    return gerar_universo(seed=42, destino=destino)


def test_correlacao_psd(u):
    corr = u["correlacao_acoes"].set_index("ticker").to_numpy()
    assert corr.shape == (20, 20)
    assert np.allclose(corr, corr.T)
    assert np.allclose(np.diag(corr), 1.0)
    assert np.linalg.eigvalsh(corr).min() >= -1e-10


def test_correlacao_realizada_positiva(u):
    pa = u["precos_acoes"].pivot(index="data", columns="ticker", values="preco")
    c = np.log(pa).diff().dropna().corr().to_numpy()
    assert c[np.triu_indices(20, 1)].mean() > 0.1


def test_vinte_acoes_504_dias(u):
    assert u["precos_acoes"]["ticker"].nunique() == 20
    assert len(u["calendario"]) == N_DIAS
    assert u["precos_acoes"].groupby("ticker").size().eq(N_DIAS).all()


def test_sem_precos_negativos_ou_nulos(u):
    assert (u["precos_acoes"]["preco"] > 0).all()
    rf = u["precos_renda_fixa"]
    assert (rf["pu"] > 0).all() and rf["pu"].notna().all()
    assert (u["curva_pre"]["taxa"] > 0).all()
    assert (u["eventos"]["valor_por_unidade"] > 0).all()


def test_pu_renda_fixa_plausivel(u):
    rf = u["precos_renda_fixa"]
    tit = rf[rf["ticker"].str.startswith(("LTN", "NTNB", "ALEN", "BSAN", "GROD", "DLOG", "EVAR"))]
    assert tit["pu"].between(200, 3000).all()


def _assert_dias_uteis(serie):
    d = pd.to_datetime(serie).to_numpy().astype("datetime64[D]")
    assert np.is_busday(d, holidays=calendario.FERIADOS).all()


def test_datas_em_dia_util(u):
    _assert_dias_uteis(u["calendario"]["data"])
    for tabela, cols in {
        "precos_acoes": ["data"], "precos_renda_fixa": ["data"], "fluxo_cotistas": ["data"],
        "eventos": ["data_com", "data_pagamento"], "ativos": ["vencimento"],
    }.items():
        for c in cols:
            _assert_dias_uteis(u[tabela][c].dropna())


def test_calendario_exclui_feriados_conhecidos():
    assert not calendario.eh_dia_util("2023-04-07")  # Sexta-feira Santa
    assert not calendario.eh_dia_util("2023-02-21")  # Carnaval
    assert not calendario.eh_dia_util("2023-12-25")
    assert calendario.eh_dia_util("2023-01-02")


def test_universo_de_ativos(u):
    a = u["ativos"]
    assert a["classe"].value_counts().to_dict() == {
        "ACAO": 20, "TITULO_PUBLICO": 8, "DEBENTURE": 5, "FUTURO_DI": 3,
    }
    deb = a[a["classe"] == "DEBENTURE"]
    assert set(deb["indexador"]) == {"CDI", "IPCA"}
    assert set(deb["rating"]) == {"AAA", "AA+", "AA-", "A+", "A-"}
    assert a.loc[a.ticker.str.startswith("LTN"), "vencimento"].nunique() == 4


def test_eventos_consistentes(u):
    ev = u["eventos"]
    assert {"DIVIDENDO", "CUPOM", "AMORTIZACAO"} <= set(ev["tipo"])
    assert (ev["data_pagamento"] >= ev["data_com"]).all()
    assert set(ev["ticker"]) <= set(u["ativos"]["ticker"])


def test_regulamentos_coerentes(u, destino):
    fundos = carregar_fundos(destino)
    assert {f.classe for f in fundos} == set(ClasseFundo)
    regs = {f.regulamento.model_dump_json() for f in fundos}
    assert len(regs) == 3  # regulamentos distintos
    for f in fundos:
        assert cnpj_valido(f.cnpj)
        r = f.regulamento
        assert sum(r.limite_classe.values()) >= 1
        assert r.alavancagem_maxima >= 1
    fia = next(f for f in fundos if f.classe == ClasseFundo.FIA)
    assert fia.regulamento.minimo_classe[ClasseAtivo.ACAO] >= 0.67
    mm = next(f for f in fundos if f.classe == ClasseFundo.MULTIMERCADO)
    assert mm.regulamento.alavancagem_maxima > 1


def test_cotistas_e_fluxo(u):
    assert len(u["cotistas"]) == 200
    fl = u["fluxo_cotistas"]
    assert set(fl["tipo"]) == {"APLICACAO", "RESGATE"}
    assert set(fl["fundo_cnpj"]) == set(u["fundos"]["cnpj"])
    # resgates nunca excedem o saldo aplicado
    f = fl.sort_values(["id_cotista", "data"]).assign(
        s=lambda d: d.valor.where(d.tipo == "APLICACAO", -d.valor))
    assert (f.groupby("id_cotista")["s"].cumsum() >= -1e-6).all()


def test_determinismo(u, tmp_path):
    outro = gerar_universo(seed=42, destino=tmp_path)
    for nome in ("precos_acoes", "precos_renda_fixa", "fluxo_cotistas", "eventos"):
        pd.testing.assert_frame_equal(u[nome], outro[nome])
    diferente = gerar_universo(seed=7, destino=tmp_path / "b")
    assert not diferente["precos_acoes"]["preco"].equals(u["precos_acoes"]["preco"])


def test_parquets_gravados(u, destino):
    for nome in u:
        assert (destino / f"{nome}.parquet").exists()


# ------------------------------------------------------------------ validações de domínio

def test_regulamento_incoerente_rejeitado():
    base = dict(limite_emissor=0.1, limite_classe={ClasseAtivo.ACAO: 1.0},
                liquidez_minima=0.05, alavancagem_maxima=1.0)
    LimitesRegulamento(**base)
    with pytest.raises(ValidationError):
        LimitesRegulamento(**{**base, "minimo_classe": {ClasseAtivo.ACAO: 1.0}, "limite_classe": {ClasseAtivo.ACAO: 0.5, ClasseAtivo.TITULO_PUBLICO: 0.5}})
    with pytest.raises(ValidationError):
        LimitesRegulamento(**{**base, "limite_classe": {ClasseAtivo.ACAO: 0.5}})  # não soma 100%
    with pytest.raises(ValidationError):
        LimitesRegulamento(**{**base, "alavancagem_maxima": 0.5})
    with pytest.raises(ValidationError):  # rating pior com limite maior
        LimitesRegulamento(**{**base, "limite_rating": {"AAA": 0.1, "A-": 0.3}})


def test_fia_exige_67pct_em_acoes():
    reg = LimitesRegulamento(limite_emissor=0.1, limite_classe={ClasseAtivo.ACAO: 1.0},
                             liquidez_minima=0.05, alavancagem_maxima=1.0)
    with pytest.raises(ValidationError):
        Fundo(cnpj="99.000.001/0001-00", nome="x", classe=ClasseFundo.FIA, regulamento=reg,
              data_inicio="2023-01-02")


def test_ativo_validacoes():
    with pytest.raises(ValidationError):
        Ativo(ticker="X", classe=ClasseAtivo.DEBENTURE, emissor="e", vencimento="2030-01-01")
    with pytest.raises(ValidationError):
        Ativo(ticker="X", classe=ClasseAtivo.ACAO, emissor="e", vencimento="2030-01-01")
    assert RATINGS.index("AAA") == 0
