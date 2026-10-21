"""Serviços do app (boletagem manual, workflow de quebras, lâmina) e teste de fumaça das páginas Streamlit."""

import hashlib
import importlib
import os
import shutil
from pathlib import Path

import pandas as pd
import pytest

from gestorasim.boletagem.manual import TABELAS_MANUAIS, BoletagemManual
from gestorasim.conciliacao.workflow import TransicaoInvalida, WorkflowQuebras
from gestorasim.relatorios.lamina import gerar_lamina, periodos

RAIZ = Path(__file__).resolve().parents[1]
PAGINAS = ["visao_geral", "boletagem", "carteira", "risco", "enquadramento", "passivo_cota", "conciliacao", "trilha",
           "lamina"]


@pytest.fixture()
def dados_copia(ciclo_curto, tmp_path):
    """Cópia do ciclo curto: testes que gravam (manual, workflow) não contaminam a sessão."""
    _, _, origem = ciclo_curto
    for f in Path(origem).glob("*.parquet"):
        shutil.copy(f, tmp_path / f.name)
    for f in Path(origem).glob("*.json"):
        shutil.copy(f, tmp_path / f.name)
    return tmp_path


def hash_oficial(pasta: Path) -> dict[str, str]:
    return {f.name: hashlib.sha1(f.read_bytes()).hexdigest() for f in sorted(pasta.glob("*.parquet"))
            if f.stem not in TABELAS_MANUAIS and f.stem != "workflow_quebras"}


# ======================================================================= boletagem manual

def test_ordem_manual_nao_altera_a_serie_oficial(dados_copia):
    antes = hash_oficial(dados_copia)
    bm = BoletagemManual(dados_copia)
    data = bm.datas[10]
    fundos = {f.classe.value: f.cnpj for f in bm.fundos}
    pre = bm.pre_trade(data, "LTN270104", "C", {fundos["MULTIMERCADO"]: 500})
    assert set(pre["status"]) == {"OK"} and {"regra", "valor_pos_ordem", "limite", "folga", "status"} <= set(pre.columns)
    assert hash_oficial(dados_copia) == antes and not any((dados_copia / f"{n}.parquet").exists() for n in TABELAS_MANUAIS)
    r = bm.enviar(data, "LTN270104", "C", {fundos["MULTIMERCADO"]: 500}, "walter")
    assert r["estado"] == "ALOCADA"
    assert hash_oficial(dados_copia) == antes  # a série oficial continua byte a byte igual
    for nome in TABELAS_MANUAIS:
        assert (dados_copia / f"{nome}.parquet").exists()
    om = pd.read_parquet(dados_copia / "ordens_manuais.parquet")
    assert om["tag"].eq("MANUAL").all() and om["origem"].eq("MANUAL").all() and om["ordem_id"].iloc[0] == "MAN0001"
    led = pd.read_parquet(dados_copia / "ledger_manual.parquet")
    assert led["tag"].eq("MANUAL").all() and led["origem"].str.startswith("MAN").all()
    al = pd.read_parquet(dados_copia / "alocacoes_manuais.parquet")
    assert al["quantidade_alocada"].sum() == 500 and al["preco_medio"].iloc[0] == pytest.approx(
        r["execucoes"]["preco"].mul(r["execucoes"]["quantidade"]).sum() / 500)


def test_ordens_manuais_acumulam_na_mesma_data_e_respeitam_lote(dados_copia):
    bm = BoletagemManual(dados_copia)
    data = bm.datas[12]
    mm = next(f.cnpj for f in bm.fundos if f.classe.value == "MULTIMERCADO")
    bm.enviar(data, "LTN270104", "C", {mm: 100}, "a")
    bm.enviar(data, "LTN270104", "C", {mm: 200}, "b")
    om = pd.read_parquet(dados_copia / "ordens_manuais.parquet")
    assert om["ordem_id"].tolist() == ["MAN0001", "MAN0002"]
    # a 2ª ordem já parte da posição da 1ª: o pré-trade enxerga as 300 unidades
    novo = BoletagemManual(dados_copia)
    q0 = novo._mesa(data).ledger.quantidades(mm)["LTN270104"]
    oficial = novo.posicoes.query("fundo_cnpj == @mm and data == @data and ticker == 'LTN270104'")["quantidade"]
    assert q0 == (int(oficial.iloc[0]) if len(oficial) else 0) + 300
    with pytest.raises(ValueError, match="lote"):
        bm.pre_trade(data, "ALFA3", "C", {mm: 150})
    with pytest.raises(ValueError, match="quantidade"):
        bm.pre_trade(data, "ALFA3", "C", {mm: 0})


def test_ordem_manual_violadora_fica_bloqueada_e_override_exige_justificativa(dados_copia):
    bm = BoletagemManual(dados_copia)
    data = bm.datas[10]
    fia = next(f.cnpj for f in bm.fundos if f.classe.value == "FIA")
    pre = bm.pre_trade(data, "DLOG14", "C", {fia: 5000})  # debênture num FIA
    assert (pre["status"] == "VIOLADO").any() and "limite_classe" in set(pre.loc[pre["status"] == "VIOLADO", "regra"])
    r = bm.enviar(data, "DLOG14", "C", {fia: 5000}, "walter")
    assert r["estado"] == "BLOQUEADA" and r["alocacoes"].empty
    with pytest.raises(ValueError, match="justificativa"):
        bm.enviar(data, "DLOG14", "C", {fia: 5000}, "walter", ("walter", "ok"))  # justificativa curta demais
    r2 = bm.enviar(data, "DLOG14", "C", {fia: 5000}, "walter", ("diretor.risco", "Excecao aprovada em comite"))
    assert r2["estado"] == "ALOCADA"
    aud = r2["auditoria"]
    assert "LIBERADA_OVERRIDE" in set(aud["para"]) and aud.loc[aud["para"] == "LIBERADA_OVERRIDE", "usuario"].iloc[0] == "diretor.risco"


# ======================================================================= workflow das quebras

def test_workflow_nao_permite_resolvida_sem_justificada(tmp_path):
    wf = WorkflowQuebras(tmp_path)
    assert wf.status_atual(["Q00001"]).iloc[0] == "ABERTA"
    with pytest.raises(TransicaoInvalida, match="justificar"):
        wf.transicionar("Q00001", "RESOLVIDA", "ana", "resolvido direto")
    assert wf.historico().empty
    wf.transicionar("Q00001", "JUSTIFICADA", "ana", "cota digitada errada pelo adm")
    wf.transicionar("Q00001", "RESOLVIDA", "bia", "administrador reenviou o arquivo")
    assert wf.status_atual(["Q00001"]).iloc[0] == "RESOLVIDA"
    with pytest.raises(TransicaoInvalida):
        wf.transicionar("Q00001", "ABERTA", "ana", "reabrir depois de resolvida")  # RESOLVIDA é final
    h = wf.historico()
    assert h["status"].tolist() == ["JUSTIFICADA", "RESOLVIDA"] and h["usuario"].tolist() == ["ana", "bia"]
    assert h["seq"].tolist() == [1, 2] and h["timestamp"].notna().all()


def test_workflow_exige_usuario_e_comentario_e_quebra_existente(tmp_path):
    wf = WorkflowQuebras(tmp_path)
    for usuario, comentario in (("", "comentario valido"), ("  ", "comentario valido"), ("ana", ""), ("ana", "ok")):
        with pytest.raises(TransicaoInvalida):
            wf.transicionar("Q1", "JUSTIFICADA", usuario, comentario)
    with pytest.raises(TransicaoInvalida, match="inexistente"):
        wf.transicionar("QX", "JUSTIFICADA", "ana", "comentario valido", existentes=["Q1", "Q2"])
    with pytest.raises(TransicaoInvalida, match="desconhecido"):
        wf.transicionar("Q1", "FINALIZADA", "ana", "comentario valido")
    wf.transicionar("Q1", "JUSTIFICADA", "ana", "comentario valido")
    wf.transicionar("Q1", "ABERTA", "ana", "reaberta por engano")  # justificada pode ser reaberta
    assert wf.status_atual(["Q1", "Q2"]).to_dict() == {"Q1": "ABERTA", "Q2": "ABERTA"}


def test_workflow_acopla_ao_motor(ciclo_curto, tmp_path):
    _, res, _ = ciclo_curto
    q = res["quebras_conciliacao"]
    assert len(q) >= 3  # ciclo curto com taxa de injeção alta
    wf = WorkflowQuebras(tmp_path)
    wf.transicionar(q["id_quebra"].iloc[0], "JUSTIFICADA", "ana", "dividendo creditado no dia seguinte")
    t = wf.quebras_com_status(q)
    assert t["status_workflow"].iloc[0] == "JUSTIFICADA" and (t["status_workflow"].iloc[1:] == "ABERTA").all()
    assert (q["status"] == "ABERTA").all()  # o status do motor é a origem; o workflow vive à parte


# ======================================================================= lâmina

def test_lamina_mensal_conteudo_e_tres_casas(ciclo_curto):
    _, res, _ = ciclo_curto
    fundos = res["cota_diaria"][["fundo_cnpj", "fundo"]].drop_duplicates()
    cnpj = fundos.loc[fundos["fundo"].str.contains("Macro"), "fundo_cnpj"].iloc[0]
    dados = {**res, "ativos": pd.read_parquet(ciclo_curto[2] / "ativos.parquet")}
    html = gerar_lamina(dados, cnpj, "2023-01", "MULTIMERCADO")
    for trecho in ("Sim Macro Multimercado", "Rentabilidade", "Composição por classe", "Maiores posições", "Risco",
                   "CDI", "PL líquido", "2023"):
        assert trecho in html
    cota = res["cota_diaria"].query("fundo_cnpj == @cnpj").sort_values("data")
    ult_jan = cota[cota["data"] <= "2023-01-31"].iloc[-1]
    assert f"{ult_jan['cota_liquida']:.3f}" in html and f"{ult_jan['pl_liquido']:.3f}" in html
    assert "n/d" in html  # 12 meses: histórico insuficiente
    rent = periodos(res["cota_diaria"].query("fundo_cnpj == @cnpj"), "2023-01")
    assert rent["mês"][0] == pytest.approx(ult_jan["cota_liquida"] / cota["cota_liquida"].iloc[0] - 1)
    assert rent["12 meses"] == (None, None)
    fev = gerar_lamina(dados, cnpj, "2023-02", "MULTIMERCADO")
    assert fev != html and "<html" in fev
    with pytest.raises(KeyError):
        gerar_lamina(dados, "00.000.000/0000-00", "2023-01")


# ======================================================================= app Streamlit

@pytest.mark.parametrize("modulo", PAGINAS)
def test_paginas_importam_e_tem_render(modulo):
    m = importlib.import_module(f"app.paginas.{modulo}")
    assert callable(m.render)


def test_scripts_do_app_rodam_sem_erro(ciclo_curto, monkeypatch):
    """Executa Home e as 8 páginas com o AppTest do Streamlit sobre os parquets de um ciclo curto."""
    from streamlit.testing.v1 import AppTest

    _, _, destino = ciclo_curto
    monkeypatch.setenv("GESTORASIM_DADOS", str(destino))
    scripts = [RAIZ / "app" / "Home.py", *sorted((RAIZ / "app" / "pages").glob("*.py"))]
    assert len(scripts) == 9
    for s in scripts:
        at = AppTest.from_file(str(s), default_timeout=120).run()
        assert not at.exception, f"{s.name}: {[str(e.value)[:300] for e in at.exception]}"
        assert not at.error, f"{s.name}: {[e.value[:200] for e in at.error]}"
        assert len(at.title) == 1


def test_app_sem_dados_orienta_o_usuario(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("GESTORASIM_DADOS", str(tmp_path))
    at = AppTest.from_file(str(RAIZ / "app" / "Home.py"), default_timeout=60).run()
    assert not at.exception and any("orquestrador" in w.value for w in at.warning)


def test_boletagem_da_interface_pre_trade_e_envio(dados_copia, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("GESTORASIM_DADOS", str(dados_copia))
    at = AppTest.from_file(str(RAIZ / "app" / "pages" / "1_Boletagem.py"), default_timeout=120).run()
    assert not at.exception
    at.number_input(key=at.number_input[1].key).set_value(500).run()  # 2º fundo (Multimercado)
    assert not at.exception and len(at.dataframe) >= 1  # pré-trade ao vivo na tela
    antes = hash_oficial(dados_copia)
    next(b for b in at.button if b.label == "Enviar ordem").click().run()
    assert not at.exception and any("ALOCADA" in s.value or "BLOQUEADA" in s.value for s in [*at.success, *at.warning])
    assert hash_oficial(dados_copia) == antes and (dados_copia / "ordens_manuais.parquet").exists()


def test_conciliacao_da_interface_segue_o_workflow(dados_copia, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("GESTORASIM_DADOS", str(dados_copia))
    at = AppTest.from_file(str(RAIZ / "app" / "pages" / "6_Conciliacao.py"), default_timeout=120).run()
    assert not at.exception
    at.text_input(key="conc_usuario").set_value("ana")
    at.text_area(key="conc_comentario").set_value("resolvido sem justificar")
    next(b for b in at.button if b.label == "Resolver").click().run()
    assert any("justificar" in e.value for e in at.error)  # RESOLVIDA direto é recusada na tela
    assert not (dados_copia / "workflow_quebras.parquet").exists()
    at.text_area(key="conc_comentario").set_value("cota digitada errada pelo administrador")
    next(b for b in at.button if b.label == "Justificar").click().run()
    h = pd.read_parquet(dados_copia / "workflow_quebras.parquet")
    assert h["status"].tolist() == ["JUSTIFICADA"] and h["usuario"].iloc[0] == "ana"
