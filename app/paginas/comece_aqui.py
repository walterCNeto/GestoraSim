"""Comece aqui: roteiro de 7 passos para ver, de ponta a ponta, o que uma plataforma de gestão faz."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app import dados
from app import formatacao as F

TITULO = "Comece aqui"
REAL = ("a rotina de uma gestora tem sempre o mesmo fio: decidir e boletar ordens, checar regras antes e depois, medir risco, "
        "reconciliar com o administrador, cuidar do passivo e prestar contas. Este roteiro percorre esse fio em 7 passos.")
GLOSSARIO = ("Pré-trade", "Override", "Desenquadramento ativo", "Desenquadramento passivo", "Cota paralela", "Come-cotas")


def _ir(data, fundo) -> None:
    """Callback dos botões: ajusta os filtros globais da barra lateral."""
    st.session_state["gs_data"] = pd.Timestamp(data)
    if fundo:
        st.session_state["gs_fundo"] = fundo


def fenomenos() -> dict:
    """Procura nos dados gravados um exemplo de cada fenômeno (as datas variam com o dataset)."""
    nomes = dados.fundos()
    nome = lambda cnpj: nomes.get(str(cnpj).split(",")[0], None)  # noqa: E731
    out: dict = {}
    o = dados.tabela_ou_vazia("ordens_serie")
    if len(o):
        b = o[(o["estado"] == "BLOQUEADA") & o["origem"].fillna("").str.startswith("VIOLADORA")].sort_values(["data_serie", "ordem_id"])
        if b.empty:
            b = o[o["estado"] == "BLOQUEADA"].sort_values(["data_serie", "ordem_id"])
        if len(b):
            r = b.iloc[0]
            out["ordem"] = dict(data=r["data_serie"], ordem=r["ordem_id"], ticker=r["ticker"], lado="compra" if r["lado"] == "C" else "venda",
                                qtd=r["quantidade_alvo_original"], fundo=nome(r["fundos"]), origem=r["origem"])
    ep = dados.tabela_ou_vazia("episodios")
    if len(ep):
        for classe in ("PASSIVO", "ATIVO"):
            e = ep[ep["classe"] == classe].sort_values("inicio")
            if len(e):
                r = e.iloc[0]
                out[classe.lower()] = dict(data=r["inicio"], fundo=nomes.get(r["fundo_cnpj"]), regra=r["regra"], detalhe=r["detalhe"],
                                           ordem=r["ordem_principal"], fim=r["fim"])
    ex = dados.tabela_ou_vazia("excecoes_var")
    if len(ex):
        r = ex.sort_values("data").iloc[0]
        out["excecao"] = dict(data=r["data"], fundo=r["fundo"])
    q = dados.tabela_ou_vazia("quebras_conciliacao")
    if len(q):
        qq = q[q["tipo"] == "CAIXA"] if (q["tipo"] == "CAIXA").any() else q
        r = qq.sort_values("data").iloc[0]
        out["quebra"] = dict(data=r["data"], fundo=nomes.get(r["fundo_cnpj"]), id=r["id_quebra"], tipo=r["tipo"], item=r["item"])
    t = dados.tabela_ou_vazia("tributos")
    if len(t) and (t["tipo"] == "COME_COTAS").any():
        d = t[t["tipo"] == "COME_COTAS"].sort_values("data")
        r = d.iloc[0]
        out["come_cotas"] = dict(data=r["data"], fundo=nomes.get(r["fundo_cnpj"]), n=int((d["data"] == r["data"]).sum()))
    datas = dados.datas_pregao()
    if datas:
        out["lamina"] = dict(data=datas[-1], fundo=None)
    return out


def _passo(n: int, titulo: str, texto: str, pagina: str, rotulo: str, sugestao: dict | None, detalhe: str = "") -> None:
    with st.container(border=True):
        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown(f"#### {n}. {titulo}")
            st.write(texto)
            if detalhe:
                st.caption(detalhe)
        with c2:
            if sugestao is None:
                st.caption("Este fenômeno não ocorre no dataset carregado.")
            else:
                st.markdown(f"**Data sugerida:** {F.data(sugestao['data'])}" + (f" · {sugestao['fundo']}" if sugestao.get("fundo") else ""))
                st.button("Usar esta data e fundo", key=f"ir_{n}", on_click=_ir, args=(sugestao["data"], sugestao.get("fundo")))
            st.page_link(pagina, label=rotulo, icon="➡️")


def render() -> None:
    st.markdown(
        "O **GestoraSim** simula, com dados 100% fictícios, a operação de uma gestora com três fundos (Ações, Multimercado e "
        "Renda Fixa Crédito Privado): ordens, regras do regulamento, risco, contabilidade da cota, passivo e conciliação. "
        "Siga os passos abaixo; cada botão ajusta o fundo e a data da barra lateral para o dia em que o fenômeno ocorre.")
    if dados.existe("cota_diaria"):
        datas = dados.datas_pregao()
        st.caption(f"Dataset: {F.data(datas[0])} a {F.data(datas[-1])} ({F.qtd(len(datas))} pregões) · pasta `{dados.pasta()}`"
                   f"{' (demo reduzido)' if dados.modo_demo() else ''}.")
    else:
        dados.exigir("cota_diaria")
    f = fenomenos()

    o = f.get("ordem")
    _passo(1, "Boletagem com violação e override",
           "Boleta uma ordem que fere o regulamento: o pré-trade mostra a regra violada, o fundo é excluído e a ordem fica "
           "BLOQUEADA. Depois libere-a com override, informando usuário e justificativa.",
           "pages/3_Boletagem.py", "Abrir Boletagem", o,
           "" if not o else f"Exemplo: {o['ordem']} ({o['origem']}) — {o['lado']} de {F.qtd(o['qtd'])} {o['ticker']}. "
                            f"Na página, escolha o ativo {o['ticker']}, lado compra e a quantidade no fundo indicado.")
    _passo(2, "Trilha de auditoria",
           "Veja a mesma ordem como linha do tempo: criada, pré-trade, bloqueio ou liberação, envio, execução e rateio, "
           "cada passo com usuário, hora e motivo.",
           "pages/8_Trilha_de_auditoria.py", "Abrir Trilha de auditoria", o,
           "" if not o else f"Selecione {o['ordem']} em 'Ordem do dia'.")
    ativo = f.get("ativo")
    _passo(3, "Risco no dia seguinte (D+1)",
           "A ordem muda a carteira e o risco muda no dia seguinte: acompanhe o VaR contra o limite da política, o DV01 por vértice e o stress.",
           "pages/4_Risco.py", "Abrir Risco", ativo or f.get("excecao"),
           (f"No dia sugerido o VaR do {ativo['fundo']} rompeu o limite da política depois de {ativo['ordem'] or 'ordens do dia'}."
            if ativo else "O backtest mostra as exceções do VaR como marcas × no gráfico."))
    p = f.get("passivo")
    _passo(4, "Enquadramento: desenquadramento ativo × passivo",
           "Compare dois tipos de violação: o ATIVO nasce de uma ordem do gestor; o PASSIVO nasce do mercado ou do fluxo e dá prazo "
           "para reenquadrar. A linha do tempo mostra os episódios e a ordem principal causadora.",
           "pages/5_Enquadramento.py", "Abrir Enquadramento", p or ativo,
           "" if not p else f"Episódio passivo: {p['regra']} ({p['detalhe']}) em {p['fundo']}"
                            + (f"; episódio ativo em {F.data(ativo['data'])}." if ativo else "."))
    q = f.get("quebra")
    _passo(5, "Conciliação com o administrador",
           "O arquivo do administrador traz divergências injetadas. Abra a quebra, leia o diagnóstico automático, justifique e resolva; "
           "veja também a cota paralela em bp.",
           "pages/7_Conciliação.py", "Abrir Conciliação", q,
           "" if not q else f"Quebra {q['id']} ({q['tipo']} · {q['item']}). Use os botões Justificar e depois Resolver.")
    c = f.get("come_cotas")
    _passo(6, "Passivo: aplicações, resgates e come-cotas",
           "Acompanhe a cota bruta, a líquida e a linha d'água, as movimentações dos cotistas e o come-cotas, que cancela cotas "
           "para pagar IR no semestre.",
           "pages/6_Passivo_e_cota.py", "Abrir Passivo e cota", c,
           "" if not c else f"Come-cotas de {F.data(c['data'])}: {F.qtd(c['n'])} lotes tributados no fundo {c['fundo']} (e nos demais).")
    _passo(7, "Lâmina",
           "Feche o mês com a lâmina do fundo: rentabilidade contra o benchmark, composição, VaR e as 10 maiores posições, "
           "pronta para download em HTML.",
           "pages/9_Lâmina_mensal.py", "Abrir Lâmina mensal", f.get("lamina"))

    st.divider()
    st.markdown("Para o panorama geral, volte à **Visão geral**:")
    st.page_link("pages/1_Visão_geral.py", label="Abrir Visão geral", icon="🏠")
    st.page_link("pages/2_Carteira.py", label="Abrir Carteira", icon="📊")
