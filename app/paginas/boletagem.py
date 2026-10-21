"""Boletagem manual em três passos: ordem → pré-trade → envio ou override.

As ordens manuais vão para um ledger separado (`*_manuais.parquet`, tag MANUAL) e não alteram a série oficial.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app import dados, graficos
from app import formatacao as F
from app.layout import selo, stepper
from gestorasim.boletagem.manual import BoletagemManual

TITULO = "Boletagem"
REAL = ("é a mesa do gestor: digitar a ordem, ver na hora o efeito sobre cada regra do regulamento (pré-trade), "
        "enviar, e, se for o caso, liberar com justificativa — tudo registrado na trilha de auditoria.")
GLOSSARIO = ("Pré-trade", "Override", "Rateio", "Desenquadramento ativo")
_REGRAS_MINIMO = {"minimo_classe", "liquidez_minima", "caixa_suficiente", "caixa_minimo_projetado"}
_REGRAS_REAIS = {"caixa_suficiente", "caixa_minimo_projetado", "sem_venda_descoberto"}
SOFT = 0.85


@st.cache_resource(show_spinner="Carregando posições, preços e volumes…")
def _servico(pasta: str, escrita: str, mtime: float) -> BoletagemManual:
    return BoletagemManual(pasta, destino_escrita=escrita)


def _uso(r) -> float:
    """Uso do limite (1 = no limite; > 1 = violado). Regras de mínimo invertem a razão."""
    v, lim = float(r.valor_pos_ordem), float(r.limite)
    if r.regra in _REGRAS_MINIMO:
        return (0.0 if r.status != "VIOLADO" else 1.5) if v <= 0 or lim <= 0 else min(lim / v, 1.5)
    if lim <= 0:
        return 0.0 if v <= 0 else 1.5
    return min(v / lim, 1.5)


def _barras_folga(pre: pd.DataFrame) -> None:
    pre = pre.assign(uso=[_uso(r) for r in pre.itertuples()]).sort_values("uso", ascending=False)
    rot, hov, cores = [], [], []
    for r in pre.itertuples():
        f = F.brl_mi if r.regra in _REGRAS_REAIS else F.pct
        rot.append(f"{r.fundo.split()[-1]} · {r.regra} ({r.detalhe})")
        hov.append(f"valor pós-ordem {f(r.valor_pos_ordem)} · limite {f(r.limite)} · folga {f(r.folga)} · {r.status}")
        cores.append(F.VERMELHO if r.status == "VIOLADO" else F.AMBAR if r.uso >= SOFT else F.VERDE)
    import plotly.graph_objects as go

    fig = go.Figure(go.Bar(y=rot, x=pre["uso"], orientation="h", marker_color=cores, customdata=hov,
                           hovertemplate="%{y}<br>%{customdata}<extra></extra>"))
    fig.add_vline(x=1.0, line=dict(color=F.VERMELHO, dash="dash"), annotation_text="limite", annotation_font_color=F.VERMELHO)
    fig.update_layout(height=max(220, 26 * len(rot) + 70), margin=dict(l=10, r=10, t=30, b=10), separators=",.",
                      template="plotly_white", showlegend=False)
    fig.update_xaxes(tickformat=".0%", range=[0, 1.55], title_text="uso do limite (100% = no limite; barra cheia além = violação)")
    fig.update_yaxes(autorange="reversed")
    graficos.mostrar(fig, "bol_folga")


def render() -> None:
    dados.exigir("posicoes_diarias", "pl_diario", "cota_diaria", "mtm_painel", "ativos", "fundos", "calendario")
    mtime = (dados.pasta() / "pl_diario.parquet").stat().st_mtime
    bm = _servico(str(dados.pasta()), str(dados.pasta_gravavel()), mtime)
    dados.aviso_escrita()
    st.info("Ordens manuais ficam em um ledger **separado** (tag MANUAL): a série oficial não é alterada. "
            "O estado de partida é o fechamento oficial da data escolhida na barra lateral.")

    # ---- passo 1 --------------------------------------------------------------------------------------------
    st.subheader("Passo 1 — Ordem")
    data = dados.escolher_data(list(bm.datas))
    c1, c2, c3 = st.columns(3)
    ticker = c1.selectbox("Ativo", bm.ativos_negociaveis(data), key="bol_ticker")
    lado = c2.radio("Lado", ["C", "V"], horizontal=True, format_func=lambda x: "Compra" if x == "C" else "Venda", key="bol_lado")
    lote = bm.lote(ticker)
    c3.metric("Preço de fechamento (R$)", F.num(bm.preco(data, ticker)), help=f"Data da ordem: {F.data(data)} (barra lateral).")
    st.caption(f"Data da ordem: **{F.data(data)}** · lote padrão {F.qtd(lote)}. Quantidade por fundo (0 = não participa):")
    qtds = {}
    cols = st.columns(len(bm.fundos))
    for col, f in zip(cols, bm.fundos):
        qtds[f.cnpj] = col.number_input(f.nome, min_value=0, step=lote, value=0, key=f"bol_q_{f.cnpj}")
    if not any(qtds.values()):
        st.info("Informe uma quantidade em pelo menos um fundo para ver o pré-trade.")
        st.stop()

    # ---- passo 2 --------------------------------------------------------------------------------------------
    st.subheader("Passo 2 — Pré-trade (carteira simulada pós-ordem)")
    try:
        pre = bm.pre_trade(data, ticker, lado, qtds)
    except ValueError as e:
        st.error(str(e))
        st.stop()
    violadas = pre[pre["status"] == "VIOLADO"]
    _barras_folga(pre)
    if violadas.empty:
        st.success("Nenhuma regra violada: a ordem passa no pré-trade.")
    else:
        fundos_viol = ", ".join(sorted(set(violadas["fundo"])))
        st.error(f"{len(violadas)} violação(ões) — fundos: {fundos_viol}. Fundos que violam são excluídos da ordem; "
                 "se todos violam, a ordem fica BLOQUEADA e só segue com override justificado.")
    with st.expander("Detalhe das regras (valores e limites)"):
        pre2 = pre.assign(unid=pre["regra"].isin(_REGRAS_REAIS))
        pre2["valor_txt"] = [F.brl_mi(r.valor_pos_ordem) if r.unid else F.pct(r.valor_pos_ordem) for r in pre2.itertuples()]
        pre2["limite_txt"] = [F.brl_mi(r.limite) if r.unid else F.pct(r.limite) for r in pre2.itertuples()]
        pre2["folga_txt"] = [F.brl_mi(r.folga) if r.unid else F.pct(r.folga) for r in pre2.itertuples()]
        F.exibir(pre2, {"fundo": ("Fundo", "txt"), "regra": ("Regra", "txt"), "detalhe": ("Detalhe", "txt"),
                        "valor_txt": ("Valor pós-ordem", "txt"), "limite_txt": ("Limite", "txt"), "folga_txt": ("Folga", "txt"),
                        "status": ("Status", "txt")}, badges=("status",), altura=300)

    # ---- passo 3 --------------------------------------------------------------------------------------------
    st.subheader("Passo 3 — Envio ou override")
    usuario = st.text_input("Usuário", value="gestor.manual", key="bol_usuario")
    just = st.text_area("Justificativa do override (obrigatória para liberar uma ordem bloqueada; mínimo 10 caracteres)",
                        key="bol_just", placeholder="Ex.: reenquadramento autorizado pelo comitê de risco em 09/01.")
    ok_just = len(just.strip()) >= 10 and bool(usuario.strip())
    if not violadas.empty and not ok_just:
        st.warning("Para liberar ordem com violação é preciso informar usuário e justificativa (visível na trilha de auditoria).")
    if not violadas.empty and ok_just:
        st.markdown(f"Justificativa registrada para o override: _{just.strip()}_")
    b1, b2 = st.columns(2)
    resultado = None
    if b1.button("Enviar ordem", type="primary", help="Segue o fluxo normal: violadores saem; se todos violam, BLOQUEADA."):
        resultado = bm.enviar(data, ticker, lado, qtds, usuario)
    if b2.button("Enviar com override", disabled=not ok_just, help="Habilita com usuário e justificativa preenchidos."):
        resultado = bm.enviar(data, ticker, lado, qtds, usuario, (usuario, just))
    if resultado is not None:
        estado = resultado["estado"]
        (st.success if estado == "ALOCADA" else st.warning)(f"Estado final: {estado}")
        aud = resultado["auditoria"]
        if len(aud):
            seq = [(r.para, f"{r.usuario} · {r.motivo}") for r in aud[aud["tipo"] == "TRANSICAO"].itertuples()]
            if seq:
                st.markdown("**Ciclo de vida da ordem**")
                stepper(seq)
        for titulo, chave, spec in (
                ("Trilha de auditoria", "auditoria", {"seq": ("Seq", "qtd"), "timestamp": ("Timestamp", "txt"), "usuario": ("Usuário", "txt"),
                                                      "tipo": ("Tipo", "txt"), "de": ("De", "txt"), "para": ("Para", "txt"),
                                                      "motivo": ("Motivo", "txt")}),
                ("Execução (fills)", "execucoes", None), ("Rateio entre fundos", "alocacoes", None)):
            if len(resultado[chave]):
                st.markdown(f"**{titulo}**")
                if spec:
                    F.exibir(resultado[chave], spec, badges=("para",))
                else:
                    st.dataframe(F.tres_casas(resultado[chave]), hide_index=True)

    st.subheader("Ordens manuais gravadas")
    om = dados.tabela_gravavel("ordens_manuais")
    if len(om):
        om = om.sort_values("data", ascending=False)
        F.exibir(om.assign(preco_medio=om["preco_medio"].astype(float)) if "preco_medio" in om else om,
                 {c: (c, "data" if c == "data" else "num" if str(om[c].dtype).startswith("float") else "txt") for c in om.columns},
                 badges=("estado",))
    else:
        st.caption("Nenhuma ordem manual gravada ainda.")
