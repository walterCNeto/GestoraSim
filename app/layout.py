"""Elementos de layout comuns: cabeçalho de página, linha de KPIs, glossário e selos de status."""

from __future__ import annotations

import streamlit as st

from app import formatacao as F

GLOSSARIO: dict[str, str] = {
    "DV01": "Variação do valor da carteira (R$) para uma alta de 1 bp (0,01%) nas taxas de juros. Negativo: perde se os juros sobem. "
            "Aqui é calculado por vértice da curva, com reavaliação completa dos títulos.",
    "CS01": "Variação do valor (R$) para uma alta de 1 bp no spread de crédito do rating. Mede o risco de crédito das debêntures.",
    "VaR": "Value at Risk: perda que não deve ser excedida em 99% dos dias (1 dia), em % do PL líquido. "
           "Paramétrico (delta-normal, covariância EWMA) ou histórico (252 pregões).",
    "ES": "Expected Shortfall: perda média nos dias piores que o VaR (aqui, 97,5% histórico). Enxerga a cauda que o VaR ignora.",
    "Stress": "Cenário extremo com reavaliação completa da carteira (ex.: choque de juros e bolsa). Resultado em % do PL.",
    "Backtest": "Compara o VaR de ontem com o P&L de hoje. Exceção = perda maior que o VaR. Com 99%, espera-se ~1% de exceções "
                "(testes de Kupiec e zonas de Basileia). P&L limpo: só fatores de mercado; sujo: inclui carrego e eventos.",
    "Linha d'água": "Maior valor de cota já atingido pelo cotista (high-water mark). A taxa de performance só incide sobre "
                    "o que superar a linha d'água.",
    "Come-cotas": "Antecipação semestral de IR (maio e novembro): a administradora cancela cotas para pagar o imposto "
                  "sobre o rendimento, mesmo sem resgate.",
    "Desenquadramento ativo": "Causado por ordens do dia: sem elas a carteira estaria enquadrada. É falha do gestor e "
                              "deve ser evitado na pré-trade.",
    "Desenquadramento passivo": "Causado por mercado, eventos ou fluxo do passivo, sem ordem do gestor. Há prazo "
                                "(dias úteis) para reenquadrar; estourado, vira VENCIDO.",
    "Cota paralela": "A cota que o gestor calcula em paralelo à do administrador. A diferença (em bp) revela erros de "
                     "preço, provisão ou posição.",
    "Pré-trade": "Checagem das regras de enquadramento antes de enviar a ordem, simulando a carteira pós-ordem.",
    "Override": "Liberação manual de uma ordem bloqueada, exigindo usuário e justificativa gravados na trilha de auditoria.",
    "Rateio": "Divisão da quantidade executada entre os fundos participantes (maior resto, limitada à meta de cada fundo).",
    "PL líquido": "Patrimônio líquido do fundo = ativos − taxas a pagar − resgates a pagar. Cota = PL líquido / nº de cotas.",
    "Dias para zerar": "Quantos dias para vender a posição a 20% do volume diário de negociação.",
    "Quebra": "Divergência entre a carteira/cota do gestor e o arquivo do administrador. Ciclo: ABERTA → JUSTIFICADA → RESOLVIDA.",
}


def cabecalho(titulo: str, real: str = "", glossario: tuple[str, ...] = ()) -> None:
    """Título, 'o que esta tela faz numa gestora real' e expander 'Como ler esta página'."""
    st.title(titulo)
    if real:
        st.caption(f"**Numa gestora real:** {real}")
    if glossario:
        with st.expander("Como ler esta página"):
            for termo in glossario:
                st.markdown(f"**{termo}** — {GLOSSARIO[termo]}")


def kpis(itens: list[tuple], por_linha: int | None = None) -> None:
    """Linha de st.metric. Cada item: (rótulo, valor_texto, delta_texto|None, delta_color, ajuda|None).
    delta_color: 'normal' (alta é boa), 'inverse' (alta é ruim: risco) ou 'off'."""
    n = por_linha or len(itens)
    for i in range(0, len(itens), n):
        _linha_kpis(itens[i:i + n], n)


def _linha_kpis(itens: list[tuple], n: int) -> None:
    cols = st.columns(n)
    for col, item in zip(cols, itens):
        rotulo, valor, delta, cor, ajuda = (*item, *(None, None, None, "normal", None)[len(item):])
        col.metric(rotulo, valor, delta=delta, delta_color=cor or "normal", help=ajuda)


def selo(txt) -> str:
    return F.badge_html(txt)


def selo_md(txt) -> None:
    st.markdown(F.badge_html(txt), unsafe_allow_html=True)


def semaforo(nivel: str) -> str:
    """Bolinha colorida para texto: OK, SOFT/ALERTA, HARD/VIOLADO."""
    return {F.VERDE: "🟢", F.AMBAR: "🟠", F.VERMELHO: "🔴"}.get(F.cor_status(nivel), "⚪")


def stepper(etapas: list[tuple[str, str]]) -> None:
    """Linha do tempo de estados como etapas: [(estado, legenda)]. A cor segue o status; a última etapa ganha destaque."""
    from html import escape

    blocos = []
    for i, (estado, legenda) in enumerate(etapas):
        cor = F.cor_status(estado)
        borda = "3px" if i == len(etapas) - 1 else "1px"
        blocos.append(
            f"<div style='flex:1 1 150px;min-width:140px;border:{borda} solid {cor};background:{F.FUNDO_CLARO[cor]};"
            f"border-radius:8px;padding:6px 10px'><div style='color:{cor};font-weight:700;font-size:.9em'>"
            f"{i + 1}. {escape(str(estado))}</div><div style='font-size:.78em;color:#444;word-break:break-word'>"
            f"{escape(str(legenda))}</div></div>")
        if i < len(etapas) - 1:
            blocos.append("<div style='align-self:center;color:#6b7480;font-size:1.3em'>→</div>")
    st.markdown("<div style='display:flex;flex-wrap:wrap;gap:6px;align-items:stretch;margin-bottom:.8rem'>" + "".join(blocos) + "</div>",
                unsafe_allow_html=True)
