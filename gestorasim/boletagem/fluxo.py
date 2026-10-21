"""Mesa de boletagem: conduz cada ordem pelo ciclo de vida completo e registra tudo."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from gestorasim.boletagem import execucao
from gestorasim.boletagem.auditoria import Trilha
from gestorasim.boletagem.modelos import EstadoOrdem as E
from gestorasim.boletagem.modelos import OrdemAgregada
from gestorasim.boletagem.rateio import ratear
from gestorasim.dominio import Fundo
from gestorasim.enquadramento import avaliar_pre_trade, lote, violacoes
from gestorasim.posicao import Ledger, aplicar_execucao

USUARIO_SISTEMA = "sistema.pretrade"
USUARIO_MESA = "mesa.execucao"


class Mesa:
    def __init__(self, ledger: Ledger, fundos: list[Fundo], meta: pd.DataFrame, data,
                 precos: pd.Series, volume: pd.Series) -> None:
        self.ledger, self.meta, self.data = ledger, meta, pd.Timestamp(data)
        self.fundos = {f.cnpj: f for f in fundos}
        self.precos, self.volume = precos, volume
        self.trilha = Trilha(datetime.combine(self.data.date(), datetime.min.time()).replace(hour=9))
        self.ordens: dict[str, OrdemAgregada] = {}
        self._pretrade: list[dict] = []
        self._execucoes: list[dict] = []
        self._alocacoes: list[dict] = []

    # ------------------------------------------------------------------ etapas
    def criar(self, ordem: OrdemAgregada) -> None:
        if ordem.id in self.ordens:
            raise ValueError(f"ordem duplicada: {ordem.id}")
        self.ordens[ordem.id] = ordem
        self.trilha.transicionar(ordem, E.CRIADA, ordem.usuario,
                                 f"{ordem.lado.value} {ordem.ticker} alvo={ordem.quantidade_alvo_total} "
                                 f"em {len(ordem.alvos)} fundo(s); origem={ordem.origem}")

    def pre_trade(self, ordem: OrdemAgregada) -> bool:
        """Avalia cada fundo. Fundos que violam são excluídos (evento EXCLUIDO_PRE_TRADE) e os demais
        seguem; a ordem só fica BLOQUEADA, com os alvos originais, se todos violarem."""
        aprovados, excluidos, n_regras = {}, [], 0
        for cnpj, qtd in ordem.alvos.items():
            f = self.fundos[cnpj]
            res = avaliar_pre_trade(self.ledger.quantidades(cnpj), self.ledger.caixa_projetado(cnpj),
                                    f.regulamento, self.meta, self.precos, ticker=ordem.ticker,
                                    lado=ordem.lado.value, quantidade=qtd)
            n_regras += len(res)
            for r in res:
                self._pretrade.append({"ordem_id": ordem.id, "fundo_cnpj": cnpj, **r.asdict()})
            viol = violacoes(res)
            if viol:
                excluidos.append((cnpj, f.nome, viol))
            else:
                aprovados[cnpj] = qtd

        def descrever(viol):
            return ", ".join(f"{r.regra}[{r.detalhe}] {r.valor_pos_ordem:.3f} vs {r.limite:.3f}"
                             for r in viol[:4])

        if not aprovados:
            resumo = "; ".join(f"{nome}: {descrever(v)}" for _, nome, v in excluidos[:3])
            self.trilha.transicionar(ordem, E.BLOQUEADA, USUARIO_SISTEMA,
                                     f"todos os {len(excluidos)} fundo(s) violam: {resumo}")
            return False
        for cnpj, nome, viol in excluidos:
            self.trilha.evento(ordem, "EXCLUIDO_PRE_TRADE", USUARIO_SISTEMA,
                               f"{nome} ({cnpj}) excluído, alvo {ordem.alvos[cnpj]}: {descrever(viol)}")
        ordem.alvos = aprovados
        self.trilha.transicionar(ordem, E.PRE_TRADE_OK, USUARIO_SISTEMA,
                                 f"{n_regras} regras avaliadas; {len(aprovados)} fundo(s) liberado(s), "
                                 f"{len(excluidos)} excluído(s)")
        return True

    def liberar_override(self, ordem: OrdemAgregada, usuario: str, justificativa: str) -> None:
        self.trilha.transicionar(ordem, E.LIBERADA_OVERRIDE, usuario, justificativa)

    def enviar_e_executar(self, ordem: OrdemAgregada) -> None:
        self.trilha.transicionar(ordem, E.ENVIADA, USUARIO_MESA, "ordem enviada ao mercado simulado")
        fech = float(self.precos[ordem.ticker])
        vol = int(self.volume[ordem.ticker])
        fills = execucao.simular_fills(ordem.quantidade_alvo_total, ordem.lado.value, fech, vol,
                                       ordem.classe.value)
        ordem.fills = fills
        ordem.quantidade_executada = sum(q for q, _ in fills)
        ordem.preco_medio = execucao.preco_medio(fills)
        for i, (q, p) in enumerate(fills, 1):
            self._execucoes.append({"ordem_id": ordem.id, "fill": i, "data": self.data, "ticker": ordem.ticker,
                                    "lado": ordem.lado.value, "quantidade": q, "preco": p,
                                    "fechamento": fech, "volume_dia": vol, "participacao": q / vol})
        total = ordem.quantidade_alvo_total
        estado = E.EXECUTADA if ordem.quantidade_executada == total else E.EXECUTADA_PARCIAL
        self.trilha.transicionar(
            ordem, estado, USUARIO_MESA,
            f"{ordem.quantidade_executada}/{total} a PM={ordem.preco_medio:.3f} em {len(fills)} fill(s); "
            f"participação={ordem.quantidade_executada / vol:.3f} do volume")

    def ratear(self, ordem: OrdemAgregada) -> None:
        aloc, criterio, cancelado = ratear(ordem.alvos, ordem.quantidade_executada, lote(ordem.classe))
        if cancelado:
            ordem.quantidade_cancelada = cancelado
            self.trilha.evento(ordem, "CANCELAMENTO_EXCEDENTE", USUARIO_MESA,
                               f"{cancelado} unidade(s) executadas além dos tetos por fundo; canceladas")
        for cnpj, q in aloc.items():
            self._alocacoes.append({
                "ordem_id": ordem.id, "fundo_cnpj": cnpj, "ticker": ordem.ticker, "lado": ordem.lado.value,
                "quantidade_alvo": ordem.alvos[cnpj], "quantidade_alocada": q,
                "preco_medio": ordem.preco_medio, "financeiro": q * ordem.preco_medio,
            })
            if q > 0:
                aplicar_execucao(self.ledger, data=self.data, fundo_cnpj=cnpj, ticker=ordem.ticker,
                                 classe=ordem.classe.value, lado=ordem.lado.value, quantidade=q,
                                 preco=ordem.preco_medio, origem=ordem.id,
                                 preco_referencia=float(self.precos[ordem.ticker]))
        self.trilha.transicionar(ordem, E.ALOCADA, USUARIO_MESA,
                                 f"{criterio}; PM único={ordem.preco_medio:.3f}")

    # ------------------------------------------------------------------ orquestração
    def processar(self, ordem: OrdemAgregada) -> OrdemAgregada:
        self.criar(ordem)
        if not self.pre_trade(ordem):
            if ordem.pedido_override is None:
                return ordem
            self.liberar_override(ordem, *ordem.pedido_override)
        self.enviar_e_executar(ordem)
        self.ratear(ordem)
        return ordem

    # ------------------------------------------------------------------ saídas
    def tabelas(self) -> dict[str, pd.DataFrame]:
        ordens = pd.DataFrame([{
            "ordem_id": o.id, "data": pd.Timestamp(o.data), "ticker": o.ticker, "classe": o.classe.value,
            "lado": o.lado.value, "origem": o.origem, "n_fundos": len(o.alvos_originais),
            "n_fundos_excluidos": len(o.alvos_originais) - len(o.alvos),
            "quantidade_alvo_original": sum(o.alvos_originais.values()),
            "quantidade_alvo": o.quantidade_alvo_total, "quantidade_cancelada": o.quantidade_cancelada, "quantidade_executada": o.quantidade_executada,
            "preco_medio": o.preco_medio, "estado": o.estado.value,
        } for o in self.ordens.values()])
        return {
            "ordens": ordens,
            "execucoes": pd.DataFrame(self._execucoes, columns=[
                "ordem_id", "fill", "data", "ticker", "lado", "quantidade", "preco", "fechamento",
                "volume_dia", "participacao"]),
            "alocacoes": pd.DataFrame(self._alocacoes, columns=[
                "ordem_id", "fundo_cnpj", "ticker", "lado", "quantidade_alvo", "quantidade_alocada",
                "preco_medio", "financeiro"]),
            "auditoria": self.trilha.para_df(),
            "pretrade": pd.DataFrame(self._pretrade),
            "ledger": self.ledger.para_df(),
        }
