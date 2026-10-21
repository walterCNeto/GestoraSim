"""Engine do passivo de um fundo: cotistas, cota, taxas, tributos.

Convenções (todas parametrizáveis em `parametros.py`):
- A aplicação entra no caixa e emite cotas na data de conversão, à cota líquida de fechamento do dia (antes do fluxo).
- O resgate é convertido na data de conversão (cotas canceladas à cota líquida do dia) e vira `resgates_a_pagar`
  (valor bruto, incluindo o IR retido) até a data de pagamento, quando sai do caixa.
- Taxa de administração: provisão diária = PL líquido do dia anterior × ((1+tx)^(1/252) − 1); paga no 1º pregão do
  mês seguinte pelo saldo acumulado.
- Taxa de performance (método do ativo, linha d'água): sobre a cota bruta (antes da performance), provisão por cota
  = taxa × max(0, cota bruta − linha d'água), com linha d'água = cota de referência × (benchmark / benchmark de
  referência). Cobrança no último pregão de jun/dez: paga em caixa e, se houve cobrança, a referência passa a ser a
  cota líquida do dia (high-water mark). Sem cobrança, a referência se mantém.
- A provisão de performance (valor total) não é recalculada dentro do dia quando o número de cotas muda por fluxos,
  o que mantém PL líquido = cotas × cota líquida exatamente.
- Come-cotas (RF/MM): último pregão de maio/novembro, por lote, cancelando cotas sem alterar a cota.

Ciclo do dia (orquestrador): `abrir_dia` paga o que já é devido e conhecido (resgates convertidos antes e a taxa de
administração do mês anterior) e `fechar_dia` converte aplicações/resgates com a cota do dia, apura taxas e tributos.
Chamar só `fechar_dia` (uso antigo) faz as duas fases e devolve todos os fluxos do dia.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from gestorasim.cota.parametros import ParametrosPassivo, RegrasFundo
from gestorasim.cota.tributos import Lote, come_cotas, resgatar_fifo
from gestorasim.dados import calendario
from gestorasim.dominio import Fundo

COLUNAS_MOV = ["id_mov", "id_cotista", "fundo_cnpj", "tipo", "data_solicitacao", "valor_solicitado",
               "data_conversao", "data_pagamento"]
COLUNAS_MOV_COMPLETAS = COLUNAS_MOV + ["cota_conversao", "cotas", "valor_bruto", "ir", "iof", "valor_liquido", "status"]
_NAN = float("nan")


def e_primeiro_du_do_mes(data: pd.Timestamp) -> bool:
    return calendario.somar_du(data, -1).month != data.month


def e_ultimo_du_do_mes(data: pd.Timestamp) -> bool:
    return calendario.somar_du(data, 1).month != data.month


def agendar(fluxos: pd.DataFrame, regras: RegrasFundo, prefixo: str) -> pd.DataFrame:
    """Datas de conversão e de pagamento de cada movimentação, pelas regras do fundo."""
    linhas = []
    for n, f in enumerate(fluxos.sort_values(["data", "id_cotista"]).itertuples(), 1):
        d = pd.Timestamp(f.data)
        if f.tipo == "APLICACAO":
            conv, pag = calendario.somar_du(d, regras.conv_aplicacao_du), pd.NaT
        else:
            if regras.conv_resgate_tipo == "corridos":
                conv = calendario.proximo_dia_util(d + pd.Timedelta(days=regras.conv_resgate_n))
            else:
                conv = calendario.somar_du(d, regras.conv_resgate_n)
            pag = calendario.somar_du(conv, regras.pag_resgate_du)
        linhas.append((f"{prefixo}-{n:05d}", f.id_cotista, f.fundo_cnpj, f.tipo, d, float(f.valor), conv, pag))
    return pd.DataFrame(linhas, columns=COLUNAS_MOV)


@dataclass
class ResultadoDia:
    fluxos: list[tuple[float, str]]  # (valor no caixa, tipo): + entra, − sai
    cota: dict
    provisao: dict
    tributos: list[dict] = field(default_factory=list)


class PassivoFundo:
    def __init__(self, fundo: Fundo, regras: RegrasFundo, params: ParametrosPassivo, fluxos: pd.DataFrame) -> None:
        self.fundo, self.r, self.p = fundo, regras, params
        agenda = agendar(fluxos, regras, fundo.cnpj[7:10])
        self._mov: list[dict] = [
            {**rec, "cota_conversao": _NAN, "cotas": _NAN, "valor_bruto": _NAN, "ir": _NAN, "iof": _NAN,
             "valor_liquido": _NAN, "status": "AGENDADO"} for rec in agenda.to_dict("records")]
        self._por_conv: dict[pd.Timestamp, list[int]] = {}
        for k, m in enumerate(self._mov):
            self._por_conv.setdefault(m["data_conversao"], []).append(k)
        self._resg_ag = [k for k, m in enumerate(self._mov) if m["tipo"] == "RESGATE"]  # candidatos à agenda de saídas
        self._id_idx = {m["id_mov"]: k for k, m in enumerate(self._mov)}
        self.lotes: dict[str, list[Lote]] = {}
        self.cotas = 0.0
        self.adm_saldo = 0.0
        self.perf_saldo = 0.0
        self.resg_pend: list[dict] = []
        self.la_base = self.idx_base = None
        self.pl_liq_prev = self.A_prev = None
        self._n_lote = 0
        self.cota_rows: list[dict] = []
        self.prov_rows: list[dict] = []
        self.tributos: list[dict] = []
        self.ultima_cota = params.cota_inicial
        self._aberto_em: pd.Timestamp | None = None
        self._fluxos_abertura: list[tuple[float, str]] = []
        self._adm_pago_abertura = 0.0

    @property
    def mov(self) -> pd.DataFrame:
        return pd.DataFrame(self._mov, columns=COLUNAS_MOV_COMPLETAS)

    # ------------------------------------------------------------------ consultas
    def cotas_do_cotista(self, cotista: str) -> float:
        return sum(l.cotas for l in self.lotes.get(cotista, []))

    @property
    def passivos_total(self) -> float:
        """Passivos do fundo na data (taxas a pagar, performance provisionada e resgates a pagar)."""
        return self.adm_saldo + self.perf_saldo + sum(r["gross"] for r in self.resg_pend)

    def saidas_previstas(self, hoje: pd.Timestamp, datas: list[pd.Timestamp]) -> list[float]:
        """Saídas de caixa (negativas) esperadas nas `datas`: resgates convertidos a pagar, pedidos de resgate já
        feitos (pelo valor solicitado) e a taxa de administração acumulada no 1º pregão do mês."""
        out = [0.0] * len(datas)
        for r in self.resg_pend:
            if r["pag"] in datas:
                out[datas.index(r["pag"])] -= r["gross"]
        for k in self._resg_ag:
            m = self._mov[k]
            if m["status"] == "AGENDADO" and m["data_solicitacao"] <= hoje and m["data_pagamento"] in datas:
                out[datas.index(m["data_pagamento"])] -= m["valor_solicitado"]
        for k, d in enumerate(datas):
            if e_primeiro_du_do_mes(d):
                out[k] -= self.adm_saldo
        return out

    # ------------------------------------------------------------------ abertura: pagamentos já devidos
    def abrir_dia(self, data: pd.Timestamp) -> list[tuple[float, str]]:
        """Paga resgates já convertidos (com pagamento hoje) e a taxa de administração acumulada (1º pregão do
        mês). Reduz os passivos e devolve os fluxos de caixa a lançar no ledger."""
        data = pd.Timestamp(data)
        self._aberto_em, self._fluxos_abertura, self._adm_pago_abertura = data, [], 0.0
        if self.pl_liq_prev is None:  # primeiro dia: nada devido
            return []
        if e_primeiro_du_do_mes(data) and self.adm_saldo > 0:
            self._adm_pago_abertura = self.adm_saldo
            self._fluxos_abertura.append((-self.adm_saldo, "ADM"))
            self.adm_saldo = 0.0
        for r in [r for r in self.resg_pend if r["pag"] == data]:
            self._fluxos_abertura.append((-r["gross"], "RESGATE"))
            self._mov[self._id_idx[r["id"]]]["status"] = "PAGO"
            self.resg_pend.remove(r)
        return list(self._fluxos_abertura)

    # ------------------------------------------------------------------ fechamento do dia
    def fechar_dia(self, data: pd.Timestamp, A: float, bench: float, abertura_lancada: bool | None = None) -> ResultadoDia:
        """`A` = PL dos ativos (caixa projetado + mercado). Se a abertura já foi lançada no ledger (orquestrador),
        `A` já está líquido dos pagamentos de abertura; senão, `fechar_dia` faz a abertura e devolve todos os fluxos."""
        regras, p = self.r, self.p
        data = pd.Timestamp(data)
        if abertura_lancada is None:
            abertura_lancada = self._aberto_em == data
        if not abertura_lancada:
            self.abrir_dia(data)
        abertura = sum(v for v, _ in self._fluxos_abertura)
        ini = self.pl_liq_prev is None
        primeiro_du, ultimo_du = (e_primeiro_du_do_mes(data), e_ultimo_du_do_mes(data))
        A_pos = A if abertura_lancada else A + abertura  # PL dos ativos já líquido dos pagamentos de abertura
        if ini:
            self.cotas = A_pos / p.cota_inicial
            self._novo_lote(p.cotista_capital_inicial, data, self.cotas, p.cota_inicial, A_pos)
            self.la_base, self.idx_base = p.cota_inicial, bench
            adm_dia, pnl_ativos = 0.0, 0.0
        else:
            adm_dia = self.pl_liq_prev * ((1 + regras.taxa_adm) ** (1 / 252) - 1)
            self.adm_saldo += adm_dia
            pnl_ativos = A_pos - self.A_prev - abertura  # a abertura (pagamentos) não é P&L
        resg_antes = sum(r["gross"] for r in self.resg_pend)
        pl_pre = A_pos - self.adm_saldo - resg_antes
        q_bruta = pl_pre / self.cotas
        linha = self.la_base * bench / self.idx_base
        perf_cota = regras.taxa_perf * max(0.0, q_bruta - linha)
        cota = q_bruta - perf_cota
        perf_novo = perf_cota * self.cotas
        despesa_perf = perf_novo - self.perf_saldo
        fluxos: list[tuple[float, str]] = []
        tributos: list[dict] = []
        aplic = res_liq = ir_resg = cc_total = 0.0

        for k in self._por_conv.get(data, []):
            m = self._mov[k]
            if m["status"] != "AGENDADO" or m["data_solicitacao"] > data:
                continue
            if m["tipo"] == "APLICACAO":
                novas = m["valor_solicitado"] / cota
                self._novo_lote(m["id_cotista"], data, novas, cota, m["valor_solicitado"])
                self.cotas += novas
                fluxos.append((m["valor_solicitado"], "APLICACAO"))
                aplic += m["valor_solicitado"]
                self._marcar(k, cota=cota, cotas=novas, bruto=m["valor_solicitado"], ir=0.0, iof=0.0)
            else:
                posse = self.cotas_do_cotista(m["id_cotista"])
                a_resgatar = min(m["valor_solicitado"] / cota, posse)
                if a_resgatar <= 1e-12:
                    m["status"] = "SEM_SALDO"
                    continue
                partes, ir, iof = resgatar_fifo(self.lotes[m["id_cotista"]], a_resgatar, cota, data, regras.ir,
                                                p.aliquota_fia, p.iof_ligado)
                bruto = a_resgatar * cota
                self.cotas -= a_resgatar
                self.resg_pend.append({"id": m["id_mov"], "gross": bruto, "pag": m["data_pagamento"]})
                res_liq += bruto - ir - iof
                ir_resg += ir + iof
                self._marcar(k, cota=cota, cotas=-a_resgatar, bruto=bruto, ir=ir, iof=iof)
                for pt in partes:
                    for tipo, v in (("IR_RESGATE", pt["ir"]), ("IOF_RESGATE", pt["iof"])):
                        if v:
                            tributos.append({"data": data, "fundo_cnpj": self.fundo.cnpj, "cotista": pt["cotista"],
                                             "lote_id": pt["lote_id"], "tipo": tipo, "base": pt["base"],
                                             "aliquota": pt["aliquota"], "valor": v, "cotas_canceladas": pt["cotas"],
                                             "id_mov": m["id_mov"]})

        if regras.come_cotas and ultimo_du and data.month in p.meses_come_cotas:
            for cotista, lotes in self.lotes.items():
                for lote in lotes:
                    imposto, canc = come_cotas(lote, cota, p.aliquota_come_cotas)
                    if imposto:
                        cc_total += imposto
                        self.cotas -= canc
                        tributos.append({"data": data, "fundo_cnpj": self.fundo.cnpj, "cotista": cotista,
                                         "lote_id": lote.id, "tipo": "COME_COTAS",
                                         "base": imposto / p.aliquota_come_cotas, "aliquota": p.aliquota_come_cotas,
                                         "valor": imposto, "cotas_canceladas": canc, "id_mov": ""})
            if cc_total:
                fluxos.append((-cc_total, "COME_COTAS"))

        perf_pago = 0.0
        if not ini and ultimo_du and data.month in p.meses_performance and perf_novo > 0:
            perf_pago = perf_novo
            fluxos.append((-perf_pago, "PERFORMANCE"))
            self.la_base, self.idx_base = cota, bench  # high-water mark: nova linha d'água
        self.perf_saldo = perf_novo - perf_pago

        for r in [r for r in self.resg_pend if r["pag"] == data]:  # convertidos hoje e pagos hoje (RF)
            fluxos.append((-r["gross"], "RESGATE"))
            self._mov[self._id_idx[r["id"]]]["status"] = "PAGO"
            self.resg_pend.remove(r)

        A_fecha = A_pos + sum(v for v, _ in fluxos)
        resg_pagar = sum(r["gross"] for r in self.resg_pend)
        pl_liq = A_fecha - self.adm_saldo - resg_pagar - self.perf_saldo
        variacao = 0.0 if ini else pl_liq - self.pl_liq_prev
        taxas = adm_dia + despesa_perf
        esperado = pnl_ativos + aplic - res_liq - ir_resg - taxas - cc_total
        self.pl_liq_prev, self.A_prev, self.ultima_cota = pl_liq, A_fecha, cota
        cota_row = {
            "data": data, "fundo_cnpj": self.fundo.cnpj, "fundo": self.fundo.nome,
            "pl_ativos": A_fecha, "adm_a_pagar": self.adm_saldo, "perf_provisao": self.perf_saldo,
            "resgates_a_pagar": resg_pagar, "pl_liquido": pl_liq, "cotas": self.cotas,
            "cota_bruta": q_bruta, "cota_liquida": cota, "benchmark": bench, "linha_dagua": linha,
            "aplicacoes": aplic, "resgates_liquidos": res_liq, "tributos_retidos": ir_resg + cc_total,
            "ir_resgate": ir_resg, "come_cotas": cc_total, "taxa_adm": adm_dia, "taxa_perf": despesa_perf,
            "taxas": taxas, "pnl_ativos": pnl_ativos, "variacao_pl_liquido": variacao,
            "erro_fechamento_pl": 0.0 if ini else variacao - esperado,
        }
        prov = {"data": data, "fundo_cnpj": self.fundo.cnpj, "adm_dia": adm_dia, "adm_pago": self._adm_pago_abertura,
                "adm_a_pagar": self.adm_saldo, "perf_dia": despesa_perf, "perf_pago": perf_pago,
                "perf_saldo": self.perf_saldo, "resgates_a_pagar": resg_pagar}
        self.cota_rows.append(cota_row)
        self.prov_rows.append(prov)
        self.tributos += tributos
        devolvidos = fluxos if abertura_lancada else list(self._fluxos_abertura) + fluxos
        self._aberto_em = None
        return ResultadoDia(devolvidos, cota_row, prov, tributos)

    # ------------------------------------------------------------------ internos
    def _novo_lote(self, cotista: str, data: pd.Timestamp, cotas: float, cota: float, capital: float) -> None:
        self._n_lote += 1
        self.lotes.setdefault(cotista, []).append(
            Lote(f"{self.fundo.cnpj[7:10]}-L{self._n_lote:05d}", cotista, data, cotas, cota, capital, cota))

    def _marcar(self, k: int, cota: float, cotas: float, bruto: float, ir: float, iof: float) -> None:
        self._mov[k].update(cota_conversao=cota, cotas=cotas, valor_bruto=bruto, ir=ir, iof=iof,
                            valor_liquido=bruto - ir - iof, status="CONVERTIDO")

    def lotes_df(self, data: pd.Timestamp) -> pd.DataFrame:
        linhas = [(self.fundo.cnpj, l.id, l.cotista, l.data, l.cotas, l.cota_aplicacao, l.capital, l.cc_pago,
                   l.base_cc, l.cotas * self.ultima_cota)
                  for ls in self.lotes.values() for l in ls]
        return pd.DataFrame(linhas, columns=["fundo_cnpj", "lote_id", "cotista", "data_aplicacao", "cotas",
                                             "cota_aplicacao", "capital", "come_cotas_pago", "cota_base_cc",
                                             "valor_atual"]).assign(data_posicao=data)
