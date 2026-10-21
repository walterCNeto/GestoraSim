"""Convenções operacionais compartilhadas entre pacotes."""

from gestorasim.dominio.modelos import ClasseAtivo

# Prazo de liquidação financeira (dias úteis após a operação).
PRAZO_LIQUIDACAO_DU = {
    ClasseAtivo.ACAO: 2,
    ClasseAtivo.TITULO_PUBLICO: 0,
    ClasseAtivo.DEBENTURE: 0,  # tratada como título de renda fixa (D+0)
    ClasseAtivo.FUTURO_DI: 1,  # ajuste diário liquidado em D+1
}

# Fração do volume diário que se consegue negociar sem mover o mercado.
PARTICIPACAO_LIQUIDEZ = 0.20
