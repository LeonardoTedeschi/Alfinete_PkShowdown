"""
shared/log_filters.py — silenciar avisos conhecidos e inofensivos do poke-env.

PORQUE EXISTE
-------------
O poke-env emite um WARNING por cada efeito de protocolo que nao reconhece:

    Unexpected effect 'HEALREPLACEMENT' received. Effect.UNKNOWN will be used
    instead.

`HEALREPLACEMENT` e o efeito de Healing Wish e Lunar Dance. O poke-env mapeia-o
para `Effect.UNKNOWN` e continua: o servidor aplica o efeito na mesma e o resultado
(o substituto entra com vida cheia) chega pelas mensagens de HP, logo o
`current_hp_fraction` fica correto e o estado que o agente le esta certo. Nenhuma
camada do instinto consulta `Effect` por nome.

Ou seja, e ruido, nao e dano. Mas numa corrida de 30.000 batalhas enche o terminal
e esconde mensagens que importam.

PORQUE NAO SE BAIXA O NIVEL GLOBAL
-----------------------------------
Por-se `logging.getLogger("poke-env").setLevel(logging.ERROR)` calaria tambem os
avisos de rejeicao de equipa, que foram exatamente o que permitiu diagnosticar o
travamento do `avaliar_generalizacao` a 26/08/2026. Filtra-se por MENSAGEM, nao por
nivel: tudo o resto continua visivel.

NOTA DE VERSAO
--------------
A partir de poke-env 0.14.0 ("Ignore some rare messages", PR #881) este aviso
deixa de ser emitido. O filtro fica na mesma: e inofensivo se a mensagem nunca
aparecer, e protege quem correr o projeto com uma versao antiga.

USO
---
No topo de qualquer script, depois dos imports:

    from shared.log_filters import silenciar_avisos_conhecidos
    silenciar_avisos_conhecidos()
"""

import logging

# Fragmentos de mensagem a filtrar. Acrescentar SO depois de confirmar que o efeito
# nao altera a leitura do estado nem a decisao do agente.
MENSAGENS_IGNORADAS = (
    "HEALREPLACEMENT",   # Healing Wish / Lunar Dance
)


class FiltroAvisosConhecidos(logging.Filter):
    def filter(self, record):
        try:
            msg = record.getMessage()
        except Exception:
            return True
        return not any(frag in msg for frag in MENSAGENS_IGNORADAS)


_aplicado = False


def silenciar_avisos_conhecidos():
    """Idempotente: pode ser chamada em varios scripts sem duplicar o filtro."""
    global _aplicado
    if _aplicado:
        return
    logging.getLogger("poke-env").addFilter(FiltroAvisosConhecidos())
    _aplicado = True
