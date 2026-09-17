"""
shared/diagnostico.py — interruptor unico da instrumentacao temporaria.

PORQUE EXISTE
-------------
O projeto tem duas necessidades opostas e ate aqui resolvia-as mal:

  TREINO           400.000 batalhas. Um `print` por turno torna o log ilegivel e
                   ja produziu um diagnostico errado, porque os avisos do
                   InstinctBot (que corre como ADVERSARIO no mesmo processo e no
                   mesmo stdout) pareciam vir do agente.

  BATALHA MANUAL   e a UNICA ferramenta que apanha defeitos que nenhuma metrica
                   mostra (6.35, 6.44). Aqui o `print` nao e ruido, e o produto.

Em 03/09 os avisos passaram a estar desligados por omissao para limpar o treino, e
isso apagou-os TAMBEM nas batalhas manuais — exatamente onde eram precisos. Este
modulo separa as duas coisas com um interruptor so, em vez de cada ficheiro ler a
variavel de ambiente a sua maneira.

COMO LIGAR
----------
    $env:ALFINETE_DIAGNOSTICO = "1"      # PowerShell
    python -m scripts.jogar_contra_instinto

O `jogar_contra_instinto.py` liga-o SOZINHO, porque uma batalha manual sem
instrumentacao nao serve para o fim a que se destina.

Para ligar dentro do codigo (testes, medicoes pontuais):

    from shared import diagnostico
    diagnostico.ligar()

REGRA DE USO
------------
Isto e para instrumentacao TEMPORARIA de diagnostico, nao para logging permanente.
Cada tag deve ter um dono e uma pergunta a responder; quando a pergunta estiver
respondida, a chamada sai do codigo.

TAGS EM USO (03/09/2026)
------------------------
    [HAZ]      `masking.is_hazard_already_set` — porque e que o Stealth Rock
               repetido nao e podado. Ver 6.35 bug 3 e 6.45.
    [QUAR]     `execution._registar_saida` e `_penalizacao_quarentena` — porque e
               que a quarentena nao segura o carrossel de trocas. Ver 6.44.
    [DECISAO]  `instinct_player.choose_move` — que INTENCAO produziu o objeto que
               foi mesmo jogado. Responde as duas perguntas acima de uma vez, se
               o caminho furado nao for nenhum dos dois instrumentados.
    [IMUNE]    `instinct_player._avisar_imune` — golpe de multiplicador zero
               descartado.
"""

import os

_LIGADO = None


def _ler_ambiente():
    valor = os.environ.get("ALFINETE_DIAGNOSTICO", "")
    return str(valor).strip().lower() not in ("", "0", "false", "no")


def ligado():
    """O modo diagnostico esta activo?

    Le a variavel de ambiente UMA vez e guarda o resultado: a alternativa era um
    `os.environ` por turno em caminhos que correm 14 milhoes de vezes num ciclo.
    """
    global _LIGADO
    if _LIGADO is None:
        _LIGADO = _ler_ambiente()
    return _LIGADO


def ligar():
    """Liga o modo diagnostico em tempo de execucao (ignora o ambiente)."""
    global _LIGADO
    _LIGADO = True


def desligar():
    global _LIGADO
    _LIGADO = False


def log(tag, mensagem):
    """Imprime `[TAG] mensagem` se o modo diagnostico estiver ligado.

    NUNCA levanta. Uma instrumentacao que rebenta a batalha que estava a observar
    seria pior que nao existir.
    """
    if not ligado():
        return
    try:
        print(f"[{tag}] {mensagem}")
    except Exception:
        pass
