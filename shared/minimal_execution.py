"""
shared/minimal_execution.py — executor MINIMO, para o agente Ash.

Existe para responder a uma critica metodologica: o Green nao e um Q-Learning "puro".
Ele nao usa a policy (ranking e poda de acoes abstratas), mas USA todo o executor do
instinto — estimativa de dano, poda de golpes inuteis, prioridade, recuo, previsao de
troca, escolha pontuada de switches e de lead. Sao ~1.079 linhas de conhecimento de
dominio escrito a mao, cerca de 66% do total.

O Ash mantem EXATAMENTE a mesma percepcao (estado de 15 dimensoes) e o mesmo espaco de
37 acoes abstratas — sem isso a comparacao com o Blue e o Green deixaria de ser direta.
O que muda e so a TRADUCAO da intencao em jogada concreta.

Criterio adotado: **percepcao partilhada, decisao diferenciada.**

O que este executor FAZ (minimo indispensavel para a abstracao funcionar):
  - traduz uma categoria abstrata num golpe concreto dessa categoria
  - para ataques, escolhe por poder base x precisao (aritmetica simples, nao
    estimativa de dano com tipos, stats, clima e item)
  - respeita IMUNIDADES POR TIPO (regra do jogo: um golpe imune nao e uma escolha
    valida, e nao saber isso seria ruido puro e nao ausencia de estrategia)
  - mantem a distincao entre troca DEFENSIVA e OFENSIVA por efetividade de tipos, para
    que as duas acoes abstratas continuem a significar coisas diferentes

O que este executor NAO FAZ (e o instinto faz):
  - estimativa de dano com stats, boosts, clima, item, STAB
  - poda de golpes inuteis alem da imunidade (Protect consecutivo, hazards repetidos,
    status ja aplicado, cura com HP cheio, etc.)
  - logica de prioridade, recuo, auto-debuff, previsao de troca do adversario
  - pontuacao de switches por ameaca e sobrevivencia
  - escolha de lead (o Ash escolhe a ordem ao acaso)
"""

import random


class ExecutorMinimo:
    """Traducao ingenua de intencao abstrata em jogada concreta."""

    def __init__(self, physics):
        self.physics = physics

    # ------------------------------------------------------------------
    # helpers minimos
    # ------------------------------------------------------------------

    @staticmethod
    def _precisao(move):
        """Precisao em fracao. None/True significam 'nunca falha'."""
        a = getattr(move, "accuracy", None)
        if a is None or a is True:
            return 1.0
        try:
            a = float(a)
        except (TypeError, ValueError):
            return 1.0
        return a / 100.0 if a > 1.0 else a

    @staticmethod
    def _multiplicador(move, alvo):
        """Efetividade de tipo do golpe contra o alvo. 0.0 = imune."""
        try:
            return float(alvo.damage_multiplier(move))
        except Exception:
            return 1.0

    def _nao_imune(self, move, alvo):
        """Regra do jogo, nao estrategia: um golpe imune nao e opcao valida."""
        if getattr(move, "base_power", 0) <= 0:
            return True                      # golpes de status nao tem imunidade de dano
        return self._multiplicador(move, alvo) > 0.0

    def _golpes_da_categoria(self, battle, categoria):
        """Golpes disponiveis que pertencem a categoria abstrata pedida."""
        alvo = battle.opponent_active_pokemon
        saida = []
        for m in battle.available_moves:
            try:
                if self.physics.classify_move(m).name != categoria:
                    continue
            except Exception:
                continue
            if alvo is not None and not self._nao_imune(m, alvo):
                continue
            saida.append(m)
        return saida

    # ------------------------------------------------------------------
    # lead e trocas
    # ------------------------------------------------------------------

    def get_best_lead(self, battle):
        """Ordem de equipa ALEATORIA (o instinto pontua por matchup)."""
        n = len(getattr(battle, "team", {}) or {}) or 6
        ordem = list(range(1, n + 1))
        random.shuffle(ordem)
        return "/team " + "".join(str(i) for i in ordem)

    def get_post_faint_switch(self, battle, history=None):
        """Troca pos-faint ALEATORIA (o instinto pontua sobrevivencia e ameaca)."""
        switches = list(getattr(battle, "available_switches", []) or [])
        return random.choice(switches) if switches else None

    def _troca_por_tipo(self, battle, defensiva):
        """Troca decidida SO por efetividade de tipos.

        Mantida para que SWITCH_DEFENSIVE e SWITCH_OFFENSIVE continuem a significar
        coisas diferentes (senao as duas acoes abstratas colapsavam numa so). Usa
        apenas a tabela de tipos, nao estimativa de dano nem analise de ameaca.
        """
        switches = list(getattr(battle, "available_switches", []) or [])
        if not switches:
            return None
        opp = battle.opponent_active_pokemon
        if opp is None:
            return random.choice(switches)

        melhor, melhor_score = None, None
        for cand in switches:
            score = 0.0
            try:
                if defensiva:
                    # Quanto MENOS dano os tipos do oponente fazem a este candidato.
                    for t in (opp.types or []):
                        if t is None:
                            continue
                        score -= cand.damage_multiplier(t)
                else:
                    # Quanto MAIS os tipos do candidato ameacam o oponente.
                    for t in (cand.types or []):
                        if t is None:
                            continue
                        score += opp.damage_multiplier(t)
            except Exception:
                score = 0.0
            if melhor_score is None or score > melhor_score:
                melhor, melhor_score = cand, score
        return melhor or random.choice(switches)

    # ------------------------------------------------------------------
    # traducao principal
    # ------------------------------------------------------------------

    def get_best_execution_object(self, base_action, battle, history=None):
        """Intencao abstrata -> jogada concreta, sem conhecimento tatico."""
        categoria = str(base_action).replace("_MEC", "")

        if categoria == "SWITCH_DEFENSIVE":
            return self._troca_por_tipo(battle, defensiva=True)
        if categoria == "SWITCH_OFFENSIVE":
            return self._troca_por_tipo(battle, defensiva=False)

        candidatos = self._golpes_da_categoria(battle, categoria)
        if not candidatos:
            # Categoria indisponivel: qualquer golpe legal serve (o cerebro pediu algo
            # que nao existe neste turno).
            legais = [m for m in battle.available_moves
                      if battle.opponent_active_pokemon is None
                      or self._nao_imune(m, battle.opponent_active_pokemon)]
            if legais:
                return random.choice(legais)
            switches = list(getattr(battle, "available_switches", []) or [])
            return random.choice(switches) if switches else None

        if categoria.startswith("ATTACK"):
            # Aritmetica simples: poder base x precisao. NAO e estimativa de dano
            # (que usaria stats, tipos, STAB, boosts, clima e item).
            return max(candidatos,
                       key=lambda m: getattr(m, "base_power", 0) * self._precisao(m))

        # Categorias de status/suporte: escolha ao acaso dentro da categoria.
        return random.choice(candidatos)
