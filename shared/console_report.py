"""
shared/console_report.py — relatorio de progresso do treino no terminal.

Mostra os resultados parciais de forma agrupada e legivel, para se identificar de
relance o que e progresso, o que e aprendizado, o que e qualidade de jogo e o que e
custo. As colunas estao separadas por " | " em quatro grupos:

  PROGRESSO   : batalhas concluidas e tempo decorrido
  APRENDIZADO : win rate, variacao, estados na Q-table, epsilon, visitas por estado
  QUALIDADE   : margem de vitoria, duracao da batalha, auto-ties
  CUSTO       : latencia de decisao por turno

Nao mexe em logging nem suprime nada: apenas formata. So usa ASCII, para nao dar
UnicodeEncodeError no terminal do Windows (cp1252).

Uso:
    rel = RelatorioConsola(agente="BLUE", descricao="Hibrido (Q-Learning + Instinto)")
    rel.cabecalho(config={...}, caminhos={...})
    rel.bloco(batalhas=1000, metricas={...})     # uma vez por bloco
    rel.resumo_final(extra={...})
"""

LARGURA = 116


def _num(valor, default=0):
    """Converte para numero de forma tolerante (metrica ausente nao rebenta nada)."""
    try:
        if valor is None:
            return default
        return float(valor)
    except (TypeError, ValueError):
        return default


def _sparkline(valores):
    """Mini-grafico ASCII da evolucao de uma serie: '_' baixo -> '#' alto."""
    if not valores:
        return ""
    lo, hi = min(valores), max(valores)
    if hi - lo < 1e-9:
        return "-" * len(valores)
    niveis = "_-=#"
    return "".join(niveis[int((v - lo) / (hi - lo) * (len(niveis) - 1))] for v in valores)


class RelatorioConsola:
    """Acumula e imprime os resultados parciais de um treino, agrupados por tema."""

    # Grupos de colunas: (titulo do grupo, [(chave, titulo, largura, formato), ...])
    GRUPOS = [
        ("PROGRESSO", [
            ("batalhas", "Batalhas", 9, "{:>9,.0f}"),
            ("tempo_s", "Tempo", 6, "{:>5.0f}s"),
        ]),
        ("APRENDIZADO", [
            ("win_rate", "WR%", 6, "{:>6.1f}"),
            # VarWR = variacao do Win Rate face ao bloco anterior, em pontos
            # percentuais. Positivo = melhorou, negativo = piorou.
            ("delta_wr", "VarWR", 7, "{:>+7.1f}"),
            ("estados", "Estados", 9, "{:>9,.0f}"),
            ("epsilon", "Eps", 6, "{:>6.3f}"),
            ("visitas", "Vis", 5, "{:>5.1f}"),
            # Conf% = percentagem de estados com visitas suficientes para o valor
            # aprendido ser fiavel (o "quanto do que sabe e solido").
            ("confianca", "Conf%", 6, "{:>6.1f}"),
            # Reward = recompensa media POR BATALHA no bloco. Deve subir com o
            # aprendizado; e o sinal mais direto de que o agente esta a melhorar.
            ("reward", "Reward", 9, "{:>9,.0f}"),
        ]),
        ("QUALIDADE", [
            ("margem_media", "Marg", 5, "{:>5.1f}"),
            ("duracao_media", "Dur", 5, "{:>4.0f}t"),
            ("auto_ties", "Ties", 5, "{:>5.0f}"),
        ]),
        ("CUSTO", [
            # Tempo por JOGADA: quanto o agente demora a decidir um turno. E a metrica
            # de viabilidade em producao (um jogo real nao tolera decisoes lentas).
            ("latencia_ms", "ms/Jogada", 9, "{:>9.2f}"),
            # Tempo por BATALHA: custo de relogio real por batalha neste bloco
            # (inclui o servidor e a concorrencia, nao apenas a decisao).
            ("seg_por_batalha", "s/Batalha", 9, "{:>9.2f}"),
        ]),
    ]

    REPETIR_CABECALHO_A_CADA = 12  # blocos

    def __init__(self, agente, descricao=""):
        self.agente = agente
        self.descricao = descricao
        self._historico = []
        self._wr = []
        self._blocos = 0
        self._wr_anterior = None

    # ------------------------------------------------------------------
    # cabecalho
    # ------------------------------------------------------------------

    def cabecalho(self, config=None, caminhos=None):
        print()
        print("=" * LARGURA)
        titulo = f"  TREINO {self.agente}"
        if self.descricao:
            titulo += f"  |  {self.descricao}"
        print(titulo)
        print("=" * LARGURA)
        for rotulo, valor in (config or {}).items():
            print(f"  {rotulo:<20}: {valor}")
        if caminhos:
            print("  " + "-" * (LARGURA - 4))
            for rotulo, valor in (caminhos or {}).items():
                print(f"  {rotulo:<20}: {valor}")
        print("=" * LARGURA)
        self._cabecalho_tabela()

    def _larguras_grupo(self):
        """Largura ocupada por cada grupo (soma das colunas + espacos internos)."""
        larguras = []
        for _nome, cols in self.GRUPOS:
            larg = sum(w for _, _, w, _ in cols) + (len(cols) - 1)
            larguras.append(larg)
        return larguras

    def _cabecalho_tabela(self):
        larguras = self._larguras_grupo()
        # Linha 1: nome dos grupos, centrado sobre as respetivas colunas.
        faixa = " | ".join(nome.center(larg)
                           for (nome, _), larg in zip(self.GRUPOS, larguras))
        # Linha 2: titulos das colunas.
        titulos = " | ".join(
            " ".join(f"{t:>{w}}" for _, t, w, _ in cols) for _nome, cols in self.GRUPOS)
        print()
        print(faixa)
        print("-" * len(titulos))
        print(titulos)
        print("-" * len(titulos))
        if self._blocos == 0:
            print("  (VarWR = variacao do WR vs bloco anterior, em pontos percentuais | "
                  "Conf% = estados com dados fiaveis)")

    # ------------------------------------------------------------------
    # bloco de progresso
    # ------------------------------------------------------------------

    def bloco(self, batalhas, metricas):
        """Imprime uma linha de progresso. Chaves ausentes em `metricas` aparecem
        como 0 em vez de rebentar."""
        wr = _num(metricas.get("win_rate"))
        delta = 0.0 if self._wr_anterior is None else (wr - self._wr_anterior)
        self._wr_anterior = wr
        self._wr.append(wr)
        self._blocos += 1

        dados = dict(metricas)
        dados["batalhas"] = batalhas
        dados["delta_wr"] = delta

        # Tempo por batalha DESTE bloco: (tempo agora - tempo antes) / batalhas do
        # bloco. Derivado aqui para o script nao ter de o calcular.
        tempo_agora = _num(metricas.get("tempo_s"))
        if self._historico:
            tempo_antes = _num(self._historico[-1].get("tempo_s"))
            batalhas_antes = _num(self._historico[-1].get("batalhas"))
        else:
            tempo_antes, batalhas_antes = 0.0, 0.0
        n_bloco = batalhas - batalhas_antes
        dados["seg_por_batalha"] = ((tempo_agora - tempo_antes) / n_bloco) if n_bloco > 0 else 0.0

        self._historico.append(dados)

        if self._blocos > 1 and (self._blocos - 1) % self.REPETIR_CABECALHO_A_CADA == 0:
            self._cabecalho_tabela()

        partes = []
        for _nome, cols in self.GRUPOS:
            celulas = []
            for chave, _t, largura, fmt in cols:
                try:
                    celulas.append(fmt.format(_num(dados.get(chave))))
                except (ValueError, TypeError):
                    celulas.append(" " * largura)
            partes.append(" ".join(celulas))
        linha = " | ".join(partes)

        # Sinais a direita, fora da tabela, para nao desalinhar as colunas.
        sinais = []
        if _num(dados.get("auto_ties")) > 0:
            sinais.append(f"ties={_num(dados.get('auto_ties')):.0f}")
        if _num(dados.get("duracao_media")) > 100:
            sinais.append("batalhas longas")
        if sinais:
            linha += "   << " + ", ".join(sinais)

        print(linha, flush=True)

    # ------------------------------------------------------------------
    # resumo final
    # ------------------------------------------------------------------

    def resumo_final(self, extra=None):
        print()
        print("=" * LARGURA)
        print(f"  RESUMO DO TREINO  |  {self.agente}")
        print("=" * LARGURA)

        if not self._historico:
            print("  (nenhum bloco concluido)")
            print("=" * LARGURA)
            return

        primeiro, ultimo = self._historico[0], self._historico[-1]
        wr_ini, wr_fim = _num(primeiro.get("win_rate")), _num(ultimo.get("win_rate"))
        wr_max, wr_min = max(self._wr), min(self._wr)
        ties = sum(_num(h.get("auto_ties")) for h in self._historico)

        def sec(titulo):
            print("  " + "-" * (LARGURA - 4))
            print(f"  {titulo}")
            print("  " + "-" * (LARGURA - 4))

        def linha(rotulo, valor):
            print(f"    {rotulo:<28}: {valor}")

        sec("PROGRESSO")
        linha("Blocos concluidos", f"{len(self._historico)}")
        linha("Batalhas totais", f"{_num(ultimo.get('batalhas')):,.0f}")
        linha("Tempo total", f"{_num(ultimo.get('tempo_s')):,.0f} s")

        sec("APRENDIZADO")
        linha("Win Rate inicial -> final", f"{wr_ini:.1f}% -> {wr_fim:.1f}%   "
                                           f"({wr_fim - wr_ini:+.1f} pp)")
        linha("Win Rate min / max", f"{wr_min:.1f}% / {wr_max:.1f}%")
        linha("Estados na Q-table", f"{_num(ultimo.get('estados')):,.0f}")
        linha("Epsilon final", f"{_num(ultimo.get('epsilon')):.3f}")
        linha("Visitas medias por estado", f"{_num(ultimo.get('visitas')):.2f}")
        linha("Confianca (estados fiaveis)", f"{_num(ultimo.get('confianca')):.1f}%")
        r_ini = _num(primeiro.get("reward"))
        r_fim = _num(ultimo.get("reward"))
        linha("Reward/batalha inicio -> fim", f"{r_ini:,.0f} -> {r_fim:,.0f}   "
                                             f"({r_fim - r_ini:+,.0f})")
        print(f"    {'Evolucao do Win Rate':<28}: {_sparkline(self._wr)}")
        print(f"    {'':<28}  ('_' = {wr_min:.1f}%   '#' = {wr_max:.1f}%)")

        sec("QUALIDADE DE JOGO")
        linha("Margem media de vitoria", f"{_num(ultimo.get('margem_media')):.2f} pokemon")
        linha("Duracao media da batalha", f"{_num(ultimo.get('duracao_media')):.0f} turnos")
        linha("Auto-ties no treino", f"{ties:.0f}")

        sec("CUSTO")
        linha("Tempo por jogada (decisao)", f"{_num(ultimo.get('latencia_ms')):.2f} ms")
        linha("Tempo por batalha", f"{_num(ultimo.get('seg_por_batalha')):.2f} s")
        n_tot = _num(ultimo.get("batalhas"))
        if n_tot > 0:
            linha("Tempo medio por batalha (total)",
                  f"{_num(ultimo.get('tempo_s')) / n_tot:.2f} s")

        avisos = []
        if wr_fim < wr_ini - 2:
            avisos.append("Win Rate mais baixo no fim do que no inicio. Com epsilon alto "
                          "isto e sobretudo ruido de exploracao, nao necessariamente "
                          "perda de aprendizado.")
        if ties > 0:
            avisos.append(f"{ties:.0f} batalha(s) sem vencedor (auto-tie): nao geram "
                          "recompensa terminal, logo nao ensinam o desfecho.")
        if _num(ultimo.get("duracao_media")) > 100:
            avisos.append("Duracao media acima de 100 turnos: batalhas a arrastar-se.")
        if _num(ultimo.get("visitas")) < 2:
            avisos.append("Menos de 2 visitas por estado: Q-table ainda muito esparsa.")
        if avisos:
            sec("AVISOS")
            for a in avisos:
                print(f"    - {a}")

        if extra:
            sec("FICHEIROS E RESULTADO")
            for rotulo, valor in extra.items():
                linha(rotulo, valor)

        print("=" * LARGURA)

    @property
    def historico(self):
        return list(self._historico)
