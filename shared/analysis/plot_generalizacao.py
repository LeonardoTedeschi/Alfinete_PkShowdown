"""Diagnosticos e graficos da avaliacao externa contra Cynthia.

03/10/2026: substitui o leitor do historico legado contra InstinctBot.
Nao aplica o antigo vies de 2,35 pp nem interpreta 50% como acaso.
Writer: DiagnosticoDecisoes observa brain.decide_action e o retorno do executor.
Passagem: avaliar_condicao -> registar -> salvar_diagnosticos.
Reader: gerar_grafico le o CSV principal e seus CSVs acompanhantes.

Uso avulso (na raiz do projeto):
    python -m shared.analysis.plot_generalizacao
    python -m shared.analysis.plot_generalizacao --csv CAMINHO_DO_CSV

Todos os arquivos acompanham o CSV da tentativa em
artefatos/logs/Generalizacao/<Agente>VsCynthia/<Ciclo>/.
CSV antigo gera apenas desempenho; cobertura ausente nunca vira zero.
Instrumentacao observacional: nao congela escritas internas do brain, nao altera
decisoes e nao salva checkpoints. A referencia de visitas e a carga inicial.
"""

import argparse
import csv
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GENERALIZACAO_DIR = ROOT / "artefatos" / "logs" / "Generalizacao"
FAIXAS = ("Ausente", "0 visitas", "1 visita", "2 a 4", "5 a 19", "20+ visitas")


def _faixa(conhecido, visitas):
    if not conhecido:
        return FAIXAS[0]
    if visitas == 0:
        return FAIXAS[1]
    if visitas == 1:
        return FAIXAS[2]
    if visitas < 5:
        return FAIXAS[3]
    if visitas < 20:
        return FAIXAS[4]
    return FAIXAS[5]


def _acao(resultado):
    base, mecanica = resultado
    return str(base) + ("_MEC" if mecanica == "ACTIVATE" else "")


class DiagnosticoDecisoes:
    """Contagens exclusivas desta condicao; nunca reutiliza N(s,a) do treino.

Referencia: chaves e visitas existentes ANTES da primeira batalha. Copia apenas
esse indice, nao duplica os vetores Q. As metricas Q descrevem a entrada de cada
chamada, pois a politica original ainda pode fazer heranca _MEC em memoria.
"""

    def __init__(self, jogador):
        self.brain = getattr(jogador, "brain", None)
        self.estados = {}
        self.execucoes = Counter()
        self.erros_decisao = 0
        self.erros_executor = 0
        self.chamadas_executor = 0
        self.chamadas_choose_move = 0
        self.erros_choose_move = 0
        self.referencia = None
        self._restaurar = []
        if self.brain is None:
            return
        b = self.brain
        self.referencia = {s: int(b.visit_counts.get(s, 0)) for s in b.q_table}
        self.limiar = int(getattr(b, "LIMIAR_MADURO", 20))
        self.indices = {a: i for i, a in enumerate(b.actions)}
        original = b.decide_action

        def observar(state, valid_actions, ranking_list):
            abs_state = b._get_abstract_state(state)
            visitas = int(b.visit_counts.get(abs_state, 0))
            q = b.q_table.get(abs_state)
            antes = tuple(float(x) for x in q) if q is not None else None
            indices = [self.indices[a] for a in valid_actions]
            zeros_entrada = antes is None or all(antes[i] == 0 for i in indices)
            try:
                resultado = original(state, valid_actions, ranking_list)
            except Exception:
                self.erros_decisao += 1
                raise
            # O brain decide o cold start APOS a heranca _MEC. Observar o vetor
            # resultante reproduz esse criterio sem repetir sorteios nem a policy.
            depois = tuple(float(x) for x in b.q_table[abs_state])
            fallback = visitas == 0 or all(depois[i] == 0 for i in indices)
            r = self.estados.get(abs_state)
            if r is None:
                conhecido = abs_state in self.referencia
                v0 = self.referencia.get(abs_state, 0)
                r = dict(conhecido=conhecido, visitas=v0,
                         faixa=_faixa(conhecido, v0), escolhas=Counter(),
                         validas=set(), fallback=0, zeros=0, mudancas=0,
                         criacoes=0, exploratorias=0)
                self.estados[abs_state] = r
            r["escolhas"][_acao(resultado)] += 1
            r["validas"].update(valid_actions)
            r["fallback"] += int(fallback)
            r["zeros"] += int(zeros_entrada)
            r["criacoes"] += int(antes is None)
            r["mudancas"] += int(antes is not None and antes != depois)
            r["exploratorias"] += int(bool(getattr(b, "ultima_foi_exploratoria", False)))
            return resultado

        self._substituir(b, "decide_action", observar)
        executor = getattr(jogador, "executor", None)
        if executor is not None and hasattr(executor, "get_best_execution_object"):
            executar = executor.get_best_execution_object

            def observar_executor(base_action, battle, *args, **kwargs):
                self.chamadas_executor += 1
                try:
                    obj = executar(base_action, battle, *args, **kwargs)
                except Exception:
                    self.erros_executor += 1
                    raise
                # Nao chama classify_move novamente: evita interferencia e falsas
                # equivalencias entre intencoes taticas e categorias de golpes.
                if obj is None:
                    tipo, identificador = "SEM_OBJETO", ""
                elif any(obj is p for p in (getattr(battle, "available_switches", None) or [])):
                    tipo = "TROCA"
                    identificador = str(getattr(obj, "species", "?"))
                elif hasattr(obj, "id"):
                    tipo, identificador = "GOLPE", str(obj.id)
                else:
                    tipo, identificador = "OUTRO", type(obj).__name__
                self.execucoes[(str(base_action), tipo, identificador)] += 1
                return obj

            self._substituir(executor, "get_best_execution_object", observar_executor)

    def _substituir(self, obj, nome, novo):
        proprio = nome in vars(obj)
        anterior = vars(obj).get(nome)
        self._restaurar.append((obj, nome, proprio, anterior))
        setattr(obj, nome, novo)

    def fechar(self):
        for obj, nome, proprio, anterior in reversed(self._restaurar):
            if proprio:
                setattr(obj, nome, anterior)
            else:
                delattr(obj, nome)
        self._restaurar.clear()
        # A classificacao inicial de cada estado observado ja esta nas linhas.
        self.referencia = None
        self.brain = None

    def linhas_estados(self):
        linhas = []
        for state, r in self.estados.items():
            counts = r["escolhas"]
            n = sum(counts.values())
            entropia = -sum((v / n) * math.log(v / n) for v in counts.values())
            dominante = max(counts, key=counts.get)
            linhas.append({
                "Estado": repr(state), "Presente_Checkpoint": int(r["conhecido"]),
                "Visitas_Checkpoint": r["visitas"], "Faixa_Checkpoint": r["faixa"],
                "Decisoes_Medidas": n, "Acoes_Escolhidas": len(counts),
                "Acoes_Validas_Vistas": len(r["validas"]),
                "Cobertura_Acoes_pct": 100 * len(counts) / max(1, len(r["validas"])),
                "Acao_Dominante": dominante, "Dominante_pct": 100 * counts[dominante] / n,
                "Entropia": entropia, "Acoes_Efetivas": math.exp(entropia),
                "Decisoes_Fallback_Inicial": r["fallback"],
                "Decisoes_Q_Validos_Zerados_Entrada": r["zeros"],
                "Decisoes_Exploratorias": r["exploratorias"],
                "Entradas_Criadas_Em_Decide": r["criacoes"],
                "Chamadas_Com_Alteracao_Q_Em_Decide": r["mudancas"],
            })
        return sorted(linhas, key=lambda x: (-x["Decisoes_Medidas"], x["Estado"]))


def _escrever(caminho, campos, linhas):
    with Path(caminho).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(linhas)


def _ler(caminho):
    with Path(caminho).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def salvar_diagnosticos(caminho_csv, treino, holdout):
    """Grava acompanhantes com o mesmo prefixo e diretorio da tentativa."""
    principal = Path(caminho_csv)
    prefixo = principal.with_suffix("")
    resumos = []
    for condicao, resultado in (("Treino", treino), ("Holdout", holdout)):
        d = resultado.get("diagnostico")
        if d is None:
            continue
        linhas = d.linhas_estados()
        n = sum(r["Decisoes_Medidas"] for r in linhas)
        resumo = {"Condicao": condicao, "Status": "MEDIDO" if linhas else
                  ("SEM_DECISOES" if hasattr(d, "limiar") else "NAO_APLICAVEL"),
                  "Limiar_Maduro": getattr(d, "limiar", ""),
                  "Chamadas_Choose_Move": d.chamadas_choose_move,
                  "Erros_Choose_Move": d.erros_choose_move,
                  "Erros_Decisao": d.erros_decisao,
                  "Decisoes_Cerebro": n, "Estados_Observados": len(linhas)}
        for faixa in FAIXAS:
            resumo["Decisoes_" + faixa] = sum(r["Decisoes_Medidas"] for r in linhas
                                              if r["Faixa_Checkpoint"] == faixa)
        maduros = sum(r["Decisoes_Medidas"] for r in linhas
                      if r["Presente_Checkpoint"] and r["Visitas_Checkpoint"] >= d.limiar)
        conhecidos = sum(r["Decisoes_Medidas"] for r in linhas if r["Presente_Checkpoint"])
        for chave, total in (("Conhecidos_pct", conhecidos), ("Maduros_pct", maduros),
                             ("Fallback_Inicial_pct", sum(r["Decisoes_Fallback_Inicial"] for r in linhas))):
            resumo[chave] = 100 * total / n if n else ""
        for campo in ("Cobertura_Acoes_pct", "Dominante_pct", "Acoes_Efetivas"):
            resumo[campo + "_Ponderado"] = sum(r[campo] * r["Decisoes_Medidas"] for r in linhas) / n if n else ""
        for campo in ("Decisoes_Q_Validos_Zerados_Entrada", "Decisoes_Exploratorias",
                      "Entradas_Criadas_Em_Decide", "Chamadas_Com_Alteracao_Q_Em_Decide"):
            resumo[campo] = sum(r[campo] for r in linhas)
        resumo["Chamadas_Executor"] = d.chamadas_executor
        resumo["Erros_Executor"] = d.erros_executor
        resumos.append(resumo)
        # Mesmo sem decisoes, o cabecalho explicita a ausencia de observacoes.
        campos = list(linhas[0]) if linhas else ["Estado", "Decisoes_Medidas"]
        _escrever(f"{prefixo}_{condicao}_acoes_estado.csv", campos, linhas)
        acoes = Counter()
        for r in d.estados.values():
            acoes.update(r["escolhas"])
        _escrever(f"{prefixo}_{condicao}_acoes.csv", ["Acao", "Decisoes", "Percentual"],
                  ({"Acao": a, "Decisoes": v, "Percentual": 100 * v / n}
                   for a, v in acoes.most_common()))
        _escrever(f"{prefixo}_{condicao}_executor.csv",
                  ["Intencao", "Tipo_Objeto", "Objeto", "Chamadas"],
                  ({"Intencao": a, "Tipo_Objeto": t, "Objeto": o, "Chamadas": v}
                   for (a, t, o), v in d.execucoes.most_common()))
    if resumos:
        _escrever(f"{prefixo}_cobertura.csv", list(resumos[0]), resumos)


def gerar_grafico(caminho_csv):
    """Desempenho e, quando disponivel, cobertura e decisoes desta tentativa."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = Path(caminho_csv)
    linhas = _ler(p)
    if not linhas:
        return None
    r = linhas[-1]
    if "Delta_Treino_Menos_Holdout_pp" not in r:
        raise ValueError(f"CSV fora do protocolo Cynthia atual: {p}")
    condicoes = ("Treino", "Holdout")
    cores = ("#3576B8", "#D78336")
    fig, axs = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    wr = [float(r["WR_" + c]) for c in condicoes]
    ns = [int(r.get("Batalhas_" + c) or r["Batalhas_Por_Condicao"]) for c in condicoes]
    erros = [1.96 * math.sqrt((w / 100) * (1 - w / 100) / n) * 100
             for w, n in zip(wr, ns)]
    axs[0, 0].bar(condicoes, wr, color=cores, yerr=erros, capsize=5)
    for i, (w, e, n) in enumerate(zip(wr, erros, ns)):
        axs[0, 0].text(i, min(98, w + e + 2), f"{w:.2f}% | n={n:,}", ha="center")
    delta = float(r["Delta_Treino_Menos_Holdout_pp"])
    ep = float(r["EP_Diferenca_pp"])
    sigma = f"{abs(delta) / ep:.2f} EP" if ep else "EP arredondado a zero"
    axs[0, 0].set(title=f"Vitorias: IC 95% aproximado\nDelta {delta:+.2f} pp | EP {ep:.2f} pp | {sigma}",
                   ylabel="Vitorias (%)", ylim=(0, 105))
    cobertura = p.with_name(p.stem + "_cobertura.csv")
    dados = {x["Condicao"]: x for x in _ler(cobertura)} if cobertura.exists() else {}
    disponivel = all(dados.get(c, {}).get("Status") == "MEDIDO" for c in condicoes)
    if disponivel:
        for i, c in enumerate(condicoes):
            d = dados[c]
            n = int(d["Decisoes_Cerebro"])
            ys = [100 * int(d["Decisoes_" + f]) / n for f in FAIXAS]
            xs = [j + (-0.19 if i == 0 else 0.19) for j in range(len(FAIXAS))]
            axs[0, 1].bar(xs, ys, width=.38, label=c, color=cores[i])
            for x, y in zip(xs, ys):
                axs[0, 1].text(x, y + 1, f"{y:.1f}", ha="center", fontsize=8)
        axs[0, 1].set_xticks(range(len(FAIXAS)), FAIXAS, rotation=18)
        axs[0, 1].set(title="Decisoes por visitas no checkpoint inicial",
                      ylabel="Decisoes do cerebro (%)", ylim=(0, 110))
        axs[0, 1].legend()
        rotulos = ("Conhecidos", "Maduros", "Fallback inicial")
        campos = ("Conhecidos_pct", "Maduros_pct", "Fallback_Inicial_pct")
        for i, c in enumerate(condicoes):
            ys = [float(dados[c][k]) for k in campos]
            xs = [j + (-.19 if i == 0 else .19) for j in range(3)]
            axs[1, 0].bar(xs, ys, width=.38, color=cores[i], label=c)
            for x, y in zip(xs, ys):
                axs[1, 0].text(x, y + 1, f"{y:.1f}%", ha="center", fontsize=9)
        axs[1, 0].set_xticks(range(3), rotulos)
        axs[1, 0].set(title="Uso da tabela e regra inicial\nIndicadores se sobrepoem; nao somar",
                      ylabel="Decisoes do cerebro (%)", ylim=(0, 110))
        axs[1, 0].legend()
        texto = []
        for c in condicoes:
            d = dados[c]
            texto.append(f"{c}: {int(d['Decisoes_Cerebro']):,} decisoes; "
                         f"{int(d['Estados_Observados']):,} estados\n"
                         f"  Cobertura de acoes: {float(d['Cobertura_Acoes_pct_Ponderado']):.2f}%\n"
                         f"  Dominancia: {float(d['Dominante_pct_Ponderado']):.2f}%\n"
                         f"  Chamadas com alteracao Q: {d['Chamadas_Com_Alteracao_Q_Em_Decide']}\n"
                         f"  Entradas criadas em decide: {d['Entradas_Criadas_Em_Decide']}\n")
        axs[1, 1].text(.02, .97, "\n".join(texto) +
                       "Contagens incluem estados imaturos e maduros.\n"
                       "Executor: CSV mostra objetos retornados, nao prova obediencia.\n"
                       "Q: mede alteracoes em decide_action, nao todas as escritas.",
                       va="top", fontsize=10, transform=axs[1, 1].transAxes)
        axs[1, 1].axis("off")
    else:
        for ax in (axs[0, 1], axs[1, 0], axs[1, 1]):
            ax.axis("off")
            ax.text(.5, .5, "Cobertura nao disponivel nesta tentativa.\n"
                    "CSV antigo ou agente sem decisoes do cerebro.\n"
                    "Nao e possivel reconstruir a partir do WR.",
                    ha="center", va="center", transform=ax.transAxes)
    fig.suptitle(f"{r.get('Agente', '')} vs {r.get('Regua', 'Cynthia')} | "
                 f"{r.get('Ciclo', '')} | tentativa {r.get('Tentativa', '')} | seed {r.get('Semente', '')}\n"
                 "Treino e holdout identificam pools; ambas as condicoes sao avaliacao",
                 fontsize=13)
    destino = p.with_name(p.stem + "_dashboard.png")
    fig.savefig(destino, dpi=140)
    plt.close(fig)
    # Frequencia REAL de intencoes nesta avaliacao, nao argmax da Q-table.
    for c in condicoes:
        arq = p.with_name(p.stem + f"_{c}_acoes.csv")
        acoes = _ler(arq) if arq.exists() else []
        if not acoes:
            continue
        acoes.sort(key=lambda x: int(x["Decisoes"]), reverse=True)
        fig, ax = plt.subplots(figsize=(11, max(4, len(acoes) * .27)), layout="constrained")
        ax.barh([x["Acao"] for x in acoes], [float(x["Percentual"]) for x in acoes], color=cores[condicoes.index(c)])
        ax.invert_yaxis()
        ax.set(title=f"{r.get('Agente', '')} vs Cynthia | {c} | intencoes escolhidas",
               xlabel="Decisoes do cerebro (%)")
        fig.savefig(p.with_name(p.stem + f"_{c}_acoes.png"), dpi=140)
        plt.close(fig)
    return str(destino)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, help="CSV principal de uma tentativa")
    ap.add_argument("--diretorio", type=Path, default=GENERALIZACAO_DIR)
    args = ap.parse_args()
    arquivos = [args.csv] if args.csv else sorted(args.diretorio.rglob("Generalizacao_*.csv"))
    gerados = 0
    for p in arquivos:
        if not p.is_file():
            ap.error(f"Arquivo inexistente: {p}")
        with p.open(newline="", encoding="utf-8-sig") as f:
            campos = next(csv.reader(f), [])
        if "Delta_Treino_Menos_Holdout_pp" not in campos:
            continue
        destino = gerar_grafico(p)
        if destino:
            gerados += 1
            print(f"Grafico: {destino}")
    if not gerados:
        print("Nenhum CSV do protocolo Cynthia encontrado.")


if __name__ == "__main__":
    main()
