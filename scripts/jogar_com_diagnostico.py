"""Batalhas manuais observaveis, sem aprendizado (colocar em scripts/).

python -m scripts.jogar_com_diagnostico --agente green --humano Vylleon --batalhas 3
python -m scripts.jogar_com_diagnostico --agente blue --brain artefatos/brains/blue_brain.pkl
python -m scripts.jogar_com_diagnostico --agente instinto

Mantem desafios diretos, formato gen9nationaldex, time fixo/sorteado, servidor
local/oficial, timer e diagnostico do projeto. Nao modifica os agentes de treino.
Epsilon=0 nao elimina desempates/estados desconhecidos nem o shuffle do Green.
Logs: artefatos/logs/DiagnosticoManual/<agente>_<data>/.
JSONL contem decisoes, Q, filtros observados, execucao, tempos e snapshots.
Os snapshots seguintes mostram consequencias observadas, nao causalidade isolada.
Para detalhes de motivos dos filtros, ver tambem terminal.log e sondas existentes.
"""
import argparse
import asyncio
from datetime import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def label(obj):
    return getattr(obj, 'id', None) or getattr(obj, 'species', None) or str(obj)


def pokemon(mon):
    if mon is None:
        return None
    return {k: getattr(mon, k, None) for k in
            ('species', 'current_hp_fraction', 'fainted', 'status', 'boosts',
             'effects', 'ability', 'item', 'types')}


def snapshot(b):
    return dict(batalha=b.battle_tag, turno=b.turn,
                ativo=pokemon(b.active_pokemon), adversario=pokemon(b.opponent_active_pokemon),
                time={k: pokemon(v) for k, v in b.team.items()},
                time_adversario={k: pokemon(v) for k, v in b.opponent_team.items()},
                movimentos=[dict(id=m.id, categoria=str(m.category),
                                 potencia=m.base_power, pp=getattr(m, 'current_pp', None))
                            for m in b.available_moves],
                trocas=[label(m) for m in b.available_switches],
                troca_forcada=b.force_switch, clima=b.weather,
                campo=getattr(b, 'fields', {}), hazards=b.side_conditions,
                hazards_adversario=b.opponent_side_conditions,
                mecanicas={k: getattr(b, k, None) for k in
                           ('can_mega_evolve', 'can_z_move', 'can_tera', 'can_dynamax')})


def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [safe(v) for v in value]
    if hasattr(value, 'tolist'):
        return safe(value.tolist())
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


class Recorder:
    def __init__(self, path):
        self.file = open(path, 'w', encoding='utf-8', buffering=1)
        self.lock = threading.RLock()
        self.last = time.monotonic()
        self.failure = None

    def emit(self, event, **data):
        with self.lock:
            self.file.write(json.dumps(safe(dict(evento=event,
                            horario=datetime.now().astimezone().isoformat(), **data)),
                            ensure_ascii=False, allow_nan=False) + '\n')
            if event not in ('pulso', 'alerta_sem_progresso'):
                self.last = time.monotonic()


class Tee:
    def __init__(self, console, file, lock):
        self.console, self.file, self.lock = console, file, lock
    def write(self, text):
        with self.lock:
            self.console.write(text)
            self.file.write(text)
            self.file.flush()
        return len(text)
    def flush(self):
        self.console.flush()
        self.file.flush()
    def __getattr__(self, name):
        return getattr(self.console, name)


class DiagnosticMixin:
    """Instrumenta o comportamento existente; nao corrige silenciosamente a politica."""
    def _learn_from_previous(self, *args, **kwargs):
        # Avaliacao: nenhuma atualizacao de reward, traces, replay ou Q.
        return None

    def _aplicar_update_terminal(self, *args, **kwargs):
        return None

    def _get_actions_and_ranking(self, battle, hist):
        t = time.perf_counter()
        result = super()._get_actions_and_ranking(battle, hist)
        if self._trace is not None:
            self._trace['opcoes'] = dict(validas=list(result[0]), ranking=list(result[1]),
                                        ms=(time.perf_counter()-t)*1000)
        return result

    def _log_choose_error(self):
        # Captura inclusive erros que a base recupera por fallback.
        self.rec.emit('erro_choose_move', traceback=traceback.format_exc())
        print(traceback.format_exc(), flush=True)

    def choose_random_move(self, battle):
        self.rec.emit('fallback_aleatorio', batalha=battle.battle_tag, turno=battle.turn)
        return super().choose_random_move(battle)

    def teampreview(self, battle):
        t = time.perf_counter()
        order = super().teampreview(battle)
        self.rec.emit('lead', batalha=battle.battle_tag, ordem=str(order),
                      ms=(time.perf_counter()-t)*1000)
        return order

    def choose_move(self, battle):
        t = time.perf_counter()
        self._trace = dict(snapshot=snapshot(battle), filtros=[], execucoes=[],
                           historico_entrada=dict(getattr(self, '_history', {}).get(battle.battle_tag, {})))
        try:
            order = super().choose_move(battle)
            self._trace['ordem'] = getattr(order, 'message', str(order))
            return order
        except Exception:
            self._trace['erro'] = traceback.format_exc()
            self.rec.failure = self._trace['erro']
            raise
        finally:
            self._trace['ms_total'] = (time.perf_counter()-t)*1000
            self.rec.emit('decisao', **self._trace)
            print(f"[DECISAO] {battle.battle_tag} T{battle.turn} "
                  f"{self._trace.get('cerebro', {}).get('escolha', 'sem escolha tabular')} "
                  f"-> {self._trace.get('ordem', 'ERRO')} "
                  f"| {self._trace['ms_total']:.2f} ms", flush=True)
            if 'cerebro' in self._trace:
                c = self._trace['cerebro']
                print(f"  [Q] visitas={c['visitas']} inicial={c['regra_inicial']} "
                      f"valores={c['q_usados']}", flush=True)
            self._trace = None

    def _battle_finished_callback(self, battle):
        super()._battle_finished_callback(battle)
        self.rec.emit('fim_batalha', snapshot=snapshot(battle), venceu=battle.won,
                      perdeu=battle.lost)
        # Nenhum processamento terminal de aprendizado; apenas libera memoria.
        getattr(self, '_history', {}).pop(battle.battle_tag, None)


def instrument(bot, rec):
    bot.rec, bot._trace = rec, None
    masker = bot.instinct.masker if hasattr(bot, 'instinct') else bot.masker
    executor = getattr(bot, 'executor', bot.instinct.executor)
    def wrap(owner, name, kind):
        original = getattr(owner, name)
        def traced(*args, **kwargs):
            t = time.perf_counter()
            try:
                result = original(*args, **kwargs)
            except Exception:
                rec.emit('erro_componente', componente=name, traceback=traceback.format_exc())
                raise
            if bot._trace is not None:
                if kind == 'filtro':
                    bot._trace['filtros'].append(dict(funcao=name, movimento=label(args[0]),
                                                     removido=bool(result)))
                else:
                    hist = args[2] if len(args) > 2 else kwargs.get('history')
                    bot._trace['execucoes'].append(dict(intencao=args[0], objeto=label(result),
                        parametros=kwargs, ultima_acao_no_executor=(hist or {}).get('last_action'),
                        ms=(time.perf_counter()-t)*1000))
            return result
        setattr(owner, name, traced)
    wrap(masker, 'is_move_useless', 'filtro')
    wrap(masker, 'is_hazard_already_set', 'filtro')
    wrap(executor, 'get_best_execution_object', 'executor')
    policy = bot.instinct.policy
    original_profile = policy.get_instinct_profile
    def profile(*args, **kwargs):
        t = time.perf_counter()
        result = original_profile(*args, **kwargs)
        if bot._trace is not None:
            bot._trace['perfil_instinto'] = dict(primaria=result[0], confianca=result[1],
                ranking=result[2], mascara=result[3], letal=result[4],
                ms=(time.perf_counter()-t)*1000)
        return result
    policy.get_instinct_profile = profile
    if not hasattr(bot, 'brain'):
        return
    brain = bot.brain
    brain.epsilon = 0.0
    original = brain.decide_action
    # Desliga contadores de treino; os diagnosticos desta sessao sao independentes.
    brain._record_action_choice = lambda *args: None
    def decide(state, valid, ranking):
        key = brain._get_abstract_state(state)
        old = brain.q_table.get(key)
        saved = None if old is None else old.copy()
        had_visits = key in brain.visit_counts
        visits = brain.visit_counts.get(key, 0)
        before = {} if saved is None else {a: float(saved[brain.actions.index(a)]) for a in valid}
        t = time.perf_counter()
        try:
            result = original(state, valid, ranking)
            used = brain.q_table[key]
            values = {a: float(used[brain.actions.index(a)]) for a in valid}
            if bot._trace is not None:
                bot._trace['cerebro'] = dict(estado=state, conhecido=old is not None,
                    visitas=visits, q_antes=before, q_usados=values, validas=list(valid),
                    ranking=list(ranking), escolha=result, epsilon=brain.epsilon,
                    exploratoria=getattr(brain, 'ultima_foi_exploratoria', None),
                    regra_inicial=(visits == 0 or all(v == 0 for v in values.values())),
                    empatadas_no_maximo=[a for a,v in values.items() if v == max(values.values())],
                    ms=(time.perf_counter()-t)*1000)
            return result
        finally:
            # decide_action cria estados e heranca _MEC: restaurar tambem essas escritas.
            if old is None:
                brain.q_table.pop(key, None)
            else:
                old[:] = saved
                brain.q_table[key] = old
            if had_visits:
                brain.visit_counts[key] = visits
            else:
                brain.visit_counts.pop(key, None)
    brain.decide_action = decide


async def run(args, rec, folder):
    from poke_env import AccountConfiguration, ServerConfiguration
    from poke_env.ps_client.server_configuration import ShowdownServerConfiguration
    from shared import diagnostico
    from shared.env.teams_train import RandomTeamFromPool, TEAMS_LIST
    if args.agente == 'instinto':
        from instinct.instinct_player import InstinctBot as Parent
    elif args.agente == 'blue':
        from qlearning.hybrid_agent import HybridAgent as Parent
    else:
        from qlearning.pure_agent import PureAgent as Parent
    Agent = type('ManualDiagnosticAgent', (DiagnosticMixin, Parent), {})
    if args.time is not None and not 0 <= args.time < len(TEAMS_LIST):
        raise ValueError(f'--time deve estar entre 0 e {len(TEAMS_LIST)-1}')
    checkpoint = None
    kw = {}
    if args.agente != 'instinto':
        checkpoint = Path(args.brain or ROOT / 'artefatos' / 'brains' / f'{args.agente}_brain.pkl').resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f'Checkpoint ausente: {checkpoint}')
        initial_hash = digest(checkpoint)
        kw.update(brain_file=str(checkpoint), epsilon=0.0)
    else:
        kw['diagnostico'] = True
    diagnostico.ligar()
    bot = Agent(account_configuration=AccountConfiguration(args.bot, os.getenv('SHOWDOWN_PASS') if args.oficial else None),
        server_configuration=ShowdownServerConfiguration if args.oficial else ServerConfiguration(
            'ws://localhost:8000/showdown/websocket', 'http://localhost:8000/'),
        battle_format='gen9nationaldex', team=TEAMS_LIST[args.time] if args.time is not None else RandomTeamFromPool(TEAMS_LIST),
        max_concurrent_battles=1, start_timer_on_battle_start=not args.sem_timer,
        log_level=25, **kw)
    if checkpoint and not bot.brain.q_table:
        raise RuntimeError('Checkpoint nao carregou uma Q-table nao vazia; avaliacao cancelada.')
    instrument(bot, rec)
    sources = {}
    for relative in ['qlearning/base_agent.py', 'qlearning/brain.py', 'qlearning/hybrid_agent.py',
                     'qlearning/pure_agent.py', 'instinct/execution.py', 'shared/physics.py',
                     'instinct/masking.py', 'instinct/policy.py']:
        path = ROOT / relative
        if path.is_file(): sources[relative] = digest(path)
    rec.emit('configuracao', agente=args.agente, bot=args.bot, humano=args.humano,
             checkpoint=str(checkpoint), checkpoint_sha256=initial_hash if checkpoint else None,
             aprendizado=False, epsilon=0 if checkpoint else None, time=args.time,
             timer=not args.sem_timer, seed=args.seed, codigo_sha256=sources)
    print(f"\n{args.agente.upper()} | bot: {args.bot} | desafios: {args.batalhas} | humano: {args.humano or 'qualquer'}")
    print('Navegador: Find a user -> nome do bot -> Challenge -> [Gen 9] National Dex')
    print(f'Logs: {folder}\nAprendizado desligado; checkpoint nao sera salvo.')
    async def monitor():
        while True:
            await asyncio.sleep(5)
            if rec.failure:
                raise RuntimeError('Falha na decisao. Consulte decisoes.jsonl e terminal.log; sessao interrompida.')
            idle = time.monotonic()-rec.last
            if idle >= args.alerta:
                pending = [dict(batalha=b.battle_tag, turno=b.turn) for b in list(bot.battles.values()) if not b.finished]
                rec.emit('alerta_sem_progresso', segundos=idle, pendentes=pending,
                         concluidas=bot.n_finished_battles)
                print(f'[ESPERA] {idle:.0f}s sem decisao/fim. Pendentes: {pending}. Pode ser espera pelo humano.', flush=True)
                await asyncio.sleep(max(0, args.alerta-5))
    challenge = asyncio.create_task(bot.accept_challenges(args.humano, args.batalhas))
    watcher = asyncio.create_task(monitor())
    try:
        done, _ = await asyncio.wait([challenge, watcher], return_when=asyncio.FIRST_COMPLETED)
        for task in done: await task
    finally:
        for task in [challenge, watcher]:
            if not task.done(): task.cancel()
        await asyncio.gather(challenge, watcher, return_exceptions=True)
        rec.emit('fim_sessao', concluidas=bot.n_finished_battles, vitorias=bot.n_won_battles,
                 checkpoint_inalterado=(digest(checkpoint)==initial_hash) if checkpoint else None)
        print(f'[FIM] {bot.n_won_battles} vitorias / {bot.n_finished_battles} concluidas')


def cli():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--agente', choices=['blue', 'green', 'instinto'], default='green')
    ap.add_argument('--brain', help='checkpoint .pkl; default artefatos/brains/<agente>_brain.pkl')
    ap.add_argument('--humano')
    ap.add_argument('--bot')
    ap.add_argument('--oficial', action='store_true', help='senha via SHOWDOWN_PASS')
    ap.add_argument('--batalhas', type=int, default=1)
    ap.add_argument('--time', type=int)
    ap.add_argument('--sem-timer', action='store_true')
    ap.add_argument('--seed', type=int, help='semente Python/NumPy; nao controla RNG do servidor')
    ap.add_argument('--alerta', type=float, default=60, help='segundos sem decisao/fim antes do aviso')
    ap.add_argument('--logs', type=Path, default=ROOT/'artefatos'/'logs'/'DiagnosticoManual')
    args = ap.parse_args()
    args.bot = args.bot or {'blue':'AlfineteBlueDiag','green':'AlfineteGreenDiag','instinto':'AlfineteInstintoDiag'}[args.agente]
    normalize = lambda s: ''.join(c for c in s.lower() if c.isascii() and c.isalnum())
    if args.humano and normalize(args.humano) == normalize(args.bot):
        ap.error('Use nomes diferentes para humano e bot.')
    if args.batalhas < 1 or args.alerta < 5:
        ap.error('--batalhas >= 1 e --alerta >= 5')
    if args.seed is not None:
        import numpy as np
        random.seed(args.seed); np.random.seed(args.seed)
    folder = args.logs / f'{args.agente}_{datetime.now():%Y%m%d_%H%M%S_%f}'
    folder.mkdir(parents=True, exist_ok=False)
    rec = Recorder(folder/'decisoes.jsonl')
    stdout, stderr = sys.stdout, sys.stderr
    terminal = open(folder/'terminal.log', 'w', encoding='utf-8', buffering=1)
    sys.stdout, sys.stderr = Tee(stdout, terminal, rec.lock), Tee(stderr, terminal, rec.lock)
    try:
        if sys.platform == 'win32':
            loop = asyncio.SelectorEventLoop()
            asyncio.set_event_loop(loop)
            try: loop.run_until_complete(run(args, rec, folder))
            finally: loop.close()
        else:
            asyncio.run(run(args, rec, folder))
    except Exception:
        rec.emit('erro_sessao', traceback=traceback.format_exc())
        traceback.print_exc()
        return 1
    finally:
        sys.stdout, sys.stderr = stdout, stderr
        terminal.close(); rec.file.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(cli())
