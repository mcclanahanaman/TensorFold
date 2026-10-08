"""N concurrent agent sessions with dynamic turn lengths against a TensorFold server's /v1/capacity (FR-OPS-17).

Usage: python tools/thor/session_stress.py URL OUT.json [--sessions 8] [--turns 6] [--plan release|tiny]
       [--extra 0] [--extra-turns 3] [--require-refusal] [--seed 20261007] [--max-tokens 64] [--window 262144]
       [--turn-timeout 900] [--max-wait 600] [--sample-s 5] [--stagger-s 2] [--model qwen3.8-flash-next]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

WORDS = (
    'token', 'cache', 'window', 'prefix', 'session', 'server', 'agent', 'turn', 'prompt', 'memory', 'budget',
    'latency', 'throughput', 'stream', 'message', 'context', 'eviction', 'admission', 'retry', 'timeout',
    'summary', 'note', 'material', 'answer', 'reply', 'delta', 'chunk', 'usage', 'quota', 'limit', 'header',
    'route', 'client', 'thread', 'worker', 'sampler', 'timeline', 'record', 'row', 'class', 'reason', 'flag',
    'policy', 'fence', 'kernel', 'driver', 'daemon', 'socket', 'buffer', 'index', 'entry', 'value', 'key',
    'name', 'file', 'path', 'tree', 'branch', 'commit', 'diff', 'test', 'result', 'report', 'metric', 'graph',
)

SESSION_CLASS = ('short', 'short', 'short', 'long', 'long', 'long', 'huge', 'huge')
SIZES = {'first': {'short': (8000, 20000), 'long': (30000, 60000), 'huge': (60000, 120000)},
         'later': {'short': (500, 6000), 'long': (2000, 20000), 'huge': (20000, 60000)}}


def prose(tokens, seed) -> str:
    rng = random.Random(seed)
    out = []
    while len(out) < tokens * 0.72:
        sentence = [rng.choice(WORDS) for _ in range(rng.randint(6, 16))]
        out += sentence[:-1] + [sentence[-1] + '.']
    return ' '.join(out)


SYSTEM = {'role': 'system', 'content': 'You are a careful coding agent. Read the material, then answer briefly. '
                                       + prose(2000, 1)}


def get_json(url, timeout):
    req = urllib.request.Request(url, method='GET')
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode('utf-8'))


def post_json(url, body, timeout):
    req = urllib.request.Request(url, data=json.dumps(body).encode('utf-8'),
                                 headers={'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode('utf-8'))


def classify(k, err, cached, resumable, preview_cached, ev_after, ev_before) -> str:
    if k == 0:
        return 'first'
    if err:
        return 'error'
    if resumable is None:
        return 'unexplained'
    if cached is not None and cached >= 0.99 * resumable:
        return 'warm'
    if (preview_cached or 0) < 0.99 * resumable:
        return 'announced_cold'
    if (ev_after or 0) > (ev_before or 0):
        return 'raced'
    return 'unexplained'


def do_turn(i, k, messages, args, prev_prompt, extra=False, preview=True):
    body = {'model': args.model, 'messages': messages, 'max_tokens': args.max_tokens, 'temperature': 0,
            'reasoning_effort': 'low'}
    t_start = time.time()
    waited = 0.0
    retries = 0
    refusals = []
    pv = {}
    while preview:                                  # skipped entirely when preview is False (extra session)
        try:
            pv = post_json(args.url + '/v1/capacity', body, timeout=args.turn_timeout)
        except (OSError, ValueError) as exc:            # a preview the server could not answer is this turn's error
            return ({'session': i, 'turn': k, 'prompt_tokens': None, 'cached_tokens': None, 'completion_tokens': None,
                     'prev_prompt_tokens': prev_prompt, 'preview_prompt_tokens': None, 'preview_cached_tokens': None,
                     'evicted_before': None, 'evicted_after': None, 'ttft_s': None,
                     'total_s': round(time.time() - t_start, 3), 'retries': retries, 'waited_s': round(waited, 3),
                     'error': f'preview: {exc}'[:300], 'killed_mid_stream': False,
                     'timeout': isinstance(exc, (TimeoutError, socket.timeout)), 'class': 'error',
                     'extra': extra, 'refusals': refusals}, '')
        if pv.get('reason') == 'length':
            cls = classify(k, None, None, prev_prompt, pv.get('cached_tokens'), None, None)
            row = {'session': i, 'turn': k, 'ended': 'length', 'preview_prompt_tokens': pv.get('prompt_tokens'),
                   'error': None, 'killed_mid_stream': False, 'timeout': False, 'retries': retries,
                   'waited_s': round(waited, 3), 'class': cls, 'extra': extra, 'refusals': refusals}
            return row, ''
        if pv.get('admissible_now'):
            break
        nap = min(float(pv.get('retry_after_s') or 1.0), 5.0)
        time.sleep(nap)
        waited += nap
        if waited > args.max_wait:
            break
    p_tokens = pv.get('prompt_tokens')
    p_cached = pv.get('cached_tokens')
    ev_before = pv.get('evicted_kept_sessions_total')

    chat_body = dict(body)
    chat_body['stream'] = True
    chat_body['stream_options'] = {'include_usage': True}
    data = json.dumps(chat_body).encode('utf-8')
    err = None
    killed = False
    tmo = False
    ttft = None
    usage = None
    parts = []
    ev_after = None
    while True:
        t_send = time.time()
        try:
            req = urllib.request.Request(args.url + '/v1/chat/completions', data=data,
                                         headers={'Content-Type': 'application/json'}, method='POST')
            resp = urllib.request.urlopen(req, timeout=args.turn_timeout)
        except urllib.error.HTTPError as e:
            if e.code in (429, 503):
                try:
                    blob = json.loads(e.read().decode('utf-8'))
                except (OSError, ValueError):
                    blob = None
                reason = None
                if isinstance(blob, dict) and isinstance(blob.get('error'), dict):
                    reason = blob['error'].get('reason')
                refusals.append({'code': e.code, 'reason': reason, 'retry_after': e.headers.get('Retry-After') or None,
                                 'at_s': round(time.time() - t_start, 3)})
                retries += 1
                nap = min(float(e.headers.get('Retry-After') or 1), 10.0)
                time.sleep(nap)
                waited += nap
                if waited > args.max_wait:
                    err = f'http {e.code} after max wait'
                    break
                continue
            err = f'http {e.code}'
            break
        except (TimeoutError, socket.timeout):
            err = 'timeout'
            tmo = True
            break
        except (OSError, urllib.error.URLError) as e:
            err = f'conn: {e}'
            break
        try:
            with resp:
                for raw in resp:
                    line = raw.decode('utf-8', 'replace').strip()
                    if not line.startswith('data: '):
                        continue
                    payload = line[6:].strip()
                    if payload == '[DONE]':
                        break
                    try:
                        obj = json.loads(payload)
                    except ValueError:
                        continue
                    if isinstance(obj, dict) and obj.get('error') is not None:
                        killed = True
                        eobj = obj['error']
                        err = str(eobj.get('message') or eobj) if isinstance(eobj, dict) else str(eobj)
                        break
                    ch = obj.get('choices') or [{}]
                    d = (ch[0] or {}).get('delta') or {}
                    if d.get('content') or d.get('reasoning_content'):
                        if ttft is None:
                            ttft = time.time() - t_send
                        if d.get('content'):
                            parts.append(d['content'])
                    if obj.get('usage'):
                        usage = obj['usage']
        except (TimeoutError, socket.timeout):
            err = 'timeout'
            tmo = True
        except (OSError, urllib.error.URLError, ValueError) as e:
            err = err or f'read: {e}'
        try:
            ev_after = get_json(args.url + '/v1/capacity', timeout=args.turn_timeout).get(
                'evicted_kept_sessions_total')
        except (OSError, ValueError):
            ev_after = None
        break

    pt = usage.get('prompt_tokens') if usage else None
    ct = 0
    if usage:
        ct = (usage.get('prompt_tokens_details') or {}).get('cached_tokens') or 0
    comp = usage.get('completion_tokens') if usage else None
    cls = classify(k, err, ct, prev_prompt, p_cached, ev_after, ev_before)
    row = {'session': i, 'turn': k, 'prompt_tokens': pt, 'cached_tokens': ct, 'completion_tokens': comp,
           'prev_prompt_tokens': prev_prompt, 'preview_prompt_tokens': p_tokens, 'preview_cached_tokens': p_cached,
           'evicted_before': ev_before, 'evicted_after': ev_after, 'ttft_s': ttft,
           'total_s': round(time.time() - t_start, 3), 'retries': retries, 'waited_s': round(waited, 3),
           'error': err, 'killed_mid_stream': killed, 'timeout': tmo, 'class': cls, 'extra': extra,
           'refusals': refusals}
    return row, ''.join(parts)


def summarize(rows, timeline, require_refusal=False):
    sent = [r for r in rows if not r.get('ended')]
    follow = [r for r in sent if r['turn'] > 0 and not r.get('error')]

    def held(thr):
        counts = [sum(1 for t in (s.get('kept_tokens') or []) if (t or 0) >= thr) for s in timeline]
        return max(counts, default=0)

    refused_by = {}
    refused_malformed = 0
    for r in rows:
        for x in (r.get('refusals') or []):
            if x.get('reason') is not None:
                refused_by[x['reason']] = refused_by.get(x['reason'], 0) + 1
            ra = x.get('retry_after')
            good = isinstance(ra, str) and ra.isdecimal() and int(ra) >= 1
            if not good or x.get('reason') not in ('slots', 'memory'):
                refused_malformed += 1
    shares = [r['cached_tokens'] / r['prompt_tokens'] for r in follow if r.get('prompt_tokens')]
    s = {'sessions': len({r['session'] for r in rows}), 'turns_sent': len(sent), 'followups': len(follow),
         'warm': sum(1 for r in follow if r['class'] == 'warm'),
         'announced_cold': sum(1 for r in follow if r['class'] == 'announced_cold'),
         'raced': sum(1 for r in follow if r['class'] == 'raced'),
         'unexplained': sum(1 for r in follow if r['class'] == 'unexplained'),
         'errors': sum(1 for r in rows if r.get('error')),
         'killed': sum(1 for r in rows if r.get('killed_mid_stream')),
         'timeouts': sum(1 for r in rows if r.get('timeout')),
         'length_ended': sum(1 for r in rows if r.get('ended') == 'length'),
         'refused': sum(len(r.get('refusals') or []) for r in rows),
         'refused_by': {reason: refused_by[reason] for reason in sorted(refused_by)},
         'refusals_malformed': refused_malformed,
         'extra_sessions': len({r['session'] for r in rows if r.get('extra')}),
         'extra_turns_done': sum(1 for r in rows if r.get('extra') and not r.get('ended')
                                 and not r.get('error')),
         'retries': sum(r.get('retries', 0) for r in rows),
         'waited_s': round(sum(r.get('waited_s', 0) for r in rows), 3),
         'held_long_max': held(32768), 'held_long_max_64k': held(65536),
         'worst_followup_cached_share': round(min(shares), 4) if shares else None}
    s['pass'] = (s['errors'] == 0 and s['killed'] == 0 and s['timeouts'] == 0 and s['unexplained'] == 0
                 and s['refusals_malformed'] == 0 and (not require_refusal or s['refused'] >= 1))
    return s


def dump(out_path, args, rows, timeline):
    rs = sorted(rows, key=lambda r: (r['session'], r['turn']))
    doc = {'args': vars(args), 'rows': rs, 'timeline': timeline,
           'summary': summarize(rs, timeline, require_refusal=args.require_refusal)}
    out_path.write_text(json.dumps(doc, indent=1), encoding='utf-8')


def run_session(i, args, rows, timeline, state, lock, out_path):
    try:
        time.sleep(i * args.stagger_s)
        rng = random.Random(args.seed * 1000 + i)
        messages = [SYSTEM]
        last_prompt = None
        prev_text = ''
        extra = i >= args.sessions
        for k in range(args.extra_turns if extra else args.turns):
            if args.plan == 'tiny':
                size = rng.randint(20, 60)
            else:
                lo, hi = SIZES['first' if k == 0 else 'later']['short' if extra else SESSION_CLASS[i % 8]]
                size = int(math.exp(rng.uniform(math.log(lo), math.log(hi))))
            pseed = rng.randint(1, 10 ** 9)
            if k == 0:
                content = f'[s{i}] Summarize this material in one line.\n' + prose(size, pseed)
            else:
                messages.append({'role': 'assistant', 'content': prev_text})
                content = f'[s{i}] Turn {k} note:\n' + prose(size, pseed)
            messages.append({'role': 'user', 'content': content})
            row, text = do_turn(i, k, messages, args, last_prompt, extra=extra, preview=not extra)
            if row.get('prompt_tokens') is not None:
                last_prompt = row['prompt_tokens']
            prev_text = text                            # this turn's reply (empty when it was reasoning only)
            line = (f"[stress] s{i} t{k} class={row['class']} prompt={row.get('prompt_tokens')} "
                    f"cached={row.get('cached_tokens')} ttft={row.get('ttft_s')}s total={row.get('total_s')}s "
                    f"retries={row.get('retries')} refusals={len(row.get('refusals') or [])} "
                    f"waited={row.get('waited_s')}s")
            with lock:
                rows.append(row)
                dump(out_path, args, rows, timeline)
            print(line, flush=True)
            if row.get('ended') or row.get('error'):
                break
    except Exception as exc:                            # noqa: BLE001  a harness fault is a failed session, not a hidden one
        with lock:
            rows.append({'session': i, 'turn': -1, 'error': f'harness: {exc!r}'[:300], 'killed_mid_stream': False,
                         'timeout': False, 'retries': 0, 'waited_s': 0.0, 'class': 'error'})
        print(f'[stress] s{i} harness error {exc!r}', flush=True)
    finally:
        with lock:
            state['finished'] += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('url')
    ap.add_argument('out')
    ap.add_argument('--sessions', type=int, default=8)
    ap.add_argument('--extra', type=int, default=0)
    ap.add_argument('--turns', type=int, default=6)
    ap.add_argument('--extra-turns', type=int, default=3)
    ap.add_argument('--plan', choices=['release', 'tiny'], default='release')
    ap.add_argument('--seed', type=int, default=20261007)
    ap.add_argument('--max-tokens', type=int, default=64)
    ap.add_argument('--window', type=int, default=262144)
    ap.add_argument('--turn-timeout', type=float, default=900)
    ap.add_argument('--max-wait', type=float, default=600)
    ap.add_argument('--sample-s', type=float, default=5)
    ap.add_argument('--stagger-s', type=float, default=2)
    ap.add_argument('--model', default='qwen3.8-flash-next')
    ap.add_argument('--require-refusal', action='store_true')
    args = ap.parse_args()
    out_path = Path(args.out)
    rows, timeline = [], []
    state = {'finished': 0}
    lock = threading.Lock()

    def sampler():
        while True:
            cap = {}
            try:
                cap = get_json(args.url + '/v1/capacity', timeout=30)
            except (OSError, ValueError):
                pass
            ks = cap.get('kept_sessions') or []
            with lock:
                timeline.append({'t': time.time(), 'live_streams': cap.get('live_streams'),
                                 'overloaded': cap.get('overloaded'),
                                 'free_cache_bytes': cap.get('free_cache_bytes'),
                                 'evicted_kept_sessions_total': cap.get('evicted_kept_sessions_total'),
                                 'kept_tokens': [e.get('tokens') for e in ks]})
                done = state['finished'] >= args.sessions + args.extra
            if done:
                break
            time.sleep(args.sample_s)

    ths = [threading.Thread(target=run_session,
                            args=(i, args, rows, timeline, state, lock, out_path), daemon=True)
           for i in range(args.sessions + args.extra)]
    for t in ths:
        t.start()
    st = threading.Thread(target=sampler, daemon=True)
    st.start()
    for t in ths:
        t.join()
    st.join()
    with lock:
        dump(out_path, args, rows, timeline)
        s = summarize(sorted(rows, key=lambda r: (r['session'], r['turn'])), timeline,
                      require_refusal=args.require_refusal)
    print(('session_stress: PASS ' if s['pass'] else 'session_stress: FAIL ')
          + ' '.join(f'{k}={v}' for k, v in s.items()))
    sys.exit(0 if s['pass'] else 1)


if __name__ == '__main__':
    main()
