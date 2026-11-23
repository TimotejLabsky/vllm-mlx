"""Replay opencode sessions answered by Qwen3.8-27B-8bit and simulate
model-free n-gram (prompt-lookup / suffix) drafting against the tokens the
model actually produced. Laptop only, tokenizer only.

For each assistant message: context = every earlier part of the session
(user text, assistant reasoning/text, tool input JSON, tool output), output =
this message's reasoning + text + tool input JSON. At each output position the
longest suffix (n = NMAX..NMIN tokens) of context+output-so-far is looked up;
the tokens after its most recent earlier occurrence are the draft (<= k).
Accepted = common prefix with what the model really emitted. A round emits
accepted + 1 tokens and costs c(1 + len(draft)) plain steps.

usage: sim_ngram.py <db> <tokenizer.json> <curve json {S: x_of_S1}>
"""

import json
import sqlite3
import sys
from collections import defaultdict

from tokenizers import Tokenizer

MODEL = "Qwen3.8-27B-8bit"
NMIN, NMAX = 3, 8

db, tokp, curve_json = sys.argv[1], sys.argv[2], sys.argv[3]
measured = {int(k): v for k, v in json.loads(curve_json).items()}
pts = sorted(measured)
curve = {}
for s in range(1, pts[-1] + 1):  # linear interpolation between measured widths
    lo = max(p for p in pts if p <= s)
    hi = min(p for p in pts if p >= s)
    curve[s] = (
        measured[lo]
        if lo == hi
        else measured[lo] + (measured[hi] - measured[lo]) * (s - lo) / (hi - lo)
    )
tok = Tokenizer.from_file(tokp)
con = sqlite3.connect(db)


def enc(s):
    return tok.encode(s, add_special_tokens=False).ids if s else []


def part_text(d):
    t = d.get("type")
    if t in ("text", "reasoning"):
        return d.get("text") or ""
    if t == "tool":
        st = d.get("state") or {}
        inp = json.dumps({"name": d.get("tool"), "arguments": st.get("input")})
        return inp, (st.get("output") or "")
    return ""


sessions = [
    r[0]
    for r in con.execute(
        "select distinct session_id from message where json_extract(data,'$.modelID')=?",
        (MODEL,),
    )
]

# per message: (context tokens, output tokens)
samples = []
for sid in sessions:
    msgs = con.execute(
        "select id, data from message where session_id=? order by time_created, id",
        (sid,),
    ).fetchall()
    hist = []
    for mid, mdata in msgs:
        m = json.loads(mdata)
        parts = [
            json.loads(p[0])
            for p in con.execute(
                "select data from part where message_id=? order by id", (mid,)
            )
        ]
        ours = m.get("role") == "assistant" and m.get("modelID") == MODEL
        out = []
        after = []  # tool outputs arrive after the generation
        for d in parts:
            if d.get("type") == "compaction":
                continue
            pt = part_text(d)
            if isinstance(pt, tuple):
                out.append(enc(pt[0]))
                after.append(enc(pt[1]))
            elif pt:
                out.append(enc(pt))
        out_toks = [t for seg in out for t in seg]
        if ours and out_toks:
            samples.append((list(hist), out_toks))
        hist.extend(out_toks)
        for seg in after:
            hist.extend(seg)

print(
    f"sessions={len(sessions)} messages={len(samples)} "
    f"out_tokens={sum(len(o) for _, o in samples)} "
    f"median_ctx={sorted(len(c) for c, _ in samples)[len(samples)//2]}"
)


def simulate(ctx, out, k, min_match=NMIN):
    H = []
    idx = [None] + [dict() for _ in range(NMAX)]  # idx[n][ngram] -> continuation pos

    def push(t):
        j = len(H)
        H.append(t)
        for n in range(1, NMAX + 1):
            if j - n >= 0:
                idx[n][tuple(H[j - n : j])] = j

    for t in ctx:
        push(t)
    i = 0
    tokens = cost = rounds = drafted_rounds = drafted = accepted = 0
    full = 0
    while i < len(out):
        draft = []
        L = len(H)
        for n in range(NMAX, min_match - 1, -1):
            if L >= n:
                p = idx[n].get(tuple(H[L - n : L]))
                if p is not None:
                    draft = H[p : p + k]
                    break
        draft = draft[: max(0, min(k, len(out) - i - 1))]
        a = 0
        while a < len(draft) and draft[a] == out[i + a]:
            a += 1
        emit = a + 1
        cost += curve[1 + len(draft)]
        rounds += 1
        if draft:
            drafted_rounds += 1
            drafted += len(draft)
            accepted += a
            full += a == len(draft)
        for t in out[i : i + emit]:
            push(t)
        i += emit
        tokens += emit
    return dict(
        tokens=tokens,
        cost=cost,
        rounds=rounds,
        drafted_rounds=drafted_rounds,
        drafted=drafted,
        accepted=accepted,
        full=full,
    )


for k in sorted(s - 1 for s in pts if s > 1):
    agg = defaultdict(float)
    for ctx, out in samples:
        for key, v in simulate(ctx, out, k).items():
            agg[key] += v
    print(
        f"k={k:2d} (S={k+1:2d}, c={curve[k+1]:.2f}): speedup {agg['tokens']/agg['cost']:.3f}x  "
        f"draft-rounds {agg['drafted_rounds']/agg['rounds']:.1%}  "
        f"accepted/drafted {agg['accepted']/max(1,agg['drafted']):.1%}  "
        f"accepted/draft-round {agg['accepted']/max(1,agg['drafted_rounds']):.2f}  "
        f"full-accept {agg['full']/max(1,agg['drafted_rounds']):.1%}  "
        f"tokens/round {agg['tokens']/agg['rounds']:.3f}"
    )


# ---- policy: wide n-gram verify only on a long match, else a fallback round
MTP_ALPHA, MTP_COST = 0.80, curve[2] + 0.06  # live #167 acceptance; head = 3.2 ms


def policy(ctx, out, K, T, fallback):
    H = []
    idx = [None] + [dict() for _ in range(NMAX)]

    def push(t):
        j = len(H)
        H.append(t)
        for n in range(1, NMAX + 1):
            if j - n >= 0:
                idx[n][tuple(H[j - n : j])] = j

    for t in ctx:
        push(t)
    i = 0
    tokens = cost = wide = 0.0
    while i < len(out):
        draft = []
        L = len(H)
        if L >= T:
            p = idx[T].get(tuple(H[L - T : L]))
            if p is not None:
                draft = H[p : p + K][: max(0, len(out) - i - 1)]
        if draft:
            a = 0
            while a < len(draft) and draft[a] == out[i + a]:
                a += 1
            emit, c = a + 1, curve[1 + len(draft)]
            wide += 1
        elif fallback == "mtp":  # expected value of one MTP round
            emit, c = 1, MTP_COST
            tokens += MTP_ALPHA  # expected bonus token, position advance ignored
        else:
            emit, c = 1, 1.0
        for t in out[i : i + emit]:
            push(t)
        i += emit
        tokens += emit
        cost += c
    return tokens, cost, wide


base_mtp = (1 + MTP_ALPHA) / MTP_COST
print(f"\nMTP-only model-level baseline: {base_mtp:.3f}x")
for fb in ("plain", "mtp"):
    for T in (4, 6, 8):
        for K in (7, 11, 15, 31):
            tk = cs = w = 0.0
            for ctx, out in samples:
                a, b, c = policy(ctx, out, K, T, fb)
                tk, cs, w = tk + a, cs + b, w + c
            extra = f" ({tk / cs / base_mtp:.3f}x over MTP-only)" if fb == "mtp" else ""
            print(
                f"fallback={fb:5s} match>={T} K={K:2d}: {tk / cs:.3f}x over plain{extra}  wide rounds {w:.0f}"
            )
