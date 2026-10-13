#!/usr/bin/env python
"""HeritAgent coordination metrics (read-only analysis of the comms ledger + raw API JSON).

Usage: python coordination_metrics.py [COORD_DIR]
Outputs: COORD_DIR/metrics.json and COORD_DIR/metrics_tables.md

Data sources
  - raw/ic_<N>.json : issue-comment endpoint for number N (PR conversation for PR numbers, exact timestamps)
  - comms-ledger.jsonl : heuristic role/kind/tests extraction (used for cross-checks and test counts)
  - PR_META : created_at/merged_at copied from mcp__github__pull_request_read (live read, 2026-10-12)
  - SUPPLEMENT : 3 comments posted AFTER the ledger snapshot (live read), flagged as such
Roles are content-heuristic (all comments share one GitHub login); no agent attribution is attempted.
"""
import json, re, sys, statistics as st
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict, Counter

D = Path(sys.argv[1] if len(sys.argv) > 1 else
         "<scratch>/out/research/coordination")
RAW = D / "raw"
PRS = [1, 2, 21, 23, 24, 25, 26, 27, 28]

def ts(s): return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
def mins(a, b): return round((b - a).total_seconds() / 60.0, 1)

PR_META = {  # (created_at, merged_at) from live PR reads; PR28 still open
    1: ("2026-10-02T18:24:41Z", "2026-10-02T19:16:40Z"),
    2: ("2026-10-02T18:25:05Z", "2026-10-02T19:16:53Z"),
    21: ("2026-10-02T19:45:04Z", "2026-10-02T21:01:36Z"),
    23: ("2026-10-02T21:33:46Z", "2026-10-03T16:40:07Z"),
    24: ("2026-10-03T17:14:16Z", "2026-10-04T10:39:52Z"),
    25: ("2026-10-04T17:53:32Z", "2026-10-07T16:21:22Z"),
    26: ("2026-10-07T16:48:18Z", "2026-10-08T12:33:16Z"),
    27: ("2026-10-08T13:03:55Z", "2026-10-12T14:20:54Z"),
    28: ("2026-10-12T15:47:27Z", None),
}
SUPPLEMENT = {  # comments newer than the ledger snapshot (live read of PR #28 / issue #20)
    28: [
        dict(id=6133007770, created_at="2026-10-12T21:32:48Z", role="builder", kind="fix", tests=438,
             sha="c6ce4c1", blockers=None, note="builder: final 5-item pass, 438/438 local, re-review requested"),
        dict(id=6133043400, created_at="2026-10-12T21:35:14Z", role="critic", kind="blocking", tests=None,
             sha="c6ce4c1", blockers=2, note="critic: 2 type-strict blockers (stage_pass bool; exact typed GenerationSettings)"),
    ]}
# PR26 round-1 critic review is absent from every GitHub thread (see report); builder answered it at 17:15Z.
HIDDEN_ROUNDS = {26: [dict(inferred_before="2026-10-07T17:15:00Z", evidence="builder 'four blocking groups patched' with no critic comment on #26/#10/#20")]}

# Manual item-count overrides where the regex cannot parse unnumbered/prose reviews (verified against the review text / ledger blocker_groups)
ITEM_OVERRIDE = {6213: None}  # placeholder, keyed below by (pr, "HH:MM")
ITEM_OVERRIDE = {(1, "19:06"): 1, (2, "19:06"): 1, (26, "11:58"): 3, (26, "12:24"): 1}
# issue-thread builder handoff test counts (first builder report per issue) = baseline of each PR's trajectory
HANDOFF_TESTS = {21: 61, 23: 127, 24: 178, 25: 236, 26: 283, 27: 348, 28: 406}

ledger = [json.loads(l) for l in open(D / "comms-ledger.jsonl")]
tests_by_id = {r["comment_id"]: r["tests_reported"] for r in ledger if r["tests_reported"] is not None}

STOP = re.compile(r"^\s*(#{2,4}\s*)?(non-blocking|smaller|small\b|test/ci|closed on this head|confirmed closed)", re.I)

def classify(body):
    head = body[:160]
    if "Research/Critic" in head:
        return "critic", ("clear" if "FINAL CLEAR" in head else "blocking")
    if re.match(r"\s*(##\s*)?Builder", head):
        return "builder", "fix"
    return "other", "other"

def split_items(body):
    """Return list of blocking item texts (excluding non-blocking / closed sections)."""
    lines = body.split("\n")
    # drop 'closed' sections: keep from 'Still blocking' if present
    for i, l in enumerate(lines):
        if re.match(r"^\s*(#{2,4}\s*)?(still blocking|still blocking:)", l, re.I) or l.strip().lower() == "still blocking:":
            lines = lines[i + 1:]; break
    items, cur, mode = [], None, None
    has_h3 = any(re.match(r"^###\s*\d+\)", l) for l in lines)
    for l in lines:
        if STOP.match(l) and not re.match(r"^###\s*\d+\)", l):
            if cur is not None: items.append("\n".join(cur)); cur = None
            if re.match(r"^\s*(#{2,4}\s*)?(non-blocking|smaller|small\b|test/ci)", l, re.I): break
            continue
        if has_h3:
            if re.match(r"^###\s*\d+\)", l):
                if cur is not None: items.append("\n".join(cur))
                cur = [l]
            elif cur is not None: cur.append(l)
        else:
            m = re.match(r"^(\d+)[.)]\s", l)
            if m and (len(items) + (1 if cur is not None else 0) + 1 == int(m.group(1))):
                if cur is not None: items.append("\n".join(cur))
                cur = [l]
            elif cur is not None: cur.append(l)
    if cur is not None: items.append("\n".join(cur))
    return items

CATS = [
    ("immutability/aliasing", r"immutab|frozen|mutable|deepcopy|deep-copy|snapshot|alias|mappingproxy|defensive cop|shared dict"),
    ("atomicity/commit", r"atomic|all-or-nothing|exclusive creation|check-then-open|fail-atomic|live (cig )?(store|commit)|commit"),
    ("provenance/audit evidence", r"provenance|audit|reconstruct|durable|trajectory (id|ref)|evidence record|identify/reconstruct"),
    ("schema/contract validation", r"schema|load_cig|load_somatic|fail[- ]closed|validat|enforce|preflight|coheren|parity"),
    ("determinism/reproducibility", r"determinis|reproducib|seed|temperature|generationsettings|settings|monotonic|timestamp|canonical"),
    ("hashing/identity", r"hash|digest|sha256|identity|unique|id grammar|trajectory_id|version identity"),
    ("lifecycle/state machine", r"lifecycle|state machine|terminal|replay parity|stale|append guard|journal"),
    ("type strictness", r"isinstance|\bbool\b|typed|type-strict|int,|float"),
    ("leakage/oracle labels", r"leak|oracle|evaluation-only|task_type|backbone_tested"),
    ("pre-reg/protocol constants", r"pre-?reg|locked|retun|threshold|sensitivity|protocol"),
]

def cat_counts(items):
    c = Counter()
    for it in items:
        low = it.lower()
        for name, pat in CATS:
            if re.search(pat, low): c[name] += 1
    return c

out = {"prs": {}, "overall": {}}
all_fix_lat, all_crit_lat = [], []
cat_total = Counter(); cat_by_pr = {}
for n in PRS:
    raw = json.load(open(RAW / f"ic_{n}.json"))
    ev = []
    for c in raw:
        role, kind = classify(c["body"])
        items = split_items(c["body"]) if role == "critic" and kind == "blocking" else []
        hh = c["created_at"][11:16]
        if (n, hh) in ITEM_OVERRIDE and role == "critic" and kind == "blocking": items = ["(override)"] * ITEM_OVERRIDE[(n, hh)]
        has_sha = bool(re.search(r"\b[0-9a-f]{7}\b", c["body"]))
        ev.append(dict(id=c["id"], t=ts(c["created_at"]), role=role, kind=kind, n_items=len(items), items=items,
                       tests=tests_by_id.get(str(c["id"])), body_len=len(c["body"]), body=c["body"],
                       sha=has_sha))
    for s in SUPPLEMENT.get(n, []):
        ev.append(dict(id=s["id"], t=ts(s["created_at"]), role=s["role"], kind=("fix" if s["role"] == "builder" else s["kind"]),
                       n_items=s["blockers"] or 0, items=[], tests=s["tests"], body_len=0, body="", sha=True, supplement=True))
    ev.sort(key=lambda e: e["t"])
    created = ts(PR_META[n][0]); merged = ts(PR_META[n][1]) if PR_META[n][1] else None
    crit_block = [e for e in ev if e["role"] == "critic" and e["kind"] == "blocking"]
    crit_clear = [e for e in ev if e["role"] == "critic" and e["kind"] == "clear"]
    builder = [e for e in ev if e["role"] == "builder"]
    hidden = len(HIDDEN_ROUNDS.get(n, []))
    # builder fix latency: critic blocking -> next builder comment
    fix_lat, crit_lat = [], []
    for e in crit_block:
        nxt = next((b for b in builder if b["t"] > e["t"]), None)
        if nxt: fix_lat.append(mins(e["t"], nxt["t"]))
    # critic latency: PR open / builder report -> next critic comment
    triggers = [created] + [b["t"] for b in builder]
    for t0 in triggers:
        nxt = next((c for c in ev if c["role"] == "critic" and c["t"] > t0), None)
        if nxt and not (t0 != created and any(b["t"] > t0 and b["t"] < nxt["t"] for b in builder)):
            crit_lat.append(mins(t0, nxt["t"]))
    # hidden round: builder's first reply at 17:15 has no preceding critic -> its 'fix latency' from open is 27 min (not a real fix latency)
    if hidden: crit_lat = crit_lat[1:]  # open->first visible critic comment is not a latency: round-1 review is invisible  # first PR-open->critic latency is undefined for PR26 (no visible round-1 review)
    tests = ([HANDOFF_TESTS[n]] if n in HANDOFF_TESTS else []) + [e["tests"] for e in builder if e["tests"] is not None]
    items_all = [it for e in crit_block for it in e["items"]]
    cc = cat_counts(items_all); cat_by_pr[n] = dict(cc); cat_total.update(cc)
    # protocol-5 adherence on builder reports (PR threads, snapshot only)
    p5 = dict(sha=0, unresolved=0, rereview=0, n=0)
    for b in builder:
        if b.get("supplement"): continue
        p5["n"] += 1
        p5["sha"] += bool(re.search(r"`[0-9a-f]{7,40}`", b["body"]))
        p5["unresolved"] += bool(re.search(r"unresolved", b["body"], re.I))
        p5["rereview"] += bool(re.search(r"re-?review|review requested", b["body"], re.I))
    still_share = None
    later = [e for e in crit_block[1:]]
    if later:
        li = [it for e in later for it in e["items"]]
        if li: still_share = round(sum(bool(re.search(r"\bstill\b|remain|not yet|does not yet|incomplete", it, re.I)) for it in li) / len(li), 2)
    out["prs"][n] = dict(
        opened=PR_META[n][0], merged=PR_META[n][1],
        open_to_merge_h=round(mins(created, merged) / 60, 2) if merged else None,
        visible_critic_blocking_rounds=len(crit_block), inferred_hidden_rounds=hidden,
        total_blocking_rounds_incl_hidden=len(crit_block) + hidden,
        builder_reports=len(builder), critic_clear=len(crit_clear),
        items_per_round=[e["n_items"] for e in crit_block],
        review_body_chars_per_round=[e["body_len"] for e in crit_block],
        fix_latency_min=fix_lat, critic_latency_min=crit_lat,
        clear_to_merge_s=(round((merged - crit_clear[-1]["t"]).total_seconds()) if merged and crit_clear else None),
        tests_trajectory=tests, carryover_item_share_after_round1=still_share,
        protocol5=p5, nonblocking_notes=sum(bool(re.search(r"non-blocking|smaller|small (cleanup|hardening)", e["body"], re.I)) for e in crit_block),
        category_counts=dict(cc),
    )
    all_fix_lat += fix_lat; all_crit_lat += crit_lat

def summ(x):
    x = [v for v in x if v is not None]
    return dict(n=len(x), median=round(st.median(x), 1), mean=round(st.mean(x), 1), min=min(x), max=max(x),
                p_le_30min=round(sum(v <= 30 for v in x) / len(x), 2), over_8h=sum(v > 480 for v in x)) if x else {}

done = [p for p in out["prs"].values() if p["merged"]]
out["overall"] = dict(
    fix_latency_min=summ(all_fix_lat), critic_latency_min=summ(all_crit_lat),
    open_to_merge_h=summ([p["open_to_merge_h"] for p in done]),
    total_visible_blocking_rounds_merged=sum(p["visible_critic_blocking_rounds"] for p in done),
    total_blocking_rounds_incl_hidden_merged=sum(p["total_blocking_rounds_incl_hidden"] for p in done),
    merged_prs=len(done), merged_with_zero_blocking_rounds=sum(p["total_blocking_rounds_incl_hidden"] == 0 for p in done),
    clear_to_merge_s={n: p["clear_to_merge_s"] for n, p in out["prs"].items()},
    category_totals=dict(cat_total.most_common()),
    ledger_rows=len(ledger),
    rows_by_thread_type=dict(Counter(("issue#20" if r["item"] == "issue#20" else ("pr" if r["item"].startswith("pr#") else "issue_other")) for r in ledger)),
    note="categories are regex-heuristic over item text (see report for the manual categorisation); PR1/PR2 are doc/ADR PRs and included.",
)
# issue#20 share, duplicate posting
n20 = sum(r["item"] == "issue#20" for r in ledger); out["overall"]["issue20_share"] = round(n20 / len(ledger), 3)
out["overall"]["posted_to_3_threads_per_event"] = True
# timeline from ledger (PR threads + #20 milestones), UTC
out["timeline"] = [dict(t=r["created_at"], item=r["item"], role=r["role"], kind=r["kind"], shas=r["commits_mentioned"][:2], tests=r["tests_reported"], summary=r["summary"][:80])
                   for r in ledger if r["item"].startswith("pr#") or r["kind"] in ("handoff_start",)]
json.dump(out, open(D / "metrics.json", "w"), indent=1, default=str)

# ----- markdown tables -----
L = ["| PR | issue | opened | merged | open->merge (h) | critic blocking rounds (visible+hidden) | items/round | fix latency min (critic->builder) | critic latency min (builder->critic) | tests trajectory | clear->merge s |", "|---|---|---|---|---|---|---|---|---|---|---|"]
issue_of = {1: "#3", 2: "#5", 21: "#4", 23: "#7", 24: "#8", 25: "#9", 26: "#10", 27: "#11", 28: "#12"}
for n, p in out["prs"].items():
    L.append(f"| #{n} | {issue_of[n]} | {p['opened'][5:16]}Z | {(p['merged'] or 'OPEN')[5:16]} | {p['open_to_merge_h']} | {p['visible_critic_blocking_rounds']}+{p['inferred_hidden_rounds']} | {p['items_per_round']} | {p['fix_latency_min']} | {p['critic_latency_min']} | {p['tests_trajectory']} | {p['clear_to_merge_s']} |")
(D / "metrics_tables.md").write_text("\n".join(L) + "\n\nOVERALL\n" + json.dumps(out["overall"], indent=1, default=str) + "\n")
print("\n".join(L)); print(json.dumps(out["overall"], indent=1, default=str))
for n, p in out["prs"].items(): print(n, "p5", p["protocol5"], "carry", p["carryover_item_share_after_round1"], "chars", p["review_body_chars_per_round"], "nonblock", p["nonblocking_notes"])
