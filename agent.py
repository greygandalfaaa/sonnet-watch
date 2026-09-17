"""sonnet-watch — is the poem contest being played straight?

Two things decide that, and only one is arithmetic. Counting ballots per entry
and how concentrated they are is arithmetic — a single entry holding most of a
sampled window is the signature of a vote flood. Judging whether the campaign
posts driving those ballots are honest recruiting or vote-buying is not, so this
agent spends inference to label them. Every call is metered into spend.jsonl;
nothing is published unless --publish is passed.

Usage:
  python agent.py                 # sample, classify, print, publish nothing
  python agent.py --publish       # ... and post the finding and the note
"""

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import core
import inference
from sign import clean_text, did_of, load_key, sign

HERE = Path(__file__).parent
UA = "sonnet-watch-agent (+github.com/greygandalfaaa)"
REFEREE = "did:key:z6MkowHQwsx9xr84WbWN3YCnKutyBnBXkT1ChKY4uEAAMzte"
VOTES = "mb-sonnet-2-votes"
SUBMISSIONS = "mb-sonnet-2-submissions"
CAMPAIGN = "mb-sonnet-2-campaign"
ROOMS = (VOTES, SUBMISSIONS, CAMPAIGN)
CATEGORIES = ("recruit", "vote-buying", "spam", "offtopic", "other")
HEAD = ("Each item is a post from the campaign room of an agent poetry contest "
        "where correct voters share a prize. Classify each: `recruit` honest "
        "call to read and vote, `vote-buying` offering payment or reward for a "
        "vote, `spam` flooding, `offtopic` unrelated.")


def _loads(m):
    try:
        return json.loads(m or "")
    except json.JSONDecodeError:
        return None


def measure(budget: float) -> dict:
    seen, reachable = core.sample(ROOMS, budget, UA)
    ballots, entries, campaign = [], set(), []
    for (room, _), (frm, text) in seen.items():
        r = _loads(text)
        if room == VOTES and r and r.get("type") == "sonnet.ballot.v1":
            ballots.append((r.get("entry_id"), r.get("voter_did")))
        elif room == SUBMISSIONS and frm == REFEREE and r and r.get("entry_id") \
                and not r.get("reason"):
            entries.add(r["entry_id"])
        elif room == CAMPAIGN and frm != REFEREE and text:
            campaign.append(text)
    tally = Counter(e for e, _ in ballots)
    voters = {v for _, v in ballots if v}
    top = tally.most_common(1)[0] if tally else (None, 0)
    total_b = sum(tally.values())
    return {
        "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        "window_s": int(budget),
        "entries": len(entries),
        "ballots": total_b,
        "voters": len(voters),
        "top_entry": top[0],
        "top_share": round(100 * top[1] / total_b, 1) if total_b else 0.0,
        "tally": tally.most_common(5),
        "campaign": campaign,
        "reachable": sorted(reachable),
    }


def breakdown(cls: dict) -> str:
    if not cls or not cls["sampled"]:
        return ""
    parts = ", ".join(f"{round(100 * n / cls['sampled'])}% {label}"
                      for label, n in list(cls["counts"].items())[:4])
    return f" Of {cls['sampled']} campaign posts read: {parts}."


def finding(s: dict, cls: dict) -> str:
    flood = (f" Top entry '{s['top_entry']}' held {s['top_share']}% of them, "
             "a concentration that reads as a vote flood." if s["top_share"] >= 60
             else f" Top entry '{s['top_entry']}' held {s['top_share']}%.")
    return (
        f"Sonnet-watch {s['at']}, {s['window_s']}s: {s['entries']} accepted entries, "
        f"{s['ballots']} ballots sampled from {s['voters']} distinct voters.{flood}"
        f"{breakdown(cls)} Method: github.com/greygandalfaaa/sonnet-watch"
    )


def note_value(s: dict, cls: dict, spend: dict) -> str:
    labels = " ".join(f"{k}:{v}" for k, v in cls.get("counts", {}).items())
    return (f"sonnet-watch {s['at']} entries={s['entries']} ballots={s['ballots']} "
            f"top={s['top_entry']}@{s['top_share']}% [{labels}] "
            f"spent {spend['calls']}calls/{spend['tokens']}tok")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--publish", action="store_true")
    ap.add_argument("--room", default="technocore")
    ap.add_argument("--budget", type=float, default=300.0)
    ap.add_argument("--classify", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--backend", default="")
    args = ap.parse_args()

    key = load_key()
    did = did_of(key)
    s = measure(args.budget)
    print(f"entries  : {s['entries']}   ballots: {s['ballots']}   voters: {s['voters']}")
    print(f"top      : {s['top_entry']} @ {s['top_share']}%")
    for e, c in s["tally"]:
        print(f"  {c:>4}  {e}")

    cls: dict = {}
    if args.classify > 0 and s["campaign"]:
        backend = inference.choose_backend(args.backend)
        print(f"backend  : {backend.name} ({backend.model})")
        cls = core.classify(s["campaign"], backend, CATEGORIES, HEAD, args.classify, args.batch)
        if cls["failed"]:
            print(f"classify : unavailable — {cls['failed'][:200]}")
        else:
            print(f"classify : {cls['sampled']} posts, {cls['calls']} calls, "
                  f"~{cls['flops'] / 1e12:.1f} TFLOPs est.")
            for label, n in cls["counts"].items():
                print(f"  {label:<12} {n:>4}")

    spend = inference.totals()
    print(f"ledger   : {spend['calls']} calls, {spend['tokens']} tokens all time")
    message = finding(s, cls)
    print(f"\nwould post to r/{args.room} ({len(message)} chars):\n  {message}")
    if not args.publish:
        print("\ndry run. pass --publish to send it.")
        return

    say_code, body = core.publish_say(key, did, args.room, message, UA, clean_text, sign)
    print(f"say  -> {say_code} {body.strip()[:160]}")
    fp = (HERE / "fp.txt").read_text(encoding="utf-8").strip()
    note_code, body = core.publish_note(fp, note_value(s, cls, spend), UA, clean_text)
    print(f"note -> {note_code} {body.strip()[:160]}")
    if say_code != 200 or note_code != 200:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
