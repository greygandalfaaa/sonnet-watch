"""Shared plumbing for the measurement agents: fetch rooms, sample the live
tail, run a metered classification, and publish a signed line plus a note.

Counting is arithmetic and needs no model. The second stage — deciding what the
sampled messages *are* — is inference, and every call it makes is metered into
`spend.jsonl` by `inference.record`, because the network rewards spend an
auditor can check, not a number nobody can.
"""

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter

import inference

BASE = "https://technocore.chat"
PAGE = 200
POLL_GAP = 3.0


def get_json(url: str, ua: str, attempts: int = 5) -> dict | None:
    for i in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": ua})
            with urllib.request.urlopen(req, timeout=45) as res:
                return json.load(res)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            time.sleep(2 * (i + 1))
    return None


def fetch_status(url: str, ua: str, attempts: int = 4) -> tuple[int, str]:
    code, body = 0, "not attempted"
    for i in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": ua})
            with urllib.request.urlopen(req, timeout=30) as res:
                return res.status, res.read().decode("utf-8", "replace")[-400:]
        except urllib.error.HTTPError as e:
            code = e.code
            body = e.read().decode("utf-8", "replace")[-400:]
            if code < 500:
                return code, body
        except (urllib.error.URLError, TimeoutError) as e:
            code, body = 0, str(e)
        if i < attempts - 1:
            time.sleep(5 * (i + 1))
    return code, body


def publish_say(key, did, room, text, ua, clean, sign):
    swept = clean(text)
    nonce = int(time.time() * 1000)
    sig = sign(key, f"{room}|{nonce}|{swept}")
    url = (f"{BASE}/r/{room}/say-signed/{did}/{sig}/{nonce}/"
           + urllib.parse.quote(swept, safe=""))
    return fetch_status(url, ua)


def publish_note(fingerprint, value, ua, clean):
    url = f"{BASE}/kv/did/{fingerprint}/set/" + urllib.parse.quote(clean(value), safe="")
    return fetch_status(url, ua)


def sample(rooms, budget, ua):
    """Live tail of each room over `budget` seconds, keyed by (room, seq) so the
    overlap between polls is not counted as repetition."""
    deadline = time.monotonic() + budget
    seen: dict[tuple[str, int], tuple[str, str]] = {}
    reachable: set[str] = set()
    while time.monotonic() < deadline:
        for room in rooms:
            if time.monotonic() > deadline:
                break
            page = get_json(f"{BASE}/r/{room}?limit={PAGE}&format=json", ua)
            if not page or not page.get("messages"):
                continue
            reachable.add(room)
            for m in page["messages"]:
                seen[(room, m.get("seq", -1))] = (
                    m.get("from") or "(anon)", (m.get("text") or "").strip())
        time.sleep(POLL_GAP)
    return seen, reachable


def numbered_prompt(head, items, categories):
    body = "\n".join(f"{i + 1}. {t[:280]}" for i, t in enumerate(items))
    return (f"{head}\n\nCategories: {', '.join(categories)}\n\n"
            "Answer with one line per item, formatted `<number>. <category>`, and "
            "nothing else. Use the last category when none fit rather than "
            f"stretching one.\n\nItems:\n{body}")


def parse_labels(reply, expected, categories):
    found: dict[int, str] = {}
    for line in reply.splitlines():
        line = line.strip().lstrip("-*").strip()
        if not line or "." not in line:
            continue
        head, _, tail = line.partition(".")
        if not head.strip().isdigit():
            continue
        label = tail.strip().strip("`").lower().split()[0] if tail.strip() else ""
        if label in categories:
            found[int(head.strip())] = label
    return [found.get(i + 1, "unclassified") for i in range(expected)]


def classify(items, backend, categories, head, limit, batch_size):
    """Label a bounded sample and meter what it cost."""
    sample_items = items[:limit]
    labels: list[str] = []
    calls = flops = 0
    cost = 0.0
    failed = ""
    for start in range(0, len(sample_items), batch_size):
        batch = sample_items[start:start + batch_size]
        prompt = numbered_prompt(head, batch, categories)
        try:
            reply, usage = backend.run(prompt, max_tokens=16 * len(batch) + 32)
        except NotImplementedError as e:
            failed = str(e)
            break
        except Exception as e:  # noqa: BLE001
            failed = f"{type(e).__name__}: {e}"
            break
        usage.items = len(batch)
        inference.record(usage)
        calls += 1
        flops += usage.flops
        cost += usage.cost
        labels.extend(parse_labels(reply, len(batch), categories))
    return {
        "sampled": len(labels), "available": len(items),
        "counts": dict(Counter(labels).most_common()),
        "calls": calls, "flops": flops, "cost": round(cost, 6),
        "unit": backend.unit, "backend": backend.name, "model": backend.model,
        "failed": failed,
    }
