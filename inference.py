"""The inference this agent buys, and the ledger that proves it.

Counting messages is arithmetic and needs no model. Saying what those messages
*are* does: the categories below cannot be decided by a regular expression,
because agents invent their own phrasing and the interesting cases are exactly
the ones no pattern anticipated. That is the work this module runs.

Backends are interchangeable on purpose. Today a local or hosted model serves
the request. On the Flop network the same call becomes a session request that
pays a miner to run it, so the swap is a backend, not a rewrite. `FlopBackend`
holds that shape now so the fields it needs are already being produced.

Every call is metered into `spend.jsonl`, one row each. The network rewards what
an agent spends on inference, and a spend claim nobody can audit is worth
nothing, so the ledger is written before the figure is ever published.
"""

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path

HERE = Path(__file__).parent
LEDGER = HERE / "spend.jsonl"

# A forward pass costs about two floating-point operations per parameter per
# token. It is an estimate, not a measurement, and it is labelled as one
# wherever it is published — but a Flop session request has to name the compute
# it wants in FLOPs, so the agent needs this number before the network exists.
FLOPS_PER_PARAM_PER_TOKEN = 2

# Parameter counts for models this agent is likely to be pointed at. An unknown
# model falls back to the default rather than guessing high and inflating every
# figure that follows.
PARAMS = {
    "llama3.2": 3_000_000_000,
    "llama3.1": 8_000_000_000,
    "qwen2.5": 7_000_000_000,
    "mistral": 7_000_000_000,
    "phi3": 3_800_000_000,
    "gemma2": 9_000_000_000,
}
DEFAULT_PARAMS = 7_000_000_000


@dataclass
class Usage:
    """What one inference call consumed."""

    at: str
    backend: str
    model: str
    items: int
    prompt_tokens: int
    completion_tokens: int
    flops: int
    cost: float
    unit: str


def params_for(model: str) -> int:
    base = model.split(":")[0].lower()
    for known, n in PARAMS.items():
        if base.startswith(known):
            return n
    return DEFAULT_PARAMS


def estimate_tokens(text: str) -> int:
    """Roughly four characters per token.

    The backends that report real token counts overwrite this; it exists so a
    backend that reports nothing still lands a defensible number in the ledger
    instead of a zero that would understate the spend.
    """
    return max(1, len(text) // 4)


def post_json(url: str, payload: dict, headers: dict, timeout: int = 120) -> dict:
    body = json.dumps(payload).encode("utf-8")
    hdrs = {"Content-Type": "application/json", **headers}
    req = urllib.request.Request(url, data=body, headers=hdrs, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as res:
        return json.load(res)


class Backend:
    """A place that will run a prompt and charge for it."""

    name = "none"
    model = "-"
    unit = "none"

    def available(self) -> bool:
        return False

    def run(self, prompt: str, max_tokens: int) -> tuple[str, Usage]:
        raise NotImplementedError

    def _usage(self, prompt: str, reply: str, items: int,
               prompt_tokens: int = 0, completion_tokens: int = 0,
               cost: float = 0.0) -> Usage:
        pt = prompt_tokens or estimate_tokens(prompt)
        ct = completion_tokens or estimate_tokens(reply)
        return Usage(
            at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            backend=self.name,
            model=self.model,
            items=items,
            prompt_tokens=pt,
            completion_tokens=ct,
            flops=FLOPS_PER_PARAM_PER_TOKEN * params_for(self.model) * (pt + ct),
            cost=cost,
            unit=self.unit,
        )


class OllamaBackend(Backend):
    """A model on a GPU you can reach over HTTP.

    This is the closest thing available to what Flop will do: the agent hands a
    job to a machine that owns the hardware and gets a completion back. Only the
    settlement is missing.
    """

    name = "ollama"
    unit = "local"

    def __init__(self, base: str, model: str) -> None:
        self.base = base.rstrip("/")
        self.model = model

    def available(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.base}/api/tags")
            with urllib.request.urlopen(req, timeout=4) as res:
                json.load(res)
            return True
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
            return False

    def run(self, prompt: str, max_tokens: int) -> tuple[str, Usage]:
        out = post_json(
            f"{self.base}/api/chat",
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                # Temperature zero because the same room should classify the
                # same way twice; a drifting label makes the published series
                # meaningless.
                "options": {"temperature": 0, "num_predict": max_tokens},
            },
            {},
        )
        reply = (out.get("message") or {}).get("content", "")
        return reply, self._usage(
            prompt, reply, items=0,
            prompt_tokens=out.get("prompt_eval_count", 0),
            completion_tokens=out.get("eval_count", 0),
        )


class OpenAICompatBackend(Backend):
    """Any endpoint that speaks `/chat/completions`."""

    name = "openai-compat"
    unit = "usd"

    def __init__(self, base: str, model: str, key: str,
                 usd_per_mtok_in: float, usd_per_mtok_out: float) -> None:
        self.base = base.rstrip("/")
        self.model = model
        self.key = key
        self.in_rate = usd_per_mtok_in
        self.out_rate = usd_per_mtok_out

    def available(self) -> bool:
        return bool(self.key and self.base)

    def run(self, prompt: str, max_tokens: int) -> tuple[str, Usage]:
        out = post_json(
            f"{self.base}/chat/completions",
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": max_tokens,
            },
            {"Authorization": f"Bearer {self.key}"},
        )
        reply = out["choices"][0]["message"]["content"]
        usage = out.get("usage") or {}
        pt = usage.get("prompt_tokens", 0)
        ct = usage.get("completion_tokens", 0)
        cost = (pt * self.in_rate + ct * self.out_rate) / 1_000_000
        return reply, self._usage(prompt, reply, 0, pt, ct, cost)


class FlopBackend(Backend):
    """The shape a Flop session request will take.

    Per the draft whitepaper a request pins five fields — model-weight hash, max
    latency, compute in FLOPs, a confidentiality flag, and the fee — a miner
    accepts and returns a proof of useful inference, and the agent settles in
    FLOP. `session_request` below builds exactly that, so the fields are already
    being computed against real jobs.

    It cannot run yet: Flop Testnet is Q4 2026, the Yellow Paper is not final,
    and no faucet, client or published weight hash exists. Refusing here is the
    honest answer; a stub that pretended to spend would put a false figure in
    the ledger, which is the one thing the ledger exists to prevent.
    """

    name = "flop"
    unit = "FLOP"

    def __init__(self, model: str, weight_hash: str = "",
                 max_latency_ms: int = 60_000, confidential: bool = False,
                 fee: float = 0.0) -> None:
        self.model = model
        self.weight_hash = weight_hash
        self.max_latency_ms = max_latency_ms
        self.confidential = confidential
        self.fee = fee

    def available(self) -> bool:
        return False

    def session_request(self, prompt: str, max_tokens: int) -> dict:
        tokens = estimate_tokens(prompt) + max_tokens
        return {
            "model_weight_hash": self.weight_hash,
            "max_latency_ms": self.max_latency_ms,
            "flops": FLOPS_PER_PARAM_PER_TOKEN * params_for(self.model) * tokens,
            "confidential": self.confidential,
            "fee": self.fee,
        }

    def run(self, prompt: str, max_tokens: int) -> tuple[str, Usage]:
        raise NotImplementedError(
            "Flop Testnet is Q4 2026: no mempool to post a session request to, "
            "no faucet, and no published model-weight hash. "
            f"The request this job would post: {self.session_request(prompt, max_tokens)}"
        )


def choose_backend(name: str = "") -> Backend:
    """Pick a backend from the environment, or say plainly that there is none.

    Falling back to a silent no-op would let a run report zero spend as though
    it were a real measurement, so an unavailable backend stays unavailable and
    the caller decides what to say about it.
    """
    name = (name or os.environ.get("AGENT_INFERENCE") or "auto").lower()
    model = os.environ.get("AGENT_MODEL", "llama3.2")

    if name in ("ollama", "auto"):
        b = OllamaBackend(os.environ.get("OLLAMA_HOST", "http://localhost:11434"), model)
        if b.available() or name == "ollama":
            return b
    if name in ("openai", "openai-compat", "auto"):
        b = OpenAICompatBackend(
            os.environ.get("AGENT_API_BASE", ""),
            model,
            os.environ.get("AGENT_API_KEY", ""),
            float(os.environ.get("AGENT_USD_PER_MTOK_IN", "0")),
            float(os.environ.get("AGENT_USD_PER_MTOK_OUT", "0")),
        )
        if b.available():
            return b
    if name == "flop":
        return FlopBackend(model, os.environ.get("FLOP_WEIGHT_HASH", ""))
    return Backend()


def record(usage: Usage, path: Path = LEDGER) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(usage)) + "\n")


def totals(path: Path = LEDGER) -> dict:
    """What this agent has spent on inference, all time."""
    calls = items = pt = ct = flops = 0
    cost = 0.0
    unit = "none"
    try:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    # A half-written final line after a killed run must not
                    # discard every row before it.
                    continue
                calls += 1
                items += row.get("items", 0)
                pt += row.get("prompt_tokens", 0)
                ct += row.get("completion_tokens", 0)
                flops += row.get("flops", 0)
                cost += row.get("cost", 0.0)
                unit = row.get("unit", unit)
    except FileNotFoundError:
        pass
    return {
        "calls": calls, "items": items, "tokens": pt + ct,
        "flops": flops, "cost": round(cost, 6), "unit": unit,
    }
