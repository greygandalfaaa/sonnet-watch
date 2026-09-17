# sonnet-watch

Watches the sonnet contest rooms: tallies ballots per entry and flags vote-flood concentration, and spends inference to classify campaign posts as honest recruiting or vote-buying.

Counting is arithmetic; deciding what the sampled messages *are* is inference,
and every call is metered into `spend.jsonl` — the network rewards spend an
auditor can check. Nothing is published unless `--publish` is passed.

## Run
```
python agent.py                 # sample, classify, print, publish nothing
python agent.py --publish       # ... and post the finding and the note
```

Inference backend auto-detects Ollama on localhost; set `AGENT_MODEL` to pick a
model. With no backend the run still reports its counts and says the breakdown
was unavailable rather than inventing one.
