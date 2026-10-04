"""Experiment grid for the BiLSTM/BiGRU approach.

    python src/bilstm_experiments.py --shard 0/2   # run in two terminals
    python src/bilstm_experiments.py --shard 1/2

Runs whose metrics.json already exists are skipped, so the grid is resumable.

Phase 1 (standard text, seed 42) - design choices, selected on validation macro-F1:
    embeddings: random-init vs GloVe-Twitter (fine-tuned) vs GloVe (frozen)
    imbalance:  no class weights vs sqrt-balanced vs fully balanced
    recurrent cell: LSTM vs GRU
Phase 2 (keyword-masked text) - can the models classify from context alone?
Phase 3 - extra seeds for the selected configurations, to measure run-to-run variance
          (the hold-out has only ~20 tweets in the two rarest classes).
"""

import argparse
import os

from bilstm_gru import RESULTS, run

EXPERIMENTS = [
    # phase 1
    dict(rnn="lstm", emb="random", weights="sqrt"),
    dict(rnn="lstm", emb="glove", weights="sqrt"),
    dict(rnn="lstm", emb="glove", weights="sqrt", trainable_emb=False),
    dict(rnn="lstm", emb="glove", weights="none"),
    dict(rnn="lstm", emb="glove", weights="balanced"),
    dict(rnn="gru", emb="glove", weights="sqrt"),
    # phase 2
    dict(rnn="lstm", emb="random", weights="sqrt", masked=True),
    dict(rnn="lstm", emb="glove", weights="sqrt", masked=True),
    dict(rnn="gru", emb="glove", weights="sqrt", masked=True),
    # phase 3
    *[dict(rnn=r, emb="glove", weights="sqrt", masked=m, seed=s)
      for s in (1, 2) for m in (False, True) for r in ("lstm", "gru")],
]


def run_name(cfg: dict) -> str:
    c = {"trainable_emb": True, "masked": False, "seed": 42, "units": 128, **cfg}
    return (f"{'masked_' if c['masked'] else ''}bi{c['rnn']}_{c['emb']}"
            f"{'' if c['trainable_emb'] else '-frozen'}_w-{c['weights']}_u{c['units']}_s{c['seed']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--shard", default="0/1", help="i/n: run every n-th experiment starting at i")
    a = p.parse_args()
    i, n = map(int, a.shard.split("/"))
    for k, cfg in enumerate(EXPERIMENTS):
        name = run_name(cfg)
        if k % n != i or (RESULTS / "runs" / name / "metrics.json").exists():
            continue
        print(f"\n=== [{k + 1}/{len(EXPERIMENTS)}] {name} ===", flush=True)
        run(**cfg, name=name)
