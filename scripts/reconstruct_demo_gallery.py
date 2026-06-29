#!/usr/bin/env python
"""Minimal end-to-end sanity check for the demo gallery.

Loads the frozen model + tokenizer, pulls the pre-computed embedding gallery
from the Hub, and greedily decodes every row from its compression embedding
(no other input). Prints the reconstruction next to the original with a
token-by-token ANSI diff so you can see at a glance whether the round-trip
holds. This is the same operation the demo notebook does on each ▶ Reconstruct
click -- collapsed into a script so you can verify the Hub artifact before
opening Colab.

Usage:
    python scripts/reconstruct_demo_gallery.py
    python scripts/reconstruct_demo_gallery.py --dtype bfloat16        # A100 / H100
    python scripts/reconstruct_demo_gallery.py --repo_id <user>/<name> # custom gallery
"""

from __future__ import annotations

import argparse

import torch
from datasets import load_dataset

from progressive_cramming.demo import load_frozen_model, reconstruct_text

DEFAULT_REPO = "LarryLovestein/progressive_cramming_demo_gallery"
DEFAULT_MODEL = "unsloth/Llama-3.2-1B"

# ANSI colours -- match the notebook's green/red diff convention.
GREEN = "\033[32m"
RED = "\033[31m"
GRAY = "\033[90m"
RESET = "\033[0m"


def _strip_bos(ids: list[int], bos_id: int | None) -> list[int]:
    if bos_id is None or not ids or ids[0] != bos_id:
        return ids
    return ids[1:]


def _coloured_diff(gt_ids: list[int], gen_ids: list[int], tokenizer) -> str:
    """Render gen_ids token-by-token; colour each token by whether it matches gt at the same index."""
    parts: list[str] = []
    for i, tid in enumerate(gen_ids):
        piece = tokenizer.decode([tid]).replace("\n", "⏎")
        if i >= len(gt_ids):
            colour = GRAY  # past the original span -- free continuation
        elif tid == gt_ids[i]:
            colour = GREEN
        else:
            colour = RED
        parts.append(f"{colour}{piece}{RESET}")
    return "".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo_id", default=DEFAULT_REPO, help="HF dataset id to load.")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Frozen model checkpoint.")
    ap.add_argument(
        "--dtype", default="float16", choices=["float16", "bfloat16", "float32"],
        help="Inference dtype (float16 = T4/Colab; bfloat16 = A100/H100; float32 = CPU).",
    )
    ap.add_argument(
        "--extra_tokens", type=int, default=4,
        help="Decode `num_tokens + extra_tokens` per row so we also see the free continuation past the original.",
    )
    args = ap.parse_args()

    print(f"Loading frozen model: {args.model}  (dtype={args.dtype})")
    model, tokenizer = load_frozen_model(args.model, dtype=args.dtype)
    device = next(model.parameters()).device
    print(f"  device: {device}  hidden_size: {model.config.hidden_size}")

    print(f"\nLoading gallery: {args.repo_id}")
    ds = load_dataset(args.repo_id, split="train")
    print(f"  {len(ds)} rows\n")
    print("=" * 80)

    for r in ds:
        # Embedding is stored as a 2D nested list of float32; torch.tensor() gives [n_cram, hidden].
        emb = torch.tensor(r["embedding"], dtype=torch.float32)

        # Greedy decode from the embedding alone, no text input.
        gen_text = reconstruct_text(
            model, tokenizer, emb, max_new_tokens=r["num_tokens"] + args.extra_tokens
        )
        gen_ids = tokenizer(gen_text, add_special_tokens=False)["input_ids"]

        # Ground-truth ids: clip the saved input_ids to the actual span (PC rows
        # may store padding past num_tokens; TC rows store only the real span).
        gt_ids = _strip_bos(list(r["input_ids"][: r["num_tokens"]]), tokenizer.bos_token_id)

        n_compare = min(len(gt_ids), len(gen_ids))
        matches = sum(1 for i in range(n_compare) if gt_ids[i] == gen_ids[i])
        pct = 100.0 * matches / n_compare if n_compare else 0.0

        kind = r["kind"]
        method = r["method"]
        domain = r["domain"]
        title = r["title"]
        print(f"[{kind:7s} {method:22s} {domain:10s}]  {title}")
        print(
            f"  stored: tok={r['num_tokens']:3d}  horizon={r['horizon']}  "
            f"conv={r['final_convergence']:.3f}  steps={r['steps_taken']}"
        )
        print(f"  decode match first {n_compare} tokens: {matches}/{n_compare} ({pct:.1f}%)")
        print(f"  original: {r['text']!r}")
        print(f"  decoded : {_coloured_diff(gt_ids, gen_ids, tokenizer)}")
        print("-" * 80)


if __name__ == "__main__":
    main()
