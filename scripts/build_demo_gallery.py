#!/usr/bin/env python
"""Build (and optionally upload) the demo notebook's pre-computed embedding gallery.

The public demo notebook (Colab) does NOT cram the gallery examples live -- it loads
already-converged compression embeddings from a Hugging Face dataset and only runs the
cheap *reconstruction* (greedy decode) in the browser session. This script produces that
dataset:

  * 5 gallery examples across domains (literature / code / news / poetry / science),
    each compressed with **progressive cramming** -- the paper's contribution -- into
    a single memory embedding (one ``▶ Reconstruct`` per row). PC gives the same
    exact reconstruction TC would on these short spans, but the saved embedding sits
    at PC's measured compression horizon, matching how the paper characterises the
    method everywhere else.
  * 1 TC-vs-PC pair on one longer text: a full-cramming ("total cramming") embedding
    and a progressive-cramming run, for the side-by-side section. TC here is trained
    with ``convergence_threshold=0.99`` -- the paper's nominal protocol -- so it
    stops just short of perfect reconstruction and the residual error visibly
    cascades under greedy decoding.

Schema (one row per embedding)::

    kind            "gallery" | "tc_pc"
    domain          domain label (gallery) or pair tag (tc_pc)
    title           short human label
    method          "full_cramming" | "progressive_cramming"
    text            the (decoded) crammed span
    input_ids       token ids of the span
    embedding       the compression embedding, [n_cram, hidden] (float32, as nested lists)
    n_cram          number of memory embeddings (compression tokens)
    num_tokens      number of text tokens crammed
    horizon         progressive compression horizon in tokens (PC only, else null)
    final_convergence / information_gain_bits / steps_taken / elapsed_s
    training_config JSON string describing how the embedding was produced

Run on a GPU machine (e.g. Colab) ONCE; the demo notebook then reads the result::

    huggingface-cli login          # or set HF_TOKEN
    python scripts/build_demo_gallery.py --repo_id <user>/progressive_cramming_demo_gallery --push

Inspect locally without uploading::

    python scripts/build_demo_gallery.py --out_dir runs/demo_gallery
"""

from __future__ import annotations

import argparse
import json
import os

import torch
from datasets import Dataset

from progressive_cramming.demo import (
    DEFAULT_MODEL_CHECKPOINT,
    cram_text,
    load_frozen_model,
    progressive_cram_text,
)

# Five spans across distinct domains, ~60-90 tokens each -- a substantial paragraph,
# yet kept with a strong margin below the model's compression horizon so every gallery
# row reaches *exact* (1.0) reconstruction and decodes back cleanly on a small model.
# (Paper Tables 2-3: progressive horizons are ~335 for SmolLM2-1.7B / ~430 for
# Pythia-1.4B at full budget; a 16-layer Llama-3.2-1B sits lower, so ~70 tokens is a
# comfortable, reliable target in the demo's reduced step budget.)
GALLERY: list[dict] = [
    {
        "domain": "literature",
        "title": "PG19 — In the footsteps of King Arthur",
        "text": (
            "This desire had existed ever since, at five years old, I made acquaintance "
            "with Jack the Giantkiller, and afterwards, at fifteen or so, fell in love "
            "with my life's one hero, King Arthur. Between these two illustrious "
            "Cornishmen there exists more similarity than at first appears."
        ),
    },
    {
        "domain": "code",
        "title": "Python: quicksort",
        "text": (
            "def quicksort(items):\n"
            "    if len(items) <= 1:\n"
            "        return items\n"
            "    pivot = items[len(items) // 2]\n"
            "    left = [x for x in items if x < pivot]\n"
            "    middle = [x for x in items if x == pivot]\n"
            "    right = [x for x in items if x > pivot]\n"
            "    return quicksort(left) + middle + quicksort(right)\n"
        ),
    },
    {
        "domain": "news",
        "title": "Newswire report",
        "text": (
            "Researchers reported on Tuesday that a single learned vector can store an "
            "entire paragraph of text, which a frozen language model then reconstructs "
            "token by token. The team said the method needs no retraining and could "
            "shed light on how transformers pack information into their representations."
        ),
    },
    {
        "domain": "poetry",
        "title": "Frost — The Road Not Taken (final stanza)",
        "text": (
            "I shall be telling this with a sigh\n"
            "Somewhere ages and ages hence:\n"
            "Two roads diverged in a wood, and I—\n"
            "I took the one less traveled by,\n"
            "And that has made all the difference."
        ),
    },
    {
        "domain": "science",
        "title": "The second law of thermodynamics",
        "text": (
            "The second law of thermodynamics states that the entropy of an isolated "
            "system never decreases over time. It remains constant only in an idealized "
            "reversible process and increases in every real, irreversible one, which "
            "sets a fundamental direction for the flow of time."
        ),
    },
]

# A deliberately longer passage (~140 tokens) for the side-by-side TC-vs-PC pair: long
# enough that *total* cramming struggles to reach exact reconstruction (so greedy
# decoding visibly drifts -- the paper's brittleness finding, Table 1), while
# *progressive* cramming still pins a clean compression horizon and decodes that prefix
# exactly. Its theme doubles as a one-paragraph explainer of the method.
PAIR_TEXT = (
    "A language model processes text as a sequence of tokens, each mapped to a "
    "high-dimensional vector before it enters the network. Cramming asks a surprising "
    "question: how much of a passage can be packed into just one such vector? By "
    "freezing the model and optimizing a single embedding until the network reconstructs "
    "the original tokens, researchers found that one vector can hold hundreds of words. "
    "Yet this reconstruction is fragile. Under greedy decoding a single early mistake "
    "cascades, and the recovered text drifts away from the original, revealing that "
    "perfect reconstruction is brittle steering rather than genuine understanding."
)
PAIR_DOMAIN = "explainer"
PAIR_TITLE = "What is cramming? (long passage)"


def _embedding_to_list(embedding: torch.Tensor) -> list:
    """[n_cram, hidden] float32 tensor -> nested python lists for the dataset."""
    return embedding.to(torch.float32).cpu().numpy().tolist()


def build_gallery_rows(model, tokenizer, args) -> list[dict]:
    """Progressive-cram each gallery example into one memory embedding; return one
    row per example. The saved embedding is the converged solution at PC's
    measured horizon -- ideally the full span (every gallery text is well within
    Llama-3.2-1B's compression capacity), so greedy decoding reproduces the
    original tokens exactly."""
    rows: list[dict] = []
    for i, item in enumerate(GALLERY):
        print(f"\n[gallery {i+1}/{len(GALLERY)}] {item['domain']}: {item['title']}")
        result = progressive_cram_text(
            model,
            tokenizer,
            item["text"],
            num_mem_tokens=args.num_mem_tokens,
            max_seq_len=args.gallery_max_seq_len,
            step=args.pc_step,
            max_steps_per_token=args.pc_max_steps_per_token,
            learning_rate=args.learning_rate,
            init_method=args.init_method,
            seed=args.seed,
        )
        final_conv = result.stages[-1].final_convergence if result.stages else 0.0
        info_gain = result.stages[-1].information_gain_bits if result.stages else 0.0
        print(
            f"  PC: horizon={result.horizon}/{result.num_tokens} tokens over "
            f"{len(result.stages)} stages ({result.total_steps} steps, "
            f"{result.elapsed_s:.1f}s) reconstruction={final_conv:.3f} "
            f"info_gain={info_gain:.1f} bits"
        )
        rows.append(
            {
                "kind": "gallery",
                "domain": item["domain"],
                "title": item["title"],
                "method": "progressive_cramming",
                "text": result.text,
                "input_ids": result.input_ids,
                "embedding": _embedding_to_list(result.embedding),
                "n_cram": result.num_mem_tokens,
                "num_tokens": result.num_tokens,
                "horizon": result.horizon,
                "final_convergence": final_conv,
                "information_gain_bits": info_gain,
                "steps_taken": result.total_steps,
                "elapsed_s": result.elapsed_s,
                "training_config": json.dumps(result.training_config()),
            }
        )
    return rows


def build_tc_pc_rows(model, tokenizer, args) -> list[dict]:
    """Build the TC (full) and PC (progressive) rows for one shared, longer passage."""
    text = PAIR_TEXT
    print(f"\n[tc_pc] using the '{PAIR_DOMAIN}' long passage for the side-by-side pair")

    print("[tc_pc] total cramming (full)...")
    tc = cram_text(
        model,
        tokenizer,
        text,
        num_mem_tokens=args.num_mem_tokens,
        max_seq_len=args.pair_max_seq_len,
        learning_rate=args.learning_rate,
        max_steps=args.pair_max_steps,
        init_method=args.init_method,
        seed=args.seed,
        # Stop at the paper's nominal 0.99 protocol. On the 160-token pair this
        # permits ~1 residual error (159/160 = 0.994 satisfies; 158/160 = 0.988
        # does not), which guarantees an early-position argmax flip and the
        # autoregressive cascade that the side-by-side section is meant to show.
        convergence_threshold=args.tc_convergence_threshold,
    )
    print(
        f"  TC: tokens={tc.num_tokens} reconstruction={tc.final_convergence:.3f} "
        f"steps={tc.steps_taken} ({tc.elapsed_s:.1f}s)"
    )

    print("[tc_pc] progressive cramming...")
    pc = progressive_cram_text(
        model,
        tokenizer,
        text,
        num_mem_tokens=args.num_mem_tokens,
        max_seq_len=args.pair_max_seq_len,
        step=args.pc_step,
        max_steps_per_token=args.pc_max_steps_per_token,
        learning_rate=args.learning_rate,
        init_method=args.init_method,
        seed=args.seed,
    )
    print(
        f"  PC: horizon={pc.horizon}/{pc.num_tokens} tokens over {len(pc.stages)} stages "
        f"({pc.total_steps} steps, {pc.elapsed_s:.1f}s)"
    )

    tc_row = {
        "kind": "tc_pc",
        "domain": PAIR_DOMAIN,
        "title": f"{PAIR_TITLE} — total cramming",
        "method": "full_cramming",
        "text": tc.text,
        "input_ids": tc.input_ids,
        "embedding": _embedding_to_list(tc.embedding),
        "n_cram": tc.num_mem_tokens,
        "num_tokens": tc.num_tokens,
        "horizon": None,
        "final_convergence": tc.final_convergence,
        "information_gain_bits": tc.information_gain_bits,
        "steps_taken": tc.steps_taken,
        "elapsed_s": tc.elapsed_s,
        "training_config": json.dumps(tc.training_config()),
    }
    pc_row = {
        "kind": "tc_pc",
        "domain": PAIR_DOMAIN,
        "title": f"{PAIR_TITLE} — progressive cramming",
        "method": "progressive_cramming",
        "text": pc.text,
        "input_ids": pc.input_ids,
        "embedding": _embedding_to_list(pc.embedding),
        "n_cram": pc.num_mem_tokens,
        "num_tokens": pc.num_tokens,
        "horizon": pc.horizon,
        "final_convergence": (pc.stages[-1].final_convergence if pc.stages else 0.0),
        "information_gain_bits": (pc.stages[-1].information_gain_bits if pc.stages else 0.0),
        "steps_taken": pc.total_steps,
        "elapsed_s": pc.elapsed_s,
        "training_config": json.dumps(pc.training_config()),
    }
    return [tc_row, pc_row]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL_CHECKPOINT)
    ap.add_argument("--dtype", default="float16", help="float16 (T4) / bfloat16 (Ampere+) / float32")
    ap.add_argument("--gallery_max_seq_len", type=int, default=96, help="Max tokens crammed per gallery span.")
    ap.add_argument("--num_mem_tokens", type=int, default=1)
    ap.add_argument(
        "--pc_step",
        type=int,
        default=8,
        help="Progressive prefix growth per stage (tokens). Used for both the gallery PC runs and the PC side of the side-by-side pair.",
    )
    ap.add_argument(
        "--pc_max_steps_per_token",
        type=int,
        default=600,
        help="Per-stage optimisation budget for progressive cramming (gallery + pair).",
    )
    ap.add_argument(
        "--pair_max_seq_len",
        type=int,
        default=160,
        help="Token cap for the longer TC-vs-PC passage (intentionally beyond easy full-cramming capacity).",
    )
    ap.add_argument(
        "--pair_max_steps",
        type=int,
        default=10000,
        help="Full-cramming step ceiling for the TC side of the pair.",
    )
    ap.add_argument(
        "--tc_convergence_threshold",
        type=float,
        default=0.99,
        help="TC stop threshold for the side-by-side pair. Default 0.99 mirrors the paper's nominal protocol and on the 160-token passage permits ~1 residual error -- enough to make autoregressive drift visible. Set to 1.0 to demand exact reconstruction (and risk identical TC/PC output on short spans).",
    )
    ap.add_argument("--learning_rate", type=float, default=0.1)
    ap.add_argument("--init_method", default="random0.02")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="runs/demo_gallery", help="Where to save the dataset locally.")
    ap.add_argument("--repo_id", default=None, help="HF Hub dataset id to push to (with --push).")
    ap.add_argument("--push", action="store_true", help="Push the dataset to the Hub (needs --repo_id + auth).")
    ap.add_argument("--private", action="store_true", help="Create the Hub dataset as private.")
    args = ap.parse_args()

    if args.push and not args.repo_id:
        ap.error("--push requires --repo_id")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | model: {args.model} | dtype: {args.dtype}")
    model, tokenizer = load_frozen_model(args.model, dtype=args.dtype, device=device)

    rows = build_gallery_rows(model, tokenizer, args)
    rows.extend(build_tc_pc_rows(model, tokenizer, args))

    dataset = Dataset.from_list(rows)
    os.makedirs(args.out_dir, exist_ok=True)
    dataset.save_to_disk(args.out_dir)
    print(f"\nSaved {len(dataset)} rows to {args.out_dir}")

    if args.push:
        print(f"Pushing to https://huggingface.co/datasets/{args.repo_id} ...")
        dataset.push_to_hub(args.repo_id, private=args.private)
        print("Done.")
    else:
        print("Not pushing (pass --push --repo_id <id> to upload).")


if __name__ == "__main__":
    main()
