"""Notebook helpers used by ``notebooks/progressive_cramming_demo.ipynb``.

The gallery sections of the demo notebook need three small functions for the
embedding-to-text round trip + a coloured token-level diff. Keeping them here
(rather than as a notebook cell) means:

- The notebook stays nearly free of utility code -- only what the reader has to
  understand inline lives in cells.
- We can update visuals (e.g. tweak diff colours, switch to raw-id comparison)
  without re-shipping the notebook itself.

The helpers are deliberately minimal: they assume an already-loaded frozen
``model`` + ``tokenizer`` (cell 4 in the notebook) and a row from the demo
gallery dataset (the schema produced by ``scripts/build_demo_gallery.py``).
"""

from __future__ import annotations

import html

import torch
from IPython.display import HTML, display

from ._core import reconstruct_text


def emb_from_row(row) -> torch.Tensor:
    """Dataset row -> compression embedding tensor ``[n_cram, hidden]`` (float32).

    The embedding is stored in the Hub dataset as a 2D nested list of float32;
    ``torch.tensor`` rebuilds the correct shape without extra reshaping.
    """
    return torch.tensor(row["embedding"], dtype=torch.float32)


def _strip_bos(ids, tokenizer) -> list[int]:
    """Drop the BOS token from a generated/ground-truth id sequence if present.

    Llama-3.2 tokenizes with a leading BOS; SmolLM2 doesn't. ``strip_bos`` is a
    no-op for tokenizers without a BOS id, so callers can apply it unconditionally.
    """
    bos = tokenizer.bos_token_id
    ids = list(ids)
    if bos is not None and ids and ids[0] == bos:
        ids = ids[1:]
    return ids


def render_token_diff(gt_ids, gen_ids, title: str = "", *, tokenizer) -> str:
    """Build an HTML string colouring each generated token by whether it matches GT.

    Green = matches the original at that index, red = mismatches. Newlines are
    rendered as the visible glyph ⏎ so multi-line spans don't get smushed in the
    HTML output.
    """
    g = _strip_bos(gt_ids, tokenizer)
    h = list(gen_ids)
    spans = []
    for i in range(max(len(g), len(h))):
        a = g[i] if i < len(g) else None
        b = h[i] if i < len(h) else None
        piece = tokenizer.decode([b]) if b is not None else ""
        disp = html.escape(piece).replace("\n", "⏎")
        color = "#d6f5d6" if (a is not None and b is not None and a == b) else "#f8d0d0"
        spans.append(
            f'<span style="background:{color};padding:1px 2px;border-radius:2px">{disp}</span>'
        )
    return (
        f'<div style="font-family:monospace;line-height:1.9;margin:6px 0">'
        f'<b>{html.escape(title)}</b><br>{"".join(spans)}</div>'
    )


def reconstruct_and_show(row, label: str, *, model, tokenizer) -> None:
    """Greedy-decode from a row's embedding and render the coloured diff inline.

    This is the operation invoked by every "▶ Reconstruct" / "▶ Run side-by-side"
    button in the demo notebook. ``model`` + ``tokenizer`` are passed explicitly
    so the same helper works for both the Llama-1B section (gallery + Llama
    side-by-side pair) and the SmolLM2-360M section (the second side-by-side pair).
    """
    gen = reconstruct_text(
        model, tokenizer, emb_from_row(row),
        max_new_tokens=row["num_tokens"] + 4,
    )
    gen_ids = tokenizer(gen, add_special_tokens=False)["input_ids"]
    display(HTML(render_token_diff(row["input_ids"], gen_ids, label, tokenizer=tokenizer)))


# ─────────────────────────────────────────────────────────────────────────────
# High-level widgets — what notebook cells §3 and §4 ultimately call.
# ─────────────────────────────────────────────────────────────────────────────


def _card(*children):
    """Standard card box used by both §3 and §4."""
    import ipywidgets as widgets
    return widgets.VBox(
        children,
        layout=widgets.Layout(border="1px solid #ddd", padding="10px", margin="6px 0"),
    )


def display_gallery(gallery_repo_id: str, *, model, tokenizer) -> None:
    """Render the §3 gallery: one card per ``kind=="gallery"`` row, each with a
    ▶ Reconstruct button that greedy-decodes from the saved compression embedding.

    The frozen ``model``/``tokenizer`` must match every row's ``model_checkpoint``;
    the gallery rows are all single-model (Llama-3.2-1B in the canonical demo).
    """
    import html as _html

    import ipywidgets as widgets
    from datasets import load_dataset

    rows = [r for r in load_dataset(gallery_repo_id, split="train") if r["kind"] == "gallery"]

    def make_card(row):
        out = widgets.Output()
        btn = widgets.Button(description="▶ Reconstruct", button_style="primary")
        header = widgets.HTML(
            f"<b>{_html.escape(row['domain'])}</b> &mdash; {_html.escape(row['title'])}"
            f"<br><span style='color:#666;font-family:monospace'>"
            f"{_html.escape(row['text'][:140])}</span>"
        )

        def on_click(_):
            out.clear_output()
            with out:
                print("Reconstructing from 1 memory embedding...")
                reconstruct_and_show(
                    row, "Reconstruction (green = exact match)",
                    model=model, tokenizer=tokenizer,
                )
                print(
                    f"reconstruction={row['final_convergence']:.3f}  |  "
                    f"info gain={row['information_gain_bits']:.0f} bits  |  "
                    f"{row['num_tokens']} tokens → {row['n_cram']} embedding"
                )

        btn.on_click(on_click)
        return _card(header, btn, out)

    display(widgets.VBox([make_card(r) for r in rows]))


def display_side_by_side(
    gallery_repo_id: str,
    *,
    default_model,
    default_tokenizer,
    dtype: str = "float16",
) -> None:
    """Render the §4 side-by-side: one card per (model_checkpoint, sample) pair.

    Each card pulls the TC + PC rows for one model, hides them behind a ▶ button,
    and greedy-decodes both on click. The first card uses ``default_model`` /
    ``default_tokenizer`` (already loaded for §3); cards on a different checkpoint
    *lazy*-load their model the first time their button is clicked, and cache it
    for subsequent clicks. ``dtype`` is the precision the extra model is loaded
    in (default ``float16`` — matches what T4 Colab uses for the §3 model).
    """
    import html as _html
    from collections import defaultdict

    import ipywidgets as widgets
    from datasets import load_dataset

    from ._core import load_frozen_model

    pairs_ds = [r for r in load_dataset(gallery_repo_id, split="train") if r["kind"] == "tc_pc"]
    pairs_by_model: dict[str, dict] = defaultdict(dict)
    for r in pairs_ds:
        pairs_by_model[r["model_checkpoint"]][r["method"]] = r

    default_ckpt = next(iter(pairs_by_model)) if default_model is None else None
    # Cache of loaded frozen models, keyed by checkpoint.
    cache: dict[str, tuple] = {}
    if default_model is not None:
        # Trust the caller: default_model is the model loaded in cell 4.
        cache[_model_checkpoint_of(default_model)] = (default_model, default_tokenizer)

    def get_model(ckpt: str):
        if ckpt in cache:
            return cache[ckpt]
        print(f"Loading frozen model: {ckpt}  (dtype={dtype}) ...")
        cache[ckpt] = load_frozen_model(ckpt, dtype=dtype)
        return cache[ckpt]

    def make_card(ckpt: str, pair: dict):
        tc_row = pair["full_cramming"]
        pc_row = pair["progressive_cramming"]
        short_ckpt = ckpt.split("/", 1)[-1]
        out = widgets.Output()
        btn = widgets.Button(
            description=f"▶ Run side-by-side ({short_ckpt})",
            button_style="primary",
        )
        header = widgets.HTML(
            f"<b>Model:</b> <code>{_html.escape(ckpt)}</code> "
            f"&middot; <b>{tc_row['num_tokens']}-token span</b> "
            f"&middot; PC horizon: <b>{pc_row['horizon']}/{pc_row['num_tokens']}</b>"
        )

        def on_click(_):
            out.clear_output()
            with out:
                m, t = get_model(ckpt)
                display(HTML(
                    f"<b>Original ({_html.escape(tc_row['domain'])}):</b> "
                    f"<span style='font-family:monospace'>"
                    f"{_html.escape(tc_row['text'])}</span>"
                ))
                reconstruct_and_show(
                    tc_row, "Total cramming — whole span at once",
                    model=m, tokenizer=t,
                )
                reconstruct_and_show(
                    pc_row, "Progressive cramming — grown to the horizon",
                    model=m, tokenizer=t,
                )
                display(HTML(
                    f"<small>TC reconstruction (teacher-forced): "
                    f"<b>{tc_row['final_convergence']:.3f}</b> &middot; "
                    f"PC steps to converge: <b>{pc_row['steps_taken']}</b></small>"
                ))

        btn.on_click(on_click)
        return _card(header, btn, out)

    display(widgets.VBox([make_card(ckpt, p) for ckpt, p in pairs_by_model.items()]))


def _model_checkpoint_of(model) -> str:
    """Best-effort recovery of the checkpoint string a frozen model was loaded from."""
    name_or_path = getattr(getattr(model, "config", None), "_name_or_path", None)
    return name_or_path or ""
