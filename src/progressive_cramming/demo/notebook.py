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
import logging
import warnings

import torch
from IPython.display import HTML, display

from ._core import reconstruct_text


# Suppress the "Ignoring clean_up_tokenization_spaces=True for BPE" message that
# transformers logs (not warns) on every tokenizer call. We always pass
# ``clean_up_tokenization_spaces=False`` ourselves, so the message is non-actionable
# noise. Logging filter is idempotent: re-importing the module just re-adds the
# same filter, no effect.
class _SuppressBPECleanupWarning(logging.Filter):
    def filter(self, record):
        return "clean_up_tokenization_spaces" not in record.getMessage()


logging.getLogger("transformers").addFilter(_SuppressBPECleanupWarning())
logging.getLogger("transformers.tokenization_utils_base").addFilter(_SuppressBPECleanupWarning())

# Token-diff colours -- foreground only, NOT background, so the diff stays
# readable on both light and dark Colab themes (background fills wash the
# foreground text in dark mode).
COLOR_MATCH = "#2ea043"     # GitHub-ish green; passes contrast on white and on #0d1117
COLOR_MISMATCH = "#d44a2c"  # HSE coral
COLOR_PAST_GT = "#888888"   # neutral grey for free continuation past the GT span


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

    Foreground-only (no background fills), so legible on dark Colab themes.
    Real newlines from the tokens are preserved verbatim and the container uses
    ``white-space: pre-wrap``, so multi-line code/poetry samples render across
    real lines instead of being smashed onto one line.
    """
    g = _strip_bos(gt_ids, tokenizer)
    h = list(gen_ids)
    spans = []
    for i in range(max(len(g), len(h))):
        a = g[i] if i < len(g) else None
        b = h[i] if i < len(h) else None
        piece = tokenizer.decode([b], clean_up_tokenization_spaces=False) if b is not None else ""
        if i >= len(g):
            color = COLOR_PAST_GT
        elif a is not None and b is not None and a == b:
            color = COLOR_MATCH
        else:
            color = COLOR_MISMATCH
        # html.escape but keep newlines so white-space:pre-wrap can break them.
        disp = html.escape(piece)
        spans.append(
            f'<span style="color:{color};font-weight:600">{disp}</span>'
        )
    body = "".join(spans)
    return (
        f'<div style="font-family:monospace;line-height:1.55;margin:6px 0;'
        f'white-space:pre-wrap;word-break:break-word">'
        f'<b style="color:inherit">{html.escape(title)}</b>\n{body}</div>'
    )


def reconstruct_and_show(row, label: str, *, model, tokenizer, extra_tokens: int = 0) -> None:
    """Greedy-decode from a row's embedding and render the coloured diff inline.

    Generates ``row["num_tokens"] + extra_tokens`` tokens. The base ``num_tokens``
    is the span the embedding was trained on; tokens past it appear in grey as
    "free continuation". ``model`` + ``tokenizer`` are passed explicitly so the
    same helper works across §3 / §4 / §5.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*")
        gen = reconstruct_text(
            model, tokenizer, emb_from_row(row),
            max_new_tokens=row["num_tokens"] + extra_tokens,
        )
        gen_ids = tokenizer(
            gen, add_special_tokens=False,
        )["input_ids"]
    # Clip GT to the actual trained span -- raw input_ids carry pad tokens out to
    # max_sequence_length, which would otherwise count as mismatches and turn
    # the "free continuation" tail red instead of grey.
    gt_ids = list(row["input_ids"][: row["num_tokens"]])
    display(HTML(render_token_diff(gt_ids, gen_ids, label, tokenizer=tokenizer)))


# ─────────────────────────────────────────────────────────────────────────────
# High-level widgets -- what notebook cells §3 and §4 ultimately call.
# ─────────────────────────────────────────────────────────────────────────────


def _card(*children):
    """Standard card box used by both §3 and §4."""
    import ipywidgets as widgets
    return widgets.VBox(
        children,
        layout=widgets.Layout(border="1px solid #555", padding="10px", margin="6px 0"),
    )


def _original_block(text: str) -> str:
    """Render the full original passage with newlines preserved and dark-mode-safe colours."""
    return (
        f'<div style="font-family:monospace;font-size:0.92em;color:#888;'
        f'white-space:pre-wrap;word-break:break-word;margin-top:4px">'
        f'{html.escape(text)}</div>'
    )


def display_gallery(rows, *, model, tokenizer) -> None:
    """Render the §3 gallery: one card per ``kind=="gallery"`` row, each with a
    ▶ Reconstruct button that greedy-decodes from the saved compression embedding.

    ``rows`` is a list of dataset rows (already filtered by ``kind == "gallery"``
    in the notebook -- keeps the dataset-loading logic visible to the reader).
    The frozen ``model`` / ``tokenizer`` must match every row's
    ``model_checkpoint`` (the canonical demo uses Llama-3.2-1B for all gallery
    rows).
    """
    import ipywidgets as widgets

    legend = (
        f'<span style="color:{COLOR_MATCH};font-weight:600">green = match</span>'
        f' &middot; '
        f'<span style="color:{COLOR_MISMATCH};font-weight:600">red = mismatch</span>'
        f' &middot; '
        f'<span style="color:{COLOR_PAST_GT};font-weight:600">grey = past the original span</span>'
    )

    def make_card(row):
        out = widgets.Output()
        btn = widgets.Button(description="▶ Reconstruct", button_style="primary")
        header = widgets.HTML(
            f"<b>{html.escape(row['domain'])}</b> &mdash; {html.escape(row['title'])}"
            f"{_original_block(row['text'])}"
        )

        def on_click(_):
            out.clear_output()
            with out:
                display(HTML(
                    f"<div style='font-size:0.9em;color:#888;margin-bottom:4px'>{legend}</div>"
                ))
                # Show ~10% extra tokens past the compression horizon so the reader
                # sees the model's free continuation past the trained span (in grey).
                extra = max(1, row["num_tokens"] // 5)
                reconstruct_and_show(
                    row, "Reconstruction",
                    model=model, tokenizer=tokenizer, extra_tokens=extra,
                )
                n_cram_label = (
                    "Single compression embedding"
                    if row["n_cram"] == 1
                    else f"{row['n_cram']} compression embeddings"
                )
                display(HTML(
                    f"<div style='margin-top:4px;font-size:0.9em;color:#888'>"
                    f"{n_cram_label} &rarr; {row['num_tokens']} reconstructed tokens</div>"
                ))

        btn.on_click(on_click)
        return _card(header, btn, out)

    display(widgets.VBox([make_card(r) for r in rows]))


def _passage_label(row) -> str:
    """Strip the trailing ' — total cramming' / ' — progressive cramming' suffix
    from a tc_pc row's title to get the underlying passage label."""
    title = row["title"]
    for tail in (" — total cramming", " — progressive cramming"):
        if title.endswith(tail):
            return title[: -len(tail)]
    return title


def display_side_by_side(rows, *, models: dict) -> None:
    """Render the §4 side-by-side: one card per (model_checkpoint, sample) pair.

    Visually mirrors :func:`display_gallery` (same card layout, same legend, same
    colour scheme). Each card pairs the TC + PC rows for one ``model_checkpoint``
    and runs them through ``reconstruct_and_show`` on click.

    ``models`` is a ``{checkpoint: (model, tokenizer)}`` mapping built by the
    notebook -- one entry per frozen model the gallery references. The function
    NEVER loads a model itself; if a pair's checkpoint is missing from ``models``
    we render a placeholder card instead of crashing or printing diagnostics.
    """
    from collections import defaultdict

    import ipywidgets as widgets

    pairs_by_model: dict[str, dict] = defaultdict(dict)
    for r in rows:
        pairs_by_model[r["model_checkpoint"]][r["method"]] = r

    legend = (
        f'<span style="color:{COLOR_MATCH};font-weight:600">green = match</span>'
        f' &middot; '
        f'<span style="color:{COLOR_MISMATCH};font-weight:600">red = mismatch</span>'
        f' &middot; '
        f'<span style="color:{COLOR_PAST_GT};font-weight:600">grey = past the original span</span>'
    )

    def make_card(ckpt: str, pair: dict):
        tc_row = pair["full_cramming"]
        pc_row = pair["progressive_cramming"]
        short_ckpt = ckpt.split("/", 1)[-1]
        out = widgets.Output()

        # Header mirrors §3: domain + passage title, then short meta line, then
        # the full original text as a monospace pre-wrap block.
        header = widgets.HTML(
            f"<b>{html.escape(tc_row['domain'])}</b> &mdash; "
            f"{html.escape(_passage_label(tc_row))}"
            f"<div style='font-size:0.85em;color:#888;margin-top:2px'>"
            f"<code>{html.escape(ckpt)}</code> &middot; "
            f"{tc_row['num_tokens']} tokens &middot; "
            f"PC horizon <b>{pc_row['horizon']}/{pc_row['num_tokens']}</b></div>"
            f"{_original_block(tc_row['text'])}"
        )

        if ckpt not in models:
            placeholder = widgets.HTML(
                f"<div style='color:{COLOR_MISMATCH};font-size:0.9em;margin-top:6px'>"
                f"Model <code>{html.escape(ckpt)}</code> not loaded. "
                f"Add it to <code>models=</code> in the cell above to enable this pair."
                f"</div>"
            )
            return _card(header, placeholder)

        m, t = models[ckpt]
        btn = widgets.Button(
            description=f"▶ Run side-by-side ({short_ckpt})",
            button_style="primary",
        )

        def on_click(_):
            out.clear_output()
            with out:
                display(HTML(
                    f"<div style='font-size:0.9em;color:#888;margin-bottom:4px'>{legend}</div>"
                ))
                # ~10% extra tokens past the trained span -> visible grey continuation.
                extra = max(1, tc_row["num_tokens"] // 5)
                reconstruct_and_show(
                    tc_row, "Total cramming — whole span at once",
                    model=m, tokenizer=t, extra_tokens=extra,
                )
                reconstruct_and_show(
                    pc_row, "Progressive cramming — grown to the horizon",
                    model=m, tokenizer=t, extra_tokens=extra,
                )
                display(HTML(
                    f"<div style='margin-top:4px;font-size:0.9em;color:#888'>"
                    f"Token Cramming teacher-forced accuracy="
                    f"<b>{tc_row['final_convergence']:.3f}</b></div>"
                ))

        btn.on_click(on_click)
        return _card(header, btn, out)

    display(widgets.VBox([make_card(ckpt, p) for ckpt, p in pairs_by_model.items()]))


# ─────────────────────────────────────────────────────────────────────────────
# §5 -- interactive progressive cramming with optimisation curves (live) +
# accuracy landscape on PC1-PC2 (after run completes)
# ─────────────────────────────────────────────────────────────────────────────


class _ProgressiveLiveCurves:
    """Accumulates loss / teacher-forced reconstruction during PC and redraws live.

    Only the cheap left-panel curves are drawn during optimisation. The heavier
    accuracy landscape is computed *after* PC converges (one forward pass per
    grid cell) and rendered separately.
    """

    def __init__(self, *, redraw_every: int = 20):
        self.steps: list[int] = []
        self.losses: list[float] = []
        self.convs: list[float] = []
        self.redraw_every = redraw_every
        self._since_redraw = 0

    def __call__(self, info: dict) -> None:
        self.steps.append(info["global_step"])
        self.losses.append(info["loss"])
        self.convs.append(info["convergence"])
        self._since_redraw += 1
        if self._since_redraw >= self.redraw_every:
            self._since_redraw = 0
            self.draw()

    def draw(self) -> None:
        import matplotlib.pyplot as plt
        from IPython.display import clear_output

        clear_output(wait=True)
        fig, ax = plt.subplots(1, 1, figsize=(9, 3.6))
        ax.plot(self.steps, self.losses, color=COLOR_MISMATCH, lw=1.2, label="loss")
        ax.set_xlabel("step")
        ax.set_ylabel("loss", color=COLOR_MISMATCH)
        ax.tick_params(axis="y", labelcolor=COLOR_MISMATCH)
        axb = ax.twinx()
        axb.plot(self.steps, self.convs, color=COLOR_MATCH, lw=1.2)
        axb.set_ylabel("reconstruction", color=COLOR_MATCH)
        axb.tick_params(axis="y", labelcolor=COLOR_MATCH)
        axb.set_ylim(-0.02, 1.02)
        ax.set_title("Optimisation (loss + teacher-forced reconstruction)")
        plt.tight_layout()
        plt.show()
        plt.close(fig)


@torch.no_grad()
def _accuracy_batch(model, *, compression_flat, mem_shape, input_ids, text_embeds,
                    attention_mask, batch_size: int = 8):
    """Teacher-forced match-rate of a batch of candidate compression embeddings against ``input_ids``.

    Mirrors the ``_compute_accuracy_batch`` helper used by the paper's
    ``visualize_landscale_2pca.py``: cat compression + text embeddings, forward,
    argmax over logits at the continuation positions, count matches.
    """
    import numpy as np

    model.eval()
    device = next(model.parameters()).device
    num_grid = int(compression_flat.shape[0])
    mem_tokens, hidden = int(mem_shape[0]), int(mem_shape[1])
    denom = attention_mask.sum(dim=-1).clamp_min(1).float()

    accs: list = []
    for batch_start in range(0, num_grid, batch_size):
        batch_end = min(batch_start + batch_size, num_grid)
        batch = compression_flat[batch_start:batch_end]
        bs = int(batch.shape[0])
        comp = batch.reshape(bs, mem_tokens, hidden).to(device=device, dtype=text_embeds.dtype)
        text_bs = text_embeds.expand(bs, -1, -1)
        inputs_embeds = torch.cat([comp, text_bs], dim=1)
        comp_attn = torch.ones((bs, mem_tokens), device=device, dtype=attention_mask.dtype)
        attn_bs = attention_mask.expand(bs, -1)
        full_attn = torch.cat([comp_attn, attn_bs], dim=1)
        out = model(inputs_embeds=inputs_embeds, attention_mask=full_attn, use_cache=False)
        pred = out.logits[:, mem_tokens - 1 : -1].argmax(dim=-1)  # [bs, L]
        match = (pred == input_ids.expand(bs, -1)).to(torch.float32)
        match = match * attn_bs.to(torch.float32)
        accs.append((match.sum(dim=-1) / denom.expand(bs)).cpu().numpy())
    return np.concatenate(accs, axis=0)


def _compute_landscape(result, model, *, grid_size: int = 24, padding: float = 0.35):
    """PCA the snapshot trajectory, build a mesh grid in PC1-PC2, project each grid
    point back to the embedding space, score teacher-forced accuracy against the
    converged horizon. Returns ``(XX, YY, ZZ, coords)`` or ``None`` if there is
    not enough trajectory to fit a PCA.
    """
    import numpy as np
    from sklearn.decomposition import PCA

    if result.trajectory is None or len(result.trajectory) < 3:
        return None

    snapshots = result.trajectory.numpy()
    pca = PCA(n_components=2).fit(snapshots)
    coords = pca.transform(snapshots)
    span = (coords.max(axis=0) - coords.min(axis=0))
    pad = padding * np.maximum(span, 1e-3)
    x = np.linspace(coords[:, 0].min() - pad[0], coords[:, 0].max() + pad[0], grid_size)
    y = np.linspace(coords[:, 1].min() - pad[1], coords[:, 1].max() + pad[1], grid_size)
    XX, YY = np.meshgrid(x, y)
    grid_xy = np.stack([XX.ravel(), YY.ravel()], axis=1)
    grid_embeds = pca.inverse_transform(grid_xy).astype(np.float32)
    grid_t = torch.tensor(grid_embeds, dtype=torch.float32)

    device = next(model.parameters()).device
    horizon = result.horizon if result.horizon > 0 else result.num_tokens
    input_ids = torch.tensor([result.input_ids[:horizon]], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    text_embeds = model.get_input_embeddings()(input_ids)

    Z = _accuracy_batch(
        model,
        compression_flat=grid_t,
        mem_shape=(result.num_mem_tokens, result.hidden_size),
        input_ids=input_ids,
        text_embeds=text_embeds,
        attention_mask=attention_mask,
    ).reshape(XX.shape)
    return XX, YY, Z, coords


def _draw_landscape(XX, YY, ZZ, coords, *, horizon: int, num_tokens: int) -> None:
    """Plot the accuracy landscape with the optimisation trajectory overlaid."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6.5))
    im = ax.pcolormesh(XX, YY, ZZ, shading="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(f"teacher-forced accuracy on first {horizon}/{num_tokens} tokens")
    ax.plot(coords[:, 0], coords[:, 1], color="white", alpha=0.65, lw=1.2)
    ax.scatter(coords[:, 0], coords[:, 1], s=22, c="white", alpha=0.85,
               edgecolors="black", linewidths=0.4)
    ax.scatter([coords[0, 0]], [coords[0, 1]], s=110, c="black", marker="o",
               edgecolors="white", linewidths=1.0, zorder=10, label="init")
    ax.scatter([coords[-1, 0]], [coords[-1, 1]], s=240, marker="*",
               c=COLOR_MISMATCH, edgecolors="black", linewidths=0.8, zorder=11,
               label="converged")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Local accuracy landscape on PC1–PC2 (white = optimisation path)")
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def display_interactive_compress(
    models: dict,
    *,
    default: str | None = None,
    max_seq_len: int = 64,
    default_max_steps_per_token: int = 300,
) -> None:
    """Interactive §5 widget: cram a user-provided text via *progressive* cramming
    and watch the PCA trajectory + loss/reconstruction curves live.

    ``models`` is a ``{display_name: (model, tokenizer)}`` mapping built by the
    notebook from already-loaded frozen models (typically the Llama-3.2-1B and
    SmolLM2-360M instances used by §3/§4). The widget never loads a model itself
    -- this keeps T4 memory predictable and matches the "no GPU side-effects in
    plotting helpers" convention used in §3 and §4.

    Each ▶ Compress click runs :func:`progressive_cram_text` end to end with
    ``capture_every`` tuned to the step budget, then renders the reconstruction
    at the horizon via the same coloured diff used in §3 and §4.
    """
    import ipywidgets as widgets

    from ._core import progressive_cram_text

    if not models:
        display(HTML(
            f"<div style='color:{COLOR_MISMATCH}'>No models supplied. Pass a "
            f"<code>{{name: (model, tokenizer)}}</code> dict.</div>"
        ))
        return

    if default is None or default not in models:
        default = next(iter(models))

    text_input = widgets.Textarea(
        value="Cramming compresses a span of text into a single learnable embedding of a frozen model.",
        layout=widgets.Layout(width="100%", height="80px"),
    )
    model_dd = widgets.Dropdown(options=list(models), value=default, description="Model")
    len_slider = widgets.IntSlider(
        value=min(32, max_seq_len), min=8, max=max_seq_len, step=4, description="Max tokens",
    )
    steps_slider = widgets.IntSlider(
        value=default_max_steps_per_token, min=100, max=1000, step=50,
        description="Max steps / token",
    )
    btn = widgets.Button(description="▶ Compress", button_style="primary")
    out = widgets.Output()
    last_result: dict = {}

    legend = (
        f'<span style="color:{COLOR_MATCH};font-weight:600">green = match</span>'
        f' &middot; '
        f'<span style="color:{COLOR_MISMATCH};font-weight:600">red = mismatch</span>'
        f' &middot; '
        f'<span style="color:{COLOR_PAST_GT};font-weight:600">grey = past the original span</span>'
    )

    def on_click(_):
        out.clear_output()
        with out:
            m, t = models[model_dd.value]
            viz = _ProgressiveLiveCurves(redraw_every=max(5, steps_slider.value // 30))
            capture = max(2, steps_slider.value // 80)
            result = progressive_cram_text(
                m, t, text_input.value,
                max_seq_len=int(len_slider.value),
                max_steps_per_token=int(steps_slider.value),
                capture_every=capture,
                on_step=viz,
            )
            viz.draw()  # final loss/conv frame

            # Accuracy landscape on PC1-PC2 (post-run, one batched forward pass per grid cell).
            print("Computing accuracy landscape on PC1-PC2 ...")
            landscape = _compute_landscape(result, m, grid_size=24, padding=0.35)
            if landscape is not None:
                _draw_landscape(
                    *landscape,
                    horizon=result.horizon, num_tokens=result.num_tokens,
                )

            # Build a row-shaped dict so reconstruct_and_show works without an adapter.
            row = {
                "embedding": result.embedding.numpy().tolist(),
                "input_ids": list(result.input_ids),
                "num_tokens": result.horizon,  # decode exactly the converged horizon
                "text": result.text,
                "final_convergence": 1.0 if result.horizon else 0.0,
            }
            extra = max(1, result.horizon // 5) if result.horizon else 0
            display(HTML(
                f"<div style='font-size:0.9em;color:#888;margin-top:6px;margin-bottom:4px'>{legend}</div>"
            ))
            reconstruct_and_show(
                row, "Reconstruction at horizon",
                model=m, tokenizer=t, extra_tokens=extra,
            )
            display(HTML(
                f"<div style='margin-top:4px;font-size:0.9em;color:#888'>"
                f"horizon: <b>{result.horizon}/{result.num_tokens}</b> tokens &middot; "
                f"{result.total_steps} steps &middot; {result.elapsed_s:.1f}s &middot; "
                f"<b>{len(result.stages)}</b> stages &middot; "
                f"model: <code>{html.escape(result.model_checkpoint)}</code></div>"
            ))
            last_result["result"] = result

    btn.on_click(on_click)
    display(widgets.VBox([
        text_input,
        widgets.HBox([model_dd, len_slider, steps_slider]),
        btn,
        out,
    ]))
