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

import contextlib
import html
import io
import logging
import sys
import warnings

import torch
from IPython.display import HTML, display

from ._core import reconstruct_text


# Silence the "Ignoring clean_up_tokenization_spaces=True for BPE tokenizer" advisory
# that fires on every tokenizer call in transformers 5.x. We always pass
# ``clean_up_tokenization_spaces=False`` ourselves, so the message is non-actionable
# noise. We hit it from three angles because transformers 5.x can emit it via any
# of Python's ``logging``, ``warnings``, or a direct ``stderr`` print from the Rust
# tokenizers backend depending on which code path decoded the tokens.
class _SuppressBPECleanupWarning(logging.Filter):
    def filter(self, record):
        return "clean_up_tokenization_spaces" not in record.getMessage()


for _logger_name in (
    "transformers",
    "transformers.tokenization_utils_base",
    "transformers.tokenization_utils_fast",
):
    logging.getLogger(_logger_name).addFilter(_SuppressBPECleanupWarning())

# Blanket-lower the transformers logger to ERROR: keeps genuine errors, drops the
# ``[transformers] Ignoring ...`` advisory that ships via transformers.logging.
try:
    import transformers.utils.logging as _tf_logging
    _tf_logging.set_verbosity_error()
except Exception:  # noqa: BLE001 -- best-effort; transformers is a hard dep anyway.
    pass


class _StderrFilter(io.TextIOBase):
    """Redirect around a token'iser call: drop lines matching ``pattern``, forward the rest."""

    def __init__(self, real, pattern: str):
        self._real = real
        self._pattern = pattern
        self._buf = ""

    def write(self, s):
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if self._pattern not in line:
                self._real.write(line + "\n")
        return len(s)

    def flush(self):
        if self._buf and self._pattern not in self._buf:
            self._real.write(self._buf)
        self._buf = ""
        self._real.flush()


@contextlib.contextmanager
def _quiet_tokenizer():
    """Suppress the BPE cleanup advisory across all three emission paths."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*")
        real_stderr = sys.stderr
        sys.stderr = _StderrFilter(real_stderr, "clean_up_tokenization_spaces")
        try:
            yield
        finally:
            try:
                sys.stderr.flush()
            except Exception:  # noqa: BLE001
                pass
            sys.stderr = real_stderr

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
    with _quiet_tokenizer():
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


class _ProgressiveLiveViz:
    """Live-updating optimisation dashboard for the §5 progressive cramming widget.

    Left panel: loss + teacher-forced reconstruction curves.
    Right panel: PCA trajectory of the compression embedding, coloured by the
    current stage's ``seq_len`` (so warm-start jumps between stages are visible
    as colour discontinuities).

    Both panels redraw every ``redraw_every`` optimiser steps. The heavier
    per-stage accuracy-landscape figure is computed *after* PC converges (~30s
    of forward passes) and rendered separately by the widget.
    """

    def __init__(self, *, redraw_every: int = 20):
        self.steps: list[int] = []
        self.losses: list[float] = []
        self.convs: list[float] = []
        self.snaps: list = []
        self.snap_steps: list[int] = []
        self.snap_seqlens: list[int] = []
        self.redraw_every = redraw_every
        self._since_redraw = 0

    def __call__(self, info: dict) -> None:
        self.steps.append(info["global_step"])
        self.losses.append(info["loss"])
        self.convs.append(info["convergence"])
        emb_flat = info["embedding"].detach().reshape(-1).to(torch.float32).cpu().numpy()
        self.snaps.append(emb_flat)
        self.snap_steps.append(info["global_step"])
        self.snap_seqlens.append(info["seq_len"])
        self._since_redraw += 1
        if self._since_redraw >= self.redraw_every:
            self._since_redraw = 0
            self.draw()

    def draw(self) -> None:
        import numpy as np
        import matplotlib.pyplot as plt
        from IPython.display import clear_output
        from sklearn.decomposition import PCA

        clear_output(wait=True)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))

        # Left: loss + reconstruction curves.
        ax[0].plot(self.steps, self.losses, color=COLOR_MISMATCH, lw=1.2)
        ax[0].set_xlabel("step")
        ax[0].set_ylabel("loss", color=COLOR_MISMATCH)
        ax[0].tick_params(axis="y", labelcolor=COLOR_MISMATCH)
        axb = ax[0].twinx()
        axb.plot(self.steps, self.convs, color=COLOR_MATCH, lw=1.2)
        axb.set_ylabel("reconstruction", color=COLOR_MATCH)
        axb.tick_params(axis="y", labelcolor=COLOR_MATCH)
        axb.set_ylim(-0.02, 1.02)
        ax[0].set_title("Optimisation")

        # Right: PCA trajectory of the compression embedding, coloured by stage seq_len.
        if len(self.snaps) >= 2:
            P = PCA(n_components=2).fit_transform(np.stack(self.snaps))
            ax[1].plot(P[:, 0], P[:, 1], color="#cccccc", alpha=0.55, lw=0.8)
            sc = ax[1].scatter(
                P[:, 0], P[:, 1],
                c=self.snap_seqlens, cmap="plasma", s=22, edgecolor="none",
            )
            ax[1].scatter([P[0, 0]], [P[0, 1]], color="black", marker="o", s=70, label="init")
            ax[1].scatter([P[-1, 0]], [P[-1, 1]], color=COLOR_MISMATCH, marker="*", s=180, label="current")
            cb = fig.colorbar(sc, ax=ax[1], pad=0.02)
            cb.set_label("stage seq_len (tokens compressed)")
            ax[1].legend(loc="best", fontsize=9)
        ax[1].set_title("Embedding trajectory (PCA)")
        ax[1].set_xticks([])
        ax[1].set_yticks([])
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


def _compute_per_stage_landscape(result, model, *, grid_size: int = 28,
                                 padding: float = 0.4, batch_size: int = 8,
                                 threshold: float = 0.9):
    """For each PC stage, compute teacher-forced accuracy on a shared PC1-PC2
    grid fit on the whole optimisation trajectory -- so the "high-accuracy
    regions" of every stage live in the same plane and can be overlaid.

    Adapted from ``compression_horizon/scripts/paper/animate_trajectory.py`` +
    ``visualize_landscale_2pca.py``: cache-only per-anchor regions, then draw
    them together (see ``visual_abstract_trajectory_zoom_progressive``). We
    skip the mp4/zoom animation and render a single static composite.

    Returns a dict with ``XX``, ``YY``, ``per_stage_Z`` (list of accuracy maps
    per stage), ``coords`` (trajectory-snapshot projections), ``snap_seq_lens``,
    ``stage_seq_lens``, and ``threshold``. Returns ``None`` if the trajectory is
    too short to fit a PCA.
    """
    import numpy as np
    from sklearn.decomposition import PCA

    if result.trajectory is None or len(result.trajectory) < 3:
        return None

    snapshots = result.trajectory.numpy()
    pca = PCA(n_components=2).fit(snapshots)
    coords = pca.transform(snapshots)  # [N_snaps, 2]
    span = (coords.max(axis=0) - coords.min(axis=0))
    pad = padding * np.maximum(span, 1e-3)
    x = np.linspace(coords[:, 0].min() - pad[0], coords[:, 0].max() + pad[0], grid_size)
    y = np.linspace(coords[:, 1].min() - pad[1], coords[:, 1].max() + pad[1], grid_size)
    XX, YY = np.meshgrid(x, y)
    grid_xy = np.stack([XX.ravel(), YY.ravel()], axis=1)
    grid_embeds = pca.inverse_transform(grid_xy).astype(np.float32)
    grid_t = torch.tensor(grid_embeds, dtype=torch.float32)

    device = next(model.parameters()).device
    per_stage_Z: list = []
    stage_seq_lens = [s.seq_len for s in result.stages]
    for seq_len in stage_seq_lens:
        input_ids = torch.tensor([result.input_ids[:seq_len]], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        text_embeds = model.get_input_embeddings()(input_ids)
        Z = _accuracy_batch(
            model,
            compression_flat=grid_t,
            mem_shape=(result.num_mem_tokens, result.hidden_size),
            input_ids=input_ids,
            text_embeds=text_embeds,
            attention_mask=attention_mask,
            batch_size=batch_size,
        ).reshape(XX.shape)
        per_stage_Z.append(Z)

    return dict(
        XX=XX, YY=YY,
        per_stage_Z=per_stage_Z,
        coords=coords,
        snap_seq_lens=list(result.trajectory_seq_len) if result.trajectory_seq_len else None,
        stage_seq_lens=stage_seq_lens,
        threshold=threshold,
    )


def _draw_per_stage_landscape(landscape) -> None:
    """Overlay one filled region per stage (``accuracy > threshold``) on a shared
    PC1-PC2 plane + trajectory polyline through the snapshots.

    Same semantics as ``visual_abstract_trajectory_zoom_progressive``'s final
    frame -- every stage gets its own colour, the trajectory line stitches the
    stages together, and the converged embedding is starred at the end.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    XX = landscape["XX"]
    YY = landscape["YY"]
    per_stage_Z = landscape["per_stage_Z"]
    coords = landscape["coords"]
    snap_seq_lens = landscape["snap_seq_lens"]
    stage_seq_lens = landscape["stage_seq_lens"]
    threshold = landscape["threshold"]

    n_stages = len(per_stage_Z)
    palette = plt.get_cmap("plasma")(np.linspace(0.08, 0.92, n_stages))

    fig, ax = plt.subplots(figsize=(9.5, 7.2))
    ax.set_facecolor("#111111")

    handles = []
    for i, (Z, colour, seq_len) in enumerate(zip(per_stage_Z, palette, stage_seq_lens)):
        # Filled contour where accuracy exceeds the threshold.
        ax.contourf(XX, YY, Z, levels=[threshold, 1.01], colors=[colour], alpha=0.5)
        # Thin bright outline so overlapping regions stay legible.
        ax.contour(XX, YY, Z, levels=[threshold], colors=[colour], linewidths=1.0, alpha=0.9)
        handles.append(Patch(facecolor=colour, alpha=0.7, label=f"stage {i+1} — {seq_len} tok"))

    # Trajectory polyline + stage-coloured snapshots.
    ax.plot(coords[:, 0], coords[:, 1], color="white", alpha=0.85, lw=1.3)
    if snap_seq_lens is not None and len(snap_seq_lens) == coords.shape[0]:
        ax.scatter(
            coords[:, 0], coords[:, 1],
            c=snap_seq_lens, cmap="plasma", s=22, edgecolor="black", linewidths=0.3,
        )
    else:
        ax.scatter(coords[:, 0], coords[:, 1], s=20, c="white", edgecolor="black", linewidths=0.3)

    # Anchors: init (black dot) + converged (red star).
    ax.scatter([coords[0, 0]], [coords[0, 1]], s=150, c="white", marker="o",
               edgecolors="black", linewidths=1.2, zorder=15, label="init")
    ax.scatter([coords[-1, 0]], [coords[-1, 1]], s=280, marker="*", c=COLOR_MISMATCH,
               edgecolors="black", linewidths=1.0, zorder=16, label="converged")

    ax.set_xlabel("PC1", color="#ddd")
    ax.set_ylabel("PC2", color="#ddd")
    ax.tick_params(colors="#aaa")
    for spine in ax.spines.values():
        spine.set_color("#666")
    ax.set_title(
        f"Per-stage accuracy regions on PC1–PC2 (colour = stage; "
        f"filled where teacher-forced accuracy > {threshold:.2f})",
        color="#ddd",
    )
    ax.set_aspect("equal", adjustable="datalim")
    legend = ax.legend(handles=handles, loc="upper left", fontsize=8,
                       ncol=min(2, max(1, n_stages // 4)), framealpha=0.85)
    legend.get_frame().set_facecolor("#222")
    for text in legend.get_texts():
        text.set_color("#ddd")
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
            viz = _ProgressiveLiveViz(redraw_every=max(5, steps_slider.value // 30))
            capture = max(2, steps_slider.value // 80)
            result = progressive_cram_text(
                m, t, text_input.value,
                max_seq_len=int(len_slider.value),
                max_steps_per_token=int(steps_slider.value),
                capture_every=capture,
                on_step=viz,
            )
            viz.draw()  # final loss/conv frame

            # Per-stage accuracy regions on the shared PC1-PC2 plane. Every stage
            # gets one region (accuracy > 0.9 against that stage's prefix); the
            # trajectory line stitches them together. This is the static
            # equivalent of ``visual_abstract_trajectory_zoom_progressive``.
            print(f"Computing per-stage accuracy regions ({len(result.stages)} stages)...")
            landscape = _compute_per_stage_landscape(
                result, m, grid_size=28, padding=0.4, threshold=0.9,
            )
            if landscape is not None:
                _draw_per_stage_landscape(landscape)

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
