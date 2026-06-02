# Progressive Cramming — 3-Minute Video Speech

**Paper:** *Progressive Cramming: Reliable Token Compression and What It Reveals*
**Authors:** Dmitrii Tarasov, Timofei Lashukov, Elizaveta Goncharova, Andrey Kuznetsov
**Code:** https://github.com/FusionBrainLab/progressive_cramming

---

## The narrative arc (why it's exciting)

Three-act structure built for a short talk: **an astonishing claim → a hidden trap →
a tool that exposes what's really going on.** The trajectory animation is the centerpiece —
the one moment the audience gets to *watch* the optimizer work. The speech crescendos into
the video at ~1:30, then lands the twist (reconstruction ≠ understanding).

The video animates the **progressive-cramming optimization trajectory** for the length-1000
Llama-3.1-8B sample, projected onto PC1–PC2: a moving cursor + growing trail over the
shrinking accuracy basins (animated version of Figure
`fig:visual_abstract_optimization_trajectory_with_good_accuracy`; PC1+PC2 = 65.7% of variance).

---

## Speech plan (≈2:55, scene-by-scene)

| Time | Beat | On screen | Spoken purpose |
|------|------|-----------|----------------|
| 0:00–0:15 | **Intro & hook** | Title card | Introduce the work; the single-token question + the trap |
| 0:15–0:38 | **Task setup** | Cramming scheme | Frozen model, 1 trainable vector, autoregressive recon |
| 0:38–0:50 | **The big claim** | "1,568 tokens fit" figure | ~1500 tokens → 1 embedding; far beyond classical encoders (≤10×) |
| 0:50–1:10 | **The trap** | Reconstruction table | Full cramming = a fixed token budget — a guess, no guarantee of recovery |
| 1:10–1:35 | **The fix** | Progressive-cramming schematic | Swap fixed budget → fixed (perfect) reconstruction; token count varies per sample |
| 1:35–2:00 | **🎬 The video** | **Trajectory animation** | Watch it walk; low-dimensional path + shrinking basins |
| 2:00–2:28 | **The twist** | Downstream + MMLU + attention-knockout figures | Perfect reconstruction ≠ understanding; *why* (early layers, all families) |
| 2:28–2:40 | **Capacity** | Depth/width heatmap | Capacity scales with depth & width — bought with model size |
| 2:40–2:55 | **Takeaway + CTA** | Conclusion / end card + GitHub | Reconstruction alone isn't enough; visit the project page |

---

## Speech text

> **Delivery note:** ~400 words, energetic-but-clear pace (~150 wpm, runtime ~2:55).
> Cue markers in **[brackets]** are for your editor, not to be read aloud.

**[TITLE CARD]**
Let me introduce our work, *Progressive Cramming*. It starts with a simple question: how much can you fit into a *single* token? The answer is surprising — and it hides a trap.

**[CRAMMING SCHEME]**
First, the cramming task. The model is frozen. We add one trainable vector in front of the prompt and optimize it, per text, with gradient descent. Feed that vector back in, and it should regenerate the whole text on its own.

**[1,568 TOKENS FIT]**
And it works surprisingly well — up to fifteen hundred tokens in one vector. More than a thousand times compression, far beyond classical encoders. Transformers have huge hidden capacity.

**[RECONSTRUCTION TABLE]**
But look at *how* full cramming works. It fixes a token budget — the same count for every sample — and hopes it fits. That budget is a guess: no guarantee the text is recovered, and every text has its own limit. And the accuracy is deceptive: full cramming can hit *ninety-nine percent* teacher-forced accuracy, yet real *greedy* decoding collapses to *zero* — because those few errors land on the very first tokens, and one early miss cascades.

**[PROGRESSIVE SCHEMATIC]**
So we flip it. Instead of fixing the budget, we fix the *reconstruction* — require it perfect — and let the token count vary per sample. Grow one token at a time, stopping only when perfect reconstruction is no longer possible. An honest, per-sample measure — and it hands us the optimizer's full path through embedding space.

**[🎬 TRAJECTORY VIDEO PLAYS]**
Watch it. Each point perfectly stores the prefix so far; as tokens are added, the basin of perfect reconstruction shrinks. And the surprise: the path is *low-dimensional* — in four thousand dimensions, just two directions capture two-thirds of the motion.

**[DOWNSTREAM TABLE — HellaSwag / ARC]**
Now the real question: if it reconstructs perfectly, does the model *understand* it? No. Add the crammed embedding to HellaSwag or ARC and accuracy drops — even with the original text still in context.

**[MMLU TABLE — full\_prefix vs random]**
The sharpest test: five-shot MMLU collapses to near zero — not even a parseable answer. A *random* embedding barely changes it. So it's not adding a vector that breaks the model — it's the *optimized* one.

**[ATTENTION-KNOCKOUT FIGURE]**
Why? Attention-knockout localizes it: the embedding takes over the *early* layers, steering computation instead of storing meaning. Mask just those layers and downstream capability returns — across Llama, Pythia, and SmolLM2.

**[DEPTH & WIDTH HEATMAP]**
Where does the capacity come from? Truncate to the first few layers, finetune, and count the tokens that still cram perfectly. It grows with *depth* and *width* — the two compound. Capacity isn't magic; it's bought with model size.

**[CONCLUSION / END CARD]**
So here's what cramming reveals: perfect reconstruction can be *brittle steering* that encodes nothing the model can use. For learned compression, reconstruction alone is not enough. To explore the code, the released trajectories, and the full paper, visit our project page — the link is on screen. Thanks for watching.

**[END CARD: title · authors · github.com/FusionBrainLab/progressive_cramming]**

---

## Delivery & editing tips

- **Energy curve:** open warm (intro + hook), turn skeptical on "that budget is a guess,"
  then build momentum into the video. The video is your breath — let it run ~5–8s with just
  the visuals before you narrate over it.
- **Avoid a color claim:** the script says "the basin of perfect reconstruction" rather than
  naming a color, since the animation uses a saturation-based palette (rocket), not a single
  hue — keeps it accurate regardless of final render.
- **If you need to cut to ~2:30:** drop the depth/width beat and trim the MMLU line to
  "On generative tasks, it collapses to near zero."
- **If you have more time:** after the trajectory beat, add: *"And these aren't a single
  basin — equally good solutions sit far apart, in nearly independent directions; one
  trajectory is a thin slice."*
