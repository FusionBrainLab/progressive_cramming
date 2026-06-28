<h1 align="center">Progressive Cramming</h1>

<p align="center">
  <b>Reliable token compression into learnable memory embeddings — and what it reveals.</b><br>
  Reference implementation for the paper<br>
  <i>Progressive Cramming: Reliable Token Compression and What It Reveals</i><br>
  (Tarasov, Lashukov, Goncharova, Kuznetsov).
</p>

<p align="center">
  <a href="https://fusionbrainlab.github.io/progressive_cramming/slides/">▶ Video&nbsp;slides</a> ·
  <a href="#-interactive-demo-colab">Demo</a> ·
  <a href="#-quickstart-minutes-on-1-gpu">Quickstart</a> ·
  <a href="#-the-three-methods">Methods</a> ·
  <a href="#-reproducing-the-paper-experiments">Reproduce</a> ·
  <a href="https://huggingface.co/datasets/mrsndmn/progressive_cramming_trajectories">Dataset</a> ·
  <a href="#-citation">Cite</a>
</p>

<p align="center">
  <a href="https://colab.research.google.com/github/FusionBrainLab/progressive_cramming/blob/main/notebooks/progressive_cramming_demo.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
  </a>
</p>

---

## What is "cramming"?

**Cramming** compresses a span of text into one (or a few) *learnable memory embeddings*:
we freeze a pretrained language model and optimise a small set of input embeddings — by
gradient descent — until the frozen model reconstructs the original tokens from them.
The text never updates the model's weights; all the information is squeezed into the
optimised embedding. How much text a single embedding can hold (its *compression
horizon*), and how the optimisation gets there, is what this repository measures.

This repo ships a **lean, self-contained implementation** of the three core methods:

| Method | What it does | Selected by |
|---|---|---|
| **Full cramming** | Optimise a memory embedding to reconstruct a fixed-length text span. | *(default)* |
| **Progressive cramming** | Grow the target span stage by stage, only extending once the current span reconstructs exactly — finding the compression horizon. | `--progressive_train` |
| **Low-dim projection** | Optimise the embedding inside a learned rank-*k* subspace instead of full hidden size. | `--low_dim_train` |

> This is a focused reproducibility release: the three methods above plus their
> evaluation. The broader research codebase (ablations, compression-head training,
> topology analysis) is intentionally not included.

---

## 🎮 Interactive demo (Colab)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FusionBrainLab/progressive_cramming/blob/main/notebooks/progressive_cramming_demo.ipynb)

A self-contained notebook ([`notebooks/progressive_cramming_demo.ipynb`](./notebooks/progressive_cramming_demo.ipynb))
that runs on a free Colab **T4 GPU** with `Llama-3.2-1B`. It lets you:

- reconstruct a **pre-compressed gallery** of 5 domains (literature / code / news / poetry / science)
  from embeddings cached on the Hub — one click each;
- compare **total vs progressive cramming** side by side on one example;
- **compress your own text** and watch the optimisation live (loss/reconstruction curves
  + the embedding's PCA trajectory), then download the resulting embedding.

The gallery embeddings are produced by [`scripts/build_demo_gallery.py`](./scripts/build_demo_gallery.py)
(run once on a GPU) and published as a small Hugging Face dataset the notebook reads at runtime.

---

## 📦 Repository layout

```
progressive_cramming/
├── src/progressive_cramming/      # the installable package
│   ├── run.py                     # entry point: model+data load, trainer select, train
│   ├── train/
│   │   ├── arguments.py           # MyTrainingArguments — every CLI flag
│   │   ├── trainers/              # FullCramming / Progressive / LowDim trainers
│   │   ├── parametrization.py     # direct / low-dim / PCA embedding parametrizations
│   │   ├── embedding_init.py      # init strategies (random0.02, mvnormal, pca, ...)
│   │   ├── loss.py, inputs.py, optimization.py
│   ├── analysis/                  # convergence tracking + information gain
│   ├── inference/                 # generation from a compression embedding
│   ├── data/                      # dataset tokenization + caching
│   ├── demo.py                    # cram/reconstruct helpers for the Colab demo notebook
│   └── utils/
├── scripts/run_cramming.py        # shell wrapper around run.py
├── scripts/build_demo_gallery.py  # builds + uploads the demo's pre-computed embedding gallery
├── examples/quickstart.py         # runs all 3 methods on a small model (this README's demo)
├── notebooks/progressive_cramming_demo.ipynb  # interactive Colab demo
└── presentation/                  # the reveal.js video deck (deployed to GitHub Pages)
```

---

## 🛠 Installation

Python ≥ 3.11 and a recent PyTorch. A CUDA GPU is recommended (the quickstart runs in
minutes on one GPU; CPU works but is much slower).

```bash
git clone https://github.com/FusionBrainLab/progressive_cramming.git
cd progressive_cramming

python -m venv .venv && source .venv/bin/activate     # or use uv / conda
pip install -e .
```

This installs the `progressive_cramming` package and the `progressive-cramming`
console script. Models and datasets are pulled from the Hugging Face Hub on first use.

---

## 🚀 Quickstart (minutes on 1 GPU)

Run all three methods on a small model (`SmolLM2-135M`) over a couple of PG-19 samples
and print each method's headline metric (~10 minutes total on one GPU — a couple of
minutes per method; progressive cramming is the slowest):

```bash
python examples/quickstart.py
```

Example output (measured on one A100; numbers vary slightly with hardware/seed):

```text
QUICKSTART RESULTS
======================================================================
{
  "method": "full_cramming",
  "samples": 2,
  "mean_reconstruction": 0.992,          # ~exact token reconstruction
  "mean_steps_to_converge": 2614.0
}
{
  "method": "low_dim_projection(k=32)",
  "samples": 2,
  "mean_reconstruction": 1.0             # exact, in a rank-32 subspace
}
{
  "method": "progressive_cramming",
  "samples": 2,
  "mean_converged_horizon_tokens": 29.0, # longest prefix reconstructed exactly
  "max_sequence_length": 64,
  "mean_stages": 5.5
}
```

Metrics and tiny artifacts are written to `runs/quickstart/`. Useful flags:

```bash
python examples/quickstart.py --model HuggingFaceTB/SmolLM2-360M --seq-len 96 --samples 4
python examples/quickstart.py --methods full progressive       # subset of methods
```

**How to read the metrics**

- **`mean_reconstruction`** — fraction of tokens the frozen model decodes correctly from
  the compression embedding (teacher-forced). `1.0` = exact reconstruction.
- **`mean_converged_horizon_tokens`** — for progressive cramming, the longest prefix (in
  tokens) that a single embedding reconstructs *exactly* — the compression horizon.

---

## 🧪 The three methods

All three run through the same entry point; the trainer is chosen by CLI flags. Replace
the model/dataset/lengths as you like.

### Full cramming
```bash
python scripts/run_cramming.py \
    --model_checkpoint HuggingFaceTB/SmolLM2-135M \
    --dataset_name LarryLovestein/pg19_1k \
    --max_sequence_length 64 --limit_dataset_items 4 \
    --embedding_init_method random0.02 \
    --learning_rate 0.1 --max_optimization_steps_per_sample 4000 \
    --output_dir runs/full_demo
```

### Progressive cramming
```bash
python scripts/run_cramming.py --progressive_train 1 \
    --model_checkpoint HuggingFaceTB/SmolLM2-135M \
    --dataset_name LarryLovestein/pg19_1k \
    --max_sequence_length 64 --limit_dataset_items 4 \
    --embedding_init_method random0.02 --learning_rate 0.1 \
    --progressive_min_seq_len 1 --progressive_step 8 \
    --max_optimization_steps_per_token 500 \
    --output_dir runs/progressive_demo
```

### Low-dim projection
```bash
python scripts/run_cramming.py --low_dim_train --low_dim_size 32 \
    --model_checkpoint HuggingFaceTB/SmolLM2-135M \
    --dataset_name LarryLovestein/pg19_1k \
    --max_sequence_length 64 --limit_dataset_items 4 \
    --embedding_init_method random0.02 --learning_rate 0.1 \
    --max_optimization_steps_per_sample 4000 \
    --output_dir runs/lowdim_demo
```

Run `python scripts/run_cramming.py --help` to see every flag (defined in
`src/progressive_cramming/train/arguments.py`).

---

## 📊 Reproducing the paper experiments

The paper's numbers come from larger models, the full PG-19 benchmark, and long
optimisation budgets — they require a GPU (the original runs used one A100 per job).
The settings below are the exact recipe (mirrors the internal job launcher).

**Common setup**

| Setting | Value |
|---|---|
| Dataset | `LarryLovestein/pg19_1k` |
| Samples (`--limit_dataset_items`) | 50 |
| `--max_sequence_length` | 4096 |
| `--max_optimization_steps_per_sample` | 10000 |
| `--max_optimization_steps_per_token` | 1000 |
| `--embedding_init_method` | `random0.02` |
| `--warmup_steps` | 100 |
| Attention backend | `flash_attention_2` (install `flash-attn`) or `sdpa` |

**Per-model learning rate / batch size / low-dim size**

| Model | `--learning_rate` | `--per_device_train_batch_size` | `--low_dim_size` |
|---|---|---|---|
| `unsloth/Meta-Llama-3.1-8B` | 0.1 | 10 | 256 |
| `EleutherAI/pythia-1.4b` | 0.5 | 25 | 256 |
| `HuggingFaceTB/SmolLM2-1.7B` | 0.1 | 25 | 256 |
| `unsloth/gemma-3-4b-pt` | 0.1 | 10 | 32 |

Example — progressive cramming, SmolLM2-1.7B, paper settings:

```bash
python scripts/run_cramming.py --progressive_train 1 \
    --model_checkpoint HuggingFaceTB/SmolLM2-1.7B \
    --dataset_name LarryLovestein/pg19_1k --limit_dataset_items 50 \
    --max_sequence_length 4096 \
    --embedding_init_method random0.02 \
    --learning_rate 0.1 --per_device_train_batch_size 25 \
    --warmup_steps 100 \
    --max_optimization_steps_per_sample 10000 \
    --max_optimization_steps_per_token 1000 \
    --attn_implementation flash_attention_2 \
    --output_dir runs/progressive_smollm2_1.7b
```

For multi-GPU, launch with `accelerate launch scripts/run_cramming.py ...` — the dataset
is tokenized once by the main process and shared.

The paper's variants (beyond the three core methods) combine these flags:

- **hybrid** (activation alignment): `--loss_type cosine --hybrid_alpha 1.0 --num_alignment_layers 8`
- **hybrid + low-dim**: add `--low_dim_projection --low_dim_size <k>`
- **no-BOS**: `--no_bos_token`

---

## 🗂 Outputs

Each run writes a Hugging Face `Dataset` of per-sample (per-stage for progressive) rows
under `--output_dir`:

- `compressed_prefixes/` (full, low-dim) or `progressive_prefixes/` (progressive)
- `compression_embeddings.pt`, and for low-dim `low_dim_projection.pt`
- TensorBoard logs

Key row fields: `final_convergence` (reconstruction accuracy), `information_gain_bits`,
`embedding`, `text`, plus `stage_seq_len` / `steps_taken` for progressive. Load with:

```python
from datasets import load_from_disk
ds = load_from_disk("runs/progressive_demo/progressive_prefixes")
print(ds[0]["final_convergence"], ds[0]["stage_seq_len"])
```

### Published trajectories

You don't have to re-run the paper's experiments to inspect them. The exact
progressive-cramming trajectories behind the paper's `progressive_modifications`
table are published on the Hugging Face Hub:

**[`mrsndmn/progressive_cramming_trajectories`](https://huggingface.co/datasets/mrsndmn/progressive_cramming_trajectories)**

One config per model family (`Llama-3.1-8B`, `pythia-1.4b`, `SmolLM2-1.7B`,
`gemma-3-4b-pt`), each with a `baseline` and a `lowdim` split. Every row is one
converged stage; reconstruct a document's trajectory by taking all rows with the
same `sample_id` and sorting by `(stage_index, stage_seq_len)`.

```python
from datasets import load_dataset
ds = load_dataset("mrsndmn/progressive_cramming_trajectories", "SmolLM2-1.7B", split="baseline")
```

See the [dataset card](https://huggingface.co/datasets/mrsndmn/progressive_cramming_trajectories)
for the full column schema and a note on the saved-embedding offset.

---

## 🎬 Video presentation

A self-contained [reveal.js](https://revealjs.com) deck lives in
[`presentation/`](./presentation) and is deployed to GitHub Pages on every change (see
[`.github/workflows/deploy-presentation.yml`](./.github/workflows/deploy-presentation.yml)):

**▶ https://fusionbrainlab.github.io/progressive_cramming/slides/**
(the bare site root redirects here)

- [`presentation/README.md`](./presentation/README.md) — present locally / export to PDF/PPTX.
- [`presentation/video_speech.md`](./presentation/video_speech.md) — the ~3-minute speaker script and scene plan.

---

## 📝 Citation

```bibtex
@inproceedings{tarasov2026progressive,
  title     = {Progressive Cramming: Reliable Token Compression and What It Reveals},
  author    = {Tarasov, Dmitrii and Lashukov and Goncharova, Elena and Kuznetsov, Andrey},
  year      = {2026},
}
```

See [`CITATION.cff`](./CITATION.cff).

## 📄 License

[MIT](./LICENSE).
