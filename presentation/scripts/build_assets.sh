#!/usr/bin/env bash
# Rebuild presentation/assets/ from existing repo artifacts.
# Fully local + offline: copies the already-rendered trajectory videos and
# converts the relevant paper-figure PDFs to PNG (via pdftoppm).
#
# NOTE: reveal.js itself is vendored separately (one-time, needs network) — see
# presentation/README.md. This script does NOT touch presentation/reveal/.
#
# Re-runnable: overwrites assets in place. No new GPU compute, no re-training.
set -euo pipefail

cd "$(dirname "$0")/.."                       # -> presentation/
ROOT="$(cd .. && pwd)"                         # -> repo worktree root
ARTI="$ROOT/artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_lr_0.1"
A="assets"
mkdir -p "$A"

echo "[1/2] Copying rendered trajectory media …"
cp "$ARTI/videos/sample0_pca.mp4"                            "$A/trajectory_pca.mp4"
cp "$ARTI/videos/sample0_pca.gif"                            "$A/trajectory_pca.gif"
cp "$ARTI/visualizations/visual_abstract_pc1_pc2_joined.png" "$A/trajectory_landscape.png"

echo "[2/2] Converting paper-figure PDFs -> PNG (200 dpi) …"
fig() { pdftoppm -singlefile -r 200 -png "$ROOT/paper/figures/$1.pdf" "$A/$2"; }
fig attention_knockout_cumulative                       attention_knockout_cumulative
fig attention_knockout_per_layer                        attention_knockout_per_layer
fig aggregate_pca_components_vs_seq_len_Llama3.1-8B_all_lrs pca_components_vs_seqlen
fig aggregate_pca_reconstruction_accuracy               pca_reconstruction_accuracy

echo "Done. Assets in presentation/$A/:"
ls -1 "$A"
