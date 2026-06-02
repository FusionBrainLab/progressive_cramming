#!/usr/bin/env bash
# Render existing paper TABLES and TikZ FIGURES to tightly-cropped PNGs for the
# slide deck (so the deck reuses the real paper artifacts, not HTML look-alikes).
# Uses the standalone document class + pdflatex + pdftoppm. Local, no network.
set -euo pipefail

cd "$(dirname "$0")/.."                 # -> presentation/
PRES="$(pwd)"                            # -> presentation/
ROOT="$(cd .. && pwd)"                   # -> repo worktree root
A="$(pwd)/assets"; mkdir -p "$A"
BUILD="$(mktemp -d)"; trap 'rm -rf "$BUILD"' EXIT
DPI=350

COLORS='\definecolor{progblue}{RGB}{41,98,168}\definecolor{fullred}{RGB}{192,57,43}\definecolor{successgreen}{RGB}{39,174,96}\definecolor{failgray}{RGB}{127,140,141}'

render() {  # $1=kind(table|figure)  $2=abs source .tex/.tikz  $3=out png basename
  local kind="$1" src="$2" out="$3" w="$BUILD/$3.tex"
  if [ "$kind" = "table" ]; then
    cat > "$w" <<EOF
\documentclass[border=12pt]{standalone}
\usepackage{booktabs,amsmath,amssymb,array}
\usepackage[table]{xcolor}
${COLORS}
\begin{document}
\input{${src}}
\end{document}
EOF
  else
    cat > "$w" <<EOF
\documentclass[border=10pt]{standalone}
\usepackage{graphicx,amsmath,amssymb}
\usepackage{xcolor}
${COLORS}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,calc,shapes.geometric,fit,backgrounds}
\begin{document}
\setlength{\columnwidth}{16cm}
\input{${src}}
\end{document}
EOF
  fi
  ( cd "$BUILD" && pdflatex -interaction=nonstopmode -halt-on-error "$3.tex" >/dev/null 2>&1 ) \
    || { echo "FAILED: $3 (see $BUILD/$3.log)"; tail -20 "$BUILD/$3.log"; return 1; }
  pdftoppm -singlefile -r "$DPI" -png "$BUILD/$3.pdf" "$A/$3"
  echo "  ok  assets/$3.png"
}

echo "Rendering tables -> PNG …"
render table  "$ROOT/paper/tables/manual/compression_reconstruction_main.tex" tbl_reconstruction
render table  "$ROOT/paper/tables/semantic_evaluation.tex"                    tbl_semantic_eval
render table  "$ROOT/paper/tables/depth_size_pivot.tex"                       tbl_depth_size
render table  "$ROOT/paper/tables/solution_diversity.tex"                     tbl_solution_diversity
render table  "$PRES/tables/mmlu_modes.tex"                                   tbl_mmlu_modes        # slide-only
render table  "$PRES/tables/depth_size_heatmap.tex"                           tbl_depth_size_heatmap # slide-only (run make_depth_size_heatmap.py first)

echo "Rendering TikZ figures -> PNG …"
render figure "$ROOT/paper/figures/figure_progressive_cramming.tikz" fig_progressive

echo "Rendering repo QR code -> PNG …"
PY="/home/jovyan/.mlspace/envs/compression_horizon/bin/python"
"$PY" -c "import segno" 2>/dev/null || "$PY" -m pip install --quiet segno
"$PY" - <<PY
import segno
segno.make("https://github.com/FusionBrainLab/progressive_cramming", error="m").save(
    "$A/qr_repo.png", scale=14, border=2, dark="#172430", light="white")
print("  ok  assets/qr_repo.png")
PY

echo "Done."
