#!/usr/bin/env python
"""Generate a heatmap-shaded LaTeX table (SLIDE-ONLY, not part of the paper).

Reuses the data from the paper's main-body table `tab:depth_size_pivot`
(paper/tables/depth_size_pivot.tex): mean perfectly-crammed tokens vs. retained
first-N decoder layers and model. Each numeric cell is shaded by its value
(log-scaled `Blues`) so the increase with depth (->) and width (v) is visible at
a glance, while keeping the actual numbers in the cells.

Output: presentation/tables/depth_size_heatmap.tex  (tabular only)
Rendered to assets/tbl_depth_size_heatmap.png by render_artifacts.sh.
"""
import math
import os

import matplotlib

matplotlib.use("Agg")
from matplotlib import cm
from matplotlib.colors import to_hex

# (model label, [N=1, N=2, N=4, N=8, Full]);  None -> "--" (not run)
ROWS = [
    ("SmolLM2-1.7B (24L)", [50, 241, 404, 455, 335]),
    ("SmolLM3-3B (36L)", [20, 39, 114, 300, None]),
    ("Qwen3-4B (36L)", [79, 147, 192, 421, 512]),
    ("Qwen3-8B (36L)", [119, 180, 304, 625, 774]),
    ("Llama-3.1-8B (32L)", [97, 184, 400, None, 1438]),
]

vals = [v for _, row in ROWS for v in row if v is not None]
lmin, lmax = math.log(min(vals)), math.log(max(vals))
cmap = cm.get_cmap("Blues")


def cell(v):
    if v is None:
        return r"\cellcolor[HTML]{EFEFEF}\textcolor[HTML]{B0B0B0}{--}"
    t = (math.log(v) - lmin) / (lmax - lmin)
    r, g, b, _ = cmap(0.12 + 0.85 * t)  # skip the near-white low end
    bg = to_hex((r, g, b)).lstrip("#").upper()
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b  # white text on dark cells
    tx = "FFFFFF" if lum < 0.55 else "172430"
    return rf"\cellcolor[HTML]{{{bg}}}\textcolor[HTML]{{{tx}}}{{\textbf{{{v}}}}}"


lines = [
    r"\begin{tabular}{lrrrrr}",
    r"\toprule",
    r"\textbf{Model} & \multicolumn{4}{c}{\textbf{Retained first-$N$ layers}} & \textbf{Full} \\",
    r" & 1 & 2 & 4 & 8 & \\",
    r"\midrule",
]
for label, row in ROWS:
    lines.append(rf" {label} & " + " & ".join(cell(v) for v in row) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}", ""]

out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tables", "depth_size_heatmap.tex"))
with open(out, "w") as f:
    f.write("\n".join(lines))
print("wrote", out)
