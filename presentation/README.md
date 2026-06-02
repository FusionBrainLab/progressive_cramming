# Progressive Cramming — 3-minute video deck

A self-contained [reveal.js](https://revealjs.com) slide deck for the short video
presentation of the paper. Speech text and plan live in
[`video_speech.md`](./video_speech.md); the spec is at
`.omc/specs/deep-interview-revealjs-video-slides.md`.

## Present / view

It's fully offline — just open the file in a browser:

```bash
xdg-open presentation/index.html      # or: open … (macOS)
```

Or serve it (some browsers restrict autoplay/notes from `file://`):

```bash
cd presentation && python -m http.server 8000   # -> http://localhost:8000
```

### Controls
- **→ / Space** advance · **←** back · **Esc / o** slide overview
- **S** speaker view (the full speech is in each slide's speaker notes)
- **↓** from the final slide → hidden **Q&A backup** slides (capacity scaling,
  attention-mass-vs-causal, solution diversity, PCA curves)
- **F** fullscreen · **B** blackout

## Export to PDF / PPTX

reveal.js exports via the browser's **print-to-PDF**, but you must use the special
`?print-pdf` URL — a plain Ctrl+P prints only the current slide and drops the layout.

**PDF (recommended):**
1. Open the deck in **Chrome/Chromium** with `?print-pdf` appended — ideally over a
   local server so every asset loads:
   ```bash
   cd presentation && python -m http.server 8000
   # then open:  http://localhost:8000/index.html?print-pdf
   ```
   Hard-refresh once (Cmd/Ctrl+Shift+R) so the latest CSS loads.
2. **Print** (Cmd/Ctrl+P) → Destination **Save as PDF**, Layout **Landscape**,
   Margins **None**, and **enable "Background graphics"** (required for the colors and
   the figure cards). Save.

Notes:
- Use **Chrome/Chromium** — Firefox/Safari handle the print stylesheet differently.
- A PDF is static, so the trajectory **video** shows its **poster frame**, not playback.
- The `@media print` block in `css/custom.css` re-flows the auto-fit (`.fit`/`.grow`)
  media for print; without it the absolutely-centered images collapse and go missing.
- Speaker-notes pages: append `&showNotes=separate-page` to the `?print-pdf` URL.

**PPTX** (no clean reveal→PPTX path):
- Make the PDF above, then `soffice --convert-to pptx slides.pdf` (LibreOffice) —
  each slide becomes a full-page image.
- For an editable deck with the **video playing**, build one PNG per slide and embed
  the mp4 on the trajectory slide via `python-pptx` (ask for a script).

## Structure
8 main slides mapped to the 7 speech beats (the trajectory beat is a vertical pair:
PC1–PC2 spatial path → optimizer-steps/effort path, where the dwell-and-leap basins
pop). Light interactivity: fragment builds, in-slide video that auto-plays on entry,
and an interactive attention-knockout slider on slide 7.

## Layout
```
index.html            the deck
css/custom.css         theme (paper accent palette, dark)
js/deck.js             video autoplay + knockout slider
reveal/                vendored reveal.js 5.2.1 (offline)
assets/                videos + figures (built by scripts/build_assets.sh)
scripts/build_assets.sh
```

## Rebuilding assets
`assets/` is generated from existing repo artifacts (no GPU, no retraining):

```bash
bash presentation/scripts/build_assets.sh      # trajectory videos + paper PDF figures -> PNG
bash presentation/scripts/render_artifacts.sh  # paper TABLES + TikZ schematics + repo QR -> PNG
```

- `build_assets.sh` copies the rendered trajectory videos from
  `artifacts/experiments_progressive/sl_4096_Meta-Llama-3.1-8B_lr_0.1/` and converts the
  relevant `paper/figures/*.pdf` to PNG via `pdftoppm`.
- `render_artifacts.sh` compiles the **real paper tables** and **TikZ schematics** to tightly
  cropped PNGs (`standalone` + `pdflatex` + `pdftoppm`) so the deck reuses the actual paper
  artifacts, and regenerates the repo **QR code** via `segno`.

## Re-vendoring reveal.js (only if `reveal/` is missing)
One-time, needs network:
```bash
tmp=$(mktemp -d) && (cd "$tmp" && npm init -y >/dev/null && npm i reveal.js@5 >/dev/null)
cp -r "$tmp/node_modules/reveal.js/dist"   presentation/reveal/dist
cp -r "$tmp/node_modules/reveal.js/plugin" presentation/reveal/plugin
rm -rf "$tmp"
```

> Note: the trajectory GIFs are large (`trajectory_pca.gif` ~17 MB). The deck uses the
> MP4s; the GIFs are only `<video>` fallbacks. If repo size matters, you can delete the
> `assets/*.gif` files — the MP4s still play.
