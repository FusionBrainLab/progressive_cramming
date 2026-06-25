/* Progressive Cramming — interactive per-token explorer.
 * Reads page/data/trajectories.json (exported by scripts/export_explorer_data.py).
 * No dependencies. Builds token spans once per sample, then only re-styles on slider moves.
 */
(function () {
  "use strict";

  var DATA = null;          // full payload
  var cur = null;           // current sample record {tokens, horizon, ...}
  var spans = [];           // <span.tok> per displayed token, index-aligned with cur.tokens
  var firstWall = -1;       // index of first token beyond the horizon (or -1)

  var el = {
    model: document.getElementById("selModel"),
    variant: document.getElementById("selVariant"),
    sample: document.getElementById("selSample"),
    readout: document.getElementById("readout"),
    slider: document.getElementById("slider"),
    sliderVal: document.getElementById("sliderVal"),
    tokens: document.getElementById("tokens"),
    strip: document.getElementById("stepStrip"),
    note: document.getElementById("expNote"),
    tooltip: document.getElementById("tooltip")
  };

  function opt(value, label) {
    var o = document.createElement("option");
    o.value = value; o.textContent = label; return o;
  }

  function curModel() { return DATA.models[+el.model.value]; }
  function curVariant() { return curModel().variants[+el.variant.value]; }

  function fillModels() {
    el.model.innerHTML = "";
    DATA.models.forEach(function (m, i) { el.model.appendChild(opt(i, m.model)); });
    fillVariants();
  }
  function fillVariants() {
    el.variant.innerHTML = "";
    curModel().variants.forEach(function (v, i) { el.variant.appendChild(opt(i, v.variant)); });
    fillSamples();
  }
  function fillSamples() {
    el.sample.innerHTML = "";
    curVariant().samples.forEach(function (s, i) {
      el.sample.appendChild(opt(i, "sample " + s.sample_id + "  (horizon " + s.horizon + ")"));
    });
    loadSample();
  }

  function fmt(n) { return n == null ? "&mdash;" : (Math.round(n).toLocaleString()); }

  function loadSample() {
    cur = curVariant().samples[+el.sample.value];
    if (!cur) return;
    var toks = cur.tokens;

    // locate the first token beyond the horizon (the "wall")
    firstWall = -1;
    for (var i = 0; i < toks.length; i++) { if (!toks[i].ok) { firstWall = i; break; } }

    // build token spans once
    el.tokens.innerHTML = "";
    spans = [];
    var frag = document.createDocumentFragment();
    toks.forEach(function (tk, i) {
      var s = document.createElement("span");
      s.className = "tok";
      s.textContent = tk.t;
      s.dataset.i = i;
      frag.appendChild(s);
      spans.push(s);
    });
    el.tokens.appendChild(frag);

    // slider over the absorbable (within-horizon) tokens
    var maxAbsorb = firstWall === -1 ? toks.length : firstWall;
    if (maxAbsorb < 1) maxAbsorb = 1;
    el.slider.min = 1;
    el.slider.max = maxAbsorb;
    el.slider.value = maxAbsorb;       // start fully crammed

    update();   // styles tokens, fills readout, and renders the strip
  }

  function update() {
    var p = +el.slider.value;          // tokens absorbed so far
    var toks = cur.tokens;
    for (var i = 0; i < toks.length; i++) {
      var cls = "tok";
      if (!toks[i].ok) {
        cls += (i === firstWall) ? " wall" : " pending";
      } else if (i < p - 1) {
        cls += " crammed";
      } else if (i === p - 1) {
        cls += " frontier";
      } else {
        cls += " pending";
      }
      spans[i].className = cls;
    }

    // readout driven by the frontier token
    var f = toks[p - 1] || {};
    var totalSteps = 0;
    for (var j = 0; j < toks.length; j++) { if (toks[j].ok && toks[j].k != null) totalSteps += toks[j].k; }

    el.sliderVal.innerHTML = "token " + (f.L != null ? f.L : p) +
      (cur.capped ? " &middot; showing first " + cur.n_tokens_shown + " of " + cur.total_tokens : "");

    el.readout.innerHTML =
      ro("gold", fmt(cur.horizon), "tokens", "compression horizon") +
      ro("", f.s == null ? "&mdash;" : f.s.toFixed(2), "bits", "frontier-token surprisal") +
      ro("", fmt(f.k), "steps", "to absorb this token") +
      ro("", fmt(totalSteps), "steps", "total, to the horizon");

    renderStrip(p);
  }

  function ro(kind, v, unit, label) {
    return '<div class="ro ' + kind + '"><div class="v">' + v +
      (unit ? ' <small>' + unit + '</small>' : '') + '</div><div class="k">' + label + '</div></div>';
  }

  // ---- step-cost strip (canvas) ----
  // Draws the per-token step-cost profile plus the gold frontier marker in one pass.
  // IMPORTANT: size the backing store from CONSTANT css dims (clientWidth + STRIP_H),
  // never from the canvas's own width/height — those are the (already dpr-scaled)
  // backing store and re-reading them compounds every redraw until the canvas
  // "exceeds max size".
  var stripGeom = null;
  var STRIP_H = 70; // px, matches #stepStrip CSS height
  function renderStrip(p) {
    var c = el.strip;
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    var W = Math.max(50, Math.min(Math.round(c.clientWidth) || 900, 4096));
    var H = STRIP_H;
    var pxW = Math.round(W * dpr), pxH = Math.round(H * dpr);
    if (c.width !== pxW) c.width = pxW;
    if (c.height !== pxH) c.height = pxH;
    var g = c.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);

    var toks = cur.tokens, n = toks.length;
    var pad = 6, h = H - pad * 2;
    var maxK = 1;
    for (var i = 0; i < n; i++) { if (toks[i].k != null && toks[i].k > maxK) maxK = toks[i].k; }
    var scale = function (k) { return Math.log1p(k) / Math.log1p(maxK); }; // log to tame spikes

    stripGeom = { W: W, H: H, n: n };

    // baseline
    g.strokeStyle = "#E5EAEE"; g.beginPath(); g.moveTo(0, H - pad); g.lineTo(W, H - pad); g.stroke();

    for (var x = 0; x < W; x++) {
      var i0 = Math.floor(x / W * n), i1 = Math.max(i0 + 1, Math.floor((x + 1) / W * n));
      var acc = 0, cnt = 0, ok = true, wall = false;
      for (var t = i0; t < i1 && t < n; t++) {
        if (toks[t].k != null) { acc += toks[t].k; cnt++; }
        if (!toks[t].ok) { ok = false; if (t === firstWall) wall = true; }
      }
      var mean = cnt ? acc / cnt : 0;
      var bh = scale(mean) * h;
      g.strokeStyle = wall ? "#C0392B" : (ok ? "#2E86C1" : "#C9D2D8");
      g.globalAlpha = ok ? 0.85 : 0.6;
      g.beginPath(); g.moveTo(x + 0.5, H - pad); g.lineTo(x + 0.5, H - pad - Math.max(bh, wall ? h * 0.6 : 0)); g.stroke();
    }
    g.globalAlpha = 1;

    // gold frontier marker
    if (p != null && n) {
      var fx = (p - 0.5) / n * W;
      g.strokeStyle = "#D4AC0D"; g.lineWidth = 2;
      g.beginPath(); g.moveTo(fx, 0); g.lineTo(fx, H); g.stroke();
      g.lineWidth = 1;
    }
  }

  // ---- tooltip ----
  function showTip(e) {
    var s = e.target;
    if (!s.classList || !s.classList.contains("tok")) return;
    var i = +s.dataset.i; var tk = cur.tokens[i];
    var state = !tk.ok ? (i === firstWall ? "beyond horizon — the wall" : "beyond horizon")
      : "absorbed at stage L=" + tk.L;
    el.tooltip.innerHTML =
      '<span class="tt-tok">' + escapeHtml(tk.t === " " ? "␠" : tk.t) + '</span> &nbsp;<span style="opacity:.7">pos ' + tk.L + '</span>' +
      '<div class="tt-row">surprisal <b>' + (tk.s == null ? "—" : tk.s.toFixed(2) + " bits") + '</b></div>' +
      '<div class="tt-row">steps to converge <b>' + (tk.k == null ? "—" : Math.round(tk.k).toLocaleString()) + '</b></div>' +
      '<div class="tt-row" style="opacity:.7">' + state + '</div>';
    el.tooltip.style.opacity = 1;
    moveTip(e);
    s.classList.add("hot");
  }
  function moveTip(e) {
    var t = el.tooltip, pad = 14;
    var x = e.clientX + pad, y = e.clientY + pad;
    var r = t.getBoundingClientRect();
    if (x + r.width > window.innerWidth - 8) x = e.clientX - r.width - pad;
    if (y + r.height > window.innerHeight - 8) y = e.clientY - r.height - pad;
    t.style.left = x + "px"; t.style.top = y + "px";
  }
  function hideTip(e) {
    el.tooltip.style.opacity = 0;
    if (e.target.classList) e.target.classList.remove("hot");
  }
  function escapeHtml(s) {
    return s.replace(/[&<>"]/g, function (c) { return ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]; });
  }

  // ---- strip click → set horizon ----
  function stripToSlider(e) {
    if (!stripGeom) return;
    var rect = el.strip.getBoundingClientRect();
    var x = (e.clientX - rect.left) / rect.width;
    var p = Math.round(x * stripGeom.n);
    p = Math.max(+el.slider.min, Math.min(+el.slider.max, p));
    el.slider.value = p; update();
  }

  function wire() {
    el.model.addEventListener("change", fillVariants);
    el.variant.addEventListener("change", fillSamples);
    el.sample.addEventListener("change", loadSample);
    el.slider.addEventListener("input", update);
    el.tokens.addEventListener("mouseover", showTip);
    el.tokens.addEventListener("mousemove", moveTip);
    el.tokens.addEventListener("mouseout", hideTip);
    var dragging = false;
    el.strip.addEventListener("mousedown", function (e) { dragging = true; stripToSlider(e); });
    window.addEventListener("mousemove", function (e) { if (dragging) stripToSlider(e); });
    window.addEventListener("mouseup", function () { dragging = false; });
    el.strip.style.cursor = "ew-resize";
    window.addEventListener("resize", function () { if (cur) { update(); } });
  }

  function start(data) {
    DATA = data;
    if (!DATA || !DATA.models || !DATA.models.length) {
      el.note.innerHTML = '<span class="err">No trajectory data found.</span>';
      return;
    }
    el.note.innerHTML = "Drag the slider or click the step-cost strip to move the horizon. Hover a token for its surprisal and step cost. " +
      "Green = absorbed, gold = the frontier token, red = the wall (first token the model cannot cram).";
    fillModels();
    wire();
  }

  // load data (works over http/Pages; for local file:// preview, serve the folder)
  if (window.__TRAJECTORIES__) {
    start(window.__TRAJECTORIES__);
  } else {
    fetch("data/trajectories.json")
      .then(function (r) { if (!r.ok) throw new Error(r.status); return r.json(); })
      .then(start)
      .catch(function (err) {
        el.note.innerHTML = '<span class="err">Could not load <code>data/trajectories.json</code> (' + err +
          ').</span> If you opened this file directly, serve the folder instead: ' +
          '<code>cd page &amp;&amp; python -m http.server</code>, then open <code>http://localhost:8000</code>.';
      });
  }
})();
