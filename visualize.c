/* visualize.c - dump a single Transformer LM forward pass to a self-contained
 * HTML file with heatmaps for every internal tensor.
 *
 * Usage:
 *   ./visualize --load <model.bin> [--seed "<text>"] [--out viz.html]
 *
 * The program loads a checkpoint trained by train_text / the `transformer`
 * CLI, runs ONE LM forward pass over the seed (treating it as both src and
 * tgt with a causal mask, mirroring `run_one_step` in text_lm.c), and then
 * writes:
 *   1. all per-layer activations from the TransformerCache (LN1/LN2/LN3
 *      outputs, residuals, FFN hidden, layer outputs, ...);
 *   2. all per-head attention weight matrices (encoder self-attn, decoder
 *      self-attn with the causal triangle, decoder cross-attn);
 *   3. token & positional embeddings;
 *   4. final logits over the vocabulary, with the Top-K predictions for
 *      the last position.
 *
 * The HTML embeds the JSON trace inline, so the output file works fully
 * offline (just open with file://) without any external assets.
 */

#include "nn_math.h"
#include "text_lm.h"
#include "tokenizer.h"
#include "transformer.h"

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ----- argv parsing ----------------------------------------------------- */
typedef struct {
    const char *load_path;
    const char *seed;
    const char *out_path;
    int max_tokens;
} Args;

static void print_usage(const char *argv0) {
    fprintf(stderr,
        "Usage:\n"
        "  %s --load <model.bin> [--seed \"text\"] [--out viz.html] [--max-tokens N]\n"
        "\n"
        "Runs ONE LM forward pass on the seed and writes a self-contained HTML\n"
        "page that visualises every internal tensor (token embeddings, positional\n"
        "encoding, per-layer LN/attention/FFN outputs, and final logits).\n",
        argv0);
}

static int parse_args(int argc, char **argv, Args *a) {
    a->load_path = NULL;
    a->seed = "the cat";
    a->out_path = "viz.html";
    a->max_tokens = 12;
    for (int i = 1; i < argc; i++) {
        const char *k = argv[i];
        const char *v = (i + 1 < argc) ? argv[i + 1] : NULL;
        if (!strcmp(k, "--help") || !strcmp(k, "-h")) { print_usage(argv[0]); return 0; }
        if (!strcmp(k, "--load")       && v) { a->load_path = v; i++; continue; }
        if (!strcmp(k, "--seed")       && v) { a->seed      = v; i++; continue; }
        if (!strcmp(k, "--out")        && v) { a->out_path  = v; i++; continue; }
        if (!strcmp(k, "--max-tokens") && v) { a->max_tokens = atoi(v); i++; continue; }
        fprintf(stderr, "Unknown arg: %s\n", k);
        print_usage(argv[0]);
        return -1;
    }
    if (!a->load_path) {
        fprintf(stderr, "--load <model.bin> is required.\n");
        print_usage(argv[0]);
        return -1;
    }
    if (a->max_tokens < 1) a->max_tokens = 1;
    return 1;
}

/* ----- JSON helpers ----------------------------------------------------- */
/* All emitters write directly into the HTML file inside a <script> tag,
 * so we can stream values without buffering the whole trace in memory. */

static void json_escape_str(FILE *fp, const char *s) {
    fputc('"', fp);
    if (!s) { fputc('"', fp); return; }
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        unsigned char c = *p;
        switch (c) {
            case '"':  fputs("\\\"", fp); break;
            case '\\': fputs("\\\\", fp); break;
            case '\n': fputs("\\n",  fp); break;
            case '\r': fputs("\\r",  fp); break;
            case '\t': fputs("\\t",  fp); break;
            default:
                if (c < 0x20) fprintf(fp, "\\u%04x", c);
                else fputc(c, fp);
        }
    }
    fputc('"', fp);
}

/* Emit a numeric matrix as {"shape":[rows,cols],"data":[...]} using %g
 * (compact, ~6 chars per value). */
static void emit_matrix(FILE *fp, int rows, int cols, const float *data) {
    fprintf(fp, "{\"shape\":[%d,%d],\"data\":[", rows, cols);
    int n = rows * cols;
    for (int i = 0; i < n; i++) {
        if (i) fputc(',', fp);
        float v = data[i];
        if (v != v) fputs("0", fp);                    /* NaN  -> 0 (defensive) */
        else if (v == (float)(int)v && fabsf(v) < 1e6f) fprintf(fp, "%d", (int)v);
        else fprintf(fp, "%.5g", v);
    }
    fputs("]}", fp);
}

/* For a Tensor3D with batch_size==1 we collapse the batch dimension and
 * emit just (seq_len, d_model). */
static void emit_tensor2d(FILE *fp, const Tensor3D *t) {
    emit_matrix(fp, t->seq_len, t->d_model, t->data);
}

/* Slice attention weights for one head out of the (1, num_heads, q*k) buffer. */
static void emit_attn_head(FILE *fp, const Tensor3D *attn, int head, int q_len, int kv_len) {
    const float *base = attn->data + (size_t)head * q_len * kv_len;
    emit_matrix(fp, q_len, kv_len, base);
}

/* ----- HTML template ---------------------------------------------------- */
/* The template is split in two parts: the head/CSS/JS prelude is written
 * before the JSON payload, the closing markup afterwards. */

static const char *HTML_HEAD =
"<!doctype html>\n"
"<html lang=\"en\">\n<head>\n"
"<meta charset=\"utf-8\">\n"
"<title>Transformer Forward Pass - Visualisation</title>\n"
"<style>\n"
"  :root { --bg:#0f1115; --panel:#171a21; --border:#262b35; --fg:#e6e6e6; --muted:#8b95a7; --accent:#6cb6ff; }\n"
"  html,body { margin:0; padding:0; background:var(--bg); color:var(--fg); font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; font-size:14px; line-height:1.5; }\n"
"  h1,h2,h3,h4 { font-weight:600; margin:0; }\n"
"  h1 { font-size:22px; padding:24px 28px 8px; }\n"
"  h2 { font-size:18px; color:var(--accent); margin:24px 0 12px; }\n"
"  h3 { font-size:15px; color:var(--fg); margin:12px 0 6px; }\n"
"  h4 { font-size:13px; color:var(--muted); margin:8px 0 4px; font-weight:500; text-transform:uppercase; letter-spacing:0.5px; }\n"
"  .container { max-width:1280px; margin:0 auto; padding:0 28px 60px; }\n"
"  .meta { color:var(--muted); padding:0 28px 16px; max-width:1280px; margin:0 auto; }\n"
"  .meta code { background:var(--panel); padding:2px 6px; border-radius:4px; color:var(--fg); }\n"
"  details { background:var(--panel); border:1px solid var(--border); border-radius:6px; margin:10px 0; padding:12px 16px; }\n"
"  details[open] { padding-bottom:18px; }\n"
"  summary { cursor:pointer; font-weight:600; color:var(--fg); list-style:none; }\n"
"  summary::before { content:'\\25B8'; color:var(--accent); margin-right:8px; transition:transform 0.15s; display:inline-block; }\n"
"  details[open] > summary::before { transform:rotate(90deg); }\n"
"  .desc { color:var(--muted); font-size:13px; margin:4px 0 10px; }\n"
"  .heatmap-wrap { overflow:auto; padding:8px 0; }\n"
"  canvas { image-rendering:pixelated; border:1px solid var(--border); display:block; }\n"
"  .stats { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; color:var(--muted); font-size:12px; padding:4px 0; }\n"
"  .legend { display:flex; align-items:center; gap:8px; font-family:ui-monospace,monospace; font-size:11px; color:var(--muted); margin-top:4px; }\n"
"  .legend-bar { display:inline-block; width:120px; height:10px; border:1px solid var(--border); }\n"
"  .grid2 { display:grid; grid-template-columns:repeat(auto-fit, minmax(280px, 1fr)); gap:12px; }\n"
"  .head-card { background:#1d2129; border:1px solid var(--border); border-radius:6px; padding:8px 10px; }\n"
"  .head-card h4 { margin-top:0; }\n"
"  .tokens { display:flex; flex-wrap:wrap; gap:6px; margin:8px 0 14px; font-family:ui-monospace,monospace; font-size:12px; }\n"
"  .tok { background:#1d2129; border:1px solid var(--border); padding:4px 8px; border-radius:4px; }\n"
"  .tok b { color:var(--accent); margin-right:6px; }\n"
"  .topk { font-family:ui-monospace,monospace; font-size:13px; }\n"
"  .topk-row { display:flex; align-items:center; gap:8px; padding:3px 0; }\n"
"  .topk-tok { width:90px; color:var(--accent); }\n"
"  .topk-prob { width:64px; color:var(--muted); text-align:right; }\n"
"  .topk-bar { background:#1d2129; height:14px; border-radius:3px; flex:1; overflow:hidden; }\n"
"  .topk-bar > div { background:var(--accent); height:100%; }\n"
"  .axis-x, .axis-y { font-family:ui-monospace,monospace; font-size:10px; color:var(--muted); }\n"
"  .axis-y { writing-mode:vertical-rl; transform:rotate(180deg); }\n"
"  /* ----- timeline toolbar ----- */\n"
"  .toolbar { position:sticky; top:0; z-index:20; background:rgba(12,14,18,0.96); backdrop-filter:blur(6px); border-bottom:1px solid var(--border); padding:10px 28px; display:flex; align-items:center; gap:12px; flex-wrap:wrap; }\n"
"  .toolbar button { background:var(--panel); color:var(--fg); border:1px solid var(--border); border-radius:4px; padding:6px 10px; cursor:pointer; font-size:13px; font-family:inherit; min-width:36px; }\n"
"  .toolbar button:hover { background:#222a36; }\n"
"  .toolbar button.primary { background:var(--accent); color:#0c0e12; border-color:var(--accent); font-weight:600; min-width:64px; }\n"
"  .toolbar button.primary:hover { background:#86c4ff; }\n"
"  .toolbar .progress { flex:1; min-width:200px; height:8px; background:var(--panel); border-radius:4px; overflow:hidden; cursor:pointer; }\n"
"  .toolbar .progress > div { background:var(--accent); height:100%; width:0; transition:width 0.25s; }\n"
"  .toolbar .step-label { font-family:ui-monospace,SFMono-Regular,monospace; font-size:12px; color:var(--muted); min-width:90px; text-align:right; }\n"
"  .toolbar .step-name { font-size:13px; color:var(--fg); flex-basis:100%; padding-top:4px; }\n"
"  .toolbar .step-name b { color:var(--accent); margin-right:6px; }\n"
"  .toolbar label { font-size:12px; color:var(--muted); display:flex; align-items:center; gap:6px; }\n"
"  .toolbar input[type=range] { width:100px; accent-color:var(--accent); }\n"
"  /* highlight pulse on the currently playing step */\n"
"  .step-active { box-shadow:0 0 0 2px var(--accent), 0 0 18px rgba(108,182,255,0.45); transition:box-shadow 0.3s; }\n"
"  @keyframes pulseBorder { 0%,100% { box-shadow:0 0 0 2px var(--accent), 0 0 8px rgba(108,182,255,0.3); } 50% { box-shadow:0 0 0 2px var(--accent), 0 0 22px rgba(108,182,255,0.7); } }\n"
"  .step-pulsing { animation:pulseBorder 1.2s ease-in-out infinite; }\n"
"</style>\n"
"</head>\n<body>\n"
"<div class=\"toolbar\" id=\"toolbar\">\n"
"  <button id=\"btn-prev\"  title=\"Previous step (Left arrow)\">&#9664; Prev</button>\n"
"  <button id=\"btn-play\"  class=\"primary\" title=\"Play / Pause (Space)\">&#9654; Play</button>\n"
"  <button id=\"btn-next\"  title=\"Next step (Right arrow)\">Next &#9654;</button>\n"
"  <button id=\"btn-reset\" title=\"Restart\">&#8634; Reset</button>\n"
"  <div class=\"progress\" id=\"progress\"><div></div></div>\n"
"  <div class=\"step-label\" id=\"step-label\">0 / 0</div>\n"
"  <label>Speed <input id=\"speed\" type=\"range\" min=\"300\" max=\"3000\" value=\"1200\" step=\"100\"></label>\n"
"  <div class=\"step-name\" id=\"step-name\"><b>Idle</b>Press Play to walk through the entire forward pass.</div>\n"
"</div>\n"
"<h1>Transformer Forward Pass</h1>\n"
"<div class=\"meta\" id=\"meta\"></div>\n"
"<div class=\"container\" id=\"app\"></div>\n"
"<script id=\"trace-data\" type=\"application/json\">\n";

/* JS rendering layer: pure vanilla, no external libs. */
static const char *HTML_TAIL =
"\n</script>\n"
"<script>\n"
"const TRACE = JSON.parse(document.getElementById('trace-data').textContent);\n"
"\n"
"// Viridis-ish colourmap, 0..1 -> rgb.\n"
"const CMAP = [[68,1,84],[71,44,122],[59,81,139],[44,113,142],[33,144,140],[39,173,129],[92,200,99],[170,220,50],[253,231,37]];\n"
"function vir(t) {\n"
"  if (!isFinite(t)) t = 0;\n"
"  t = Math.max(0, Math.min(1, t));\n"
"  const f = t * (CMAP.length - 1);\n"
"  const i = Math.floor(f), a = CMAP[i], b = CMAP[Math.min(CMAP.length-1, i+1)];\n"
"  const k = f - i;\n"
"  return [a[0]+(b[0]-a[0])*k, a[1]+(b[1]-a[1])*k, a[2]+(b[2]-a[2])*k];\n"
"}\n"
"// Diverging blue-white-red for signed data.\n"
"function div(t) {\n"
"  if (!isFinite(t)) t = 0;\n"
"  t = Math.max(-1, Math.min(1, t));\n"
"  if (t >= 0) { const k = t; return [255, 255-150*k, 255-200*k]; }\n"
"  else { const k = -t; return [255-200*k, 255-150*k, 255]; }\n"
"}\n"
"\n"
"function stats(arr) {\n"
"  let mn = Infinity, mx = -Infinity, sum = 0, n = arr.length;\n"
"  for (let i=0;i<n;i++) { const v = arr[i]; if (v<mn) mn=v; if (v>mx) mx=v; sum+=v; }\n"
"  return {min:mn, max:mx, mean:sum/n};\n"
"}\n"
"\n"
"function fmt(x) { if (Math.abs(x)<1e-3 || Math.abs(x)>=1e4) return x.toExponential(2); return x.toFixed(3); }\n"
"\n"
"// Render a (rows x cols) heatmap onto a canvas. `signed` chooses cmap.\n"
"function heatmap(parent, mat, opts) {\n"
"  opts = opts || {};\n"
"  const [rows, cols] = mat.shape;\n"
"  const data = mat.data;\n"
"  const s = stats(data);\n"
"  const signed = opts.signed !== undefined ? opts.signed : (s.min < 0 && s.max > 0);\n"
"  const peak = Math.max(Math.abs(s.min), Math.abs(s.max), 1e-9);\n"
"  const cell = opts.cell || Math.max(2, Math.min(28, Math.floor(720 / Math.max(cols, 8))));\n"
"  const W = cols * cell, H = rows * cell;\n"
"  const wrap = document.createElement('div'); wrap.className = 'heatmap-wrap';\n"
"  const canvas = document.createElement('canvas');\n"
"  canvas.width = W; canvas.height = H;\n"
"  const ctx = canvas.getContext('2d');\n"
"  const img = ctx.createImageData(W, H);\n"
"  for (let r = 0; r < rows; r++) {\n"
"    for (let c = 0; c < cols; c++) {\n"
"      const v = data[r*cols + c];\n"
"      let rgb;\n"
"      if (signed) rgb = div(v / peak);\n"
"      else rgb = vir((v - s.min) / Math.max(1e-9, s.max - s.min));\n"
"      // Fill the cell block in the ImageData directly.\n"
"      for (let dy = 0; dy < cell; dy++) {\n"
"        for (let dx = 0; dx < cell; dx++) {\n"
"          const px = ((r*cell+dy)*W + (c*cell+dx)) * 4;\n"
"          img.data[px]   = rgb[0]|0;\n"
"          img.data[px+1] = rgb[1]|0;\n"
"          img.data[px+2] = rgb[2]|0;\n"
"          img.data[px+3] = 255;\n"
"        }\n"
"      }\n"
"    }\n"
"  }\n"
"  ctx.putImageData(img, 0, 0);\n"
"  wrap.appendChild(canvas);\n"
"  // hover tooltip\n"
"  const tip = document.createElement('div'); tip.className='stats'; tip.textContent='hover for value';\n"
"  canvas.addEventListener('mousemove', e => {\n"
"    const rect = canvas.getBoundingClientRect();\n"
"    const x = Math.floor((e.clientX - rect.left) / cell);\n"
"    const y = Math.floor((e.clientY - rect.top) / cell);\n"
"    if (x>=0 && x<cols && y>=0 && y<rows) {\n"
"      tip.textContent = `[${y},${x}] = ${fmt(data[y*cols+x])}`;\n"
"    }\n"
"  });\n"
"  parent.appendChild(wrap);\n"
"  parent.appendChild(tip);\n"
"  const info = document.createElement('div'); info.className = 'stats';\n"
"  info.textContent = `shape ${rows}x${cols}  min=${fmt(s.min)}  max=${fmt(s.max)}  mean=${fmt(s.mean)}  cmap=${signed?'diverging':'viridis'}`;\n"
"  parent.appendChild(info);\n"
"}\n"
"\n"
"function section(parent, title, desc, openByDefault) {\n"
"  const det = document.createElement('details');\n"
"  if (openByDefault) det.setAttribute('open','');\n"
"  const sum = document.createElement('summary'); sum.textContent = title;\n"
"  det.appendChild(sum);\n"
"  if (desc) { const d = document.createElement('div'); d.className='desc'; d.textContent = desc; det.appendChild(d); }\n"
"  parent.appendChild(det);\n"
"  // Tag every section as a timeline step. The renderer below collects\n"
"  // them in document order (so encoder layer 0 -> ... -> logits) and the\n"
"  // toolbar walks through them sequentially.\n"
"  det.dataset.stepTitle = title;\n"
"  det.dataset.stepDesc  = desc || '';\n"
"  return det;\n"
"}\n"
"\n"
"function renderTokens(parent, ids, tokens) {\n"
"  const wrap = document.createElement('div'); wrap.className='tokens';\n"
"  ids.forEach((id,i) => {\n"
"    const sp = document.createElement('span'); sp.className='tok';\n"
"    const b = document.createElement('b'); b.textContent = '#'+i;\n"
"    sp.appendChild(b); sp.appendChild(document.createTextNode(`${id} \\u2192 ${tokens[i]}`));\n"
"    wrap.appendChild(sp);\n"
"  });\n"
"  parent.appendChild(wrap);\n"
"}\n"
"\n"
"function renderAttn(parent, attn, tokens) {\n"
"  // attn = {heads:[mat,...]}\n"
"  const grid = document.createElement('div'); grid.className='grid2'; parent.appendChild(grid);\n"
"  attn.heads.forEach((mat, h) => {\n"
"    const card = document.createElement('div'); card.className='head-card';\n"
"    const t = document.createElement('h4'); t.textContent = `head ${h}  (rows=query, cols=key)`;\n"
"    card.appendChild(t);\n"
"    heatmap(card, mat, {signed:false});\n"
"    grid.appendChild(card);\n"
"  });\n"
"}\n"
"\n"
"function renderTopK(parent, logitsMat, vocab, K) {\n"
"  const [rows, cols] = logitsMat.shape;\n"
"  // last position\n"
"  const off = (rows-1) * cols;\n"
"  const logits = logitsMat.data.slice(off, off+cols);\n"
"  // softmax\n"
"  let mx = -Infinity; for (const v of logits) if (v>mx) mx=v;\n"
"  let sum = 0; const probs = logits.map(v => { const e = Math.exp(v - mx); sum += e; return e; });\n"
"  for (let i=0;i<probs.length;i++) probs[i] /= sum;\n"
"  const idx = probs.map((p,i)=>[p,i]).sort((a,b)=>b[0]-a[0]).slice(0, K);\n"
"  const wrap = document.createElement('div'); wrap.className='topk';\n"
"  idx.forEach(([p,i])=> {\n"
"    const row = document.createElement('div'); row.className='topk-row';\n"
"    const tk = document.createElement('div'); tk.className='topk-tok'; tk.textContent = JSON.stringify(vocab[i]||('#'+i));\n"
"    const pp = document.createElement('div'); pp.className='topk-prob'; pp.textContent = (p*100).toFixed(2)+'%';\n"
"    const bar = document.createElement('div'); bar.className='topk-bar';\n"
"    const fill = document.createElement('div'); fill.style.width = (p*100).toFixed(1)+'%'; bar.appendChild(fill);\n"
"    row.appendChild(tk); row.appendChild(pp); row.appendChild(bar);\n"
"    wrap.appendChild(row);\n"
"  });\n"
"  parent.appendChild(wrap);\n"
"}\n"
"\n"
"// ----- main render -----\n"
"const meta = document.getElementById('meta');\n"
"meta.innerHTML = `Model <code>${TRACE.model.path}</code> &middot; d_model=${TRACE.model.d_model} &middot; heads=${TRACE.model.num_heads} &middot; enc/dec=${TRACE.model.encoder_layers}/${TRACE.model.decoder_layers} &middot; vocab=${TRACE.model.vocab_size} &middot; seed=${JSON.stringify(TRACE.input.seed)} &middot; tokens=${TRACE.input.src_ids.length}`;\n"
"\n"
"const app = document.getElementById('app');\n"
"\n"
"// Input section\n"
"{\n"
"  const s = section(app, '1. Input tokenisation', 'The encoder and decoder both consume the same token sequence (LM mode). Each token id is mapped to a row of the embedding matrix.', true);\n"
"  renderTokens(s, TRACE.input.src_ids, TRACE.input.src_tokens);\n"
"}\n"
"\n"
"// Embeddings\n"
"{\n"
"  const s = section(app, '2. Token embedding (lookup)', 'Each row corresponds to one input token. Cells show the d_model-dim embedding vector. Columns = embedding dimensions.', false);\n"
"  heatmap(s, TRACE.src_embedding);\n"
"}\n"
"{\n"
"  const s = section(app, '3. Positional encoding (sinusoidal)', 'Sinusoidal table evaluated at the seed positions. Note the alternating sin/cos columns producing the characteristic stripes.', false);\n"
"  heatmap(s, TRACE.src_pe);\n"
"}\n"
"{\n"
"  const s = section(app, '4. Embedding + PE (encoder input)', 'Sum of the previous two. Fed into the first encoder layer.', false);\n"
"  heatmap(s, TRACE.src_post_pe);\n"
"}\n"
"\n"
"// Encoder layers\n"
"TRACE.encoder_layers.forEach((L, i) => {\n"
"  const root = section(app, `Encoder layer ${i}`, 'Pre-LN block: y1 = x + Attn(LN(x)); y2 = y1 + FFN(LN(y1))', i===0);\n"
"  { const s = section(root, 'LN1 output (input to self-attention)', '', false); heatmap(s, L.ln1_out); }\n"
"  { const s = section(root, 'Self-attention weights (per head, after softmax)', 'rows = query position, cols = key position. Bright = high attention.', i===0);\n"
"    renderAttn(s, L.self_attn, TRACE.input.src_tokens); }\n"
"  { const s = section(root, 'Post-attention residual (x + Attn(LN(x)))', '', false); heatmap(s, L.residual); }\n"
"  { const s = section(root, 'LN2 output (input to FFN)', '', false); heatmap(s, L.ln2_out); }\n"
"  { const s = section(root, `FFN hidden activations (d_ff=${L.ffn_hidden_pre.shape[1]})`, 'Activation values BEFORE the second linear projection. Often very sparse / heavy-tailed.', false); heatmap(s, L.ffn_hidden_pre); }\n"
"  { const s = section(root, 'Layer output (encoder representation for this depth)', '', false); heatmap(s, L.ff_out); }\n"
"});\n"
"\n"
"// Decoder embeddings (same seed in LM mode)\n"
"{\n"
"  const s = section(app, 'Decoder input: embedding + PE', '', false);\n"
"  heatmap(s, TRACE.tgt_post_pe);\n"
"}\n"
"\n"
"// Decoder layers\n"
"TRACE.decoder_layers.forEach((L, i) => {\n"
"  const root = section(app, `Decoder layer ${i}`, 'Pre-LN block: y1 = x + SelfAttn(LN(x)); y2 = y1 + CrossAttn(LN(y1), enc_out); y3 = y2 + FFN(LN(y2))', i===0);\n"
"  { const s = section(root, 'LN1 output (input to masked self-attention)', '', false); heatmap(s, L.ln1_out); }\n"
"  { const s = section(root, 'Masked self-attention weights (causal, per head)', 'Lower-triangular pattern: position i can only attend to positions <= i.', i===0);\n"
"    renderAttn(s, L.self_attn, TRACE.input.src_tokens); }\n"
"  { const s = section(root, 'After self-attention residual', '', false); heatmap(s, L.residual1); }\n"
"  { const s = section(root, 'LN2 output (input to cross-attention)', '', false); heatmap(s, L.ln2_out); }\n"
"  { const s = section(root, 'Cross-attention weights (decoder query -> encoder key)', 'rows = decoder positions, cols = encoder positions. Shows what each decoder step looks at in the source.', i===0);\n"
"    renderAttn(s, L.cross_attn, TRACE.input.src_tokens); }\n"
"  { const s = section(root, 'After cross-attention residual', '', false); heatmap(s, L.residual2); }\n"
"  { const s = section(root, 'LN3 output (input to FFN)', '', false); heatmap(s, L.ln3_out); }\n"
"  { const s = section(root, `FFN hidden activations (d_ff=${L.ffn_hidden_pre.shape[1]})`, '', false); heatmap(s, L.ffn_hidden_pre); }\n"
"  { const s = section(root, 'Layer output', '', false); heatmap(s, L.ff_out); }\n"
"});\n"
"\n"
"// Final logits + top-K\n"
"{\n"
"  const s = section(app, 'Final logits (decoder_out @ logit_head^T)', `Shape ${TRACE.logits.shape[0]} x ${TRACE.logits.shape[1]}. Each row is the unnormalised distribution over the vocabulary for one output position.`, true);\n"
"  heatmap(s, TRACE.logits, {signed:true});\n"
"}\n"
"{\n"
"  const s = section(app, `Top-${Math.min(15, TRACE.model.vocab_size)} predictions for the LAST position`, 'Softmax over the final logits row. This is what the model would sample from to produce the next token.', true);\n"
"  renderTopK(s, TRACE.logits, TRACE.vocab, Math.min(15, TRACE.model.vocab_size));\n"
"}\n"
"\n"
"// ===== Timeline auto-player =====\n"
"// Collects every <details> in document order. Pressing Play walks through\n"
"// them: opens the section (and all ancestors), scrolls it into view, and\n"
"// briefly pulses its border. Speed slider controls dwell time.\n"
"const STEPS = Array.from(document.querySelectorAll('details[data-step-title]'));\n"
"let stepIdx = -1;\n"
"let playing = false;\n"
"let timer = null;\n"
"\n"
"const $prev   = document.getElementById('btn-prev');\n"
"const $play   = document.getElementById('btn-play');\n"
"const $next   = document.getElementById('btn-next');\n"
"const $reset  = document.getElementById('btn-reset');\n"
"const $progress = document.getElementById('progress');\n"
"const $progressFill = $progress.firstElementChild;\n"
"const $label  = document.getElementById('step-label');\n"
"const $name   = document.getElementById('step-name');\n"
"const $speed  = document.getElementById('speed');\n"
"\n"
"function clearActive() {\n"
"  document.querySelectorAll('.step-active, .step-pulsing').forEach(el => {\n"
"    el.classList.remove('step-active'); el.classList.remove('step-pulsing');\n"
"  });\n"
"}\n"
"\n"
"function gotoStep(i, opts) {\n"
"  if (i < 0 || i >= STEPS.length) return;\n"
"  stepIdx = i;\n"
"  const target = STEPS[i];\n"
"  // Open every ancestor <details> so the target is actually visible.\n"
"  for (let p = target; p; p = p.parentElement) {\n"
"    if (p.tagName === 'DETAILS' && !p.hasAttribute('open')) p.setAttribute('open','');\n"
"  }\n"
"  clearActive();\n"
"  target.classList.add('step-active');\n"
"  if (playing) target.classList.add('step-pulsing');\n"
"  // Scroll so the section header sits just under the toolbar.\n"
"  const tbH = document.getElementById('toolbar').offsetHeight + 12;\n"
"  const rect = target.getBoundingClientRect();\n"
"  const top = rect.top + window.scrollY - tbH;\n"
"  window.scrollTo({ top, behavior: opts && opts.instant ? 'auto' : 'smooth' });\n"
"  // Update toolbar UI.\n"
"  $label.textContent = `${i+1} / ${STEPS.length}`;\n"
"  $progressFill.style.width = ((i+1) / STEPS.length * 100).toFixed(2) + '%';\n"
"  $name.innerHTML = `<b>Step ${i+1}</b>${target.dataset.stepTitle}` + (target.dataset.stepDesc ? ` &mdash; <span style=\"color:var(--muted)\">${target.dataset.stepDesc}</span>` : '');\n"
"}\n"
"\n"
"function setPlaying(p) {\n"
"  playing = p;\n"
"  $play.innerHTML = playing ? '&#10074;&#10074; Pause' : '&#9654; Play';\n"
"  if (playing) {\n"
"    if (stepIdx >= STEPS.length - 1) stepIdx = -1;       // restart from the top\n"
"    tick();\n"
"  } else {\n"
"    clearTimeout(timer); timer = null;\n"
"    document.querySelectorAll('.step-pulsing').forEach(el => el.classList.remove('step-pulsing'));\n"
"  }\n"
"}\n"
"\n"
"function tick() {\n"
"  if (!playing) return;\n"
"  const next = stepIdx + 1;\n"
"  if (next >= STEPS.length) { setPlaying(false); return; }\n"
"  gotoStep(next);\n"
"  timer = setTimeout(tick, parseInt($speed.value, 10));\n"
"}\n"
"\n"
"$prev.addEventListener('click', () => { setPlaying(false); gotoStep(Math.max(0, stepIdx-1)); });\n"
"$next.addEventListener('click', () => { setPlaying(false); gotoStep(Math.min(STEPS.length-1, stepIdx+1)); });\n"
"$play.addEventListener('click', () => setPlaying(!playing));\n"
"$reset.addEventListener('click', () => { setPlaying(false); gotoStep(0, {instant:true}); });\n"
"// Click the progress bar to scrub.\n"
"$progress.addEventListener('click', e => {\n"
"  const r = $progress.getBoundingClientRect();\n"
"  const ratio = (e.clientX - r.left) / r.width;\n"
"  const i = Math.max(0, Math.min(STEPS.length-1, Math.floor(ratio * STEPS.length)));\n"
"  setPlaying(false); gotoStep(i);\n"
"});\n"
"// Keyboard shortcuts: space=play/pause, arrows=step.\n"
"document.addEventListener('keydown', e => {\n"
"  if (e.target.tagName === 'INPUT') return;\n"
"  if (e.code === 'Space')      { e.preventDefault(); setPlaying(!playing); }\n"
"  else if (e.code === 'ArrowRight') { setPlaying(false); gotoStep(Math.min(STEPS.length-1, stepIdx+1)); }\n"
"  else if (e.code === 'ArrowLeft')  { setPlaying(false); gotoStep(Math.max(0, stepIdx-1)); }\n"
"});\n"
"\n"
"// Initialise: show the first step in the toolbar but don't auto-scroll yet.\n"
"if (STEPS.length > 0) {\n"
"  $label.textContent = `0 / ${STEPS.length}`;\n"
"  $name.innerHTML = `<b>Idle</b>Press Play to walk through ${STEPS.length} computation steps, or use \\u2190 / \\u2192 / Space.`;\n"
"}\n"
"</script>\n"
"</body>\n</html>\n";

/* ----- main forward pass + dump ---------------------------------------- */

int main(int argc, char **argv) {
    Args a;
    int pa = parse_args(argc, argv, &a);
    if (pa <= 0) return pa == 0 ? 0 : 1;

    Transformer *model = NULL;
    TrainingState *ts = NULL;
    if (text_lm_load(a.load_path, &model, &ts) != 0 || !model) {
        fprintf(stderr, "Failed to load model: %s\n", a.load_path);
        return 1;
    }
    if (model->config.vocab_size <= 0) {
        fprintf(stderr, "Loaded model is in legacy d_model-input mode; visualizer requires LM mode (vocab_size > 0).\n");
        return 1;
    }

    char tok_path[1024];
    snprintf(tok_path, sizeof(tok_path), "%s.tok", a.load_path);
    Tokenizer *tok = tokenizer_load(tok_path);
    if (!tok) {
        fprintf(stderr, "Failed to load tokenizer: %s\n", tok_path);
        return 1;
    }

    /* Encode the seed. We cap at max_tokens to keep the visualisation
     * page light (a 50-token attention matrix per head per layer adds up
     * quickly). The cap also stays well within model->config.max_len. */
    int cap = model->config.max_len > 0 ? model->config.max_len : 32;
    if (a.max_tokens < cap) cap = a.max_tokens;
    int *ids = (int *)malloc(sizeof(int) * (size_t)cap);
    int n = tokenizer_encode(tok, a.seed, ids, cap);
    if (n <= 0) { ids[0] = TOK_BOS; n = 1; }

    int d_model    = model->config.d_model;
    int num_heads  = model->config.num_heads;
    int enc_layers = model->config.encoder_layers;
    int dec_layers = model->config.decoder_layers;
    int vocab_size = model->config.vocab_size;

    /* Independently compute embedding + PE so we can visualise them; the
     * actual forward path inside transformer.c will redo the same lookup
     * (no shortcut to the post-PE buffer is exposed). */
    Tensor3D src_emb_only = tensor_create(1, n, d_model);
    for (int s = 0; s < n; s++) {
        int id = ids[s];
        if (id < 0 || id >= model->token_embedding->rows) id = 0;
        memcpy(src_emb_only.data + (size_t)s * d_model,
               model->token_embedding->data + (size_t)id * d_model,
               sizeof(float) * (size_t)d_model);
    }
    Tensor3D pe_only = positional_encoding_forward(model->encoder->pe, n, 1);
    Tensor3D src_post_pe = tensor_clone(&src_emb_only);
    for (int i = 0; i < n * d_model; i++) src_post_pe.data[i] += pe_only.data[i];

    /* Run the LM forward pass with a cache so we can read out every
     * intermediate. We use a causal mask on both encoder and decoder so
     * the trace matches the no-cache LM generation path. */
    Tensor3D mask = mask_causal_create(n);
    TransformerCache cache = {0};
    Tensor3D logits = transformer_forward_lm(model,
                                             ids, n,
                                             ids, n,
                                             1,
                                             &mask, &mask,
                                             &cache);

    /* ------- write HTML ------- */
    FILE *fp = fopen(a.out_path, "wb");
    if (!fp) { fprintf(stderr, "Failed to open %s for writing.\n", a.out_path); return 1; }

    fputs(HTML_HEAD, fp);

    /* Open root JSON object. */
    fputs("{\n", fp);

    /* model { ... } */
    fputs("\"model\":{", fp);
    fputs("\"path\":", fp);     json_escape_str(fp, a.load_path);
    fprintf(fp, ",\"d_model\":%d,\"d_ff\":%d,\"num_heads\":%d,\"encoder_layers\":%d,\"decoder_layers\":%d,\"vocab_size\":%d,\"max_len\":%d",
            d_model, model->config.d_ff, num_heads, enc_layers, dec_layers, vocab_size, model->config.max_len);
    fputs("},\n", fp);

    /* input { ... } */
    fputs("\"input\":{\"seed\":", fp); json_escape_str(fp, a.seed);
    fputs(",\"src_ids\":[", fp);
    for (int i = 0; i < n; i++) { if (i) fputc(',', fp); fprintf(fp, "%d", ids[i]); }
    fputs("],\"src_tokens\":[", fp);
    for (int i = 0; i < n; i++) {
        if (i) fputc(',', fp);
        const char *t = tokenizer_id_to_token(tok, ids[i]);
        json_escape_str(fp, t ? t : "?");
    }
    fputs("]},\n", fp);

    /* vocab labels (for top-K). */
    fputs("\"vocab\":[", fp);
    for (int i = 0; i < vocab_size; i++) {
        if (i) fputc(',', fp);
        const char *t = tokenizer_id_to_token(tok, i);
        json_escape_str(fp, t ? t : "?");
    }
    fputs("],\n", fp);

    /* Embedding tensors. */
    fputs("\"src_embedding\":", fp); emit_tensor2d(fp, &src_emb_only); fputs(",\n", fp);
    fputs("\"src_pe\":",        fp); emit_tensor2d(fp, &pe_only);      fputs(",\n", fp);
    fputs("\"src_post_pe\":",   fp); emit_tensor2d(fp, &src_post_pe);  fputs(",\n", fp);
    /* In LM mode the decoder consumes the same token stream, so for
     * brevity we report the same buffer (after PE). */
    fputs("\"tgt_post_pe\":",   fp); emit_tensor2d(fp, &src_post_pe);  fputs(",\n", fp);

    /* Encoder layers. */
    fputs("\"encoder_layers\":[", fp);
    for (int i = 0; i < enc_layers; i++) {
        EncoderLayerCache *L = &cache.encoder_cache.layer_caches[i];
        if (i) fputc(',', fp);
        fputs("{", fp);
        fputs("\"ln1_out\":",        fp); emit_tensor2d(fp, &L->ln1_out);
        fputs(",\"residual\":",      fp); emit_tensor2d(fp, &L->residual);
        fputs(",\"ln2_out\":",       fp); emit_tensor2d(fp, &L->ln2_out);
        fputs(",\"ffn_hidden_pre\":",fp); emit_tensor2d(fp, &L->ffn_hidden_pre);
        fputs(",\"ff_out\":",        fp); emit_tensor2d(fp, &L->ff_out);
        /* attention weights: split per head */
        fputs(",\"self_attn\":{\"heads\":[", fp);
        for (int h = 0; h < num_heads; h++) {
            if (h) fputc(',', fp);
            emit_attn_head(fp, &L->self_attn_cache.attn_weights, h, n, n);
        }
        fputs("]}", fp);
        fputs("}", fp);
    }
    fputs("],\n", fp);

    /* Decoder layers. */
    fputs("\"decoder_layers\":[", fp);
    for (int i = 0; i < dec_layers; i++) {
        DecoderLayerCache *L = &cache.decoder_cache.layer_caches[i];
        if (i) fputc(',', fp);
        fputs("{", fp);
        fputs("\"ln1_out\":",        fp); emit_tensor2d(fp, &L->ln1_out);
        fputs(",\"residual1\":",     fp); emit_tensor2d(fp, &L->residual1);
        fputs(",\"ln2_out\":",       fp); emit_tensor2d(fp, &L->ln2_out);
        fputs(",\"residual2\":",     fp); emit_tensor2d(fp, &L->residual2);
        fputs(",\"ln3_out\":",       fp); emit_tensor2d(fp, &L->ln3_out);
        fputs(",\"ffn_hidden_pre\":",fp); emit_tensor2d(fp, &L->ffn_hidden_pre);
        fputs(",\"ff_out\":",        fp); emit_tensor2d(fp, &L->ff_out);
        fputs(",\"self_attn\":{\"heads\":[", fp);
        for (int h = 0; h < num_heads; h++) {
            if (h) fputc(',', fp);
            emit_attn_head(fp, &L->self_attn_cache.attn_weights, h, n, n);
        }
        fputs("]}", fp);
        fputs(",\"cross_attn\":{\"heads\":[", fp);
        for (int h = 0; h < num_heads; h++) {
            if (h) fputc(',', fp);
            emit_attn_head(fp, &L->cross_attn_cache.attn_weights, h, n, n);
        }
        fputs("]}", fp);
        fputs("}", fp);
    }
    fputs("],\n", fp);

    /* Final logits. */
    fputs("\"logits\":", fp); emit_tensor2d(fp, &logits); fputs("\n", fp);

    /* Close root JSON object. */
    fputs("}\n", fp);

    fputs(HTML_TAIL, fp);
    fclose(fp);

    fprintf(stderr,
        "Wrote %s\n"
        "  seed         : %s\n"
        "  tokens       : %d\n"
        "  encoder/dec  : %d / %d layers, %d heads, d_model=%d\n"
        "  vocab        : %d\n"
        "Open the file in any browser (no server required).\n",
        a.out_path, a.seed, n, enc_layers, dec_layers, num_heads, d_model, vocab_size);

    /* Cleanup */
    tensor_free(&logits);
    tensor_free(&mask);
    tensor_free(&src_emb_only);
    tensor_free(&pe_only);
    tensor_free(&src_post_pe);
    transformer_cache_free(&cache, enc_layers, dec_layers);
    free(ids);
    tokenizer_free(tok);
    text_lm_free_session(model, ts);
    return 0;
}
