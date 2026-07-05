#!/usr/bin/env python3
"""cadence cv_r1 インタラクティブデモ — 話者マップから選んで合成 / 複数話者ミックス

学習話者を 2 次元マップ（x-vector PCA。無ければ emb_g PCA に自動フォールバック）に配置し、
話者を選んで単一話者合成、または複数話者の emb_g 重み付き混合で virtual 話者を合成する。

起動方法（どちらか）:
  MODEL_DIR=/path/to/model_assets/model_name python app.py     # ローカル / Colab
  HF_REPO=slp-hu/cadence-cv_r1 python app.py                    # HF Hub から取得（Spaces 用）

MODEL_DIR に必要なファイル:
  *_e*_s*.safetensors / config.json / style_vectors.npy
  任意: trained_speakers.json（話者候補の限定） / speaker_map.json（x-vector 2D 座標）
"""
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
import gradio as gr

# ---------------- モデル解決 ----------------
def resolve_model_dir() -> Path:
    md = os.environ.get("MODEL_DIR")
    if md:
        return Path(md)
    repo = os.environ.get("HF_REPO")
    assert repo, "環境変数 MODEL_DIR か HF_REPO を設定すること"
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo))

MODEL_DIR = resolve_model_dir()
_cks = sorted(MODEL_DIR.glob("*_e*_s*.safetensors"),
              key=lambda p: int(re.search(r"_s(\d+)\.safetensors$", p.name).group(1)))
assert _cks, f"safetensors が無い: {MODEL_DIR}"
CKPT = _cks[-1]
CONFIG, SV_PATH = MODEL_DIR / "config.json", MODEL_DIR / "style_vectors.npy"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from style_bert_vits2.tts_model import TTSModel
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.constants import Languages
from style_bert_vits2.models.infer import get_text

bert_models.load_model(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")
bert_models.load_tokenizer(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")
model = TTSModel(model_path=CKPT, config_path=CONFIG, style_vec_path=SV_PATH, device=DEVICE)
model.load()
net_g = model.net_g
net_g.eval()
hps = model.hyper_parameters
NEUTRAL = model.get_style_vector(0, 1.0)
spk2id = hps.data.spk2id
SR = hps.data.sampling_rate

_tsj = MODEL_DIR / "trained_speakers.json"
SPEAKERS = (json.load(open(_tsj, encoding="utf-8")) if _tsj.exists() else sorted(spk2id))

# ---------------- 2D 座標（x-vector map → 無ければ emb_g PCA）----------------
def pca2(mat: np.ndarray) -> np.ndarray:
    x = mat - mat.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    y = x @ vt[:2].T
    return y / (np.abs(y).max(0, keepdims=True) + 1e-9)

_map = next((d / "speaker_map.json" for d in (MODEL_DIR, MODEL_DIR.parent)
             if (d / "speaker_map.json").exists()), MODEL_DIR / "speaker_map.json")
if _map.exists():
    _m = json.load(open(_map, encoding="utf-8"))
    COORD = {s: _m[s] for s in SPEAKERS if s in _m and s != "_meta"}
    _method = (_m.get("_meta") or {}).get("method")
    MAP_SRC = (f"x-vector {_method.upper()}（事前計算）" if _method
               else "x-vector（事前計算・手法記録なし = 旧版ファイル）")
else:
    emb = net_g.emb_g.weight.data.cpu().float().numpy()
    xy = pca2(np.stack([emb[spk2id[s]] for s in SPEAKERS]))
    COORD = {s: xy[i].tolist() for i, s in enumerate(SPEAKERS)}
    MAP_SRC = "emb_g PCA（フォールバック。x-vector map は demo/make_speaker_map.py で生成）"
# ---------------- マップ描画（画像方式）----------------
# gradio 4.44 の ScatterPlot はクリック選択イベントが実質使えないため、
# matplotlib でサーバ側描画した画像 + gr.Image のクリック（ピクセル座標）で選択する。
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_MAP_SPKS = list(COORD.keys())
_MAP_XY = np.array([COORD[s] for s in _MAP_SPKS], dtype=float)   # (N,2) in [-1,1]

# 話者メタ（任意）: {"cv_0001": "female", ...} または {"cv_0001": {"gender": "female"}, ...}
_meta_p = next((d / "speaker_meta.json" for d in (MODEL_DIR, MODEL_DIR.parent)
                if (d / "speaker_meta.json").exists()), None)
def _norm_gender(v):
    g = (v.get("gender") if isinstance(v, dict) else v) or ""
    g = str(g).lower()
    return "female" if g.startswith(("f", "女")) else "male" if g.startswith(("m", "男")) else "unknown"
GENDER = ({s: _norm_gender(v) for s, v in json.load(open(_meta_p, encoding="utf-8")).items()}
          if _meta_p else {})
G_COLOR = {"female": "#e0705f", "male": "#5b8def", "unknown": "#9aa0a6"}
G_LABEL = {"female": "female", "male": "male", "unknown": "unknown"}

def _make_fig(selected):
    sel_set = set(selected or [])
    fig, ax = plt.subplots(figsize=(6.4, 6.4), dpi=100)
    if GENDER:
        for g_key in ("female", "male", "unknown"):
            idx = [i for i, s in enumerate(_MAP_SPKS) if s not in sel_set
                   and GENDER.get(s, "unknown") == g_key]
            if idx:
                ax.scatter(_MAP_XY[idx, 0], _MAP_XY[idx, 1], s=26, c=G_COLOR[g_key],
                           alpha=0.85, linewidths=0, label=G_LABEL[g_key])
        ax.legend(loc="lower right", fontsize=8, framealpha=0.85)
    else:
        base = [s not in sel_set for s in _MAP_SPKS]
        ax.scatter(_MAP_XY[base, 0], _MAP_XY[base, 1], s=26, c="#6e9bd8", alpha=0.85, linewidths=0)
    for s in sel_set:
        if s in COORD:
            x, y = COORD[s]
            fill = G_COLOR.get(GENDER.get(s, "unknown"), "#e04b3a") if GENDER else "#e04b3a"
            ax.scatter([x], [y], s=95, c=fill, edgecolors="#c81e1e", linewidths=2.0, zorder=3)
            ax.annotate(s, (x, y), xytext=(6, 6), textcoords="offset points",
                        fontsize=9, color="#a12315", zorder=4)
    ax.set_xlim(-1.12, 1.12); ax.set_ylim(-1.12, 1.12)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("click = select nearest speaker / click again = deselect", fontsize=10, color="#555")
    fig.tight_layout(pad=0.4)
    return fig, ax

# 各話者のピクセル座標を一度だけ計算（座標は選択状態に依らず固定）
_f, _a = _make_fig([])
_f.canvas.draw()
_W, _H = _f.canvas.get_width_height()
_disp = _a.transData.transform(_MAP_XY)              # 表示座標（原点は左下）
SPK_PIX = {s: (float(px), float(_H - py)) for s, (px, py) in zip(_MAP_SPKS, _disp)}  # 画像座標（原点左上）
plt.close(_f)

def render_map(selected):
    fig, _ = _make_fig(selected)
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[:, :, :3].copy()
    plt.close(fig)
    return img

CLICK_RADIUS_PX = 16

def nearest_speaker(x, y):
    best, bd = None, 1e18
    for s, (px, py) in SPK_PIX.items():
        d = (px - x) ** 2 + (py - y) ** 2
        if d < bd:
            best, bd = s, d
    return best if bd <= CLICK_RADIUS_PX ** 2 else None

# ---------------- 元話者の参照クリップ（あれば選択に連動して再生）----------------
def _find_ref_dir():
    """参照クリップはモデル非依存（話者の属性）なので、モデル直下 → 共有置き場の順に探す"""
    for d in (MODEL_DIR / "reference_clips", MODEL_DIR.parent / "reference_clips"):
        if (d / "reference_clips.json").exists():
            return d
    return MODEL_DIR / "reference_clips"   # 見つからない場合のダミー（REF は空になる）

REF_DIR = _find_ref_dir()
_refj = REF_DIR / "reference_clips.json"
REF = json.load(open(_refj, encoding="utf-8")) if _refj.exists() else {}

MAX_REF = 4   # 同時表示する元音声プレーヤー数

def _one_clip(spk):
    """1話者分の (audio, キャプション)。無ければ (None, 理由)"""
    info = REF.get(spk)
    if not info:
        return None, (f"{spk}: 元音声サンプルなし" +
                      ("" if REF else "（reference_clips 未同梱）"))
    try:
        import soundfile as sf
        data, sr = sf.read(str(REF_DIR / info["file"]), dtype="float32")
        return (sr, data), f"{spk}: 「{info.get('text', '')}」"
    except Exception as e:
        return None, f"{spk}: 読み込み失敗（{type(e).__name__}: {e}）"

def ref_updates(sel):
    """選択中の全話者（先頭 MAX_REF 名）の元音声プレーヤー更新 + キャプション md を返す"""
    sel = list(sel or [])
    updates, lines = [], []
    for i in range(MAX_REF):
        if i < len(sel):
            audio, cap = _one_clip(sel[i])
            updates.append(gr.update(value=audio, label=f"元音声 {sel[i]}", visible=True))
            lines.append("- " + cap)
        else:
            updates.append(gr.update(value=None, visible=False))
    if len(sel) > MAX_REF:
        lines.append(f"（選択 {len(sel)} 名のうち先頭 {MAX_REF} 名のみ表示）")
    return updates, "\n".join(lines)

# ---------------- 合成コア（eval / synth ノートと同一方式）----------------
@torch.no_grad()
def _synth(text: str, spk: str, length_scale: float = 1.0, sdp_ratio: float = 0.2):
    net_g.cadence_weight_dp = net_g.cadence_weight_sdp = 0.0
    bert, ja_bert, en_bert, phone, tone, lang = get_text(text, Languages.JP, hps, DEVICE)
    out = net_g.infer(
        phone.to(DEVICE).unsqueeze(0),
        torch.LongTensor([phone.size(0)]).to(DEVICE),
        torch.LongTensor([spk2id[spk]]).to(DEVICE),
        tone.to(DEVICE).unsqueeze(0), lang.to(DEVICE).unsqueeze(0),
        ja_bert.to(DEVICE).unsqueeze(0),
        style_vec=torch.from_numpy(NEUTRAL).to(DEVICE).unsqueeze(0), cadence_vec=None,
        sdp_ratio=sdp_ratio, noise_scale=0.667, noise_scale_w=0.8, length_scale=length_scale)
    return out[0][0, 0].data.cpu().float().numpy()

@torch.no_grad()
def _synth_mix(text: str, weights: dict, length_scale: float):
    ref = next(iter(weights))
    if len(weights) == 1:
        return _synth(text, ref, length_scale)
    emb_w = net_g.emb_g.weight.data
    tot = sum(weights.values())
    g = sum((w / tot) * emb_w[spk2id[s]].clone() for s, w in weights.items())
    slot = spk2id[ref]
    backup = emb_w[slot].clone()
    emb_w[slot] = g.to(emb_w.device, emb_w.dtype)
    try:
        return _synth(text, ref, length_scale)
    finally:
        emb_w[slot] = backup

def do_synth(text, sel, wtxt, length_scale):
    if not text.strip():
        raise gr.Error("テキストを入力してください")
    if not sel:
        raise gr.Error("話者を1人以上選んでください")
    if wtxt.strip():
        try:
            ws = [float(v) for v in wtxt.replace("、", ",").split(",")]
        except ValueError:
            raise gr.Error("混合比は数値のカンマ区切りで（例: 0.7, 0.3）")
        if len(ws) != len(sel):
            raise gr.Error(f"混合比の個数 {len(ws)} が話者数 {len(sel)} と一致しません")
    else:
        ws = [1.0] * len(sel)
    wav = _synth_mix(text, dict(zip(sel, ws)), length_scale)
    label = " + ".join(f"{s}×{w:g}" for s, w in zip(sel, ws)) if len(sel) > 1 else sel[0]
    return (SR, wav), f"合成: {label}"


# ---------------- UI ----------------
with gr.Blocks(title="cadence cv_r1 demo") as demo:
    gr.Markdown(
        "## cadence cv_r1 — 話者マップ合成デモ\n"
        f"モデル: `{CKPT.name}`（{len(SPEAKERS)} 話者 / Common Voice ja, CC0）｜"
        f"マップ: {MAP_SRC}\n\n"
        "マップの点をクリック（または下のリストで選択）→ 1人なら単一話者、複数なら emb_g 混合の "
        "virtual 話者で合成します。")
    with gr.Row():
        with gr.Column(scale=3):
            map_img = gr.Image(value=render_map([SPEAKERS[0]]), interactive=False,
                               sources=[], show_download_button=False, label="学習話者マップ")
        with gr.Column(scale=2):
            sel = gr.Dropdown(choices=SPEAKERS, multiselect=True,
                              value=[SPEAKERS[0]], label="話者（複数選択で混合）")
            wtxt = gr.Textbox(label="混合比（カンマ区切り・合計は自動正規化。話者を選び直すと等分にリセット）",
                              value="1")
            text = gr.Textbox(label="テキスト", value="音声合成のテストです。今日はとても良い天気ですね。")
            length = gr.Slider(0.7, 1.5, 1.0, step=0.05, label="話速（length_scale）")
            btn = gr.Button("合成", variant="primary")
    gr.Markdown("#### 元話者サンプル（選択中の話者）")
    with gr.Row():
        ref_audios = [gr.Audio(visible=(i == 0), label="元音声") for i in range(MAX_REF)]
    ref_text = gr.Markdown()
    audio = gr.Audio(label="合成出力", autoplay=True)
    status = gr.Markdown()
    def on_sel_change(sel_v):
        ups, t = ref_updates(sel_v)
        eq = ", ".join(["1"] * max(1, len(sel_v or [])))   # 人数分の等分値を実値として見せる
        return [render_map(sel_v), *ups, t, eq]

    def on_map_click(evt: gr.SelectData, sel_v):
        try:
            x, y = float(evt.index[0]), float(evt.index[1])
        except Exception:
            return sel_v
        spk = nearest_speaker(x, y)
        if spk is None:
            return sel_v
        sel_v = list(sel_v or [])
        return [s for s in sel_v if s != spk] if spk in sel_v else sel_v + [spk]

    btn.click(do_synth, [text, sel, wtxt, length], [audio, status])
    sel.change(on_sel_change, [sel], [map_img, *ref_audios, ref_text, wtxt])
    demo.load(on_sel_change, [sel], [map_img, *ref_audios, ref_text, wtxt])   # 初期表示から反映
    map_img.select(on_map_click, [sel], [sel])   # クリック → 選択トグル →（sel.change 経由で）再描画

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0" if os.environ.get("SPACE_ID") else None)
