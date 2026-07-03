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
import pandas as pd
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
    COORD = {s: _m[s] for s in SPEAKERS if s in _m}
    MAP_SRC = "x-vector PCA（事前計算）"
else:
    emb = net_g.emb_g.weight.data.cpu().float().numpy()
    xy = pca2(np.stack([emb[spk2id[s]] for s in SPEAKERS]))
    COORD = {s: xy[i].tolist() for i, s in enumerate(SPEAKERS)}
    MAP_SRC = "emb_g PCA（フォールバック。x-vector map は demo/make_speaker_map.py で生成）"
DF = pd.DataFrame([{"speaker": s, "x": c[0], "y": c[1]} for s, c in COORD.items()])

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

def ref_clip(sel):
    """最後に選んだ話者の元音声とテキストを返す（クリップ集が無い場合は案内のみ）"""
    if not sel:
        return None, ""
    spk = sel[-1]
    info = REF.get(spk)
    if not info:
        return None, (f"（{spk} の元音声サンプルなし" +
                      ("" if REF else " — reference_clips 未同梱。demo/make_reference_clips.py で生成") + "）")
    return str(REF_DIR / info["file"]), f"元音声 **{spk}**: 「{info.get('text', '')}」"

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

def on_plot_select(evt: gr.SelectData, sel):
    """マップ上の点クリックで話者を選択に追加/除去（対応していない環境では無視される）"""
    try:
        row = DF.iloc[evt.index] if isinstance(evt.index, (int, np.integer)) else DF.iloc[evt.index[0]]
        spk = str(row["speaker"])
    except Exception:
        return sel
    sel = list(sel or [])
    return [s for s in sel if s != spk] if spk in sel else sel + [spk]

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
            plot = gr.ScatterPlot(DF, x="x", y="y", tooltip=["speaker"],
                                  height=420, label="学習話者マップ")
        with gr.Column(scale=2):
            sel = gr.Dropdown(choices=SPEAKERS, multiselect=True,
                              value=[SPEAKERS[0]], label="話者（複数選択で混合）")
            wtxt = gr.Textbox(label="混合比（カンマ区切り。空欄 = 等分）", placeholder="0.7, 0.3")
            text = gr.Textbox(label="テキスト", value="音声合成のテストです。今日はとても良い天気ですね。")
            length = gr.Slider(0.7, 1.5, 1.0, step=0.05, label="話速（length_scale）")
            btn = gr.Button("合成", variant="primary")
    with gr.Row():
        ref_audio = gr.Audio(label="元話者サンプル（最後に選んだ話者）")
        audio = gr.Audio(label="合成出力", autoplay=True)
    ref_text = gr.Markdown()
    status = gr.Markdown()
    btn.click(do_synth, [text, sel, wtxt, length], [audio, status])
    sel.change(ref_clip, [sel], [ref_audio, ref_text])
    demo.load(ref_clip, [sel], [ref_audio, ref_text])   # 初期表示でも1人目のサンプルを出す
    try:
        plot.select(on_plot_select, [sel], [sel])
    except Exception:
        pass  # 古い gradio では点クリック非対応 → ドロップダウンのみで運用

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0" if os.environ.get("SPACE_ID") else None)
