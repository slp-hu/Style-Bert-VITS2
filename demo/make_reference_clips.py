#!/usr/bin/env python3
"""話者別の参照クリップ生成 — 各学習話者の元音声を 1 発話ずつ、小さな Opus/Vorbis に圧縮して同梱用に集める

デモ（demo/app.py）と合成ノート（cv_r1_synth_colab.ipynb §2.5）が「元話者の声」を再生するための
素材を作る。全 298 話者で 10〜20 MB 程度になるので、model_assets（→ HF Hub）に同梱できる。
データは CC0（Common Voice ja）なので再配布可。

使い方（eval と同じ環境・データ配置済み・fork 直下 cwd で 1 回だけ）:
    python demo/make_reference_clips.py --data Data/cv_r1 \\
        --out model_assets/reference_clips \\
        --pick cv_0007=common_voice_ja_39875509.wav

選択規則（決定的）:
  1. esd_train.list のファイル出現順で、長さが --min-sec〜--max-sec に収まる発話を
     先頭から最大 --n-candidates 件集める（1 件も無ければ全発話が候補）。
  2. 候補のうち「内側クレストファクタ」最小のものを選ぶ（同値は esd 順で先着）。
     内側クレスト = 両端 --edge-sec ずつを除いた区間の 20*log10(peak/rms)。
     端を除くのは、録音開始/停止のクリックがピークを支配して選定を歪めるのを防ぐため
     （実例: cv_0007 — 端クリック大 + 音声極小のクリップ群では全クリップのクレストが
     同水準になり、通常のクレスト選定では不良クリップを弾けない）。

手動差し替え（--pick SPK=WAVNAME、複数指定可）:
  該当話者は候補選定をスキップして指定 wav を使う。pick されたクリップは問題事例である
  前提で、両端 --edge-sec のトリム + フェード + 静的ゲイン正規化（内側 RMS を −18 dBFS へ、
  ピーク上限 −1.5 dBFS）を適用してからエンコードする。それ以外の話者は無加工（Opus 圧縮のみ）。
  動的な loudnorm を使わないのは、フェードで絞った端を逆増幅する副作用があるため。
  注意: ffmpeg の afade=t=out は st（開始秒）を省略すると t=0 からフェードアウトして
  全体が無音化する。本スクリプトは areverse でフェードインを両側から掛ける方式で回避。
  また areverse はタイムスタンプをリセットしないため、各段に asetpts=PTS-STARTPTS を
  挟まないと後段の atrim/afade が無効化される（実測）。

出力: {out}/{spk}.ogg（Opus 32k。libopus が無ければ Vorbis、それも無ければ wav コピー）
      {out}/reference_clips.json — {spk: {"file": ..., "text": ...}}
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def read_audio_mono(p: Path):
    import numpy as np
    import soundfile as sf
    y, sr = sf.read(str(p), dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y, sr


def duration_sec(p: Path) -> float:
    try:
        import soundfile as sf
        i = sf.info(str(p))
        return i.frames / float(i.samplerate)
    except Exception:
        return -1.0


def interior_crest_db(p: Path, edge_sec: float) -> float:
    """両端 edge_sec を除いた区間のクレストファクタ [dB]。読めない/無音なら +inf（= 選ばない）"""
    import numpy as np
    try:
        y, sr = read_audio_mono(p)
    except Exception:
        return float("inf")
    e = int(edge_sec * sr)
    if len(y) > 2 * e + sr // 10:  # 内側が 0.1 秒以上残る場合のみ端を除く
        y = y[e:-e]
    peak = float(np.abs(y).max())
    rms = float(np.sqrt((y ** 2).mean()))
    if peak <= 0 or rms <= 0:
        return float("inf")
    return 20.0 * (np.log10(peak) - np.log10(rms))


def pick_gain_db(p: Path, edge_sec: float,
                 target_rms_db: float = -18.0, peak_ceiling_db: float = -1.5) -> float:
    """pick クリップ用の静的ゲイン [dB]。内側区間の RMS を target へ、ただしピーク上限を超えない"""
    import numpy as np
    y, sr = read_audio_mono(p)
    e = int(edge_sec * sr)
    if len(y) > 2 * e + sr // 10:
        y = y[e:-e]
    peak = float(np.abs(y).max())
    rms = float(np.sqrt((y ** 2).mean()))
    if peak <= 0 or rms <= 0:
        return 0.0
    g = target_rms_db - 20.0 * np.log10(rms)
    g = min(g, peak_ceiling_db - 20.0 * np.log10(peak))
    return float(g)


def ffmpeg_filter_for_pick(edge_sec: float, gain_db: float) -> str:
    """pick クリップ用: 両端トリム + 両端フェード（areverse 方式）+ 静的ゲイン"""
    e = f"{edge_sec:g}"
    return (
        f"atrim=start={e},asetpts=PTS-STARTPTS,"
        f"areverse,asetpts=PTS-STARTPTS,"
        f"atrim=start={e},asetpts=PTS-STARTPTS,afade=t=in:d={e},"
        f"areverse,asetpts=PTS-STARTPTS,afade=t=in:d={e},"
        f"volume={gain_db:.2f}dB"
    )


def encode(src: Path, out_stem: Path, afilter: str | None = None) -> str:
    """src を Opus/Vorbis/wav の順に試して out_stem.{ogg,wav} に書く。戻り値は拡張子"""
    for args, ext in ((["-c:a", "libopus", "-b:a", "32k"], ".ogg"),
                      (["-c:a", "libvorbis", "-q:a", "2"], ".ogg")):
        out = out_stem.with_suffix(ext)
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-ac", "1"]
        if afilter:
            cmd += ["-af", afilter]
        cmd += [*args, str(out)]
        try:
            r = subprocess.run(cmd, capture_output=True)
            if r.returncode == 0 and out.exists() and out.stat().st_size > 1000:
                return ext
        except FileNotFoundError:
            break  # ffmpeg 自体が無い
    out = out_stem.with_suffix(".wav")
    shutil.copy2(src, out)
    return ".wav"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="esd_train.list と wavs/ が解決できるデータルート")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-sec", type=float, default=3.0)
    ap.add_argument("--max-sec", type=float, default=8.0)
    ap.add_argument("--n-candidates", type=int, default=12)
    ap.add_argument("--edge-sec", type=float, default=0.05)
    ap.add_argument("--pick", action="append", default=[], metavar="SPK=WAVNAME",
                    help="手動差し替え（例: --pick cv_0007=common_voice_ja_39875509.wav）。"
                         "トリム+フェード+loudnorm を適用。複数指定可")
    a = ap.parse_args()

    data = Path(a.data)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    esd = data / "esd_train.list"
    assert esd.exists(), f"★esd_train.list が無い: {data}"

    picks = {}
    for kv in a.pick:
        spk, _, wav = kv.partition("=")
        assert spk and wav, f"★--pick の書式が不正: {kv}（SPK=WAVNAME）"
        picks[spk] = wav

    # esd を話者ごとに収集（出現順を保持）
    by_spk: dict[str, list[tuple[Path, str]]] = {}
    for line in esd.read_text(encoding="utf-8").splitlines():
        f = line.split("|")
        if len(f) < 4:
            continue
        wav = data / "wavs" / Path(f[0]).name
        by_spk.setdefault(f[1], []).append((wav, f[3]))

    manifest = {}
    n_fallback_ext = 0
    for spk in sorted(by_spk):
        utts = by_spk[spk]
        if spk in picks:
            match = [(w, t) for w, t in utts if w.name == picks[spk]]
            assert match, f"★--pick の wav が esd に無い: {spk}={picks[spk]}"
            src, text = match[0]
            g = pick_gain_db(src, a.edge_sec)
            ext = encode(src, out / spk, afilter=ffmpeg_filter_for_pick(a.edge_sec, g))
            print(f"{spk}: {src.name} [--pick: trim+fade+gain{g:+.1f}dB]")
        else:
            in_range = [(w, t) for w, t in utts
                        if a.min_sec <= duration_sec(w) <= a.max_sec][: a.n_candidates]
            cands = in_range if in_range else utts
            src, text = min(cands, key=lambda wt: interior_crest_db(wt[0], a.edge_sec))
            ext = encode(src, out / spk)
        if ext == ".wav":
            n_fallback_ext += 1
        manifest[spk] = {"file": f"{spk}{ext}", "text": text}

    (out / "reference_clips.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"done: {len(manifest)} 話者 → {out}")
    if n_fallback_ext:
        print(f"★注意: {n_fallback_ext} 件が wav コピー（ffmpeg/コーデック要確認）", file=sys.stderr)


if __name__ == "__main__":
    main()
