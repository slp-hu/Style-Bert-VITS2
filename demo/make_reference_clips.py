#!/usr/bin/env python3
"""話者別の参照クリップ生成 — 各学習話者の元音声を 1 発話ずつ、小さな Opus/Vorbis に圧縮して同梱用に集める

デモ（demo/app.py）と合成ノート（cv_r1_synth_colab.ipynb §2.5）が「元話者の声」を再生するための
素材を作る。全 298 話者で 10〜20 MB 程度になるので、model_assets（→ HF Hub）に同梱できる。
データは CC0（Common Voice ja）なので再配布可。

使い方（eval と同じ環境・データ配置済み・fork 直下 cwd で 1 回だけ）:
    python demo/make_reference_clips.py --data Data/cv_r1 \\
        --out model_assets/model_name/reference_clips

選択規則（決定的）: esd_train.list のファイル出現順で、長さが --min-sec〜--max-sec に
収まる最初の発話（無ければ最初の発話）。
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def duration_sec(p: Path) -> float:
    try:
        import soundfile as sf
        i = sf.info(str(p))
        return i.frames / float(i.samplerate)
    except Exception:
        return -1.0


def encode(src: Path, dst: Path) -> str:
    """opus 32k → 失敗したら vorbis → それも無理なら wav コピー。戻り値 = 実際の拡張子"""
    for args, ext in ((["-c:a", "libopus", "-b:a", "32k"], ".ogg"),
                      (["-c:a", "libvorbis", "-q:a", "2"], ".ogg")):
        out = dst.with_suffix(ext)
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-ac", "1", *args, str(out)],
                           capture_output=True, text=True)
        if r.returncode == 0 and out.exists() and out.stat().st_size > 0:
            return ext
    out = dst.with_suffix(".wav")
    shutil.copy2(src, out)
    return ".wav"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="Data/cv_r1")
    ap.add_argument("--out", default="reference_clips")
    ap.add_argument("--min-sec", type=float, default=3.0)
    ap.add_argument("--max-sec", type=float, default=8.0)
    a = ap.parse_args()

    data, out = Path(a.data), Path(a.out)
    assert (data / "esd_train.list").exists(), f"★esd_train.list が無い: {data}"
    out.mkdir(parents=True, exist_ok=True)

    # 話者ごとの候補（出現順）
    cand = {}
    for line in open(data / "esd_train.list", encoding="utf-8"):
        f = line.rstrip("\n").split("|")
        wav, spk, text = f[0], f[1], (f[3] if len(f) > 3 else "")
        rel = wav.split("Data/cv_r1/", 1)[1] if "Data/cv_r1/" in wav else wav
        cand.setdefault(spk, []).append((data / rel, text))

    manifest, total = {}, 0
    for i, (spk, lst) in enumerate(sorted(cand.items()), 1):
        pick = next(((p, t) for p, t in lst if a.min_sec <= duration_sec(p) <= a.max_sec), lst[0])
        ext = encode(pick[0], out / spk)
        f = out / f"{spk}{ext}"
        manifest[spk] = {"file": f.name, "text": pick[1]}
        total += f.stat().st_size
        if i % 25 == 0:
            print(f"  {i}/{len(cand)} 話者 ...", end="\r")
    json.dump(manifest, open(out / "reference_clips.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\n出力: {out}（{len(manifest)} 話者 / 合計 {total/1e6:.1f} MB）")
    print("→ model_assets/<model>/reference_clips/ に置くとデモ・合成ノートが自動で使う")


if __name__ == "__main__":
    main()
