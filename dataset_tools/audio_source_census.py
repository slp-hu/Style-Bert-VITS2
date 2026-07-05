#!/usr/bin/env python3
"""音源の実態調査 — cv_r1 の源クリップの帯域クラスとトリム損失を全数集計

背景（2026-07 の実測）: Common Voice ja の源クリップはサンプリング周波数が混在しており、
帯域遮断 ~11-12kHz（源 22.05/24k）のクラスも確認された。また端点トリム（top_db=30）が
突発音を含むクリップで音声本体を刈る可能性がある。ただし CV は収録 UI が前後の無音を強制するため、
トリム比（変換後/原本長）は仕様どおりの無音を拾うだけで異常検出には向かない。
本スクリプトの主指標は **話速（モーラ/秒 = kana モーラ数 ÷ トリム後長）** で、
レート誤ラベル（早口化）を物理量として直接検出する（日本語朗読は概ね 7〜9 mora/s。
12 超・3 未満を旗立て）。あわせて
  (2) 帯域クラス: wav 実体のスペクトル遮断周波数（話者ごとサンプリング）
  (3) 性別ラベル × F0 の矛盾: 話者の F0 中央値が申告性別の典型帯と強く矛盾する話者を旗立て
      （一貫して別性の声のアカウント = cv_0028/cv_0127 型のラベル異常。純度検査では
       「話者内が一貫」なため検出できない相補的チェック）
  (4) トリム比: 参考情報として出力のみ
を集計する。Mac（パイプライン出力ツリー + CV 原本がある環境）での実行を想定。

使い方:
    python dataset_tools/audio_source_census.py --manifest <work>/audio_manifest_cv.csv \\
        --durations <cv-corpus-*>/ja/clip_durations.tsv \\
        --wavroot <out>/sbv2_data/cv_r1/wavs [--sample-per-spk 2] [--out census.tsv]
"""
import argparse
import collections
import csv
from pathlib import Path

import numpy as np


def f0_median_hz(path, max_sec=5.0):
    """クリップ先頭 max_sec の F0 中央値（Hz）。無声なら None"""
    import soundfile as sf
    y, sr = sf.read(str(path), dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y[: int(max_sec * sr)]
    try:
        import pyworld
        _f0, t = pyworld.dio(y.astype(np.float64), sr, f0_floor=60, f0_ceil=500)
        f0 = pyworld.stonemask(y.astype(np.float64), _f0, t, sr)
    except Exception:
        try:
            import librosa
            f0 = librosa.yin(y, fmin=60, fmax=500, sr=sr)
        except Exception:
            return None
    v = f0[(f0 > 60) & (f0 < 500)]
    return float(np.median(v)) if v.size >= 10 else None


_SMALL = set("ゃゅょぁぃぅぇぉャュョァィゥェォヮゎ")

def mora_count(kana: str) -> int:
    """かな表記からモーラ数を近似（拗音の小書きは直前と合流。促音ッ・長音ーは1モーラ）"""
    return sum(1 for ch in kana if not ch.isspace() and ch not in _SMALL and ch not in "、。！？!?・「」『』（）()")


def cutoff_hz(path, thresh=1e-5):
    """累積エネルギー比が thresh を超える最高周波数（≒帯域遮断）"""
    import soundfile as sf
    y, sr = sf.read(str(path), dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if len(y) < 2048:
        return None
    S = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
    f = np.fft.rfftfreq(len(y), 1 / sr)
    tot = S.sum() + 1e-30
    tail = np.cumsum(S[::-1])[::-1] / tot   # f 以上のエネルギー比
    above = np.where(tail > thresh)[0]
    return float(f[above[-1]]) if len(above) else 0.0


def band_class(hz):
    if hz is None:
        return "short"
    if hz < 9000:
        return "<9k(電話帯域級?)"
    if hz < 13000:
        return "~11-12k(源22-24k)"
    if hz < 17500:
        return "~16k(源32k)"
    return ">17.5k(源44-48k)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="audio_manifest_cv.csv")
    ap.add_argument("--durations", required=True, help="CV の clip_durations.tsv")
    ap.add_argument("--wavroot", required=True, help="変換済み wav のディレクトリ")
    ap.add_argument("--sample-per-spk", type=int, default=2, help="帯域測定の話者あたり本数")
    ap.add_argument("--out", default="census.tsv")
    a = ap.parse_args()

    # 原本長（ms）
    orig = {}
    with open(a.durations, encoding="utf-8", newline="") as fh:
        rd = csv.DictReader(fh, delimiter="\t")
        kcol = "clip" if "clip" in rd.fieldnames else rd.fieldnames[0]
        vcol = next(c for c in rd.fieldnames if "dur" in c.lower())
        for r in rd:
            orig[Path(r[kcol]).stem] = float(r[vcol]) / 1000.0

    # manifest: uid / spk / dur
    rows = []
    with open(a.manifest, encoding="utf-8", newline="") as fh:
        rd = csv.DictReader(fh)
        for r in rd:
            uid = Path(r.get("uid") or r.get("wav", "")).stem
            spk, dur = r.get("spk", "?"), float(r.get("dur", 0) or 0)
            kana = r.get("kana") or ""
            nm = mora_count(kana) if kana else 0
            rate = (nm / dur) if (nm and dur > 0) else None
            o = orig.get(uid)
            rows.append((spk, uid, dur, o, (dur / o) if o else None, nm, rate))
    matched = [r for r in rows if r[4] is not None]
    print(f"manifest {len(rows)} 行 / 原本長と突合 {len(matched)}")

    # --- (1) 話速（主指標。レート誤ラベル = 早口/遅すぎ を直接検出）---
    rated = [r for r in rows if r[6] is not None]
    rates = np.array([r[6] for r in rated])
    print(f"\n== 話速（モーラ/秒）の分布（kana あり {len(rated)} 本）==")
    for q in (0.001, 0.01, 0.05, 0.50, 0.95, 0.99, 0.999):
        print(f"  p{q*100:>5.1f}: {np.quantile(rates, q):.2f}")
    fast = sorted((r for r in rated if r[6] > 12.0), key=lambda r: -r[6])
    slow = sorted((r for r in rated if r[6] < 3.0), key=lambda r: r[6])
    print(f"  旗立て: 早口（>12 mora/s）{len(fast)} 本 / 遅すぎ（<3 mora/s）{len(slow)} 本")
    for tag, lst in (("早口", fast[:15]), ("遅すぎ", slow[:10])):
        for spk, uid, dur, o, ratio, nm, rate in lst:
            print(f"    {tag}  {spk}  {uid}  {rate:.1f} mora/s（{nm}モーラ/{dur:.2f}s）")

    # --- (3) トリム比（参考。CV は前後無音が仕様のため異常検出には使わない）---
    ratios = np.array([r[4] for r in matched])
    print(f"\n== 参考: トリム後/原本 長さ比 p1={np.quantile(ratios,0.01):.2f}"
          f" / p50={np.quantile(ratios,0.5):.2f}")

    # --- (2) 帯域クラス（話者ごとサンプリング）---
    per = collections.defaultdict(list)
    for spk, uid, *_ in matched:
        if len(per[spk]) < a.sample_per_spk:
            per[spk].append(uid)
    sex_of = {}
    with open(a.manifest, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            sex_of.setdefault(r.get("spk", "?"), (r.get("sex") or "").lower())
    cls = collections.Counter()
    detail = []
    spk_f0 = collections.defaultdict(list)
    for spk, uids in sorted(per.items()):
        for uid in uids:
            p = Path(a.wavroot) / f"{uid}.wav"
            if not p.exists():
                continue
            hz = cutoff_hz(p)
            c = band_class(hz)
            cls[c] += 1
            detail.append((spk, uid, hz, c))
            f0 = f0_median_hz(p)
            if f0:
                spk_f0[spk].append(f0)
    print("\n== 帯域クラス分布（話者ごと最大", a.sample_per_spk, "本サンプル）==")
    for c, n in cls.most_common():
        print(f"  {c}: {n}")

    # --- 性別ラベル × F0 の矛盾（保守的閾値: 強い矛盾のみ旗立て）---
    conflicts = []
    for spk, f0s in sorted(spk_f0.items()):
        f0m = float(np.median(f0s))
        sex = sex_of.get(spk, "")
        if sex.startswith("f") and f0m < 140:
            conflicts.append((spk, sex, f0m, "female ラベル × 低 F0（男声疑い）"))
        elif sex.startswith("m") and f0m > 195:
            conflicts.append((spk, sex, f0m, "male ラベル × 高 F0（女声疑い）"))
    print(f"\n== 性別ラベル × F0 矛盾: {len(conflicts)} 名 ==")
    for spk, sex, f0m, why in conflicts:
        print(f"  {spk}  label={sex}  F0中央値={f0m:.0f}Hz  {why}")

    with open(a.out, "w", encoding="utf-8") as f:
        f.write("spk\tuid\torig_s\ttrim_s\tratio\tmora\tmora_per_s\tcutoff_hz\tband\n")
        cut = {(s, u): (hz, c) for s, u, hz, c in detail}
        for spk, uid, dur, o, ratio, nm, rate in rows:
            hz, c = cut.get((spk, uid), ("", ""))
            f.write(f"{spk}\t{uid}\t{'' if o is None else f'{o:.3f}'}\t{dur:.3f}"
                    f"\t{'' if ratio is None else f'{ratio:.3f}'}\t{nm}"
                    f"\t{'' if rate is None else f'{rate:.2f}'}\t{hz}\t{c}\n")
    print(f"\n出力: {a.out}")


if __name__ == "__main__":
    main()
