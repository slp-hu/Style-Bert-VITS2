#!/usr/bin/env python3
"""steps/epoch を GPU 無しで再現し、既知の step 数から batch_size を逆算する。

なぜ計算だけで分かるか:
  - `data_utils.TextAudioSpeakerLoader._filter` は spec 長を **wav のファイルサイズだけ**から出す:
        lengths[i] = os.path.getsize(audiopath) // (2 * hop_length)
    音声を一切デコードしない。stat するだけ。
  - `DistributedBucketSampler` は boundaries [32,300,...,1000] の外側を**捨て**、
    各 bucket を (num_replicas × batch_size) の倍数まで**パディング**する。
  - `__len__` = num_samples // batch_size = sum(padded_buckets) // batch_size = steps/epoch

つまり esd_train.list と wav さえあれば、任意の batch に対する steps/epoch が厳密に出る。
学習を回さずに「R1 の 8,323 steps/epoch はどの batch なら成立するか」を判定できる。

使用例（R1 の batch を逆算）:
  python dataset_tools/estimate_steps.py \
      --esd-train ~/tmp/synth/cadence/data/csj/esd_train.list \
      --config    ~/tmp/synth/cadence/data/csj/config_r1.json \
      --wav-root  ~/tmp/synth/cadence/data/csj/r1_cut \
      --wav-root  ~/tmp/synth/cadence/data/csj/wav \
      --batch 2 4 8 12 16 24 32 --target-steps 8323

  # e10_s83230 → 83230 / 10 epoch = 8323 steps/epoch
"""

from __future__ import annotations

import argparse
import bisect
import json
import os
import sys
from pathlib import Path

BOUNDARIES = [32, 300, 400, 500, 600, 700, 800, 900, 1000]  # train_ms_jp_extra.py:227 と同値


def resolve_wav(raw: str, roots: list[Path], spk: str = "") -> Path | None:
    q = Path(raw)
    if q.is_absolute() and q.is_file():
        return q
    name = q.name
    for r in roots:
        for c in ([r / name] + ([r / spk / name] if spk else []) + [r / q.parent.name / name, r / q]):
            if c.is_file():
                return c
    return None


def bucket_index(length: int) -> int:
    """DistributedBucketSampler._bisect と同じ判定（boundaries の外は -1 = 捨てる）。"""
    i = bisect.bisect_right(BOUNDARIES, length) - 1
    return i if 0 <= i < len(BOUNDARIES) - 1 else -1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--esd-train", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--wav-root", action="append", default=[], type=Path)
    ap.add_argument("--batch", nargs="+", type=int, default=[2, 4, 8, 12, 16, 24, 32])
    ap.add_argument("--replicas", type=int, default=1, help="学習時の GPU 数（n_gpus）")
    ap.add_argument("--target-steps", type=int, default=None,
                    help="既知の steps/epoch。合致する batch を判定する")
    ap.add_argument("--epochs", type=int, default=None, help="target-steps を総 step から出す場合の epoch 数")
    ap.add_argument("--total-steps", type=int, default=None, help="ckpt の global step（例 83230）")
    a = ap.parse_args()

    target = a.target_steps
    if target is None and a.total_steps and a.epochs:
        target = a.total_steps // a.epochs
        print(f"target: {a.total_steps} / {a.epochs} epoch = {target} steps/epoch")

    hop = json.load(open(a.config.expanduser(), encoding="utf-8"))["data"].get("hop_length", 512)
    roots = [r.expanduser().resolve() for r in a.wav_root]
    rows = [l.rstrip("\n").split("|") for l in open(a.esd_train.expanduser(), encoding="utf-8") if l.strip()]
    print(f"esd_train: {len(rows)} 行 / hop_length {hop}")

    lengths, missing = [], 0
    for r in rows:
        p = resolve_wav(r[0], roots, r[1])
        if p is None:
            missing += 1
            continue
        lengths.append(os.path.getsize(p) // (2 * hop))
    if missing:
        sys.exit(f"★wav が {missing} 本見つからない → --wav-root を足す")

    counts = [0] * (len(BOUNDARIES) - 1)
    dropped_short = dropped_long = 0
    for L in lengths:
        i = bucket_index(L)
        if i == -1:
            if L <= BOUNDARIES[0]:
                dropped_short += 1
            else:
                dropped_long += 1
        else:
            counts[i] += 1
    kept = sum(counts)
    print(f"bucket に残る: {kept} / {len(lengths)} = {kept/len(lengths):.2%}")
    print(f"  捨てられた: 短すぎ(≤{BOUNDARIES[0]}) {dropped_short} / "
          f"長すぎ(>{BOUNDARIES[-1]}) {dropped_long}")
    print(f"  bucket 生カウント: {counts}")

    print(f"\n{'batch':>6} {'Bucket info（パディング後）':<44} {'total':>8} {'steps/epoch':>12}")
    for b in a.batch:
        tb = a.replicas * b
        padded = [c + (tb - (c % tb)) % tb for c in counts if c > 0]
        total = sum(padded)
        steps = total // b
        mark = ""
        if target:
            mark = "  ★一致" if steps == target else f"  ({steps - target:+d})"
        info = str(padded)
        print(f"{b:>6} {info[:44]:<44} {total:>8} {steps:>12}{mark}")

    if target:
        print(f"\n判定: steps/epoch = {target} を満たす batch が上に無い場合、"
              f"学習時の esd_train.list は現在の {len(rows)} 行と違っていた"
              f"（＝学習後に再生成された）ことになる。")


if __name__ == "__main__":
    main()
