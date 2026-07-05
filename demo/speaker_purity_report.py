#!/usr/bin/env python3
"""話者純度レポート — 話者内クリップ間の x-vector 整合性でアカウント共用等を検出

Common Voice の validated はテキスト一致の検証であり、話者の同一性は検証されない
（1アカウント複数話者の「共用」があり得る。実例: cv_0127）。本スクリプトは
話者ごとに数クリップの x-vector を取り、話者内 cos 類似度の統計で「1人の声らしさ」を
採点してランキングする。下位話者を監査ノート（cv_r1_speaker_audit.ipynb）で聴いて確認する。

使い方（eval と同じ環境・データ配置済み・fork 直下 cwd。TF/CPU で数分〜十数分）:
    python demo/speaker_purity_report.py --data Data/cv_r1 --out purity_report.tsv \
        [--per-spk 8] [--drive /content/drive/MyDrive]

出力 TSV（話者内平均 cos の昇順 = 怪しい順）:
    spk  n_clips  intra_cos_mean  intra_cos_min  worst_pair
"""
import argparse
import collections
import itertools
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="Data/cv_r1")
    ap.add_argument("--out", default="purity_report.tsv")
    ap.add_argument("--per-spk", type=int, default=8)
    ap.add_argument("--drive", default="/content/drive/MyDrive", help="xvec モデルのキャッシュ先")
    ap.add_argument("--min-sec", type=float, default=1.0, help="短すぎるクリップは x-vector が不安定")
    a = ap.parse_args()

    data = Path(a.data)
    per = collections.defaultdict(list)
    for name in ("esd_train.list", "esd_val.list"):
        f = data / name
        if not f.exists():
            continue
        for line in open(f, encoding="utf-8"):
            c = line.rstrip("\n").split("|")
            rel = c[0].split("Data/cv_r1/", 1)[1] if "Data/cv_r1/" in c[0] else c[0]
            p = data / rel
            if p.exists() and len(per[c[1]]) < a.per_spk:
                per[c[1]].append(p)
    items = [(f"{spk}::{p.name}", str(p)) for spk, ps in per.items() for p in ps]
    print(f"話者 {len(per)} / クリップ {len(items)}（各話者 ≤{a.per_spk}）")

    # --- TF preflight（protobuf 版ずれの自己修復。make_speaker_map と同じ）---
    pre = subprocess.run([sys.executable, "-c", "import tensorflow"], capture_output=True, text=True)
    if pre.returncode != 0 and "runtime_version" in (pre.stderr or ""):
        print("★protobuf が TF に対して古い → protobuf>=5.28 に更新")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "protobuf>=5.28"],
                       check=True)
        pre = subprocess.run([sys.executable, "-c", "import tensorflow"], capture_output=True, text=True)
    assert pre.returncode == 0, "★TensorFlow を import できない:\n" + (pre.stderr or "")[-800:]

    with tempfile.TemporaryDirectory() as td:
        ij, oz = Path(td) / "items.json", Path(td) / "xvec.npz"
        json.dump(dict(items), open(ij, "w", encoding="utf-8"))
        r = subprocess.run([sys.executable, "xvec_extract.py",
                            "--items", str(ij), "--out", str(oz), "--drive", a.drive])
        assert r.returncode == 0, "★xvec_extract.py 失敗"
        xv = np.load(oz)

        rows = []
        for spk, ps in sorted(per.items()):
            vs, names = [], []
            for p in ps:
                k = f"{spk}::{p.name}"
                if k in xv:
                    v = xv[k]
                    vs.append(v / (np.linalg.norm(v) + 1e-9))
                    names.append(p.name)
            if len(vs) < 2:
                continue
            sims = [(float(vs[i] @ vs[j]), names[i], names[j])
                    for i, j in itertools.combinations(range(len(vs)), 2)]
            mean = float(np.mean([s for s, *_ in sims]))
            smin, n1, n2 = min(sims, key=lambda x: x[0])
            rows.append((spk, len(vs), mean, smin, f"{n1}|{n2}"))

    rows.sort(key=lambda r: r[2])  # 話者内平均 cos の昇順 = 怪しい順
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("spk\tn_clips\tintra_cos_mean\tintra_cos_min\tworst_pair\n")
        for r_ in rows:
            f.write(f"{r_[0]}\t{r_[1]}\t{r_[2]:.4f}\t{r_[3]:.4f}\t{r_[4]}\n")
    print(f"出力: {a.out}（{len(rows)} 話者、怪しい順）")
    print("\n== 話者内整合性ワースト 15 ==")
    for spk, n, mean, smin, pair in rows[:15]:
        print(f"  {spk}  n={n}  mean={mean:.3f}  min={smin:.3f}  {pair}")
    med = float(np.median([r_[2] for r_ in rows]))
    print(f"\n全体中央値 mean={med:.3f}（これから大きく下振れる話者を監査ノートで聴く）")


if __name__ == "__main__":
    main()
