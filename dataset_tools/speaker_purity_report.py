#!/usr/bin/env python3
"""話者純度レポート — 話者内クリップ間の x-vector 整合性でアカウント共用等を検出

Common Voice の validated はテキスト一致の検証であり、話者の同一性は検証されない
（1アカウント複数話者の「共用」があり得る。実例: cv_0127）。本スクリプトは
話者ごとに数クリップの x-vector を取り、話者内の構造（2分割の塊間/塊内 cos）で
候補を5型（SHARED?F0 / INTRUDER?F0 / MIX? / outlier / ok）に分類・ランキングする。
F0（チャネル不変）をクラスタ間で比較し、ΔF0 大 = ピッチ帯も別人（共用ほぼ確定）と
ΔF0 小 = 同性共用かチャネル分裂（要聴取）を切り分ける。

★仕様上の限界（cv_r1 の較正実験 2026-07 で確定）:
  本抽出器（CV_SPKREC, CMN なしで学習）はチャネルと話者を分離できないため、
  「収録環境が2種類ある1人」と「2人の共用」は同じ形（2つの塊）に見える。
  したがって本ツールは**候補の網羅 + 最疑ペア（worst_pair）の特定まで**を担い、
  確定は聴取（cv_r1_speaker_audit.ipynb）で行う（再現率重視のスクリーニング設計）。
  なお発話単位 CMN の後付けは A/B 実験で棄却済み（--cmn utt。学習時特徴と不一致で
  インポスタ cos が 0.03→0.58 に崩壊）。フラグは実験再現のために残している。

使い方（eval と同じ環境・データ配置済み・fork 直下 cwd。TF/CPU で数分〜十数分）:
    python dataset_tools/speaker_purity_report.py --data Data/cv_r1 --out purity_report.tsv \
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


def f0_median_st(path, max_sec=5.0):
    """クリップ先頭 max_sec の F0 中央値を半音（55Hz 基準）で返す。無声なら None。
    F0 はチャネル特性にほぼ不変なので、x-vector（スペクトル包絡系）と直交する話者手がかり。"""
    import soundfile as sf
    y, sr = sf.read(str(path), dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y[: int(max_sec * sr)]
    f0 = None
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
    if v.size < 10:
        return None
    return float(12.0 * np.log2(np.median(v) / 55.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="Data/cv_r1")
    ap.add_argument("--out", default="purity_report.tsv")
    ap.add_argument("--per-spk", type=int, default=8)
    ap.add_argument("--drive", default="/content/drive/MyDrive", help="xvec モデルのキャッシュ先")
    ap.add_argument("--min-sec", type=float, default=1.0, help="短すぎるクリップは x-vector が不安定")
    ap.add_argument("--cmn", choices=["none", "utt"], default="none",
                    help="xvec_extract に渡す CMN（utt はチャネル耐性の A/B 用）")
    a = ap.parse_args()

    data = Path(a.data)
    assert (data / "esd_train.list").exists(), (
        f"★esd が無い: {data}\n"
        "  データは VM ローカル（/content/Style-Bert-VITS2/Data/cv_r1、eval §0 が配置）にある。\n"
        "  Drive clone 直下ではなく `cd /content/Style-Bert-VITS2` してから実行する")
    per = collections.defaultdict(list)
    skipped = 0
    for name in ("esd_train.list", "esd_val.list"):
        f = data / name
        if not f.exists():
            continue
        for line in open(f, encoding="utf-8"):
            c = line.rstrip("\n").split("|")
            rel = c[0].split("Data/cv_r1/", 1)[1] if "Data/cv_r1/" in c[0] else c[0]
            p = data / rel
            try:
                ok = p.exists()
            except OSError:   # Drive FUSE 等の stat 不調はそのファイルだけ飛ばす
                skipped += 1
                continue
            if ok and len(per[c[1]]) < a.per_spk:
                per[c[1]].append(p)
    if skipped:
        print(f"★stat 失敗で {skipped} 件スキップ（Drive FUSE 上で実行していないか確認）")
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
                            "--items", str(ij), "--out", str(oz), "--drive", a.drive,
                            "--cmn", a.cmn])
        assert r.returncode == 0, "★xvec_extract.py 失敗"
        xv = np.load(oz)

        def split2(V):
            """決定的2分割: 最も似ていないペアを種にし、近い方の種へ割り当てる"""
            n = len(V)
            S = np.array([[float(V[i] @ V[j]) for j in range(n)] for i in range(n)])
            i0, j0 = np.unravel_index(np.argmin(S + np.eye(n) * 2), S.shape)
            A = [k for k in range(n) if S[k, i0] >= S[k, j0]]
            B = [k for k in range(n) if k not in A]
            across = float(np.mean([S[a, b] for a in A for b in B])) if A and B else 1.0
            def within(C):
                ps = [S[a, b] for a, b in itertools.combinations(C, 2)]
                return float(np.mean(ps)) if ps else 1.0
            return A, B, across, within(A), within(B)

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
            A, B, across, wA, wB = split2(vs)
            # クラスタ別 F0 中央値の差（半音）— チャネル不変の直交手がかり
            f0s = [f0_median_st(p_) for p_ in ps[: len(vs)]]
            fA = [f0s[i] for i in A if i < len(f0s) and f0s[i] is not None]
            fB = [f0s[i] for i in B if i < len(f0s) and f0s[i] is not None]
            dst = abs(float(np.median(fA)) - float(np.median(fB))) if (fA and fB) else -1.0
            rows.append((spk, len(vs), mean, smin, f"{n1}|{n2}",
                         f"{len(A)}/{len(B)}", across, min(wA, wB), dst))

        # --- 話者間ベースライン（インポスタ統計）: 弁別力は「話者内 − 話者間」の差で測る ---
        spk_vecs = {}
        for spk, ps in per.items():
            vlist = []
            for p_ in ps:
                k = f"{spk}::{p_.name}"
                if k in xv:
                    v = xv[k]
                    vlist.append(v / (np.linalg.norm(v) + 1e-9))
            if vlist:
                spk_vecs[spk] = vlist
        rng = np.random.default_rng(0)
        spks_l = list(spk_vecs)
        imp = []
        for _ in range(3000):
            s1, s2 = rng.choice(len(spks_l), 2, replace=False)
            v1 = spk_vecs[spks_l[s1]][rng.integers(len(spk_vecs[spks_l[s1]]))]
            v2 = spk_vecs[spks_l[s2]][rng.integers(len(spk_vecs[spks_l[s2]]))]
            imp.append(float(v1 @ v2))
        imp_mean, imp_p95 = float(np.mean(imp)), float(np.quantile(imp, 0.95))

    # --- 型分類: 共用疑い（両塊 ≥2 本 & 塊間がインポスタ水準 & 各塊は内部で密）/ 外れ値型 / ok ---
    TH_ACROSS = imp_mean + 0.15   # 塊間類似がこの水準以下なら「別人の塊」とみなす
    TH_TIGHT = 0.30               # 塊の内部密度の下限（none 空間の同一話者水準）
    TH_F0_ST = 3.0   # クラスタ間 F0 差がこれ以上（半音）なら「別人のピッチ帯」とみなす
    def classify(nsplit, across, wmin, dst):
        n1, n2 = map(int, nsplit.split("/"))
        if across <= TH_ACROSS and min(n1, n2) >= 2:
            # 構造は混合。F0（チャネル不変）で確度を分ける:
            #   ΔF0 大 → ピッチ帯も別人 = 共用ほぼ確定 / ΔF0 小 → 同性共用かチャネル分裂（要聴取）
            return "SHARED?F0" if dst >= TH_F0_ST else "MIX?"
        if across <= TH_ACROSS and min(n1, n2) == 1:
            # 片側1本の孤立。ΔF0 で二分する:
            #   ΔF0 大 → 他人のクリップが1本紛れ込んでいる疑い（例: cv_0028 = 7:1 混合）
            #   ΔF0 小 → 収録チャネル差の孤立（例: cv_0105）
            return "INTRUDER?F0" if dst >= TH_F0_ST else "outlier"
        return "ok"
    typed = [(*r_, classify(r_[5], r_[6], r_[7], r_[8])) for r_ in rows]
    order = {"SHARED?F0": 0, "INTRUDER?F0": 1, "MIX?": 2, "outlier": 3, "ok": 4}
    typed.sort(key=lambda r: (order[r[9]], -r[8], r[2]))   # 型 → ΔF0 大きい順 → mean

    with open(a.out, "w", encoding="utf-8") as f:
        f.write(f"# impostor_mean={imp_mean:.4f}\timpostor_p95={imp_p95:.4f}\tcmn={a.cmn}"
                f"\tth_across={TH_ACROSS:.3f}\tth_tight={TH_TIGHT}\tth_f0_st={TH_F0_ST}\n")
        f.write("spk\tn_clips\tintra_cos_mean\tintra_cos_min\tworst_pair\tsplit\tacross_cos"
                "\twithin_min\tf0_gap_st\ttype\n")
        for r_ in typed:
            f.write(f"{r_[0]}\t{r_[1]}\t{r_[2]:.4f}\t{r_[3]:.4f}\t{r_[4]}\t{r_[5]}\t{r_[6]:.4f}"
                    f"\t{r_[7]:.4f}\t{r_[8]:.2f}\t{r_[9]}\n")
    n_sf = sum(1 for r_ in typed if r_[9] == "SHARED?F0")
    n_in = sum(1 for r_ in typed if r_[9] == "INTRUDER?F0")
    n_mx = sum(1 for r_ in typed if r_[9] == "MIX?")
    n_ol = sum(1 for r_ in typed if r_[9] == "outlier")
    print(f"出力: {a.out}（{len(typed)} 話者 / SHARED?F0 {n_sf} / INTRUDER?F0 {n_in}"
          f" / MIX? {n_mx} / outlier {n_ol}）")
    print("\n== 候補上位（SHARED?F0 = 構造+ピッチとも別人 → MIX? = 要聴取）==")
    for r_ in typed[:15]:
        print(f"  {r_[9]:9s} {r_[0]}  split={r_[5]}  across={r_[6]:.3f}"
              f"  ΔF0={r_[8]:.1f}st  within_min={r_[7]:.3f}  mean={r_[2]:.3f}")
    med = float(np.median([r_[2] for r_ in typed]))
    print(f"\n話者内中央値 = {med:.3f} / インポスタ平均 = {imp_mean:.3f}（p95 = {imp_p95:.3f}）"
          f" / マージン = {med - imp_mean:.3f}")
    print("→ SHARED? を監査ノートで聴取確認（worst_pair が最疑ペア）。outlier はチャネル差の可能性が高い")


if __name__ == "__main__":
    main()
