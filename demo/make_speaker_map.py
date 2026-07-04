#!/usr/bin/env python3
"""話者マップ前計算 — 各学習話者の x-vector 平均を 2 次元 PCA して speaker_map.json を出力

デモ（demo/app.py）のマップ座標を「モデルの emb_g」ではなく「実音声の x-vector」で
作りたいときに 1 回だけ実行する。eval ノートと同じ環境（fork 直下 cwd・データ配置済み・
xvec_extract.py が fork 直下にあること）で:

    pip install umap-learn scikit-learn   # 2次元化に使用（未導入なら）
    python demo/make_speaker_map.py --data Data/cv_r1 --out speaker_map.json
    # → 出力を model_assets/<model>/speaker_map.json に置くとデモが自動で使う

2次元化は --method で選ぶ: umap（既定）/ tsne / pca。
umap → tsne → pca の順で、ライブラリが無ければ自動で次へフォールバックする。
距離は話者照合と同じ cosine。x-vector 抽出は xvec_extract.py（TF/CPU・別プロセス）に委譲する。
"""
import argparse
import collections
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def _norm(y: np.ndarray) -> np.ndarray:
    y = y - y.mean(0, keepdims=True)
    return y / (np.abs(y).max(0, keepdims=True) + 1e-9)


def embed2d(mat: np.ndarray, method: str, seed: int = 0) -> tuple[np.ndarray, str]:
    """2次元化。umap → tsne → pca の順にフォールバック。戻り値 (座標, 実際に使った手法)"""
    if method == "umap":
        try:
            import umap  # umap-learn
            y = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                          metric="cosine", random_state=seed).fit_transform(mat)
            return _norm(np.asarray(y)), "umap"
        except Exception as e:   # 未導入だけでなく、壊れた依存（例: torch と非互換な torchvision）でも落ちる
            print(f"★umap を使えない（{type(e).__name__}: {e}）→ t-SNE にフォールバック")
            print("  （torchvision 起因なら `pip uninstall -y torchvision` で umap が使えるようになる）")
            method = "tsne"
    if method == "tsne":
        try:
            from sklearn.manifold import TSNE
            perp = min(30, max(5, (len(mat) - 1) // 3))
            y = TSNE(n_components=2, perplexity=perp, metric="cosine",
                     init="pca", random_state=seed).fit_transform(mat)
            return _norm(np.asarray(y)), "tsne"
        except Exception as e:
            print(f"★t-SNE を使えない（{type(e).__name__}: {e}）→ PCA にフォールバック")
    x = mat - mat.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return _norm(x @ vt[:2].T), "pca"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="Data/cv_r1")
    ap.add_argument("--out", default="speaker_map.json")
    ap.add_argument("--per-spk", type=int, default=8, help="話者あたりの wav 数（平均を取る）")
    ap.add_argument("--drive", default="/content/drive/MyDrive", help="xvec モデルのキャッシュ先")
    ap.add_argument("--method", default="umap", choices=["umap", "tsne", "pca"],
                    help="2次元化手法（既定 umap。無ければ tsne → pca へ自動フォールバック）")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    data = Path(a.data)
    assert (data / "esd_train.list").exists(), f"★esd_train.list が無い: {data}"
    assert Path("xvec_extract.py").exists(), "★xvec_extract.py を fork 直下に置くこと"

    per = collections.defaultdict(list)
    for line in open(data / "esd_train.list", encoding="utf-8"):
        wav, spk = line.split("|")[0], line.split("|")[1]
        if len(per[spk]) < a.per_spk:
            rel = wav.split("Data/cv_r1/", 1)[1] if "Data/cv_r1/" in wav else wav
            per[spk].append(str((data / rel).resolve()))
    items = {f"{s}__{i}": p for s, ps in per.items() for i, p in enumerate(ps)}
    print(f"話者 {len(per)} / wav {len(items)}（各話者 ≤{a.per_spk}）")

    # --- preflight: xvec_extract が使う TensorFlow が import 可能か（protobuf 版ずれの自己修復）---
    # Colab の TF 2.20 は protobuf>=5.28 が必要だが、SBV2 環境構築の依存解決が日によって
    # 古い protobuf を置くことがある（ImportError: cannot import name 'runtime_version'）。
    pre = subprocess.run([sys.executable, "-c", "import tensorflow"], capture_output=True, text=True)
    if pre.returncode != 0 and "runtime_version" in (pre.stderr or ""):
        print("★protobuf が TF に対して古い → protobuf>=5.28 に更新して再試行")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade",
                        "protobuf>=5.28"], check=True)
        pre = subprocess.run([sys.executable, "-c", "import tensorflow"], capture_output=True, text=True)
    assert pre.returncode == 0, "★TensorFlow を import できない:\n" + (pre.stderr or "")[-800:]

    with tempfile.TemporaryDirectory() as td:
        ij, oz = Path(td) / "items.json", Path(td) / "xvec.npz"
        json.dump(items, open(ij, "w", encoding="utf-8"))
        r = subprocess.run([sys.executable, "xvec_extract.py",
                            "--items", str(ij), "--out", str(oz), "--drive", a.drive])
        assert r.returncode == 0, "★xvec_extract.py 失敗"
        z = np.load(oz)
        mean = {s: np.mean([z[f"{s}__{i}"] for i in range(len(ps))], axis=0)
                for s, ps in per.items()}

    spks = sorted(mean)
    xy, used = embed2d(np.stack([mean[s] for s in spks]), a.method, a.seed)
    out = {s: [round(float(x), 5), round(float(y), 5)] for s, (x, y) in zip(spks, xy)}
    import time
    out["_meta"] = {"method": used, "requested": a.method, "per_spk": a.per_spk,
                    "seed": a.seed, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"出力: {a.out}（{len(out)-1} 話者 / 手法 {used}）→ model_assets/ に置くとデモが使う")


if __name__ == "__main__":
    main()
