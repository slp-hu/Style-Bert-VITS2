#!/usr/bin/env python3
"""cv_r1 スモーク学習用サブセットバンドル生成スクリプト

全量の Data/cv_r1 ツリー（Zenodo バンドル展開済み or パイプライン出力）から、
公開 train ノートのスモークモードが使うサブセットバンドル
`cadence_cv_r1_smoke_v1.tgz` を生成する。

使い方:
    python make_smoke_bundle.py --data /path/to/Data/cv_r1
    # → カレントに cadence_cv_r1_smoke_v1.tgz を出力

    Colab から Drive マウント上のツリー（/content/drive/MyDrive/.../Data/cv_r1）を
    直接指しても動く（FUSE の一時的な I/O エラーは自動リトライで吸収する）。
    出力 tgz は必ずローカル（/content 等）に置くこと。Colab 上での実行なら完成後に
    ブラウザへ自動ダウンロードする（--no-download で抑止）。読むのは約 2,000 ファイルで
    数分程度。それより速くしたい場合は Zenodo tar をローカル展開してから指す。

選択規則（決定的・乱数なし）:
    - train の発話数が多い順に N_SPEAKERS 話者（同数はスピーカー名昇順）
    - 各話者、esd ファイル出現順に train は MAX_UTT 発話・val は VAL_UTT 発話まで

同梱物（tar 内パスは Data/cv_r1/... で全量バンドルと同一構造）:
    - esd_train.list / esd_val.list（サブセット済み）
    - config.json（全量と同一。spk2id=298 を維持 = 底モデル warm-start 検証が本番と等価）
    - サブセットの wav と、存在する cadseq sidecar
    - SMOKE_MANIFEST.json（件数など。ノートの検証ゲートが参照）

アップロード先（ノートの既定 URL に合わせる）:
    GitHub Releases: slp-hu/Style-Bert-VITS2
    tag = cv_r1-smoke-v1 / asset 名 = cadence_cv_r1_smoke_v1.tgz
"""
import argparse
import collections
import io
import json
import sys
import tarfile
import time
from pathlib import Path

# Google Drive の FUSE マウントは大量の stat/read で一時的な OSError (Errno 5) を
# 返すことがある。指数バックオフ付きリトライで吸収する（Drive 直読みで実行可能にする）。
RETRY_WAITS = (0.5, 1, 2, 4, 8)

def _retry(fn, what):
    for i, wait in enumerate((*RETRY_WAITS, None)):
        try:
            return fn()
        except OSError as e:
            if wait is None:
                raise
            print(f"  リトライ {i+1}/{len(RETRY_WAITS)} ({what}): {e}", file=sys.stderr)
            time.sleep(wait)

def exists_retry(p: Path) -> bool:
    return _retry(p.exists, f"exists {p.name}")

def read_bytes_retry(p: Path) -> bytes:
    return _retry(p.read_bytes, f"read {p.name}")

BUNDLE_VERSION = "v1"
DEFAULT_OUT = f"cadence_cv_r1_smoke_{BUNDLE_VERSION}.tgz"
SOURCE_DOI = "10.5281/zenodo.21119791"
ARC_ROOT = "Data/cv_r1"          # tar 内ルート（全量バンドルと同一）
ESD_PREFIX = "Data/cv_r1/"       # esd 行の wav パスの接頭辞


def spk(line: str) -> str:
    return line.split("|")[1]


def select_subset(tr, va, n_speakers, max_utt, val_utt):
    cnt = collections.Counter(spk(l) for l in tr)
    keep = {s for s, _ in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[:n_speakers]}
    per_t, per_v, sub_tr, sub_va = collections.Counter(), collections.Counter(), [], []
    for l in tr:
        s = spk(l)
        if s in keep and per_t[s] < max_utt:
            sub_tr.append(l); per_t[s] += 1
    for l in va:
        s = spk(l)
        if s in keep and per_v[s] < val_utt:
            sub_va.append(l); per_v[s] += 1
    return keep, sub_tr, sub_va


def wav_relpath(line: str) -> str:
    p = line.split("|")[0]
    if ESD_PREFIX not in p:
        sys.exit(f"★esd の wav パスに {ESD_PREFIX} が含まれない: {p}")
    return p.split(ESD_PREFIX, 1)[1]


def cadseq_candidates(rel: str):
    """sidecar 命名の揺れ（foo.cadseq.npy / foo.wav.cadseq.npy）の両方を候補にする"""
    r = Path(rel)
    yield str(r.with_suffix(".cadseq.npy"))
    yield rel + ".cadseq.npy"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="全量の Data/cv_r1 ツリーへのパス")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--n-speakers", type=int, default=30)
    ap.add_argument("--max-utt", type=int, default=40)
    ap.add_argument("--val-utt", type=int, default=2)
    ap.add_argument("--no-download", action="store_true",
                    help="Colab 実行時のブラウザ自動ダウンロードを抑止")
    a = ap.parse_args()

    data = Path(a.data)
    for f in ("esd_train.list", "esd_val.list", "config.json"):
        if not (data / f).exists():
            sys.exit(f"★{data / f} が無い（--data は Data/cv_r1 を指すこと）")

    tr = [l.rstrip("\n") for l in open(data / "esd_train.list", encoding="utf-8")]
    va = [l.rstrip("\n") for l in open(data / "esd_val.list", encoding="utf-8")]
    keep, sub_tr, sub_va = select_subset(tr, va, a.n_speakers, a.max_utt, a.val_utt)
    print(f"選択: 話者 {len(keep)} / train {len(sub_tr)} 行 / val {len(sub_va)} 行")

    # サブセットが参照する wav / cadseq を収集
    wav_rels, missing = [], []
    for l in sub_tr + sub_va:
        rel = wav_relpath(l)
        (wav_rels if exists_retry(data / rel) else missing).append(rel)
    if missing:
        sys.exit(f"★wav 実体が {len(missing)} 件見つからない（例: {missing[:3]}）")
    cad_rels = []
    for rel in wav_rels:
        for c in cadseq_candidates(rel):
            if exists_retry(data / c):
                cad_rels.append(c)
                break

    cfg = json.load(open(data / "config.json", encoding="utf-8"))
    n_spk2id = len(cfg.get("data", {}).get("spk2id", {}))
    manifest = {
        "bundle": Path(a.out).name,
        "version": BUNDLE_VERSION,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_doi": SOURCE_DOI,
        "selection": {"n_speakers": a.n_speakers, "max_utt": a.max_utt, "val_utt": a.val_utt,
                      "rule": "train 発話数上位・同数は話者名昇順 / ファイル出現順に上限まで"},
        "n_speakers": len(keep),
        "train_lines": len(sub_tr),
        "val_lines": len(sub_va),
        "wav_count": len(wav_rels),
        "cadseq_count": len(cad_rels),
        "spk2id_size": n_spk2id,
    }

    out = Path(a.out)
    with tarfile.open(out, "w:gz") as tar:
        def add_bytes(arc, b: bytes):
            ti = tarfile.TarInfo(f"{ARC_ROOT}/{arc}")
            ti.size = len(b); ti.mtime = int(time.time())
            tar.addfile(ti, io.BytesIO(b))
        add_bytes("esd_train.list", ("\n".join(sub_tr) + "\n").encode("utf-8"))
        add_bytes("esd_val.list", ("\n".join(sub_va) + "\n").encode("utf-8"))
        add_bytes("SMOKE_MANIFEST.json",
                  (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
        add_bytes("config.json", read_bytes_retry(data / "config.json"))
        for i, rel in enumerate(wav_rels + cad_rels, 1):
            add_bytes(rel, read_bytes_retry(data / rel))
            if i % 100 == 0:
                print(f"  追加 {i}/{len(wav_rels) + len(cad_rels)}", end="\r")
    print()

    mb = out.stat().st_size / 1e6
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"\n出力: {out} ({mb:.1f} MB)")
    print("アップロード: GitHub Releases（slp-hu/Style-Bert-VITS2）に "
          f"tag `cv_r1-smoke-{BUNDLE_VERSION}` を作成し、この名前のままアセット添付する。")
    if mb > 1800:
        print("★警告: GitHub Release のアセット上限 2 GB に接近。--max-utt を下げること。")

    # Colab 上ならブラウザへ自動ダウンロード（GitHub Releases へのアップロード用）
    if not a.no_download:
        try:
            from google.colab import files  # type: ignore
            import IPython
        except ImportError:
            pass  # Colab 以外では何もしない
        else:
            ip = IPython.get_ipython()
            if ip is None or getattr(ip, "kernel", None) is None:
                # `!python ...` のサブプロセス実行では IPython カーネル経由の DL は不可
                print("（`!python` 実行のため自動ダウンロード不可。ノートのセルで次を実行:）")
                print(f"    from google.colab import files; files.download({str(out)!r})")
            else:
                print("ブラウザへのダウンロードを開始します…（ブロックされたらブラウザの許可を確認）")
                files.download(str(out))


if __name__ == "__main__":
    main()
