#!/usr/bin/env python3
"""学習バンドル packer — esd + config + wav ルート → 契約準拠の tar。

`docs/TRAIN_BUNDLE_SPEC.md` の契約を**コード側で**保証する唯一の場所。
cv_r1 / csj_core75 / csj_r1（375）/ 共著者の CSJ を、すべて同じ経路で吐く。

設計方針:
  - **staging しない。** tarfile.add(src, arcname=...) で元の場所から直接詰める。
    375（wav 57 GB）でディスクを二重に食わないため。
  - **wav ルートを複数受ける。** 375 は `r1_cut/`（非コア 122,476）と `wav/`（コア 10,264）
    に分かれている。ここで単一ツリーへ統合する。
  - **数を焼き込まない。** 話者数・行数・カバレッジは実測して MANIFEST に書く。
    ノートの検証ゲートは MANIFEST を読む（cv_r1 の前提を CSJ に持ち込まない）。
  - **既定は flat レイアウト**（`wavs/<utt>.wav`）。理由は --layout のヘルプ参照。

使用例（Colab・csj_core75 を既存 tar の展開結果から）:
  python dataset_tools/pack_bundle.py \
      --corpus csj_core75 \
      --config    /content/x/csj/config.json \
      --esd-train /content/x/csj/esd_train.list \
      --esd-val   /content/x/csj/esd_val.list \
      --wav-root  /content/x/csj/wav \
      --out /content/out

使用例（Mac・375）:
  python dataset_tools/pack_bundle.py \
      --corpus csj_r1 \
      --config    ~/tmp/synth/cadence/data/csj/config_r1.json \
      --esd-train ~/tmp/synth/cadence/data/csj/esd_train.list \
      --esd-val   ~/tmp/synth/cadence/data/csj/esd_val.list \
      --wav-root  ~/tmp/synth/cadence/data/csj/r1_cut \
      --wav-root  ~/tmp/synth/cadence/data/csj/wav \
      --out ~/tmp/synth/bundles
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import tarfile
import time
from pathlib import Path

ESD_COLS = 7  # wav|spk|lang|text|phones|tone|word2ph
CADENCE_DIM = 32


def log(*a, **k):
    """進捗は必ず stderr へ。--wavs-to-stdout のとき stdout は tar 本体の出力路になるため、
    stdout に 1 バイトでも余計なものを書くと tar が壊れる。"""
    k["file"] = sys.stderr
    k.setdefault("flush", True)
    sys.stderr.write(" ".join(str(x) for x in a) + k.pop("end", "\n"))
    sys.stderr.flush()


class HashingWriter:
    """書きながら sha256 を計算する薄いラッパ（stdout へ流すと後から読み直せないため）。"""

    def __init__(self, raw):
        self.raw, self.h, self.n = raw, hashlib.sha256(), 0

    def write(self, b) -> int:
        self.h.update(b)
        self.n += len(b)
        return self.raw.write(b)

    def flush(self) -> None:
        self.raw.flush()


# ---------------------------------------------------------------- utilities
def sha256_file(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def read_esd(p: Path) -> list[list[str]]:
    rows = []
    for i, line in enumerate(open(p, encoding="utf-8"), 1):
        line = line.rstrip("\n")
        if not line:
            continue
        f = line.split("|")
        if len(f) != ESD_COLS:
            sys.exit(f"★{p}:{i} カラム数 {len(f)} ≠ {ESD_COLS}\n  {line[:120]}")
        rows.append(f)
    return rows


def resolve_wav(raw: str, roots: list[Path], spk: str = "") -> Path | None:
    """esd の wav パスを実体に解決する。

    esd が持つパスは「作られた当時の場所」なので、当てにしない設計にする
    （375 の esd は /Users/... の絶対パス。Colab では当然存在しない）。
    探索順: そのまま → 各ルート直下 → 各ルート/<話者ID>/ → 各ルート/<元の親ディレクトリ名>/。
    """
    q = Path(raw)
    if q.is_absolute() and q.is_file():
        return q
    name = q.name
    cands = []
    for r in roots:
        cands += [r / name]
        if spk:
            cands.append(r / spk / name)
        cands += [r / q.parent.name / name, r / q]
    for c in cands:
        if c.is_file():
            return c
    return None


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True,
                    help="バンドル名。Data/<corpus>/ になり、CORPORA のキーにもなる")
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--esd-train", required=True, type=Path)
    ap.add_argument("--esd-val", required=True, type=Path)
    ap.add_argument("--wav-root", action="append", default=[], type=Path,
                    help="wav の探索ルート。複数指定可（375 は r1_cut と wav の2本）")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--layout", choices=["flat", "by-speaker"], default="flat",
                    help="flat: wavs/<utt>.wav（既定）。by-speaker: wavs/<spk>/<utt>.wav。"
                         "★by-speaker にすると default_style.save_styles_by_dirs が"
                         "話者ごとの style を生成し num_styles が話者数+1 になる。"
                         "cv_r1 も R1(config_r1.json) も num_styles=1 なので flat が既定")
    ap.add_argument("--verify-cadseq", choices=["none", "sample", "full"], default="sample",
                    help="cadseq の shape=(len(phones), 32) を検査する。"
                         "data_utils は不一致を黙ってゼロ系列にするので、ここで潰す")
    ap.add_argument("--verify-n", type=int, default=500, help="sample 時の本数")
    ap.add_argument("--dry-run", action="store_true", help="tar を書かず、検査と統計だけ")
    ap.add_argument("--wavs-to-stdout", action="store_true",
                    help="wavs tar を**ファイルにせず stdout へ流す**。ディスクを一切消費しない。"
                         "例: pack_bundle.py ... --wavs-to-stdout | "
                         "rclone rcat gdrive:bundles/<corpus>__<run_id>_wavs.tar  "
                         "（sha256 は流しながら計算して stderr と .sha256 に出す）")
    a = ap.parse_args()

    roots = [r.expanduser().resolve() for r in a.wav_root]
    cfg_p = a.config.expanduser().resolve()
    out = a.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    cfg = json.load(open(cfg_p, encoding="utf-8"))
    splits = {"train": read_esd(a.esd_train.expanduser().resolve()),
              "val": read_esd(a.esd_val.expanduser().resolve())}
    log(f"esd: train {len(splits['train'])} / val {len(splits['val'])} 行")

    # --- run_id は内容ハッシュ（同じ入力からは同じ ID）------------------------
    h = hashlib.sha256()
    for p in (a.esd_train, a.esd_val, cfg_p):
        h.update(sha256_file(Path(p).expanduser().resolve()).encode())
    run_id = h.hexdigest()[:8]
    log("run_id:", run_id)

    # --- wav の解決とレイアウト決定 -----------------------------------------
    plan: list[tuple[Path, str, list[str]]] = []   # (src, arc_rel, esd_row)
    missing: list[str] = []
    seen: dict[str, str] = {}
    base = f"Data/{a.corpus}"
    for split, rows in splits.items():
        for r in rows:
            src = resolve_wav(r[0], roots, r[1])
            if src is None:
                missing.append(r[0])
                continue
            rel = (f"wavs/{r[1]}/{src.name}" if a.layout == "by-speaker"
                   else f"wavs/{src.name}")
            if rel in seen and seen[rel] != str(src):
                sys.exit(f"★wav 名が衝突: {rel}\n  {seen[rel]}\n  {src}\n"
                         f"  → --layout by-speaker にするか、utt 名を一意にする")
            seen[rel] = str(src)
            plan.append((src, rel, r))
    if missing:
        log(f"★wav が見つからない: {len(missing)} 本（先頭 3）")
        for m in missing[:3]:
            log("   ", m)
        sys.exit("  → --wav-root を足す")
    log(f"wav 解決: {len(plan)} 本 / レイアウト {a.layout}")

    # --- cadseq の同梱計画 + 検査 -------------------------------------------
    cad_plan: list[tuple[Path, str]] = []
    for src, rel, _ in plan:
        c = Path(str(src) + ".cadseq.npy")
        if c.is_file():
            cad_plan.append((c, rel + ".cadseq.npy"))
    cov = len(cad_plan) / max(len(plan), 1)
    log(f"cadseq: {len(cad_plan)} 本（カバレッジ {cov:.1%}）")
    if not cad_plan:
        log("★cadseq が 1 本も無い。このバンドルで学習しても cadence は効かない")

    bad = 0
    if a.verify_cadseq != "none" and cad_plan:
        import numpy as np
        idx = {rel + ".cadseq.npy": r for _, rel, r in plan}
        targets = cad_plan if a.verify_cadseq == "full" else cad_plan[:: max(1, len(cad_plan) // a.verify_n)]
        for c, arc in targets:
            row = idx[arc]
            want = len(row[4].split(" "))
            shp = np.load(c, mmap_mode="r").shape
            if not (len(shp) == 2 and shp[0] == want and shp[1] == CADENCE_DIM):
                bad += 1
                if bad <= 3:
                    log(f"★shape 不一致: {c.name} {shp} vs ({want}, {CADENCE_DIM})")
        log(f"cadseq 検査: {len(targets)} 本中 不一致 {bad}")
        if bad:
            sys.exit("★shape 不一致がある → data_utils は黙ってゼロ系列にする。"
                     "抽出側を直すこと（docs/TRAIN_BUNDLE_SPEC.md）")

    # --- config / esd の書き換え --------------------------------------------
    cfg = json.loads(json.dumps(cfg))  # deep copy
    cfg["data"]["training_files"] = f"{base}/esd_train.list"
    cfg["data"]["validation_files"] = f"{base}/esd_val.list"
    spk2id = cfg["data"].get("spk2id", {})
    esd_spk = {r[1] for _, _, r in plan}
    if not esd_spk <= set(spk2id):
        sys.exit(f"★esd の話者が spk2id に無い: {sorted(esd_spk - set(spk2id))[:5]}")

    esd_text = {}
    for split, rows in splits.items():
        lines = []
        for r in rows:
            src = resolve_wav(r[0], roots, r[1])
            rel = (f"wavs/{r[1]}/{src.name}" if a.layout == "by-speaker"
                   else f"wavs/{src.name}")
            lines.append("|".join([f"{base}/{rel}"] + r[1:]))
        esd_text[split] = "\n".join(lines) + "\n"

    manifest = {
        "corpus": a.corpus, "run_id": run_id,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "layout": a.layout,
        "n_speakers_config": len(spk2id),
        "n_speakers_esd": len(esd_spk),
        "spk2id_sorted": list(spk2id) == sorted(spk2id),
        "freeze_decoder": cfg["train"].get("freeze_decoder"),
        "num_styles": cfg["data"].get("num_styles"),
        "esd_train_lines": len(splits["train"]),
        "esd_val_lines": len(splits["val"]),
        "wav_count": len(plan),
        "cadseq_count": len(cad_plan),
        "cadseq_coverage": round(cov, 4),
        "cadseq_verified": a.verify_cadseq,
        "source_config": str(cfg_p),
        "source_wav_roots": [str(r) for r in roots],
    }
    log("\n--- MANIFEST ---")
    log(json.dumps(manifest, ensure_ascii=False, indent=2))

    if a.dry_run:
        log("\n--dry-run のため tar は書かない")
        return

    # --- meta tgz -----------------------------------------------------------
    meta_p = out / f"{a.corpus}__{run_id}_meta.tgz"
    with tarfile.open(meta_p, "w:gz") as t:
        def add_bytes(rel: str, data: bytes):
            ti = tarfile.TarInfo(f"{base}/{rel}")
            ti.size = len(data)
            ti.mtime = int(time.time())
            t.addfile(ti, io.BytesIO(data))
        add_bytes("config.json", json.dumps(cfg, ensure_ascii=False, indent=2).encode())
        add_bytes("esd_train.list", esd_text["train"].encode())
        add_bytes("esd_val.list", esd_text["val"].encode())
        add_bytes("MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2).encode())
        for i, (c, arc) in enumerate(cad_plan):
            t.add(c, arcname=f"{base}/{arc}")
            if i % 20000 == 0:
                log(f"  cadseq {i}/{len(cad_plan)}", flush=True)
    log("meta:", meta_p, f"{meta_p.stat().st_size/2**20:.1f} MiB")

    # --- wavs tar（無圧縮。wav は圧縮が効かない）-----------------------------
    wav_name = f"{a.corpus}__{run_id}_wavs.tar"
    if a.wavs_to_stdout:
        # ストリーム出力: ディスクに 57 GB を作らない。seek しない "w|" モードを使う。
        w = HashingWriter(sys.stdout.buffer)
        with tarfile.open(fileobj=w, mode="w|") as t:
            for i, (src, rel, _) in enumerate(plan):
                t.add(src, arcname=f"{base}/{rel}")
                if i % 20000 == 0:
                    log(f"  wav {i}/{len(plan)}")
        w.flush()
        wav_sha, wav_size = w.h.hexdigest(), w.n
        log(f"wavs: (stdout) {wav_size/2**30:.2f} GiB")
    else:
        wav_p = out / wav_name
        with tarfile.open(wav_p, "w") as t:
            for i, (src, rel, _) in enumerate(plan):
                t.add(src, arcname=f"{base}/{rel}")
                if i % 20000 == 0:
                    log(f"  wav {i}/{len(plan)}")
        wav_sha, wav_size = sha256_file(wav_p), wav_p.stat().st_size
        log("wavs:", wav_p, f"{wav_size/2**30:.2f} GiB")

    # --- チェックサム -------------------------------------------------------
    sums = out / f"{a.corpus}__{run_id}.sha256"
    with open(sums, "w") as f:
        f.write(f"{sha256_file(meta_p)}  {meta_p.name}\n")
        f.write(f"{wav_sha}  {wav_name}\n")
    log("sha256:", sums)
    log(f"  {wav_sha}  {wav_name}")
    log("\n完了。CORPORA に足す形:")
    log(f'''    "{a.corpus}": {{
        "label": "...",
        "fetch": "local",
        "files": [("{meta_p.name}", None, {meta_p.stat().st_size // 2}),
                  ("{wav_name}", None, {wav_size // 2})],
        "smoke": None,
        "license": "★CSJ は二次配布禁止。tar も学習済み重みも公開不可",
    }},''')


if __name__ == "__main__":
    main()
