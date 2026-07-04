#!/usr/bin/env python3
"""話者性別メタ生成 — Common Voice の validated.tsv から speaker_meta.json を作る

cv_r1 の wav 名は Common Voice 原本のクリップ名（common_voice_ja_XXXXXXXX）を保持しているため、
esd の wav 名 → validated.tsv の path 列 → client_id → gender 列、で対応表なしに性別を引ける。
話者ごとに所属クリップの gender を多数決（通常は単一 client なので全会一致）。

使い方（標準ライブラリのみ。validated.tsv がある環境ならどこでも可 — Mac の sbv2 環境等）:
    python make_speaker_meta.py --esd /path/to/esd_train.list [--esd2 esd_val.list] \\
        --tsv /path/to/cv-corpus-*/ja/validated.tsv --out speaker_meta.json

出力: {"cv_0001": "female", ...}（female / male / unknown）
→ HF モデル repo 直下へ:  huggingface-cli upload slp-hu/cadence-cv_r1 speaker_meta.json speaker_meta.json
→ Space を Restart するとマップが色分けされる。
"""
import argparse
import collections
import csv
import json
import sys
from pathlib import Path


def norm_gender(g: str) -> str:
    g = (g or "").strip().lower()
    if g.startswith(("f", "女")):   # female / female_feminine / 女性
        return "female"
    if g.startswith(("m", "男")):   # male / male_masculine / 男性
        return "male"
    return "unknown"


def clip_key(name: str) -> str:
    """common_voice_ja_37581747(.wav|.mp3 等) → 拡張子なしのクリップ名"""
    return Path(name).stem


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--esd", required=True, help="esd_train.list")
    ap.add_argument("--esd2", default=None, help="esd_val.list（任意・網羅性向上）")
    ap.add_argument("--tsv", required=True, help="Common Voice の validated.tsv")
    ap.add_argument("--out", default="speaker_meta.json")
    a = ap.parse_args()

    # 話者 → クリップ名集合
    spk_clips = collections.defaultdict(set)
    for esd in filter(None, [a.esd, a.esd2]):
        for line in open(esd, encoding="utf-8"):
            f = line.rstrip("\n").split("|")
            spk_clips[f[1]].add(clip_key(f[0].split("/")[-1]))
    need = set().union(*spk_clips.values())
    print(f"esd: 話者 {len(spk_clips)} / 参照クリップ {len(need)}")

    # validated.tsv からクリップ名 → gender（必要分だけ保持）
    clip_gender = {}
    with open(a.tsv, encoding="utf-8", newline="") as fh:
        rd = csv.DictReader(fh, delimiter="\t")
        assert "path" in rd.fieldnames and "gender" in rd.fieldnames, \
            f"★列が見つからない: {rd.fieldnames}"
        for row in rd:
            k = clip_key(row["path"])
            if k in need:
                clip_gender[k] = norm_gender(row.get("gender"))
    print(f"tsv 照合: {len(clip_gender)}/{len(need)} クリップに一致")
    if len(clip_gender) < len(need) * 0.9:
        print("★一致率が低い: validated.tsv の版が前処理時と違う可能性（他の split の tsv も確認）",
              file=sys.stderr)

    meta, stats = {}, collections.Counter()
    for spk, clips in sorted(spk_clips.items()):
        votes = collections.Counter(clip_gender.get(c, "unknown") for c in clips)
        # unknown 以外があればそちらを優先して多数決
        known = {k: v for k, v in votes.items() if k != "unknown"}
        g = max(known, key=known.get) if known else "unknown"
        meta[spk] = g
        stats[g] += 1
    json.dump(meta, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"出力: {a.out} / 内訳: {dict(stats)}")
    print("→ huggingface-cli upload slp-hu/cadence-cv_r1", a.out, "speaker_meta.json → Space を Restart")


if __name__ == "__main__":
    main()
