# 学習バンドル仕様（CV / CSJ 共通の I/O 契約）

`colab/cadence_train_colab.ipynb` の**唯一の入力**。この契約を満たす tar なら、コーパスが
何であれノートは無改修で受ける。

**この契約は `dataset_tools/pack_bundle.py` が保証する。** 手で tar を作らないこと
（手作業の規律に依存した規約は必ず破れる。実際に固定名の上書き事故が複数回起きている）。

| コーパス | 作り方 | 配布 |
|---|---|---|
| `cv_r1` | 既存（Zenodo DOI 10.5281/zenodo.21119791） | 公開・自動 DL |
| `csj_core75` / `csj_r1` | `pack_bundle.py` | **配布しない。** 各自の Drive にのみ置く |
| 共著者の CSJ | `pack_bundle.py`（同じコード） | 各自のローカルのみ |

---

## 1. 構成

```
Data/<CORPUS>/
├── config.json          JP-Extra 設定 + spk2id
├── esd_train.list       7 カラム・g2p 済み・wav パスは Data/<CORPUS>/ 相対
├── esd_val.list         同上
├── MANIFEST.json        ★このバンドルの性質の自己申告（下記）
└── wavs/
    ├── <utt>.wav                44.1 kHz / mono / 16-bit PCM
    └── <utt>.wav.cadseq.npy     cadence sidecar（wav の隣・同名 + .cadseq.npy）
```

meta（`*_meta.tgz`）と wav（`*_wavs.tar`）の 2 本に分ける。ノートは両方を同じ場所に展開して
から `config.json` の位置でルートを判定するので、**1 本でも 2 本でもよい**。

出力名は `{corpus}__{run_id}_{meta,wavs}.{tgz,tar}`。`run_id` は
`sha256(esd_train + esd_val + config)[:8]` の**内容ハッシュ**で、同じ入力からは同じ ID が出る。

---

## 2. MANIFEST.json — 数を焼き込まないための仕組み

ノートの §5 検証ゲートは **MANIFEST を読んで期待値を決める**。「298 話者」「18,015 発話」
「`freeze_decoder=True`」といった cv_r1 固有の値をノートに書かない（書くと CSJ で必ず破れる）。

```json
{
  "corpus": "...", "run_id": "...", "layout": "flat",
  "n_speakers_config": 375, "n_speakers_esd": 375,
  "spk2id_sorted": false,          // cv_r1=true / R1=false（core 0..74 → noncore 75..374）
  "freeze_decoder": false,         // cv_r1=true / R1=false ★下記
  "num_styles": 1,
  "esd_train_lines": 131765, "esd_val_lines": 975,
  "wav_count": 132740, "cadseq_count": 113908, "cadseq_coverage": 0.858,
  "cadseq_verified": "sample"
}
```

`spk2id_sorted` と `freeze_decoder` は **契約ではなく申告**。バンドルごとに違ってよい。

---

## 3. 各ファイルの契約

### esd_train.list / esd_val.list

```
wavパス|話者ID|JP|テキスト|phones|tone|word2ph
```

- **7 カラム**（Style-Bert-VITS2 の cleaned 形式）
- wav パスは **`Data/<CORPUS>/wavs/…` 相対**（fork 直下からの相対）。
  packer が書き換える — 素材側は絶対パスでよい
- **1 行 = 1 wav。** ただし **wav が esd 行数より多いことは許される**（eval split ぶんなど）。
  逆（esd が指す wav が無い）は packer が止める
- g2p は fork の `clean_text`（= `preprocess_text.py` と同一経路）で付与
- **`preprocess_text.py` は実行しない**（spk2id が再生成され config と不整合になる）

### config.json

- `data.training_files` / `validation_files` は packer が `Data/<CORPUS>/…` に書き換える
- `data.spk2id`: **順序は触らない。** cv_r1 は辞書順、R1 は core 0..74 → noncore 75..374 で
  **非辞書順**（R0 の `emb_g` を保持するための構造）。どちらも正。`spk2id_sorted` に申告するだけ
- `train.freeze_decoder`: **契約しない。** cv_r1 は `true`、R1（`config_r1.json`）は `false`。
  R1 で decoder が凍結されていたことは重み差分（`dec Δ=0`）で確認済みで、
  **凍結は config フラグでなく学習コードの `requires_grad` で決まっていた**。
  値を MANIFEST に申告するに留める
- `data.num_styles`: cv_r1 も R1 も **1**（Neutral のみ）
- 話者数は自由。`emb_g` は warm-start 時の Missing key = 新規層なので、298 でも 375 でも通る

### wavs のレイアウト ★挙動が変わる

| `--layout` | 配置 | 効果 |
|---|---|---|
| **`flat`（既定）** | `wavs/<utt>.wav` | `default_style.save_styles_by_dirs` が「サブディレクトリ 0」→ **Neutral のみ**。cv_r1 / R1 と一致 |
| `by-speaker` | `wavs/<spk>/<utt>.wav` | サブディレクトリ = 話者ごとの style を生成し、**`num_styles` が話者数+1** になる |

`default_style.py`:
```python
subdirs = [d for d in wav_dir.iterdir() if d.is_dir()]
if len(subdirs) in (0, 1):
    save_neutral_vector(...)     # ← flat はここ
```

cv_r1 も R1 も `num_styles=1` なので、**flat が既定**。`by-speaker` を選ぶのは、話者別 style を
意図的に使うときだけ。CSJ の素材が `wav/<spk>/` 構造でも、packer が flat に畳む
（utt 名が一意でない場合は衝突を検出して止まる）。

### cadence sidecar（`*.wav.cadseq.npy`）★最重要

`data_utils.py` は **`f"{audiopath}.cadseq.npy"`** で読む。`Data/<CORPUS>/wavs/utt.wav` に対し
**`utt.wav.cadseq.npy`**（`.wav` を落とした `utt.cadseq.npy` ではない）。

| 項目 | 契約 |
|---|---|
| 置き場所 | 対応する wav の**隣** |
| 命名 | `{wavファイル名}.cadseq.npy` |
| shape | **(len(phones), 32)** — `phones` は同じ esd 行の第 5 カラムを空白分割した数。**モーラ数ではない** |
| dtype | float32、各行 L2 正規化 |
| 欠落 | 許容（ゼロ系列として学習される）。cv_r1 61% / csj_core75 99.9% / csj_r1 85.8% |

> **★これが最も踏みやすい罠。**
> `data_utils.py` の読み込みは `try/except` + shape チェックで、**どちらも例外を投げず
> ゼロ系列にフォールバックする**（警告は最初の 1 本のみ、`_cadence_warned` で抑制）。
> 名前を間違えても長さを間違えても **学習は「成功」し、cadence が一切効いていないモデルが
> 出来上がる**。UTMOS も落ちないので気づけない。
> → `pack_bundle.py --verify-cadseq` が pack 時に、ノート §5 が学習前に、二重で潰す。
>
> `docs/DATASET_cv_r1.md` §5 の「shape = (モーラ数, 32)」は**誤り**（要訂正）。
> 抽出はモーラ単位でも、sidecar は音素軸に展開して保存されている。CV スモークの実測で
> 200 行サンプル中 160 本ヒット・shape 不一致 0 を確認済み。

`add_blank` の介挿は `data_utils` が行う（`cad_b[1::2] = cad`）。sidecar は**介挿前の長さ**で作る。

---

## 4. 実行例

**csj_core75（Colab）** — 75 話者の split はこの tar にしか残っていない
（Mac の `cadence/data/csj/` は 375 に上書き済み）:

```bash
tar -xzf /content/csj_cadence.tar.gz -C /content/x
python dataset_tools/pack_bundle.py --corpus csj_core75 \
  --config /content/x/csj/config.json \
  --esd-train /content/x/csj/esd_train.list --esd-val /content/x/csj/esd_val.list \
  --wav-root /content/x/csj/wav --out /content/out
```

**csj_r1（Mac・375）** — wav が 2 ルートに分かれているのを packer が統合する:

```bash
D=~/tmp/synth/cadence/data/csj
python dataset_tools/pack_bundle.py --corpus csj_r1 \
  --config $D/config_r1.json \
  --esd-train $D/esd_train.list --esd-val $D/esd_val.list \
  --wav-root $D/r1_cut --wav-root $D/wav \
  --out ~/tmp/synth/bundles
```

`--dry-run` で tar を書かずに検査と統計だけ出せる。**375 は 57 GB（`r1_cut` 46G + `wav` 11G）**
なので、まず `--dry-run` で MANIFEST を確認してから回すこと。

**`--wavs-to-stdout` — ディスクを一切消費しない経路**（375 のように出力がローカル空きを
上回る／食い潰す場合）:

```bash
python dataset_tools/pack_bundle.py --corpus csj_r1 ... --out ~/bundles --wavs-to-stdout \
  | rclone rcat gdrive:bundles/csj_r1__cdae12ce_wavs.tar
```

- packer は staging せず元ファイルから直接 tar に詰めるので、stdout に流せば**ローカルに
  57 GB は生まれない**（meta tgz だけは `--out` にファイルとして落ちる。数百 MB）
- sha256 は**流しながら**計算し、`--out/{corpus}__{run_id}.sha256` と stderr に出す
  （書き終えたファイルを読み直せないため）
- **進捗ログは全て stderr。** stdout に 1 バイトでも混ざれば tar が壊れる
- tar は seek しない `w|`（ストリーム）モード。パイプの相手は rclone でも ssh でも何でもよい

---

## 5. CSJ 側の未了（r2 の宿題）

| 項目 | 状態 |
|---|---|
| 話速フィルタ | **未実装**。`build_csj_corpus` は Tier0 相当の `articulation_rate` 検定を持たない |
| 評価話者の固定 | `sorted(pool.keys())[:80]` は「たまたま決定論的」。pool が変われば集合も変わる |
| 話者集合の整理 | 75（core）/ 231（V2 抽出器）/ 375（R1）/ 80（評価）が併存 |
| `esd_train.list` の版 | 学習後に再生成された疑い（131,765 行 vs ckpt の 8,323 steps/epoch が整合しない）。ckpt に `train_speakers` / `data_sha16` を埋めること |

## 6. ライセンス

| | データ | 学習済み重み |
|---|---|---|
| `cv_r1` | CC0（Zenodo） | AGPL-3.0 で公開可 |
| `csj_*` | **二次配布禁止。** tar をリポジトリ・共有 Drive・HF いずれにも置かない | **公開不可。** 重み・合成音声の公開には NINJAL の事前確認が必要 |

コード（fork・`colab/`・`dataset_tools/`）は CSJ 由来物を含まない限り AGPL-3.0 で公開できる。
`pack_bundle.py` も**公開可**（書き起こし・XML 抜粋・転写入り filelist を埋め込まないこと）。
