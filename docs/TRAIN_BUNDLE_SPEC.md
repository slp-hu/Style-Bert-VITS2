# 学習バンドル仕様（CV / CSJ 共通の I/O 契約）

`colab/cadence_train_colab.ipynb` の**唯一の入力**。この仕様を満たす tar を用意すれば、
コーパスが何であれノートは無改修で通る（新コーパスは §1.2 の `CORPORA` に 1 エントリ足すだけ）。

- `cv_r1`: Zenodo（DOI 10.5281/zenodo.21119791）に公開。ノートが自動取得する
- `csj_r1`: **各自が CSJ 現物から作る。配布しない。** `DRIVE_BASE/corpus_in/csj_r1.tar.gz` に置く

## 1. 中身

tar のルートは `Data/<CORPUS>/` で、Style-Bert-VITS2 の学習ツリーそのままの構造。

```
Data/<CORPUS>/
├── config.json                     JP-Extra 設定 + spk2id + freeze_decoder=true
├── esd_train.list                  学習用 filelist（7 カラム・g2p 済み）
├── esd_val.list                    検証用 filelist（同上）
├── esd.list                        （任意・分割前の原本）
└── wavs/<spk>/<utt>.wav            44.1 kHz / mono / 16-bit PCM
    <spk>/<utt>.wav.cadseq.npy      ★cadence sidecar（wav の隣・同名 + .cadseq.npy）
```

`cv_r1` は meta（`*_meta_*.tgz`）と wav（`*_wavs_*.tar`）の 2 本に分かれるが、
ノートは両方を同じ stage に展開してから `config.json` の位置でルートを判定するので、
**1 本でも 2 本でもよい**（`CORPORA[...]["files"]` に並べるだけ）。

## 2. 各ファイルの契約

### esd_train.list / esd_val.list

```
wavパス|話者ID|JP|テキスト|phones|tone|word2ph
```

- **7 カラム**（Style-Bert-VITS2 の cleaned 形式）。wav パスは **fork 直下からの相対**（`Data/<CORPUS>/wavs/...`）
- **1 行 = 1 wav。** ノートは `wav 実数 == esd 行数` を検証ゲートで見る（数はハードコードしない）
- g2p は fork の `clean_text`（= `preprocess_text.py` と同一経路）で付与する
- **`preprocess_text.py` は実行しない。** spk2id が再生成され config と不整合になる

### config.json

- `data.training_files` / `validation_files` は **`Data/<CORPUS>/…` 相対**
- `data.spk2id`: **辞書順**（合成・評価ノートが `sorted()` 前提で話者を引く）。
  esd に現れる話者はすべて spk2id に含まれること
- `train.freeze_decoder = true`
- 話者数は自由（`emb_g` は warm-start 時の Missing key = 新規層なので、CV 298 でも CSJ 375 でも通る）

### cadence sidecar（`*.wav.cadseq.npy`）★最重要

`data_utils.py` は **`f"{audiopath}.cadseq.npy"`** で読む。すなわち
`Data/<CORPUS>/wavs/spk/utt.wav` に対し **`utt.wav.cadseq.npy`**（`.wav` を落とした
`utt.cadseq.npy` ではない）。

| 項目 | 契約 |
|---|---|
| 置き場所 | 対応する wav の**隣**（フラットな `cadseq/` ディレクトリではない） |
| 命名 | `{wavファイル名}.cadseq.npy` |
| shape | **(len(phones), 32)** — `phones` は同じ esd 行の第 5 カラムを空白分割した数。**モーラ数ではない** |
| dtype | float32 |
| 値 | 各行 L2 正規化（‖cad‖ ≈ 1）。ポーズは含まない |
| 欠落 | 許容（その発話はゼロ系列として学習される。cv_r1 の付与率は 61%） |

> **★これが新コーパスで最も踏みやすい罠。**
> `data_utils.py` の読み込みは `try/except` + shape 不一致チェックで、**どちらも
> 例外を投げずゼロ系列にフォールバックする**（警告ログ 1 行のみ）。
> 名前を間違えても長さを間違えても **学習は「成功」し、cadence が一切効いていない
> モデルが出来上がる**。しかも UTMOS は落ちないので気づけない。
> → 学習ノートの §5 検証ゲートが esd を 200 行サンプルして
> 「命名が data_utils と一致するか」「shape = (音素数, 32) か」を実測する。
> **NG のまま先へ進まないこと。**

`add_blank` の介挿はノート側でなく `data_utils` が行う（`cad_b[1::2] = cad`）。
sidecar は**介挿前の長さ**で作る。

## 3. CSJ 側で追加になる作業

CSJ は `wav + audio_manifest.csv + cadence_mora.jsonl` までしか無いので、
**そこから上の 4 点（esd 7 カラム / config.json / cadseq sidecar / wavs 配置）を作る段が要る。**
`dataset_tools/` の adapter がその位置に立つ（`corpus_adapters/csj.py` → `dataset_tools/pack.py`）。

CV との差分で効くのは:

| 項目 | cv_r1 | csj_r1 |
|---|---|---|
| pause / 実発話長 | MFA TextGrid の sil | XML の `Phone`（無音 = `SpzS`/`SbS` + Phone 間ギャップ）。**IPU 時刻は使わない**（時刻破損あり） |
| manifest の `kana` 列 | 全行空 | **埋める** |
| 話速フィルタ | Tier0（`articulation_rate > 11.7` を除外） | **未実装**（`build_csj_corpus` は当時フィルタ無し）。同検定を通すこと |
| 評価話者の固定 | select 段で決定的 | `sorted(pool.keys())[:80]` は**たまたま決定論的**なだけ。pool が変われば集合も変わる → 明示的に固定する |

**成果物のパスには `run_id` を入れる**（`{name}__{run_id}.{ext}`）。固定名の上書き事故が
実際に複数回起きている（署名 npz / eval tar / TextGrid tar）。

## 4. ライセンス

| | データ | 学習済み重み |
|---|---|---|
| `cv_r1` | CC0（Zenodo） | AGPL-3.0 で公開可（底モデル `litagin/Style-Bert-VITS2-2.0-base-JP-Extra` 由来） |
| `csj_r1` | **二次配布禁止。** tar をリポジトリ・Drive 共有・HF いずれにも置かない | **公開不可。** 重み・合成音声の公開には NINJAL の事前確認が必要 |

コード（fork・`colab/`・`dataset_tools/`）は CSJ 由来物を含まない限り AGPL-3.0 で公開できる。
`corpus_adapters/csj.py` も **コードは公開可**（書き起こし・XML 抜粋・転写入り filelist を
埋め込まないこと）。
