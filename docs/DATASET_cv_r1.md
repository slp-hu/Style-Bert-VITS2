# cadence cv_r1 データセット仕様（データシート）

**cadence_cv_r1** — Common Voice 日本語（CC0）から構築した、日本語多話者 TTS /
声道リズム（cadence）研究用の完全公開データセット。

- 配布: Zenodo [DOI 10.5281/zenodo.21119791](https://doi.org/10.5281/zenodo.21119791)（CC0-1.0）
- 版: `v20260702`
- 学習コード・構築パイプライン: [slp-hu/Style-Bert-VITS2 branch `layer-b-cadence-seq`](https://github.com/slp-hu/Style-Bert-VITS2/tree/layer-b-cadence-seq)
- 学習済みモデル: [HF `slp-hu/cadence-cv_r1`](https://huggingface.co/slp-hu/cadence-cv_r1) / デモ: [HF Spaces](https://huggingface.co/spaces/slp-hu/cadence-cv_r1-demo)

## 1. 構成（配布ファイル）

| ファイル | サイズ | 内容 |
|---|---|---|
| `cadence_cv_r1_meta_v20260702.tgz` | 31.9 MiB | `config.json` / esd リスト ×3 / cadence sidecar（`.cadseq.npy` ×11,010）/ MANIFEST |
| `cadence_cv_r1_wavs_v20260702.tar` | 4.81 GiB（無圧縮） | wav ×18,015（44.1 kHz / mono / 16-bit PCM） |

tar 内部は `Data/cv_r1/...` をルートとする、Style-Bert-VITS2 の学習ツリーそのままの構造。

## 2. 規模と統計

- 話者: **298**（`cv_0001`〜`cv_0298`）
- 発話: **18,015**（train 14,226 / val 3,789 — esd 行数）。分割は乱数ではなく
  **選定段で決定的に**行われる: train = 話者あたり最大 200 発話、val = 話者照合評価用に
  切り出した eval split（話者あたり enroll 8 + trial 5、1.5〜20 秒、品質採点通過分）
- 性別内訳（Common Voice メタデータ由来・話者単位）: **female 172 / male 126**（unknown 0）
- cadence sidecar 付与率: 11,010 / 18,015 ≈ **61%**。sidecar が付くのは
  **MFA アライメント成功（TextGrid 生成）かつ MFA モーラ列と pyopenjtalk 構造特徴の
  モーラ数が一致（対応づけ成立）**した発話のみ。不成立の発話はゼロ系列として学習される
- 音声形式: 44.1 kHz / mono / 16-bit PCM。元 mp3 を `librosa.load(sr=44100, mono=True)` で
  リサンプルし、端点無音を `librosa.effects.trim(top_db=30)` でトリム後に書き出し
  （振幅正規化は行わない）

## 3. 出所と選定

- 元データ: **Mozilla Common Voice 日本語 `cv-corpus-25.0-2026-03-09/ja`**（CC0）。
  選定は `validated.tsv` のみを対象とする（+ `clip_durations.tsv` を長さ参照に使用）
- 話者選定（`cv_pipeline.py` の一直線 DAG: select → score → report → materialize → prep。
  冪等・ハッシュベースの鮮度判定つき）:
  1. **select**: `validated.tsv`（+ `clip_durations.tsv`）から、**性別ラベル明示**
     （`male_masculine` / `female_feminine`）のクリップで **strict 発話数 ≥ 30** となる適格話者を対象に
     **300 名**を決定的に選定（選定キー `cadence_cv_v1`。除外はこの段では行わない = 安定 universe）。
     話者あたり train は最大 **200 発話**（`TRAIN_CAP`。採用クリップには gender 未記入行も
     含まれ得る — フィルタは適格判定に効く。実測例: ある話者で male 28 + 未記入 5）。あわせて話者照合評価用の
     eval split（enroll 8 / trial 5 発話、1.5〜20 秒）も切り出す（TTS 学習には未使用）
  2. **score**: クリップ単位の音質採点（Tier0、下記）
  3. **report**: 採点の結果 **train 採用数 < 10** となった話者を丸ごと除外（穴埋めなし）
  4. **materialize**: 除外話者・不採用クリップを除いて 44.1 kHz wav 化 + manifest
     （`audio_manifest_cv.csv`: uid / spk / sex / split / client_id ほか）→ 最終 **298 名**
  5. **prep**: train split から `esd.list`（テキスト空の行はスキップ）

### Tier0 品質採点（クリップ単位・`tier0-v2-silence-trimmed`）

| 指標 | 定義（要約） | 閾値 |
|---|---|---|
| クリッピング率 | \|y\| ≥ 0.99 のサンプル比率 | ≤ 0.01 |
| 音量 | RMS (dBFS) | ≥ −40 |
| 無音率 | 端点トリム後、フレーム RMS < peak−40dB の比率 | ≤ 0.60 |
| 帯域幅 | 平均パワースペクトルがピーク −55 dB を上回る上限周波数 | ≥ 5,000 Hz |
| SNR | フレーム RMS の p90 / p10 (dB) | ≥ 5 |
| 最短長 | トリム後の長さ | ≥ 0.5 s |

- クリップ名は Common Voice 原本のまま保持（`common_voice_ja_XXXXXXXX.wav`）。これにより
  原本の `validated.tsv` と突合すれば client_id・性別・年代等のメタデータを引ける
  （性別メタの生成例: fork の `demo/make_speaker_meta.py`）
- 話者 ID（`cv_NNNN`）↔ Common Voice `client_id` の対応は構築時成果物
  `cv_selected_speakers.json` / `audio_manifest_cv.csv`（`talk` 列）に保存される
  （**Zenodo 配布物には非同梱**。上記のクリップ名突合で性別等は復元可能なため、
  対応表そのものの公開有無はツール公開時に判断）

## 4. 前処理パイプライン（手順1〜7）

fork 同梱のツールで再現可能（ツール一式の公開は準備中）。

1. Common Voice ja → ESD 形式（g2p は **Style-Bert-VITS2 の `clean_text`** —
   `preprocess_text.py` と同一経路 — で phones / tone / word2ph を付与。々・〻 の展開を含む）
2. MFA による強制アライメント（**アライメント**。無声化母音 = 子音単独モーラ、促音 Cː、
   長音 2 モーラ、撥音 ɴ の扱いを含む。使用モデル・辞書と実行スクリプト
   `mfa_align_cv_r1.sh` はツール公開時に同梱予定）
3. モーラ特徴量抽出
4. cadence 系列抽出（モーラ単位・32 次元潜在、CV 版抽出器 `cadence_extractor_mora_pos.pt`。
   ポーズモーラは除外、モーラごとに L2 正規化）
5. precompute（sidecar 化）
6. 学習（Style-Bert-VITS2 JP-Extra、底モデル warm-start）
7. Zenodo 配布物のパッケージング

## 5. ファイル形式

- **esd リスト**: `wavパス|話者ID|JP|テキスト|phones|tone|word2ph` の **7 カラム**
  （Style-Bert-VITS2 の cleaned 形式。g2p 済みのため `preprocess_text.py` の実行は不要 —
  むしろ実行すると spk2id が再生成され配布 config と不整合になるため**実行しないこと**）。
  wav パスは `Data/cv_r1/` からの相対
- **cadence sidecar（`*.cadseq.npy`）**: 対応する wav と同名ベースの numpy 配列、
  **shape = (モーラ数, 32)・float32・各行 L2 正規化（‖cad‖ ≈ 1）**。ポーズモーラは含まない
- **config.json**: JP-Extra 設定 + `spk2id`（298 話者）。`freeze_decoder=true`

## 6. 既知の制約・注意

- クラウドソーシング収録のため音質は多様（原音 UTMOS 平均 ≈ 2.37）。スタジオ品質 TTS の
  学習データとしてではなく、多話者・話者匿名化研究のベースラインとして設計している
- cadence sidecar は全発話には付与されていない（§2）。sidecar 欠落発話はゼロ系列として学習
- テキストは Common Voice の文（朗読調・比較的短文中心）
- **性別ラベルは Common Voice の自己申告**であり、申告と音響的実態の不一致を含み得る。
  話者内でのラベル矛盾は 0 / 298 名（実測）で、話者単位の性別付与にデータ処理由来の
  劣化はない。本データセットはラベルを聴感で上書きせず、原本の申告値を用いる（再現性優先）
- **「話者」= Common Voice のアカウント（client_id）であり、1人の人物とは限らない**。
  CV の validated は「テキストが正しく読まれているか」への賛成票による検証であり、
  話者同一性・話者数は検証対象外。アカウント共用（家族・教室等での録音）が混入し得る。
  実例: `cv_0127` は male 申告の単一アカウントだが、聴取監査で男女複数人の声を確認。
  複数人の声が同一クリップに入る例 `common_voice_ja_36362240` / `common_voice_ja_36363149` は
  いずれも **up_votes 2 / down_votes 0 で検証を正規に通過**している（validated.tsv 実測）
  — 「validated ⊅ 単一話者」の一次証拠
- **話者純度の実測（2026-07 監査）**: 2軸スクリーニング
  （話者内 x-vector の2分割構造 × クラスタ間 F0 差。`dataset_tools/speaker_purity_report.py`）で
  候補を抽出し聴取監査（`colab/cv_r1_speaker_audit.ipynb`）した結果、
  **複数人の混入を計 9 名で確認**: 複数本ずつの共用（SHARED）8 名
  （`cv_0019`, `cv_0025`, `cv_0075`, `cv_0078`, `cv_0095`, `cv_0127`, `cv_0258`, `cv_0279`） と、1 本だけ別人が紛れ込む型（INTRUDER）1 名
  （`cv_0028`）。single 確認 7 名 / 判定保留 11 名。境界例 `cv_0248`（ΔF0 3.3 半音）は
  聴取の結果 single（同一話者の環境・話し方差）。判定一覧は
  `docs/audit/purity_audit_verdicts.json` としてリポジトリに同梱。
  話者単位の分析を行う利用者は該当 9 名の除外を推奨。
  なお該当話者を含めても TTS 学習全体への影響は軽微（8/298 話者・混入クリップは
  さらにその一部）と考えられるが、話者埋め込みの解釈には注意
- **その他の源データ品質の観察**（聴取監査で確認された個別事象）: 不均衡な混合アカウント
  （例: `cv_0028` は 8 クリップ中 7 本が一つの声で、1 本だけ別人 — 塊間 cos −0.11・
  ΔF0 10.4 半音、x-vector 実測。少数の紛れ込みは話者平均を歪め、x-vector マップ上で
  男女クラスタの中間に浮く。純度検査の INTRUDER?F0 型として検出される）、
  極端なクリックノイズと極小音声レベルの共存クリップ、速度/ピッチ異常
  （レート誤ラベルと推定される早口クリップ、例 `common_voice_ja_41802387`）。
  **合成音声の投稿が混入している可能性も否定できない**（CV の検証はテキスト一致のみで、
  合成音の排除機構はない）。総じて、**本データセットを汎用の高品質 TTS に用いるには
  さらなる取捨選択が必要**であり、本リリース（r1）は多話者・話者匿名化研究の
  ベースラインとして位置づける

## 7. ライセンスと倫理

- データ加工物: **CC0-1.0**（元データの Common Voice ja も CC0）
- 話者は Common Voice に CC0 で音声を提供した話者であり、氏名等の直接識別子は含まれない。
  実在話者へのなりすまし等、提供者の意図に反する利用を行わないこと
- 学習済みモデルの重みは底モデル（JP-Extra）の AGPL-3.0 に従属（データとは別ライセンス）

## 8. 引用

論文準備中。それまでは Zenodo DOI（10.5281/zenodo.21119791）と本リポジトリを引用のこと。

---
*残作業: census 全数実行の結果反映（§2 帯域・§6 話速）。*
