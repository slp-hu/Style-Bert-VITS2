# CV 調音速度 census と Tier0 閾値（2026-07-09）

生成: `cadence_cv_articulation_rate_v4_jsonl.ipynb`
対象: cv_r1 の `cadence_mora.jsonl`（11,011 uid / 298 話者）
対になる文書: `CSJ_SPEECH_RATE_CENSUS_20260709.md`（CSJ 側・Phone ベース）

---

## 0. 一行で

CSJ 話速 census で確立した **`articulation_rate`（実発話長分母）** を CV に適用した。
朗読と自発音声で調音速度の**中心は一致**（p50: CV 8.15 / CSJ 8.40）するが、**裾だけが乖離**する。
その裾に、6 月から保留していた測定異常話者（`cv_0155` 等）が埋もれていた。

---

## 1. 指標の定義と TextGrid 廃止

`cadence_mora.jsonl` の各モーラは `d`（秒）と `pause` フラグ（`mtype='pause'`）を持つ。

| 指標 | 定義 | 対応 CSJ | 用途 |
|---|---|---|---|
| `articulation_rate` | `count(pause==0) / Σ(d for pause==0)` | `rate_speech`（Phone） | **Tier0 品質ゲート** |
| `speaking_rate` | `count(pause==0) / trim_s` | `rate_span` | 記述 |
| `pause_ratio` | `Σ(d for pause!=0) / Σ(d)` | `pause_ratio` | cadence が保存 |

**TextGrid は不要。** `Σ(d for pause==0)` は TextGrid の発話区間（`sil`/`sp`/`''` 除外）と
**完全一致**（中央差 0.0000 s、11,011 本）。

**`clip_dur`（TextGrid の xmax）は使ってはいけない。** TextGrid はトリム**前**の音声に対して
作られており、cv_r1 の wav（`librosa.effects.trim(top_db=30)` 済み、`census.tsv` の `trim_s`）と
長さが違う（中央差 1.236 s、全 11,011 本）。TextGrid ベースの `speaking_rate` / `padding` は無効だった。

ポーズモーラは 2.7%（`mtype`: normal 205k / long 14k / devoiced 10k / nasal 9k / pause 6.9k / geminate 5k）。

---

## 2. CV と CSJ を同じ土俵で

| 分位点 | CV 調音 | CSJ 調音 | 差 |
|---|---:|---:|---:|
| p50 | 8.15 | 8.40 | −0.25 |
| p90 | 9.88 | 9.63 | +0.25 |
| p95 | 10.55 | 10.14 | +0.41 |
| p99 | 12.50 | 10.75 | **+1.75** |
| p100 | 19.13 | 11.18 | **+7.95** |

**p50 は一致。p99 以上で乖離。** 中心は同じ（朗読も自発も、日本語の調音速度の中央は 8 mora/s 台）。
裾だけが CV で伸びる。これは分散拡大ではなく、**特定話者の測定異常**（§4）。

---

## 3. 従来 census（`mora_per_s` = 7.71）の正体

`census.tsv`（18,015 行）の `mora_per_s = mora / trim_s`。

- 分母 `trim_s` = トリム後クリップ長（`manifest.dur` と中央差 0.0）→ **span 系**
- 分子 `mora` = **ポーズ込み**（`census.mora == n_all` を全 11,011 行で確認）

**data sheet の「モーラ数はポーズ除外」は記述ミス。** 実データはポーズ込み（CV 側の文書誤り 1 件）。

従来 census には `articulation_rate`（実発話長分母）が無い。だから `cv_0155` 等が
span 分母では 13.1 程度に見え、**真の異常が p99 の裾に埋もれていた**。
`articulation_rate` に切り替えて初めて 12.4 として浮上する。これが本 census の核心。

---

## 4. 除外候補は 2 軸

### 4.1 話速異常（話者中央値 > CSJ p100 = 11.18）

| 話者 | art_med | art_p95 | n | 判定 |
|---|---:|---:|---:|---|
| cv_0239 | 13.95 | 18.08 | 13 | **二峰**（§4.3） |
| cv_0191 | 12.58 | 14.53 | 18 | 異常 |
| cv_0155 | 12.39 | 14.72 | 42 | 異常（42 本中 28 本が p100 超え） |
| cv_0205 | 11.83 | 15.00 | 28 | 境界〜異常 |
| cv_0157 | 11.61 | 14.70 | 17 | 境界 |
| cv_0267 | 11.41 | 13.18 | 38 | 境界（p95 が高い → per-clip） |

上位 3 は明白。下位 3 は p100 をわずかに超える程度で、**per-clip 判定**が要る
（話者ぐるみでなく一部クリップの異常なら、話者除外でなくクリップ除外）。

`cv_0155` の判定は 3 経路で一致: TextGrid 版 12.50 / ポーズ抜き jsonl 版 12.39 / 従来 census 換算 ≈16。
**測定異常で確定**（モーラ数か区間長の不整合）。6 月からの保留に決着。

### 4.2 話者混入（純度監査）

SHARED 8 名（`cv_0019`/`0025`/`0075`/`0078`/`0095`/`0127`/`0258`/`0279`）+ INTRUDER `cv_0028`。

**`cv_0028` の調音速度は正常（art_med 9.22）。** → **話速異常と話者混入は独立の軸**。
両者を統合した除外リストを r2 の select 前に適用する。

### 4.3 `cv_0239` の二峰

```
articulation_rate: 7.2 7.3 7.7 8.3 8.6 | 12.1 14.4 14.6 15.1 16.0 16.3 17.8 19.1
```

13 本中 8 本が CSJ p100 超え。5 本と 8 本で分離。純度監査の unsure と符合。
**アカウントに 2 人いるか、8 本のメタデータが破損**。話者除外でなく per-clip 分離が正しい可能性。

---

## 5. Tier0 閾値（実測で確定）

```
articulation_rate = count(pause==0) / Σ(d for pause==0)     # CSJ の Phone ベースと同型
```

| 旗 | 閾値 | 根拠 |
|---|---|---|
| 早口（clip） | `articulation_rate > 11.7` | CSJ 自発 p100 = 11.18 + 5% |
| 話者ぐるみ | 話者中央値 > 11.18 | CSJ p100。6 名該当 |
| 遅すぎ | **廃止** | 従来 `<3 mora/s` は span 分母のアーティファクト。実発話長では正常域 |

従来 `speaking_rate` の `<3` 旗は、`trim(top_db=30)` が端点しか削らず内部ポーズが分母に残るため。
`articulation_rate` では消える。遅い側は `n_mora` の下限で捕まえる。

---

## 6. r2 への帰結

- **Tier0 は `articulation_rate`、cadence は `pause_ratio`。** 品質ゲートで `pause_ratio` を落とさない。
  人手評定は pause を含む発話速度で「速い」を判断していた（CSJ 側 ρ 0.828 vs 0.653）= cadence が学習すべき情報
- **除外リストは話速 6 名 + 混入 9 名の和集合**。cv_0028 で交わらないことを確認済み
- **`SpeechIntervalProvider`**: CSJ = `Phone`（`SpzS`/`SbS` と Phone 間ギャップを無音）、
  CV = `cadence_mora.jsonl`（`pause!=0` を無音）。両者 `Σ(d for 非無音)` が実発話長。
  分子（ポーズ除外モーラ）は両者で対称（`MoraProvider` の非対称とは別）
- **`Tier0 → select` の順序**: 話速旗と混入を落としてから `MIN_UTTS` を判定する（現状は逆）

---

## 7. 文書・データの不具合（本 census で新規に判明した分）

- **CV**: data sheet「モーラ数はポーズ除外」は誤り。`census.tsv` の `mora` はポーズ込み（`== n_all`）
- **CV**: `textgrid/` の TextGrid はトリム前音声に対応。cv_r1 wav（トリム後）と長さが違う。
  `xmax` を wav 長として使うと全 11,011 本でずれる
- **fork の clone が 2 つ**: `~/tmp/synth/Style-Bert-VITS2/`（正本）と
  `~/tmp/synth/sbv2finetune/Style-Bert-VITS2/`。`census.tsv` は正本の `docs/audit/` を使う

---

## 8. 出力

```
cv_r1/census/cv_articulation_v2.csv              11,011 本 × {articulation_rate, speaking_rate, pause_ratio}
cv_r1/census/cv_articulation_by_speaker_v2.csv   298 話者
cv_r1/census/cv_articulation_meta_v2.json        閾値・除外候補・CSJ 参照値
```

## 9. 残っている宿題

1. per-clip 判定で `cv_0205`/`cv_0157`/`cv_0267`/`cv_0239` を話者除外かクリップ除外か切り分け
2. 選定監査（`cadence_csj_selection_audit.ipynb`）の実行 — `hash()` 非決定性ほか
3. `dataset_tools` 統合 + `Tier0 → select` 順序変更
4. cv_r2 の select（cv_r1 は DOI ごと残す。話速 6 名 + 混入 9 名を除外）
