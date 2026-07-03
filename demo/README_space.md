# デモ公開手順（Colab 検証 → HF Spaces 一般公開）

## 0. 前提: ライセンス（確認済み 2026-07-02）
- 底モデル HF `litagin/Style-Bert-VITS2-2.0-base-JP-Extra` = **AGPL-3.0**
  （継承元 `Stardust-minus/Bert-VITS2-Japanese-Extra` も AGPL-3.0）→ **派生の学習済み重みは
  AGPL-3.0 で公開する**（HF repo / Space の license タグを `agpl-3.0` に設定）。
  同底モデルからの公開 finetune・利用 Space（94件）が多数あり、公開は確立した慣行。
- AGPL のネットワーク条項（サービス提供時のソース開示）は fork が公開済みのため充足。
- データ加工物は CC0（Zenodo DOI 10.5281/zenodo.21119791）なので制約なし。
- モデルカードには悪用防止の注意（実在話者のなりすまし・詐欺目的の使用禁止等）を一言入れる。

## 1. Colab でデモを検証（重み非公開のまま）
`colab/cv_r1_demo_colab.ipynb` を GPU ランタイムで実行 → gradio の share リンク（72 時間有効）が
発行される。学会・共同研究者向けの一時デモはこれで足りる。

## 2. モデル重みを HF Hub へ（例: `slp-hu/cadence-cv_r1`）
アップロードするのは model_assets/<model>/ の中身:
```
huggingface-cli login
huggingface-cli repo create slp-hu/cadence-cv_r1 --type model   # 作成後 Web で license: agpl-3.0 を設定
huggingface-cli upload slp-hu/cadence-cv_r1 model_assets/model_name . \
  --include "*.safetensors" "config.json" "style_vectors.npy" "trained_speakers.json" "speaker_map.json"
```
（speaker_map.json は任意。無ければデモは emb_g PCA で座標を作る。
 生成は `pip install umap-learn` → `python demo/make_speaker_map.py`（既定 UMAP・cosine。
 `--method tsne` も可）。eval と同じ環境で 1 回だけ）

## 3. Space を作成
1. https://huggingface.co/new-space → SDK: **Gradio** / Hardware: CPU basic（無料）で作成
2. `demo/app.py`（Space では `app.py` という名前でルートに置く）と `demo/requirements.txt`
   （→ `requirements.txt`）をアップロード
3. Space の **Settings → Variables** に `HF_REPO = slp-hu/cadence-cv_r1` を追加
4. 初回ビルドは fork の pip install と deberta 取得で 10–20 分。以後はキャッシュされる

## 備考
- CPU Space での合成は 1 文あたり数秒程度。速度が要るなら ZeroGPU を申請
- Space を private で作れば URL を知る関係者のみに limited 公開もできる
