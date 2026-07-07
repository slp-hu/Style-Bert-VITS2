# docs/audit — 話者純度監査の記録（cv_r1, 2026-07）

- `purity_v2.tsv` — 純度スクリーニング結果（`dataset_tools/speaker_purity_report.py`）。
  **ファイル名は v2 だが中身は v3 形式**（ヘッダ行に `th_f0_st` あり。5 型分類 +
  F0 直交軸 + インポスタ統計）。Drive 上の同名ファイルと同一。
- `purity_audit_verdicts.json` — 聴取監査の最終判定（`colab/cv_r1_speaker_audit.ipynb` §5）。
  27 名 = shared 8 / shared(intruder) 1（`cv_0028`）/ single 7 / unsure 11。
  正本は Drive（`MyDrive/Style-Bert-VITS2/purity_audit_verdicts.json`）。更新時は両方に反映すること。

解釈・制約はデータシート `docs/DATASET_cv_r1.md` §6 を参照。
