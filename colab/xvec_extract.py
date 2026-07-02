#!/usr/bin/env python3
# xvec_extract.py — CV_SPKREC.keras による x-vector 抽出（別プロセス隔離実行）
#
# 目的:
#   torch 2.11+cu128 の SBV2 セッションを汚さないよう、TF/keras(=torch backend) を
#   この独立プロセスの中だけで import する。合成側(torch)とは同居させない。
#
# 使い方:
#   python xvec_extract.py --items items.json --out xvec.npz [--drive /content/drive/MyDrive]
#
#   items.json 形式: {"key1": "/abs/path/a.wav", "key2": "/abs/path/b.wav", ...}
#     key は自由（例 "real__cv_0001__u0" / "synth__cv_0001__u0"）。
#   出力 npz: 各 key に LDA150 ベクトル(float32, shape(150,)) を格納。
#
# 07d 準拠（chap10 と同一パラメータ）。CV_SPKREC.keras / CVLDA.npy は公開 GitHub から自動取得。

import os, sys, json, argparse, zipfile, shutil, urllib.request, pathlib

# ---- keras backend は torch（chap10 と同一。StatPool1D は keras.ops）----
os.environ.setdefault("KERAS_BACKEND", "torch")

# ---- TF を CPU 固定（重要）----
# Colab の TensorFlow ビルドは Blackwell(sm_120) の CUDA カーネルに非対応で、
# GPU を掴むと linear_to_mel_weight_matrix 等で 'CUDA_ERROR_INVALID_HANDLE' で落ちる。
# x-vector 抽出は小さな CNN で CPU で十分。TF import 前に GPU を隠す。
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"           # TF から GPU を見せない
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # 情報ログ抑制

def log(*a): print("[xvec]", *a, flush=True)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True, help="wav リスト JSON (key->wav path)")
    ap.add_argument("--out",   required=True, help="出力 npz")
    ap.add_argument("--drive", default="/content/drive/MyDrive", help="モデル/LDA 保存先の親")
    ap.add_argument("--batch", type=int, default=64)
    return ap.parse_args()

# ---- 信号処理パラメータ（07d/chap10 と一致。変更しない）----
SR = 16000; UTT_LEN_S = 2; MAXLEN = UTT_LEN_S * SR
FRAME_LEN = 512; FRAME_STEP = 256; N_MEL = 50; MEL_LO, MEL_HI = 100, 8000; N_MFCC = 30

CHAP10_ZIP_URL = "https://github.com/sp-au-mu-nl/SpeechComm/raw/refs/heads/main/data/chap10.zip"

def ensure_model_lda(drive):
    keras_model = f"{drive}/SBV2/CV_SPKREC.keras"
    cvlda_path  = f"{drive}/SBV2/CVLDA.npy"
    need = [p for p in (keras_model, cvlda_path) if not os.path.exists(p)]
    if need:
        os.makedirs(os.path.dirname(keras_model), exist_ok=True)
        zp, ex = "/content/chap10.zip", "/content/_chap10"
        log("downloading", CHAP10_ZIP_URL)
        urllib.request.urlretrieve(CHAP10_ZIP_URL, zp)
        with zipfile.ZipFile(zp) as z:
            for name in ("CV_SPKREC.keras", "CVLDA.npy"):
                z.extract(name, ex)
        if not os.path.exists(keras_model): shutil.copy(f"{ex}/CV_SPKREC.keras", keras_model)
        if not os.path.exists(cvlda_path):  shutil.copy(f"{ex}/CVLDA.npy",  cvlda_path)
    assert os.path.exists(keras_model) and os.path.exists(cvlda_path), "model/LDA 取得失敗"
    log("model/LDA ready:", os.path.getsize(keras_model)//1024, "KB /",
        os.path.getsize(cvlda_path)//1024, "KB")
    return keras_model, cvlda_path

def main():
    args = parse_args()
    items = json.load(open(args.items, encoding="utf-8"))
    keys = list(items.keys()); paths = [items[k] for k in keys]
    log("items:", len(keys))

    keras_model, cvlda_path = ensure_model_lda(args.drive)

    import numpy as np
    import tensorflow as tf
    import keras
    from keras import layers, models
    import librosa
    try:
        import vad; _vad = vad.EnergyVAD()
    except Exception as e:
        log("vad 不可（原波形フォールバックのみ）:", repr(e)[:80]); _vad = None
    log("tf", tf.__version__, "/ keras", keras.__version__, "/ backend", keras.backend.backend())

    @keras.saving.register_keras_serializable(package="MyLayers")
    class StatPool1D(layers.Layer):
        def __init__(self, **kw): super().__init__(**kw)
        def get_config(self): return super().get_config().copy()
        def call(self, X):
            mean = keras.ops.mean(X, axis=1); std = keras.ops.std(X, axis=1)
            return keras.ops.concatenate([mean, std], axis=1)
        def compute_output_shape(self, s): return (s[0], 2*s[2])

    _mel_fb = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=N_MEL, num_spectrogram_bins=FRAME_LEN//2 + 1,
        sample_rate=SR, lower_edge_hertz=MEL_LO, upper_edge_hertz=MEL_HI)

    def load_wav_16k(path):
        y, _ = librosa.load(path, sr=SR, mono=True); return y.astype(np.float32)

    def apply_vad(wf):
        m = np.max(np.abs(wf))
        if m > 0: wf = wf / m
        if _vad is not None:
            try:
                w = np.squeeze(_vad.apply_vad(np.reshape(wf, (1, -1))))
                if w.size >= FRAME_LEN: return w.astype(np.float32)
            except Exception: pass
        return wf.astype(np.float32)

    def get_mfcc(wf):
        w = tf.cast(wf, tf.float32); n = tf.shape(w)[0]
        w = tf.cond(n < MAXLEN,
                    lambda: tf.concat([w, tf.zeros([MAXLEN-n], tf.float32)], 0),
                    lambda: w[:MAXLEN])
        spec = tf.signal.stft(w, frame_length=FRAME_LEN, frame_step=FRAME_STEP)
        mel  = tf.matmul(tf.square(tf.abs(spec)), _mel_fb)
        logmel = tf.math.log(mel + 1e-6)
        return tf.signal.mfccs_from_log_mel_spectrograms(logmel)[..., :N_MFCC].numpy()

    def wav_to_mfcc(p): return get_mfcc(apply_vad(load_wav_16k(p)))

    new_model = models.load_model(keras_model, custom_objects={"StatPool1D": StatPool1D})
    xvecsModel = models.Model(inputs=new_model.inputs,
                              outputs=new_model.get_layer("dense1").output)
    xvecsModel.trainable = False
    log("x-vector dim =", xvecsModel.output_shape[-1])

    def extract(paths, batch):
        out = []
        for i in range(0, len(paths), batch):
            mf = np.stack([wav_to_mfcc(p) for p in paths[i:i+batch]]).astype(np.float32)
            out.append(xvecsModel.predict(mf, verbose=0))
            log(f"  {min(i+batch,len(paths))}/{len(paths)}")
        return np.concatenate(out, 0)

    def load_lda(path):
        obj = np.load(path, allow_pickle=True)
        return obj.item() if obj.shape == () else obj
    CVLDA = load_lda(cvlda_path)
    d = CVLDA.scalings_.shape[0]
    _ = CVLDA.transform(np.zeros((1, d), np.float32))   # unpickle 健全性
    log("LDA in_dim", d, "n_components", CVLDA.n_components)

    X = extract(paths, args.batch)          # (N,256)
    Xl = CVLDA.transform(X).astype(np.float32)  # (N,150)
    vec = {k: Xl[i] for i, k in enumerate(keys)}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, **vec)
    log("saved:", args.out, "/ keys =", len(vec), "/ dim =", Xl.shape[1])

if __name__ == "__main__":
    main()
