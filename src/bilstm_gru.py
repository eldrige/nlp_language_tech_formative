"""Approach 3: bidirectional recurrent classifier over word embeddings.

Embedding -> SpatialDropout -> BiLSTM|BiGRU (all hidden states)
          -> [masked mean-pool ; masked max-pool] -> Dense -> softmax(5)

Single run:
    python src/bilstm_gru.py --rnn lstm --emb glove --weights sqrt --seed 42
    python src/bilstm_gru.py --rnn gru --masked          # keyword-masked setting

Each run writes to results/bilstm_gru/runs/<name>/:
    config.json, history.csv, metrics.json, holdout_predictions.csv,
    zindi_submission.csv, model.keras
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras
import numpy as np
import pandas as pd
from keras import layers, ops
from sklearn.metrics import (accuracy_score, average_precision_score, classification_report,
                             confusion_matrix, f1_score, roc_auc_score)

from preprocessing import LABELS, ROOT, prepare_sequences

RESULTS = ROOT / "results" / "bilstm_gru"
GLOVE_PATH = ROOT / "embeddings" / "glove-twitter-100.gz"
MAX_LEN = 80   # covers >99% of tweets (99th percentile = 77 tokens)
EMB_DIM = 100


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
def glove_matrix(vocab, path: Path = GLOVE_PATH, dim: int = EMB_DIM) -> tuple[np.ndarray, dict]:
    """Build an embedding matrix for `vocab` from GloVe-Twitter (word2vec text format, gzipped).

    Words not in GloVe get N(0, sigma) vectors matching GloVe's scale; <pad> stays zero.
    Cached to embeddings/ keyed on a hash of the vocabulary.
    """
    key = hashlib.md5("\n".join(vocab.itos).encode("utf-8")).hexdigest()[:10]
    cache = path.parent / f"glove_matrix_{key}.npz"
    if cache.exists():
        data = np.load(cache, allow_pickle=True)
        return data["matrix"], data["coverage"].item()

    found = {}
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        next(f)  # header: "<n_words> <dim>"
        for line in f:
            word, rest = line.split(" ", 1)
            if word in vocab.stoi:
                found[word] = np.asarray(rest.split(), dtype=np.float32)

    vecs = np.stack(list(found.values()))
    rng = np.random.default_rng(0)
    matrix = rng.normal(vecs.mean(), vecs.std(), size=(len(vocab), dim)).astype(np.float32)
    matrix[0] = 0.0
    for word, vec in found.items():
        matrix[vocab.stoi[word]] = vec

    words = vocab.itos[2:]
    token_total = sum(vocab.freqs[w] for w in words)
    coverage = {
        "vocab_size": len(vocab),
        "types_in_glove": len(found),
        "type_coverage": len(found) / len(words),
        "token_coverage": sum(vocab.freqs[w] for w in found) / token_total,
        "top_missing": [w for w in words if w not in found][:30],
    }
    np.savez(cache, matrix=matrix, coverage=np.array(coverage, dtype=object))
    return matrix, coverage


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
@keras.saving.register_keras_serializable()
class MaskedGlobalMaxPool1D(layers.Layer):
    """Max over time that ignores padded positions (Keras' built-in one does not)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

    def call(self, x, mask=None):
        if mask is not None:
            x = ops.where(ops.expand_dims(mask, -1), x, ops.full_like(x, -1e9))
        return ops.max(x, axis=1)

    def compute_mask(self, inputs, mask=None):
        return None


def build_model(vocab_size: int, rnn: str = "lstm", units: int = 128,
                emb_matrix: np.ndarray | None = None, trainable_emb: bool = True,
                emb_dropout: float = 0.2, dropout: float = 0.3, lr: float = 1e-3) -> keras.Model:
    inp = layers.Input(shape=(MAX_LEN,), dtype="int32", name="token_ids")
    emb = layers.Embedding(
        vocab_size, EMB_DIM, mask_zero=True, trainable=trainable_emb, name="embedding",
        embeddings_initializer=(keras.initializers.Constant(emb_matrix) if emb_matrix is not None
                                else "uniform"),
    )(inp)
    x = layers.SpatialDropout1D(emb_dropout)(emb)
    cell = layers.LSTM if rnn == "lstm" else layers.GRU
    x = layers.Bidirectional(cell(units, return_sequences=True), name=f"bi{rnn}")(x)
    pooled = layers.Concatenate()([layers.GlobalAveragePooling1D()(x), MaskedGlobalMaxPool1D()(x)])
    h = layers.Dropout(dropout)(pooled)
    h = layers.Dense(64, activation="relu")(h)
    h = layers.Dropout(dropout)(h)
    out = layers.Dense(len(LABELS), activation="softmax", name="probs")(h)

    model = keras.Model(inp, out, name=f"bi{rnn}_classifier")
    model.compile(optimizer=keras.optimizers.Adam(lr, clipnorm=1.0),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def class_weights(y: np.ndarray, scheme: str) -> dict | None:
    """none | balanced (n / (k * n_c)) | sqrt (square root of balanced, a softer re-weighting)."""
    if scheme == "none":
        return None
    counts = np.bincount(y, minlength=len(LABELS))
    w = len(y) / (len(LABELS) * counts)
    if scheme == "sqrt":
        w = np.sqrt(w)
    return {i: float(v) for i, v in enumerate(w)}


class MacroF1EarlyStopping(keras.callbacks.Callback):
    """Tracks validation macro-F1 each epoch; stops and restores the best weights.

    Plain val_loss is dominated by the majority class, so it is a poor stopping
    signal when the minority classes are what we care about.
    """

    def __init__(self, X_val, y_val, patience: int = 3):
        super().__init__()
        self.X_val, self.y_val, self.patience = X_val, y_val, patience
        self.best, self.best_epoch, self.wait, self.best_weights = -1.0, 0, 0, None

    def on_epoch_end(self, epoch, logs=None):
        pred = self.model.predict(self.X_val, batch_size=512, verbose=0).argmax(1)
        f1 = f1_score(self.y_val, pred, average="macro")
        logs["val_macro_f1"] = f1
        print(f" - val_macro_f1: {f1:.4f}")
        if f1 > self.best:
            self.best, self.best_epoch, self.wait = f1, epoch + 1, 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(y_true: np.ndarray, probs: np.ndarray) -> dict:
    pred = probs.argmax(1)
    report = classification_report(y_true, pred, labels=range(len(LABELS)), target_names=LABELS,
                                   output_dict=True, zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "macro_f1": f1_score(y_true, pred, average="macro"),
        "weighted_f1": f1_score(y_true, pred, average="weighted"),
        "macro_roc_auc": roc_auc_score(y_true, probs, multi_class="ovr", average="macro"),
        "macro_pr_auc": average_precision_score(np.eye(len(LABELS))[y_true], probs, average="macro"),
        "per_class": {c: report[c] for c in LABELS},
        "confusion_matrix": confusion_matrix(y_true, pred, labels=range(len(LABELS))).tolist(),
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)


def run(rnn="lstm", emb="glove", trainable_emb=True, weights="sqrt", units=128, lr=1e-3,
        batch_size=128, epochs=12, patience=3, seed=42, masked=False, name=None, data=None) -> dict:
    set_seed(seed)
    data = data or prepare_sequences(max_len=MAX_LEN, masked=masked)
    vocab = data["vocab"]
    name = name or f"{'masked_' if masked else ''}bi{rnn}_{emb}{'' if trainable_emb else '-frozen'}_w-{weights}_u{units}_s{seed}"
    out_dir = RESULTS / "runs" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix, coverage = (glove_matrix(vocab) if emb == "glove" else (None, None))
    model = build_model(len(vocab), rnn=rnn, units=units, emb_matrix=matrix,
                        trainable_emb=trainable_emb, lr=lr)
    cw = class_weights(data["y_train"], weights)
    stopper = MacroF1EarlyStopping(data["X_val"], data["y_val"], patience=patience)

    t0 = time.time()
    hist = model.fit(data["X_train"], data["y_train"], validation_data=(data["X_val"], data["y_val"]),
                     batch_size=batch_size, epochs=epochs, class_weight=cw, callbacks=[stopper], verbose=2)
    train_time = time.time() - t0

    probs_val = model.predict(data["X_val"], batch_size=512, verbose=0)
    probs_test = model.predict(data["X_test"], batch_size=512, verbose=0)
    probs_zindi = model.predict(data["X_zindi"], batch_size=512, verbose=0)

    config = dict(rnn=rnn, emb=emb, trainable_emb=trainable_emb, weights=weights, units=units, lr=lr,
                  batch_size=batch_size, max_len=MAX_LEN, emb_dim=EMB_DIM, seed=seed, masked=masked,
                  class_weights=cw, vocab_size=len(vocab), params=model.count_params(),
                  best_epoch=stopper.best_epoch, epochs_run=len(hist.history["loss"]),
                  train_seconds=round(train_time, 1), glove_coverage=coverage)
    metrics = {"val": evaluate(data["y_val"], probs_val), "test": evaluate(data["y_test"], probs_test)}

    (out_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(hist.history).rename_axis("epoch").to_csv(out_dir / "history.csv")

    test_df = data["frames"]["test"]
    preds = pd.DataFrame({"Tweet_ID": test_df["Tweet_ID"], "tweet": test_df["tweet"],
                          "true": test_df["type"], "pred": [LABELS[i] for i in probs_test.argmax(1)]})
    preds[[f"prob_{c}" for c in LABELS]] = probs_test
    preds.to_csv(out_dir / "holdout_predictions.csv", index=False)

    zindi = data["frames"]["zindi"]
    pd.DataFrame({"Tweet_ID": zindi["Tweet_ID"], "type": [LABELS[i] for i in probs_zindi.argmax(1)]}) \
        .to_csv(out_dir / "zindi_submission.csv", index=False)
    model.save(out_dir / "model.keras")

    print(f"[{name}] best epoch {stopper.best_epoch} | val macro-F1 {metrics['val']['macro_f1']:.4f} "
          f"| test macro-F1 {metrics['test']['macro_f1']:.4f} acc {metrics['test']['accuracy']:.4f} "
          f"| {train_time:.0f}s")
    return {"name": name, "config": config, "metrics": metrics}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rnn", choices=["lstm", "gru"], default="lstm")
    p.add_argument("--emb", choices=["random", "glove"], default="glove")
    p.add_argument("--frozen", action="store_true", help="freeze the embedding layer")
    p.add_argument("--weights", choices=["none", "sqrt", "balanced"], default="sqrt")
    p.add_argument("--units", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--masked", action="store_true", help="mask collection-query keywords")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    run(rnn=a.rnn, emb=a.emb, trainable_emb=not a.frozen, weights=a.weights, units=a.units,
        lr=a.lr, epochs=a.epochs, seed=a.seed, masked=a.masked)
