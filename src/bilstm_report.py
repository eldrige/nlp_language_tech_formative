"""Aggregate BiLSTM/BiGRU runs into tables, figures and error-analysis files.

    python src/bilstm_report.py

Outputs (results/bilstm_gru/):
    summary_runs.csv            one row per run (all design-choice experiments)
    summary_seeds.csv           mean +- std over seeds for the selected configs
    reference_baselines.csv     keyword rule and TF-IDF+LR, standard vs masked (reference only)
    fig_*.png                   figures for the report
    errors_*.csv                misclassified hold-out tweets for error analysis
and results/predictions/<model>_holdout.csv in the shared format for the team's
evaluation harness (Tweet_ID, true, pred, prob_<class> ...).
"""

import json
import shutil

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_curve

from bilstm_gru import RESULTS
from preprocessing import LABELS, ROOT, clean_text, keyword_rule, load_data, tokenize

RUNS = RESULTS / "runs"
SHORT = {"sexual_violence": "Sexual", "Physical_violence": "Physical", "emotional_violence": "Emotional",
         "economic_violence": "Economic", "Harmful_Traditional_practice": "Harmful trad."}
# Reference categorical palette, fixed slot order (dataviz skill, light mode)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.bbox": "tight", "font.size": 9, "axes.edgecolor": INK2,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 10,
    "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
    "lines.linewidth": 2,
})

MAIN = {  # display name -> run-name prefix (seed appended)
    "BiLSTM": "bilstm_glove_w-sqrt_u128",
    "BiGRU": "bigru_glove_w-sqrt_u128",
    "BiLSTM (masked)": "masked_bilstm_glove_w-sqrt_u128",
    "BiGRU (masked)": "masked_bigru_glove_w-sqrt_u128",
}
# Colour follows the model in every figure; masked variants are also dashed / hatched.
COLOR = dict(zip(MAIN, SERIES))


def load_runs() -> pd.DataFrame:
    rows = []
    for d in sorted(RUNS.iterdir()):
        if not (d / "metrics.json").exists():
            continue
        cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
        m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
        rows.append({
            "run": d.name, "rnn": cfg["rnn"], "emb": cfg["emb"] + ("" if cfg["trainable_emb"] else "-frozen"),
            "weights": cfg["weights"], "masked": cfg["masked"], "seed": cfg["seed"],
            "params": cfg["params"], "best_epoch": cfg["best_epoch"], "train_min": cfg["train_seconds"] / 60,
            "val_macro_f1": m["val"]["macro_f1"],
            **{f"test_{k}": m["test"][k] for k in ("accuracy", "macro_f1", "weighted_f1", "macro_roc_auc", "macro_pr_auc")},
            **{f"test_f1_{SHORT[c]}": m["test"]["per_class"][c]["f1-score"] for c in LABELS},
        })
    return pd.DataFrame(rows)


def reference_baselines() -> pd.DataFrame:
    """Keyword rule and TF-IDF+LR in both settings - context for the RNN numbers.

    (The team's full baselines live in the TF-IDF approach; this is only a reference point.)
    """
    rows = []
    for masked in (False, True):
        tr, va, te, _ = load_data(masked=masked)
        if not masked:
            p = te["clean"].map(keyword_rule)
            rows.append({"model": "Keyword rule", "masked": False, "test_accuracy": accuracy_score(te.type, p),
                         "test_macro_f1": f1_score(te.type, p, average="macro")})
        vec = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=5)
        clf.fit(vec.fit_transform(tr["clean"]), tr["type"])
        p = clf.predict(vec.transform(te["clean"]))
        rows.append({"model": "TF-IDF + LR", "masked": masked, "test_accuracy": accuracy_score(te.type, p),
                     "test_macro_f1": f1_score(te.type, p, average="macro")})
    return pd.DataFrame(rows)


def seed_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, prefix in MAIN.items():
        sub = df[df.run.str.startswith(prefix + "_s")]
        if sub.empty:
            continue
        row = {"model": name, "n_seeds": len(sub)}
        for col in ["test_accuracy", "test_macro_f1", "test_weighted_f1", "test_macro_roc_auc",
                    "test_macro_pr_auc", *[f"test_f1_{SHORT[c]}" for c in LABELS]]:
            row[col] = f"{sub[col].mean():.3f} ± {sub[col].std(ddof=0):.3f}"
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_learning_curves():
    names = [n for n in MAIN if (RUNS / f"{MAIN[n]}_s42" / "history.csv").exists()]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.1))
    for i, n in enumerate(names):
        h = pd.read_csv(RUNS / f"{MAIN[n]}_s42" / "history.csv")
        ep = h["epoch"] + 1
        ls = "--" if "masked" in n else "-"
        color = COLOR[n]
        axes[0].plot(ep, h["loss"], ls, color=color, label=n)
        axes[1].plot(ep, h["val_loss"], ls, color=color, label=n)
        axes[2].plot(ep, h["val_macro_f1"], ls, color=color, marker="o", ms=3, label=n)
    for ax, t in zip(axes, ["Training loss (class-weighted)", "Validation loss", "Validation macro-F1"]):
        ax.set_title(t, loc="left")
        ax.set_xlabel("Epoch")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[2].legend(loc="lower right", fontsize=8)
    fig.savefig(RESULTS / "fig_learning_curves.png")
    plt.close(fig)


def fig_confusion():
    pairs = [(n, RUNS / f"{MAIN[n]}_s42" / "metrics.json") for n in ("BiLSTM", "BiLSTM (masked)")]
    pairs = [(n, p) for n, p in pairs if p.exists()]
    fig, axes = plt.subplots(1, len(pairs), figsize=(4.6 * len(pairs), 4), sharey=True,
                             gridspec_kw={"wspace": 0.08})
    axes = np.atleast_1d(axes)
    for k, (ax, (n, p)) in enumerate(zip(axes, pairs)):
        cm = np.array(json.loads(p.read_text(encoding="utf-8"))["test"]["confusion_matrix"])
        norm = cm / cm.sum(1, keepdims=True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.grid(False)
        for i in range(len(LABELS)):
            for j in range(len(LABELS)):
                ax.text(j, i, f"{norm[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.55 else INK)
        ticks = [SHORT[c] for c in LABELS]
        ax.set_xticks(range(5), ticks, rotation=35, ha="right")
        ax.set_yticks(range(5), ticks)
        ax.set_xlabel("Predicted")
        if k == 0:
            ax.set_ylabel("True")
        ax.set_title(f"{n}: row-normalised (count)", loc="left")
    fig.savefig(RESULTS / "fig_confusion.png")
    plt.close(fig)


def fig_per_class_f1(df: pd.DataFrame):
    names = [n for n in MAIN if not df[df.run.str.startswith(MAIN[n] + "_s")].empty]
    x = np.arange(len(LABELS))
    w = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    for i, n in enumerate(names):
        sub = df[df.run.str.startswith(MAIN[n] + "_s")]
        cols = [f"test_f1_{SHORT[c]}" for c in LABELS]
        ax.bar(x + (i - (len(names) - 1) / 2) * w, sub[cols].mean(), w * 0.92, yerr=sub[cols].std(ddof=0),
               color=COLOR[n], hatch="///" if "masked" in n else None, edgecolor="white", linewidth=0,
               label=n, error_kw={"elinewidth": 0.8, "ecolor": INK2, "capsize": 2})
    counts = pd.read_csv(ROOT / "data/splits/splits.csv").merge(
        pd.read_csv(ROOT / "data/raw/Train.csv"), on="Tweet_ID").query("split == 'test'")["type"].value_counts()
    ax.set_xticks(x, [f"{SHORT[c]}\n(n={counts[c]})" for c in LABELS])
    ax.set_ylabel("Hold-out F1")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="x", visible=False)
    ax.set_title("Per-class F1 on the hold-out set (mean ± std over seeds)", loc="left")
    ax.legend(ncol=len(names), loc="lower left", fontsize=8, bbox_to_anchor=(0, -0.38))
    fig.savefig(RESULTS / "fig_per_class_f1.png")
    plt.close(fig)


def fig_roc():
    names = [n for n in ("BiLSTM", "BiLSTM (masked)") if (RUNS / f"{MAIN[n]}_s42" / "holdout_predictions.csv").exists()]
    fig, axes = plt.subplots(1, len(names), figsize=(4.4 * len(names), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, n in zip(axes, names):
        p = pd.read_csv(RUNS / f"{MAIN[n]}_s42" / "holdout_predictions.csv")
        for i, c in enumerate(LABELS):
            fpr, tpr, _ = roc_curve(p["true"] == c, p[f"prob_{c}"])
            auc = np.trapz(tpr, fpr)
            ax.plot(fpr, tpr, color=SERIES[i], lw=1.6, label=f"{SHORT[c]} (AUC {auc:.3f})")
        ax.plot([0, 1], [0, 1], ":", color=INK2, lw=1)
        ax.set_xscale("symlog", linthresh=0.01)
        ax.set_xlim(0, 1)
        ax.set_xlabel("False positive rate (symlog)")
        ax.set_title(f"{n}: one-vs-rest ROC", loc="left")
        ax.legend(loc="lower right", fontsize=7.5)
    axes[0].set_ylabel("True positive rate")
    fig.savefig(RESULTS / "fig_roc.png")
    plt.close(fig)


def fig_length_error():
    """Error rate by tweet length: does performance degrade on longer sequences?"""
    names = [n for n in MAIN if (RUNS / f"{MAIN[n]}_s42" / "holdout_predictions.csv").exists()]
    bins = [0, 20, 35, 50, 65, 200]
    labels = ["≤20", "21–35", "36–50", "51–65", ">65"]
    fig, ax = plt.subplots(figsize=(6, 3))
    for i, n in enumerate(names):
        p = pd.read_csv(RUNS / f"{MAIN[n]}_s42" / "holdout_predictions.csv")
        length = p["tweet"].map(lambda t: len(tokenize(clean_text(t))))
        b = pd.cut(length, bins, labels=labels)
        err = (p["true"] != p["pred"]).groupby(b, observed=False).mean() * 100
        ax.plot(labels, err.values, color=COLOR[n], marker="o", ms=4, label=n,
                ls="--" if "masked" in n else "-")
    ax.set_xlabel("Tweet length (tokens)")
    ax.set_ylabel("Error rate (%)")
    ax.set_title("Hold-out error rate by sequence length", loc="left")
    ax.legend(fontsize=8)
    fig.savefig(RESULTS / "fig_length_error.png")
    plt.close(fig)


def export_predictions_and_errors():
    pred_dir = ROOT / "results" / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for n, prefix in MAIN.items():
        src = RUNS / f"{prefix}_s42" / "holdout_predictions.csv"
        if not src.exists():
            continue
        slug = n.lower().replace(" (masked)", "_masked")
        shutil.copy(src, pred_dir / f"{slug}_holdout.csv")
        p = pd.read_csv(src)
        err = p[p["true"] != p["pred"]].copy()
        err["confidence"] = err[[f"prob_{c}" for c in LABELS]].max(1)
        err["prob_true"] = [r[f"prob_{r['true']}"] for _, r in err.iterrows()]
        err.sort_values("confidence", ascending=False)[["Tweet_ID", "true", "pred", "confidence", "prob_true", "tweet"]] \
            .to_csv(RESULTS / f"errors_{slug}.csv", index=False)
    sub = RUNS / f"{MAIN['BiLSTM']}_s42" / "zindi_submission.csv"
    if sub.exists():
        shutil.copy(sub, pred_dir / "bilstm_zindi_submission.csv")


if __name__ == "__main__":
    runs = load_runs()
    runs.round(4).to_csv(RESULTS / "summary_runs.csv", index=False)
    print(runs[["run", "val_macro_f1", "test_accuracy", "test_macro_f1", "test_macro_pr_auc", "best_epoch",
                "train_min"]].round(4).to_string(index=False))
    seeds = seed_summary(runs)
    seeds.to_csv(RESULTS / "summary_seeds.csv", index=False)
    print("\n", seeds.to_string(index=False))
    ref = reference_baselines()
    ref.round(4).to_csv(RESULTS / "reference_baselines.csv", index=False)
    print("\n", ref.round(4).to_string(index=False))
    fig_learning_curves()
    fig_confusion()
    fig_per_class_f1(runs)
    fig_roc()
    fig_length_error()
    export_predictions_and_errors()
    print("\nWrote figures, summaries and error files to", RESULTS)
