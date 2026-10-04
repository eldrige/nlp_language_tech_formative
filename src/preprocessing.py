"""Shared preprocessing and tokenization pipeline for all five models.

Everyone on the team should import from this module so that every model sees
the same cleaned text and is evaluated on the same held-out rows.

    from preprocessing import load_data, clean_text, tokenize, Vocab

Framework-agnostic: returns plain Python lists / NumPy arrays.
- Baselines (TF-IDF):      use `clean_text` (or `clean_text(..., tags=False)`)
- BiLSTM/GRU, CNN:         use `tokenize` + `Vocab` + `Vocab.encode_batch`
- Transformers:            use `clean_text(..., tags=False)` and the model's own
                           subword tokenizer; still use the shared splits.
"""

from __future__ import annotations

import html
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
SPLITS_PATH = ROOT / "data" / "splits" / "splits.csv"

LABELS = [
    "sexual_violence",
    "Physical_violence",
    "emotional_violence",
    "economic_violence",
    "Harmful_Traditional_practice",
]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
SEED = 42

# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
# Tag tokens follow the GloVe-Twitter preprocessing conventions (Pennington et
# al., 2014, Stanford preprocess-twitter.rb), so our vocabulary lines up with
# the pretrained embedding vocabulary.
EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF☀-➿⬀-⯿️‍]", flags=re.UNICODE
)
QUOTES = {"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", "…": "..."}
QUOTES_RE = re.compile("|".join(map(re.escape, QUOTES)))
URL_RE = re.compile(r"https?://\S+|www\.\S+")
USER_RE = re.compile(r"@\w+")
NUMBER_RE = re.compile(r"[-+]?[.\d]*\d+[:,.\d]*")
ALLCAPS_RE = re.compile(r"\b([A-Z]{2,})\b")
ELONG_RE = re.compile(r"\b(\S*?)(\w)\2{2,}\b")          # "sooooo" -> "so <elong>"
REPEAT_PUNCT_RE = re.compile(r"([!?.]){2,}")            # "!!!" -> "! <repeat>"
HASHTAG_RE = re.compile(r"#(\w+)")
# Clitics are split Penn-Treebank style ("don't" -> do n't, "i'm" -> i 'm) because
# that is how GloVe-Twitter tokenised its corpus; keeping "don't" whole left the
# most frequent contractions without a pretrained vector.
TOKEN_RE = re.compile(
    r"<\w+>|[a-z0-9]+?(?=n't\b)|n't\b|'(?:s|m|re|ve|ll|d)\b|[a-z0-9]+|[^\sa-z0-9]", flags=re.UNICODE
)


# Collection-query keywords. EDA showed each class is almost fully determined by
# the term used to scrape it (e.g. "raped" is in 98.6% of sexual_violence tweets
# with 99.9% precision), so a model can score ~99% by keyword spotting alone.
# `mask_keywords=True` replaces all of them with one shared <kw> token so that
# models must rely on the surrounding context. The same token is used for every
# class so the mask itself carries no label information.
QUERY_KEYWORDS = {
    "sexual_violence": r"rap(?:e|ed|es|ing|ist|ists)",
    "Physical_violence": r"beat(?:s|ing|en)?|husband(?:s)?|wife|wives",
    "emotional_violence": r"insult(?:s|ed|ing)?|humiliat(?:e|ed|es|ing|ion)|verbally|public(?:ly)?",
    "economic_violence": r"fir(?:e|ed|es|ing)|job(?:s)?",
    "Harmful_Traditional_practice": r"fgm|genital|mutilat(?:e|ed|ion)|forced|marriage(?:s)?|undergo(?:es|ne)?|circumcis(?:e|ed|ion)",
}
KEYWORD_RE = re.compile(r"\b(?:" + "|".join(QUERY_KEYWORDS.values()) + r")\b")


def mask_keywords(text: str) -> str:
    """Replace collection-query keywords in *cleaned, lowercased* text with <kw>."""
    return KEYWORD_RE.sub("<kw>", text)


def keyword_rule(text: str) -> str:
    """Keyword-lookup classifier used as a reference point for 'how much is just keywords'.

    Checks the rarest classes first so that e.g. 'forced marriage ... rape'
    is not swallowed by the majority class; falls back to the majority class.
    """
    for label in reversed(LABELS):
        if re.search(r"\b(?:" + QUERY_KEYWORDS[label] + r")\b", text):
            return label
    return LABELS[0]


def clean_text(text: str, tags: bool = True) -> str:
    """Normalise a raw tweet.

    tags=True  adds GloVe-Twitter style markers (<allcaps>, <elong>, <number>, ...)
               and lowercases; intended for word-level models.
    tags=False only fixes encoding noise (HTML entities, curly quotes, emoji
               spacing, whitespace) and keeps case; intended for subword
               transformers, whose tokenizers were trained on raw text.
    """
    if not isinstance(text, str):
        return ""
    text = html.unescape(html.unescape(text))          # "&amp;amp;" appears too
    text = unicodedata.normalize("NFKC", text)
    text = QUOTES_RE.sub(lambda m: QUOTES[m.group()], text)
    text = EMOJI_RE.sub(lambda m: f" {m.group()} ", text)  # emojis as own tokens

    if tags:
        text = URL_RE.sub(" <url> ", text)
        text = USER_RE.sub(" <user> ", text)
        text = HASHTAG_RE.sub(r" <hashtag> \1 ", text)
        text = NUMBER_RE.sub(" <number> ", text)
        text = ALLCAPS_RE.sub(lambda m: f" {m.group(1).lower()} <allcaps> ", text)
        text = REPEAT_PUNCT_RE.sub(r" \1 <repeat> ", text)
        text = ELONG_RE.sub(r"\1\2 <elong>", text)
        text = text.lower()

    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    """Word-level tokenizer applied to `clean_text(text, tags=True)` output.

    Keeps tag tokens (<number>), punctuation and emojis as separate tokens and
    splits clitics (do n't, he 's).
    """
    return [t for t in TOKEN_RE.findall(text) if not t.isspace() and t not in {"️", "‍"}]


# ---------------------------------------------------------------------------
# Data loading and shared splits
# ---------------------------------------------------------------------------
def make_splits(train_df: pd.DataFrame, val_size: float = 0.1, test_size: float = 0.1,
                seed: int = SEED) -> pd.DataFrame:
    """Stratified train / val / test split over de-duplicated training data.

    The Zindi test set is unlabelled, so we carve an internal labelled
    hold-out ("test") out of Train.csv. "val" is for early stopping and
    hyperparameter tuning; "test" is touched once, for final reporting.
    """
    df = train_df.copy()
    df["norm"] = df["tweet"].map(lambda t: clean_text(t, tags=True))
    n_before = len(df)
    df = df.drop_duplicates("norm", keep="first")      # avoid train/test leakage
    print(f"Removed {n_before - len(df)} duplicate tweets")

    rest, test = train_test_split(df, test_size=test_size, stratify=df["type"], random_state=seed)
    train, val = train_test_split(rest, test_size=val_size / (1 - test_size),
                                  stratify=rest["type"], random_state=seed)
    out = pd.concat([train.assign(split="train"), val.assign(split="val"), test.assign(split="test")])
    return out[["Tweet_ID", "split"]].reset_index(drop=True)


def load_data(tags: bool = True, masked: bool = False
              ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, val, test, zindi_test) DataFrames with a `clean` column.

    Labelled frames also carry an integer `label` column (see LABELS order).
    masked=True applies `mask_keywords` (requires tags=True, i.e. lowercased text).
    """
    raw = pd.read_csv(RAW_DIR / "Train.csv")
    zindi = pd.read_csv(RAW_DIR / "Test.csv")
    if not SPLITS_PATH.exists():
        SPLITS_PATH.parent.mkdir(parents=True, exist_ok=True)
        make_splits(raw).to_csv(SPLITS_PATH, index=False)
    splits = pd.read_csv(SPLITS_PATH)

    df = raw.merge(splits, on="Tweet_ID", how="inner")
    df["clean"] = df["tweet"].map(lambda t: clean_text(t, tags=tags))
    df["label"] = df["type"].map(LABEL2ID)
    zindi["clean"] = zindi["tweet"].map(lambda t: clean_text(t, tags=tags))
    if masked:
        if not tags:
            raise ValueError("masked=True needs tags=True (keywords are matched on lowercased text)")
        df["clean"] = df["clean"].map(mask_keywords)
        zindi["clean"] = zindi["clean"].map(mask_keywords)

    parts = [df[df.split == s].reset_index(drop=True) for s in ("train", "val", "test")]
    return (*parts, zindi)


# ---------------------------------------------------------------------------
# Vocabulary and integer sequences
# ---------------------------------------------------------------------------
class Vocab:
    """Word -> index mapping built on the training split only.

    Index 0 is padding, 1 is out-of-vocabulary.
    """

    PAD, UNK = "<pad>", "<unk>"

    def __init__(self, min_freq: int = 2, max_size: int | None = None):
        self.min_freq = min_freq
        self.max_size = max_size
        self.itos: list[str] = [self.PAD, self.UNK]
        self.stoi: dict[str, int] = {}
        self.freqs: Counter = Counter()

    def fit(self, token_lists) -> "Vocab":
        for toks in token_lists:
            self.freqs.update(toks)
        words = [w for w, c in self.freqs.most_common(self.max_size) if c >= self.min_freq]
        self.itos = [self.PAD, self.UNK] + words
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        return self

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.stoi.get(t, 1) for t in tokens]

    def encode_batch(self, token_lists, max_len: int) -> np.ndarray:
        """Integer-encode and post-pad / post-truncate to `max_len`."""
        out = np.zeros((len(token_lists), max_len), dtype=np.int32)
        for i, toks in enumerate(token_lists):
            ids = self.encode(toks)[:max_len]
            out[i, : len(ids)] = ids
        return out

    def oov_rate(self, token_lists) -> float:
        total = sum(len(t) for t in token_lists)
        unk = sum(1 for toks in token_lists for t in toks if t not in self.stoi)
        return unk / max(total, 1)

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps({"min_freq": self.min_freq, "itos": self.itos}), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        v = cls(min_freq=data["min_freq"])
        v.itos = data["itos"]
        v.stoi = {w: i for i, w in enumerate(v.itos)}
        return v


def prepare_sequences(max_len: int = 80, min_freq: int = 2, masked: bool = False):
    """One-call helper for word-level neural models (BiLSTM/GRU, CNN).

    Returns a dict with X_train/X_val/X_test/X_zindi (int32 [n, max_len]),
    y_train/y_val/y_test (int), the fitted Vocab, and the DataFrames.
    """
    train, val, test, zindi = load_data(tags=True, masked=masked)
    toks = {name: df["clean"].map(tokenize).tolist()
            for name, df in [("train", train), ("val", val), ("test", test), ("zindi", zindi)]}
    vocab = Vocab(min_freq=min_freq).fit(toks["train"])
    return {
        "vocab": vocab,
        "tokens": toks,
        "frames": {"train": train, "val": val, "test": test, "zindi": zindi},
        **{f"X_{k}": vocab.encode_batch(v, max_len) for k, v in toks.items()},
        **{f"y_{k}": df["label"].to_numpy() for k, df in [("train", train), ("val", val), ("test", test)]},
    }


if __name__ == "__main__":
    # Sanity check + print the statistics used in the Methodology section.
    for s in [
        "I was RAPED by 2 men &amp; nobody believed me!!! 😭😭",
        "He beat me sooooo badly… my ex-husband",
    ]:
        print(s, "->", tokenize(clean_text(s)), "|", clean_text(s, tags=False))

    train, val, test, zindi = load_data()
    print("\nSplit sizes:", len(train), len(val), len(test), "| Zindi test:", len(zindi))
    print(pd.concat([d["type"].value_counts(normalize=True).rename(n)
                     for n, d in [("train", train), ("val", val), ("test", test)]], axis=1).round(4))
    lens = train["clean"].map(lambda t: len(tokenize(t)))
    print("\nToken length percentiles:", {p: int(np.percentile(lens, p)) for p in (50, 90, 95, 99, 100)})
