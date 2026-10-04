# Approach 3 (BiLSTM / BiGRU): results and error analysis

*Notes for Shem (Results & Discussion) and the error analysis section (Error Analysis & Limitations).
Use, cut or reword freely. All numbers are on the shared test split (n = 3,903). Raw numbers are in
`results/bilstm_gru/summary_*.csv`, figures in `results/bilstm_gru/fig_*.png`, and hold-out
predictions in the shared format in `results/predictions/`.*

## 1. Main results (mean ± std over seeds 42, 1, 2)

Configuration: GloVe-Twitter-100 embeddings (fine-tuned), square-root class weights, 128 units per
direction, mean+max pooling.

| Model | Setting | Accuracy | Macro-F1 | Macro PR-AUC | Emotional F1 | Economic F1 | HTP F1 |
|---|---|---|---|---|---|---|---|
| BiLSTM | standard | 1.000 ± 0.000 | 0.997 ± 0.003 | 1.000 | 0.995 | 0.992 | 1.000 |
| BiGRU | standard | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| BiLSTM | keyword-masked | 0.984 ± 0.001 | 0.939 ± 0.004 | 0.966 | 0.829 | 0.913 | 1.000 |
| BiGRU | keyword-masked | 0.983 ± 0.001 | 0.940 ± 0.005 | 0.966 | 0.828 | 0.943 | 0.982 |

Reference points on the same split (single run):

| Model | Setting | Accuracy | Macro-F1 |
|---|---|---|---|
| Majority class ("sexual violence") | standard | 0.834 | 0.182 |
| Keyword lookup rule | standard | 0.896 | 0.631 |
| TF-IDF (1–2-grams) + logistic regression | standard | 0.997 | 0.979 |
| TF-IDF (1–2-grams) + logistic regression | masked | 0.977 | 0.911 |
| BiLSTM, random-init embeddings | masked | 0.974 | 0.893 |

Figures:
- `fig_learning_curves.png`: training loss, validation loss and validation macro-F1 per epoch.
- `fig_confusion.png`: confusion matrices, standard vs masked.
- `fig_per_class_f1.png`: per-class F1, all four models.
- `fig_roc.png`: one-vs-rest ROC curves.
- `fig_length_error.png`: error rate by tweet length.

## 2. Design-choice experiments (seed 42, validation macro-F1)

| Variant | Val macro-F1 | Test macro-F1 |
|---|---|---|
| BiLSTM, random embeddings | 0.995 | 0.990 |
| BiLSTM, GloVe fine-tuned | 0.993 | 0.998 |
| BiLSTM, GloVe frozen | 0.992 | 1.000 |
| BiLSTM, GloVe, no class weights | 0.993 | 0.998 |
| BiLSTM, GloVe, balanced weights | 0.993 | 1.000 |
| BiGRU, GloVe | 0.995 | 1.000 |
| **masked** BiLSTM, random embeddings | 0.899 | 0.893 |
| **masked** BiLSTM, GloVe | 0.939 | 0.943 |
| **masked** BiGRU, GloVe | 0.936 | 0.939 |

In the standard setting all six variants are within 0.003 of each other on validation. That is
about one or two tweets in the minority classes, so **the standard setting cannot distinguish the
design choices**: every variant reaches the ceiling. Class weights changed the training trajectory
(different losses at every epoch) but not the end result. Only the masked setting separates the
variants.

## 3. Interpretation (suggested points for Results & Discussion)

1. **The standard task is solved by keyword spotting.** Both recurrent models are near-perfect, but
   so is TF-IDF + LR (0.979). The collection keywords are present in almost every tweet (EDA), so the
   standard numbers mostly measure keyword detection, not sequence modelling. BiGRU vs BiLSTM
   (1.000 vs 0.997) is a difference of one tweet on average and should not be read as a ranking.

2. **Masking the keywords shows what the models learn from context.** Without keywords the BiLSTM
   and BiGRU still reach 0.94 macro-F1 and 98.4% accuracy, 3 points above TF-IDF + LR on the same
   masked text (0.911). Both models read the order of the words around `<kw>` ("my `<kw>` `<kw>` me
   for drinking his…", "`<kw>` me because I refused…"), which a bag of n-grams captures only
   locally.

3. **Pretrained embeddings matter more than the recurrent cell.** On masked text, GloVe-Twitter
   adds 5 macro-F1 points over random initialisation (0.943 vs 0.893). BiLSTM vs BiGRU differs by
   0.001, well within seed noise (std 0.004–0.005). After masking, the class signal is in rarer
   context words (*divorce*, *boss*, *arranged*, *court*). A model trained from scratch sees too few
   examples of these to learn them, but GloVe already places them near related words. This matches
   the low-resource NLP finding that pretrained representations help most when labelled data is
   scarce for the signal that matters.

4. **LSTM vs GRU.** The GRU has 3% fewer parameters (1.87M vs 1.93M; most of the parameters are in
   the embedding layer) and usually peaked earlier in the standard setting (best epochs 3–6 vs 6–7).
   Accuracy is the same. This matches Chung et al. (2014): on short sequences (median 51 tokens,
   maximum 80) the LSTM's separate memory cell gives no measurable benefit. The two cells trade
   errors on the smallest classes (GRU better on economic, LSTM better on HTP), but with 19–21 test
   tweets those differences are one or two tweets.

5. **Sequence length.** In the standard setting there are no length effects (0 or 1 error in total).
   In the masked setting the error rate is lowest for short tweets (0.2% for ≤20 tokens, n = 445)
   and highest for 51–65 tokens (2.1–2.6%, n = 1,585). This is *not* the RNN forgetting long-range
   context: 80 tokens is short for a gated RNN, and the curve falls again above 65 tokens. Longer
   tweets are more likely to describe several kinds of abuse at once (see 4b below), which makes the
   single gold label ambiguous.

6. **Training dynamics.** On masked text, validation loss starts rising after epoch ~8 while
   validation macro-F1 keeps improving (`fig_learning_curves.png`). The model becomes overconfident
   on the majority class while still improving on the minority classes. This justifies early
   stopping on macro-F1 rather than on loss.

## 4. Error analysis

Files: `results/bilstm_gru/errors_*.csv`, sorted by model confidence.

**4a. Standard setting: the single remaining error is a labelling problem.** The standard BiLSTM
misclassifies exactly one test tweet and the BiGRU none. That tweet is:

> "I hope my future husband knows me well enough to NEVER propose to me in a public place, that way
> he won't have to be humiliated when I say no"

It is labelled *emotional violence* but describes no violence. It was collected because it contains
*husband*, *public* and *humiliated*. The model predicted *physical violence* (it has *husband*),
with confidence 0.86, its least confident prediction.

**4b. Masked setting: most errors involve sexual vs physical (61 BiLSTM errors, 70 BiGRU).**

| True → predicted | BiLSTM | BiGRU |
|---|---|---|
| physical → sexual | 19 | 10 |
| sexual → physical | 18 | 35 |
| emotional → sexual | 11 | 11 |
| sexual → emotional | 7 | 8 |

45 of the BiLSTM's 61 errors are also BiGRU errors. The failures are mostly driven by the data,
not by the architecture. The main causes we found:

1. **Tweets that describe more than one form of violence.** The dataset has one label per tweet,
   taken from the collection keyword, but many tweets describe several kinds of abuse:
   > "…he wouldn't take no for an answer to having sex? So he beats the crap out of me, forces me to
   > perform painful sex acts leaving me bleeding…" (gold: *physical*, predicted: *sexual*)

   > "…he started having sex with her alongside me. We fought about it but he beats me and now he's
   > thrown me out." (gold: *physical*, predicted: *sexual*)

   > "A manager telling me if I didn't sit on his lap while I worked he'd hire someone else…" (gold:
   > *economic*, predicted: *sexual*)

   These predictions are defensible. The task is really **multi-label**, and a single-label setup
   counts these predictions as errors.

2. **Keyword used in a non-violent sense** (noise from keyword collection):
   > "…my friend's mom said '[husband] beats me up almost every morning!' She meant he gets up
   > first…" (gold: *physical*)

   > "…an Anituber I unintentionally insulted who insulted me as a person in response…" (gold:
   > *emotional*)

   Several emotional-violence tweets are ordinary online arguments that contain *insulted* or
   *public*.

3. **Emotional violence is the hardest class** (F1 about 0.83 masked for both cells; 17% of its
   tweets are predicted as sexual). Its context is very varied (online fights, police harassment,
   family disputes), whereas the other classes have consistent context words (*boss/job* for
   economic, *Somalia/girls/practice* for HTP).

4. **Code-switching.** A few tweets mix English with Hindi/Urdu (e.g. "…insurance nahi thi exprire
   ho gyi then he asked to cut your challan… he insulted me"). Only 17 test tweets match a small list
   of common Hindi/Urdu function words; 1 of them is misclassified (5.9% vs 1.5% overall). The trend
   points the expected way, but the sample is far too small to draw conclusions. GloVe-Twitter is
   English-centric, so these words mostly have random vectors.

5. **Overconfidence.** 46% of masked-BiLSTM errors have confidence > 0.99. The softmax outputs are
   not calibrated, so the probabilities cannot be used to send uncertain tweets to a human reviewer
   without recalibration (e.g. temperature scaling).

## 5. Limitations of approach 3

- **Ceiling effect:** the standard setting cannot rank models or design choices, so the masked
  setting carries the comparison.
- **Small minority test sets** (19 HTP, 21 economic). One tweet moves class F1 by about 0.05. We
  report three seeds, but the test split itself is fixed. Cross-validation would give tighter
  intervals but was not affordable on CPU (about 1.5–2 min per epoch).
- **Keyword list is hand-made.** Masking removes the main query terms but not every
  correlated word. Related words such as *housewife*, *sexually*, *molested*, *hit* or *assault*
  remain, so the masked numbers are an upper bound on purely contextual performance.
- **Epoch cap.** Several masked runs peaked at epoch 11–12 of 12, so a longer budget might add a
  little.
- **English-only embeddings and word-level vocabulary.** Misspellings, code-switched words and
  post-2014 emojis get random or `<unk>` vectors. A subword model (approach 5) should handle these
  better.
- **Single-label framing.** As shown in 4b, many errors are multi-label tweets. Multi-label
  annotation would be the most valuable improvement to the data.

## 6. Future work

- Re-label a sample as multi-label and train with sigmoid outputs.
- Masked-language-model pretraining on in-domain tweets (both the train and the unlabelled Zindi
  test text) before fine-tuning.
- Temperature scaling for calibrated probabilities, then a "refer to human" threshold.
- Character- or subword-level input to handle spelling variation and code-switching.
