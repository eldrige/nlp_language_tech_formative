# Methodology

*Covers data preparation, tokenization, sequence representation, experimental
design and approach 3 (BiLSTM/BiGRU). Other approaches add a short subsection for their
approach under 4.4 in the same format.*

## 4.1 Data preparation

**Shared splits.** The Zindi test set has no labels, so we built an internal labelled hold-out
from `Train.csv` (39,650 tweets). We first removed near-duplicates. Two tweets count as duplicates
if they are identical after the normalisation in 4.2. Exact string matching finds only 8 of these,
but normalisation finds 620, mostly re-posts that differ only in HTML escaping or quote characters.
Without this step a copy of a training tweet could land in the test set. We then made a stratified
80/10/10 split (seed 42): 31,223 train, 3,904 validation and 3,903 test tweets. All three splits
keep the original class proportions (83.4% sexual, 13.9% physical, 1.7% emotional, 0.54% economic,
0.49% harmful traditional practice). Every model uses the same split file
(`data/splits/splits.csv`). Validation is used for early stopping and model selection; the test
split is used once, for the numbers in Section 5.

The rarest classes have only about 20 test tweets each (21 economic, 19 HTP). A single
misclassification changes that class's F1 by roughly 0.03–0.05, so we report the mean and standard
deviation over three random seeds rather than a single run.

**Cleaning.** The organisers had already removed user handles, URLs and hashtags, but three kinds of
noise remain. 8.3% of tweets contain HTML entities, some double-escaped (`&amp;amp;`). 11.2% contain
emojis. 17.7% contain words in all capitals. Our cleaning function (`clean_text`) does the following:

1. un-escapes HTML twice and applies Unicode NFKC normalisation;
2. maps curly quotes, dashes and ellipses to ASCII;
3. separates emojis into their own tokens, because they often carry the tone of the tweet
   (😭, 😡, 🙄);
4. marks emphasis the way GloVe-Twitter's own preprocessing does (Pennington et al., 2014): a word in
   capitals is lowercased and followed by `<allcaps>`, a lengthened word like "sooooo" becomes
   `so <elong>`, repeated punctuation like "!!!" becomes `! <repeat>`, and numbers become `<number>`.

Emphasis often signals distress or anger ("I WAS RAPED"). The tags keep that signal after
lowercasing, and the tag tokens already have vectors in the pretrained embeddings. For the
transformer (approach 5) we use a lighter mode (`tags=False`) that keeps the original case and does
not add tags, because subword tokenizers were trained on raw text.

**Keyword masking.** Exploratory analysis showed that the collection keywords almost fully determine
the labels. "raped" occurs in 98.6% of sexual-violence tweets and predicts that class with 99.9%
precision. "beats" and "husband" each cover about 99% of physical-violence tweets. A lookup rule over
these keywords reaches 89.6% accuracy (but only 0.63 macro-F1), and a TF-IDF + logistic regression model reaches 99.7%.
High scores on this data therefore say little about whether a model uses sequential context: this is
the "shortcut learning" problem that Geirhos et al. (2020) describe, and Gururangan et al. (2018) and
McCoy et al. (2019) document similar annotation artifacts. The challenge itself asks for
classification "without using keywords". We therefore evaluate every model in two settings:

- **Standard**: the cleaned text.
- **Keyword-masked**: every collection keyword and its inflections (e.g. *rape/raped/rapist*,
  *beat/beats/beaten*, *husband/wife*, *insulted/humiliated/public*, *fired/job*,
  *FGM/genital/mutilation/forced/marriage*) is replaced by one shared token `<kw>`. All classes use
  the same token, so the mask itself says nothing about the class. After masking, the most
  class-predictive words are contextual ones such as *divorce*, *court*, *housewife*, *boss*,
  *refused*, *arranged* and *child*, plus code-switched Hindi/Urdu words in the emotional-violence
  class (*mujhe*, *beizat*).

Masking is applied to train, validation and test alike, so models are trained and tested under the
same conditions.

## 4.2 Tokenization and sequence representation

**Tokenizer.** We use a regular-expression word tokenizer (`tokenize`). It keeps tag tokens, emojis
and punctuation as separate tokens and splits clitics the Penn Treebank way (*don't* → *do n't*,
*I'm* → *i 'm*; Marcus et al., 1993). The clitic rule matters. GloVe-Twitter was built with this
convention, and our first tokenizer kept *don't* and *I'm* whole, which left the most frequent
contractions without a pretrained vector. With the fix, GloVe covers 99.4% of training tokens
(up from 97.2%) and 95.1% of vocabulary types. Almost all remaining misses are emojis released after
2014 (😂, 😭, 🤣) and the word *covid*. These get random vectors that are learned during training.

**Vocabulary.** The vocabulary is built from the training split only, keeping words seen at least
twice. That gives 16,634 entries, including index 0 for padding and index 1 for unknown words. About
half of the raw word types occur only once (15,897 of 32,529), mostly misspellings, names and
elongated forms. Dropping them reduces noise and costs little: 1.7% of test tokens and 3.1% of Zindi
test tokens become `<unk>`.

**Sequence length.** Tweets are short and fairly uniform in length. After tokenization the median is
51 tokens, the 95th percentile 69 and the 99th percentile 77. The maximum (170) comes from
all-capitals tweets, where every word gains an `<allcaps>` tag. We fix the input length at 80 tokens,
padding and truncating at the end. Only 0.8% of tweets are truncated, and truncation removes only the
end of a tweet, never the start. Because the maximum length is short, recurrent models are unlikely
to suffer badly from long-range forgetting here. This informs the architecture discussion below.

**Embeddings.** Each token maps to a 100-dimensional vector. We compare random initialisation with
**GloVe-Twitter-100** (Pennington et al., 2014), which was trained on 2 billion tweets. We chose it
over general-domain embeddings because our data is informal social-media text with slang,
abbreviations and emojis. Words missing from GloVe are initialised from a normal distribution with
the mean and standard deviation of the GloVe vectors. The padding vector is fixed at zero and masked,
so the recurrent layers never see padding.

## 4.3 Experimental design (shared)

- **Model selection** uses validation macro-F1 at seed 42. **Final numbers** are the mean ± standard
  deviation over seeds 42, 1 and 2 on the test split.
- **Primary metric: macro-F1.** It weights the five classes equally. Accuracy is dominated by the
  majority class; a model that always predicts "sexual violence" already scores 83.4%. We also report
  accuracy (the Zindi leaderboard metric), weighted F1, macro one-vs-rest ROC-AUC and macro PR-AUC. We
  include PR-AUC because ROC-AUC looks optimistic when positives are rare (Saito & Rehmsmeier, 2015).
- **Class imbalance** is handled by weighting the loss, not by resampling, so the input distribution
  stays unchanged. We compare no weighting, fully balanced weights (w_c = N / (K·N_c), which puts the
  rarest class 170 times above the largest) and a softened square-root version (w_c = √(N / (K·N_c)),
  a 13× ratio). Full inverse-frequency weights can overfit the handful of minority examples, and
  softer schemes are often better (Cui et al., 2019).
- **Robustness**: every model is trained and tested in both the standard and the keyword-masked
  setting (4.1).

## 4.4 Approach 3: Bidirectional recurrent networks (BiLSTM / BiGRU)

**Why this approach.** Recurrent networks read a tweet token by token and keep a hidden state that
summarises everything read so far. That suits this task because the label often depends on who did
what to whom, and word order carries that. *"My husband beats me"* describes physical violence;
*"he beats every excuse"* does not, although both contain *beats*. Bag-of-words baselines cannot
represent that difference. LSTMs (Hochreiter & Schmidhuber, 1997) and GRUs (Cho et al., 2014) use
gates to control what the hidden state keeps or forgets, which avoids the vanishing gradients of
plain RNNs. We run the network in both directions (Schuster & Paliwal, 1997) so that each position
sees its left and right context. This helps with negation and with reported speech (*"she said he
raped her"*), where the phrase that decides the meaning can come after the keyword. We compare LSTM
with GRU because GRUs have fewer parameters and often perform similarly on short sequences (Chung
et al., 2014). The rarest class has only about 150 training tweets, so a smaller model could
generalise better.

**Architecture.**

```
token ids (80) → Embedding(16,634 × 100, mask padding) → SpatialDropout1D(0.2)
  → Bidirectional LSTM or GRU (128 units per direction, all time steps)
  → [ masked mean-pool ‖ masked max-pool ]  (512-d)
  → Dropout(0.3) → Dense(64, ReLU) → Dropout(0.3) → Dense(5, softmax)
```

We pool over all hidden states instead of using only the final state. The decisive phrase in a
tweet can appear anywhere, and a final-state summary is biased towards the end of the tweet.
Max-pooling picks the strongest signal for each feature at any position; mean-pooling summarises the
whole tweet. Concatenating both is a well-established choice for recurrent text classifiers (Conneau
et al., 2017; Howard & Ruder, 2018). We wrote a custom max-pool layer that respects the padding mask,
because Keras' built-in max-pool would also take the maximum over padding positions. Spatial dropout
drops whole embedding channels rather than single values, which regularises better for correlated
word vectors (Tompson et al., 2015).

**Training.** Adam (Kingma & Ba, 2015) with learning rate 1e-3, gradient-norm clipping at 1.0
(Pascanu et al., 2013), batch size 128, class-weighted cross-entropy and at most 12 epochs. Early
stopping monitors **validation macro-F1** (patience 3) and restores the best weights. We use this
instead of validation loss because the loss is dominated by the 83% majority class and can keep
improving while the minority classes get worse.

**Design choices tested** (seed 42, selected on validation macro-F1):

| Factor | Variants |
|---|---|
| Recurrent cell | LSTM vs GRU |
| Embeddings | random init vs GloVe-Twitter fine-tuned vs GloVe-Twitter frozen |
| Class weights | none vs square-root vs balanced |
| Input | standard vs keyword-masked |

In the standard setting all variants scored within 0.003 validation macro-F1 of each other, about
one or two minority-class tweets, so validation could not separate them. In the masked setting
GloVe beat random initialisation by 4 points (0.939 vs 0.899). The final configuration is
therefore GloVe-Twitter, fine-tuned, with square-root class weights, run as both BiLSTM and BiGRU.
We kept the square-root weights as the middle option, since none of the three weighting schemes did
better than the others. Weighting was compared only in the standard setting because of compute
limits.

We did not tune layer depth or hidden size. The assignment asks for substantially different
approaches rather than variants of one, and the key questions for this data are the input
representation and the class imbalance. Changing the network size is unlikely to answer either.

## References (for this section)

- Cho, K., et al. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. *EMNLP*.
- Chung, J., Gulcehre, C., Cho, K., & Bengio, Y. (2014). Empirical evaluation of gated recurrent neural networks on sequence modeling. *arXiv:1412.3555*.
- Conneau, A., Kiela, D., Schwenk, H., Barrault, L., & Bordes, A. (2017). Supervised learning of universal sentence representations from natural language inference data. *EMNLP*.
- Cui, Y., Jia, M., Lin, T.-Y., Song, Y., & Belongie, S. (2019). Class-balanced loss based on effective number of samples. *CVPR*.
- Geirhos, R., et al. (2020). Shortcut learning in deep neural networks. *Nature Machine Intelligence*, 2, 665–673.
- Gururangan, S., et al. (2018). Annotation artifacts in natural language inference data. *NAACL*.
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735–1780.
- Howard, J., & Ruder, S. (2018). Universal language model fine-tuning for text classification. *ACL*.
- Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *ICLR*.
- Marcus, M. P., Santorini, B., & Marcinkiewicz, M. A. (1993). Building a large annotated corpus of English: The Penn Treebank. *Computational Linguistics*, 19(2).
- McCoy, R. T., Pavlick, E., & Linzen, T. (2019). Right for the wrong reasons: Diagnosing syntactic heuristics in natural language inference. *ACL*.
- Pascanu, R., Mikolov, T., & Bengio, Y. (2013). On the difficulty of training recurrent neural networks. *ICML*.
- Pennington, J., Socher, R., & Manning, C. D. (2014). GloVe: Global vectors for word representation. *EMNLP*. (GloVe-Twitter vectors: https://nlp.stanford.edu/projects/glove/)
- Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE*, 10(3).
- Schuster, M., & Paliwal, K. K. (1997). Bidirectional recurrent neural networks. *IEEE Transactions on Signal Processing*, 45(11).
- Tompson, J., Goroshin, R., Jain, A., LeCun, Y., & Bregler, C. (2015). Efficient object localization using convolutional networks. *CVPR*.
- Zindi (2025). Gender-Based Violence Tweet Classification Challenge. https://zindi.africa/competitions/gender-based-violence-tweet-classification-challenge
- Software: TensorFlow 2.17 (Abadi et al., 2016), Keras 3 (Chollet et al., 2015), scikit-learn 1.5 (Pedregosa et al., 2011).
