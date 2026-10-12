# AI Spam Detector — Naive Bayes From Scratch

**Building AI course project | University of Helsinki / MinnaLearn**

An open-source educational Python project that classifies short English messages as **spam** or **ham** (legitimate). It demonstrates how Bayes' rule and supervised machine learning can support message filtering and help reduce repetitive moderation work.

> **Project status:** Working educational prototype with synthetic sample data. This is **not** an official university project, and the University of Helsinki does not endorse this implementation. Do not deploy it as a security filter without real-world validation.

## 1. Problem and motivation

Spam messages waste time, may contain phishing links, and can overwhelm small business support accounts. Many small organizations need an understandable starting point for filtering suspicious messages. This project shows a simple, explainable method that can be extended into a moderation workflow, for example for a Telegram bot, with human review.

## 2. Proposed AI solution

Build a Multinomial **Naive Bayes** classifier that learns word frequencies from labeled training messages. The program estimates the probabilities of `spam` and `ham` for new messages and assigns the most likely label.

- **Inputs:** a text message
- **Output:** predicted `spam` or `ham` and estimated class probabilities
- **ML task:** supervised binary text classification
- **Algorithm:** Multinomial Naive Bayes with Laplace smoothing
- **Data:** 40 hand-written, synthetic example messages (20 spam, 20 ham)
- **Implementation:** Python standard library only; no paid APIs or external dependencies

## 3. How it works

1. Lowercase text and extract word tokens.
2. Count the messages and token occurrences for each label.
3. Compute prior probability for each label.
4. Calculate log likelihoods for the words using Laplace smoothing.
5. Add log probabilities and normalize the scores.
6. Choose the label with the larger probability.

The **naive assumption** is that word occurrences are conditionally independent given the class. This makes the model efficient but ignores word order and context.

## 4. Run locally

Requires **Python 3.9+**. Clone or download the repository, then run from its root:

```bash
python app.py --text "claim your free cash prize now"
python app.py --text "please review the project report"
python app.py
```

For unit tests:

```bash
python -m unittest discover -s tests -v
```

The app uses a deterministic stratified 75%/25% train/test split of its sample messages. It prints accuracy, spam precision, spam recall, and a confusion matrix (TP/FP/TN/FN). Metrics on this tiny synthetic set do **not** establish real-world performance.

## 5. Example use cases

- A customer-support team reviewing suspicious incoming messages
- A Telegram bot that sends uncertain predictions to a human moderator
- A teaching demo for conditional probability, Naive Bayes, and model evaluation

## 6. Responsible AI and limitations

- **False positives:** legitimate customer messages might be flagged. Do not automatically block accounts based on this prototype.
- **False negatives:** malicious spam can go undetected.
- **Language limitations:** examples are English; the tokenizer and training data need adaptation for Persian and other languages.
- **Small, biased sample:** the toy data is synthetic and not representative of real conversations.
- **Security & privacy:** don't include sensitive chats, private data, credentials, or personally identifiable information in public training files. Respect messaging-platform rules.
- **Model calibration:** probabilities produced by a small naive Bayes model are not guaranteed to be well-calibrated.

## 7. Possible future improvements

- Collect a larger **consented, anonymized** dataset with realistic class proportions.
- Introduce a validation set, cross-validation, and confidence thresholds.
- Add better multilingual tokenization and Persian examples.
- Compare with TF-IDF + logistic regression as a stronger baseline.
- Integrate with Telegram/n8n using a review queue rather than automatically deleting messages.

## 8. Files

```text
app.py                        # CLI demo and evaluation
spam_classifier.py            # Multinomial Naive Bayes implementation
data/sample_messages.csv      # Synthetic labeled messages
tests/test_classifier.py      # Unit tests
README.md                     # Project report and installation
LICENSE                       # MIT license
```

## 9. Course project statement

This is an independent educational submission to the **Building AI** course. It demonstrates a realistic AI idea, a working prototype, its evaluation, and its potential social benefits and risks.

**Author:** Hamzeh Karimpour Hosseinabadi

## License

MIT License — see [LICENSE](LICENSE).
