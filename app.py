"""Command-line demonstration and evaluation of the educational classifier."""
import argparse
from pathlib import Path
from spam_classifier import NaiveBayesSpamClassifier, evaluate, load_examples, split_examples

DATA_PATH = Path(__file__).resolve().parent / "data" / "sample_messages.csv"


def main():
    parser = argparse.ArgumentParser(description="Educational Naive Bayes Spam Detector")
    parser.add_argument("--text", help="A message to classify (omit to use interactive mode)")
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="CSV with label,text columns")
    args = parser.parse_args()
    examples = load_examples(args.data)
    training, testing = split_examples(examples)
    model = NaiveBayesSpamClassifier().fit(training)
    metrics = evaluate(model, testing)
    print("AI Spam Detector — Naive Bayes (educational demonstration)")
    print(f"Training samples: {len(training)} | Test samples: {len(testing)}")
    print("Holdout metrics (small synthetic dataset; not production performance):")
    print(f"Accuracy: {metrics['accuracy']:.1%} | Spam precision: {metrics['precision_spam']:.1%} | Spam recall: {metrics['recall_spam']:.1%}")
    print(f"Confusion matrix: TP={metrics['tp']}, FP={metrics['fp']}, TN={metrics['tn']}, FN={metrics['fn']}")

    def classify(text):
        scores = model.predict_proba(text)
        print(f"Prediction: {model.predict(text).upper()} | Spam probability: {scores['spam']:.1%} | Ham probability: {scores['ham']:.1%}")
        print("Note: these probabilities are model estimates, not verified real-world risk.")

    if args.text is not None:
        classify(args.text)
        return
    print("\nEnter English messages to classify. Type 'quit' to exit.")
    while True:
        try:
            text = input("Message > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            return
        if text.lower() in {"quit", "exit"}:
            return
        if text:
            classify(text)


if __name__ == "__main__":
    main()
