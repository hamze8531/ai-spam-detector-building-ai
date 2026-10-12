import unittest
from pathlib import Path
from spam_classifier import NaiveBayesSpamClassifier, tokenize, load_examples, split_examples, evaluate

DATA = Path(__file__).resolve().parents[1] / "data" / "sample_messages.csv"

class TestSpamClassifier(unittest.TestCase):
    def setUp(self):
        self.examples = load_examples(DATA)
        self.train, self.test = split_examples(self.examples)
        self.model = NaiveBayesSpamClassifier().fit(self.train)

    def test_tokenizer(self):
        self.assertEqual(tokenize("FREE prize!!"), ["free", "prize"])

    def test_probabilities_sum_to_one(self):
        probs = self.model.predict_proba("free prize")
        self.assertAlmostEqual(sum(probs.values()), 1.0)
        self.assertGreaterEqual(probs["spam"], 0)

    def test_basic_predictions(self):
        self.assertEqual(self.model.predict("claim your free cash prize now"), "spam")
        self.assertEqual(self.model.predict("please review the project report"), "ham")

    def test_repeatable_split(self):
        self.assertEqual((self.train, self.test), split_examples(self.examples))

    def test_evaluation(self):
        metrics = evaluate(self.model, self.test)
        self.assertEqual(sum(metrics[key] for key in ("tp", "fp", "tn", "fn")), len(self.test))

    def test_requires_fit(self):
        with self.assertRaises(RuntimeError):
            NaiveBayesSpamClassifier().predict("hello")

if __name__ == "__main__":
    unittest.main()
