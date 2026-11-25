from src.langid import predict, predict_proba, train

TEXTS = [
    "Bonjour, comment allez-vous aujourd'hui ?",
    "Je vais au marché acheter du pain et des fruits.",
    "Le professeur explique le cours de mathématiques.",
    "Hello, how are you doing today?",
    "I am going to the market to buy bread and fruit.",
    "The teacher explains the mathematics lesson.",
    "I ni sogoma, i ka kɛnɛ wa?",
    "N bɛ taa sugu la ka buru san.",
    "Karamɔgɔ bɛ kalan kɛ bi.",
]
LABELS = ["fr"] * 3 + ["en"] * 3 + ["bam"] * 3


def test_fits_and_returns_known_labels():
    model = train(TEXTS, LABELS)
    preds = predict(model, TEXTS)
    assert set(preds) <= {"fr", "en", "bam"}
    assert preds == LABELS


def test_probabilities_sum_to_one():
    model = train(TEXTS, LABELS)
    probs = predict_proba(model, "Bonjour tout le monde")
    assert abs(sum(probs.values()) - 1.0) < 1e-6
