from evals.metrics import (
    first_specialist_correct,
    keyword_recall,
    specialist_accuracy,
)


def test_keyword_recall_partial():
    answer = "Paris has about 2.1 million people."
    assert keyword_recall(answer, ["Paris", "2.1 million"]) == 1.0
    assert keyword_recall(answer, ["Paris", "London"]) == 0.5
    assert keyword_recall(answer, []) == 1.0


def test_specialist_accuracy_jaccard():
    assert specialist_accuracy(["search"], ["search"]) == 1.0
    assert specialist_accuracy(["search", "code"], ["search"]) == 0.5
    assert specialist_accuracy([], []) == 1.0
    assert specialist_accuracy(["code"], ["search"]) == 0.0


def test_first_specialist_correct():
    assert first_specialist_correct(["search", "code"], ["search"]) == 1
    assert first_specialist_correct(["code"], ["search"]) == 0
    assert first_specialist_correct([], ["search"]) == 0
