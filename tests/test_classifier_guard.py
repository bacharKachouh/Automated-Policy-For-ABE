from src.classification.classifier import SecurityClassifier


def test_classify_document_empty_returns_public():
    """If no sections were classified (e.g. XML had no <Content>),
    classify_document must NOT divide by zero. It returns 'Public'."""
    # We bypass __init__ to avoid loading the BERT model in unit tests.
    clf = SecurityClassifier.__new__(SecurityClassifier)
    assert clf.classify_document({}) == "Public"
