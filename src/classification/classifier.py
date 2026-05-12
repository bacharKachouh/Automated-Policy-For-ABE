"""
Security classification of medical XML content sections.

Uses a fine-tuned BioClinicalBERT model to assign one of four labels to
each section of a patient record, then aggregates them into a document-level
label using configurable thresholds.
"""
from collections import Counter

import torch
from transformers import BertForSequenceClassification, BertTokenizer

from src.config import CLASSIFICATION_THRESHOLDS, CLASSIFIER_MODEL_PATH, SECURITY_LABEL_MAP


class SecurityClassifier:
    """
    Load a fine-tuned BioClinicalBERT model and classify medical text sections.

    Parameters
    ----------
    model_path : str or Path, optional
        Directory containing the saved model and tokenizer.
        Defaults to ``config.CLASSIFIER_MODEL_PATH``.
    """

    def __init__(self, model_path=None):
        path = str(model_path or CLASSIFIER_MODEL_PATH)
        self.model = BertForSequenceClassification.from_pretrained(path)
        self.tokenizer = BertTokenizer.from_pretrained(path)
        self.model.eval()

    def classify_section(self, section_name, text):
        """Return the security label string for a single ``section_name: text`` pair."""
        inputs = self.tokenizer(
            f"{section_name}: {text}",
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        with torch.no_grad():
            idx = self.model(**inputs).logits.argmax(dim=-1).item()
        return SECURITY_LABEL_MAP[idx]

    def classify_sections(self, content_data):
        """
        Classify all sections in *content_data*.

        Parameters
        ----------
        content_data : dict
            ``{section_tag: text}`` extracted from the XML ``<Content>`` block.

        Returns
        -------
        dict
            ``{section_tag: label}``
        """
        return {tag: self.classify_section(tag, text) for tag, text in content_data.items()}

    def classify_document(self, section_labels, thresholds=None):
        """
        Aggregate per-section labels into a single document-level label.

        If *section_labels* is empty (e.g. the XML had no <Content> element),
        returns "Public" rather than dividing by zero.
        """
        if not section_labels:
            return "Public"
        thresholds = thresholds or CLASSIFICATION_THRESHOLDS
        counts = Counter(section_labels.values())
        total = len(section_labels)
        for label, threshold in thresholds.items():
            if counts[label] / total >= threshold:
                return label
        return "Public"
