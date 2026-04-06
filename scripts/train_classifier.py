#!/usr/bin/env python3
"""
Fine-tune BioClinicalBERT for 4-class medical security classification.

Input  : ``training/data/medical_data.csv``  (columns: MedicalText, SecurityLabel)
Output : ``models/security_classifier/``

Usage
-----
    python scripts/train_classifier.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    Trainer,
    TrainingArguments,
)

from src.config import CLASSIFIER_MODEL_PATH, SECURITY_LABEL_MAP, TRAINING_DATA_DIR

BASE_MODEL = "emilyalsentzer/Bio_ClinicalBERT"
DATA_FILE = str(TRAINING_DATA_DIR / "medical_data.csv")
LABEL_MAP = {v: k for k, v in SECURITY_LABEL_MAP.items()}  # label → int


def prepare_data(tokenizer):
    data = load_dataset("csv", data_files=DATA_FILE)

    def tokenize(example):
        return tokenizer(
            example["MedicalText"], padding="max_length", truncation=True, max_length=128
        )

    data = data.map(tokenize, batched=True)
    data = data.map(lambda ex: {"label": LABEL_MAP[ex["SecurityLabel"]]})
    return data["train"].train_test_split(test_size=0.2)


def main():
    print(f"Loading base model: {BASE_MODEL}")
    tokenizer = BertTokenizer.from_pretrained(BASE_MODEL)
    model = BertForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=4)

    dataset = prepare_data(tokenizer)

    args = TrainingArguments(
        output_dir=str(CLASSIFIER_MODEL_PATH),
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,
        logging_dir=str(CLASSIFIER_MODEL_PATH / "logs"),
    )

    Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
    ).train()

    model.save_pretrained(str(CLASSIFIER_MODEL_PATH))
    tokenizer.save_pretrained(str(CLASSIFIER_MODEL_PATH))
    print(f"\nModel saved → {CLASSIFIER_MODEL_PATH}")


if __name__ == "__main__":
    main()
