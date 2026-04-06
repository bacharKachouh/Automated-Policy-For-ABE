#!/usr/bin/env python3
"""
Fine-tune GPT-2 on access policy generation.

Input  : ``training/data/access_policy_dataset.csv``
         (columns: Data_type, Sensitivity, Department, Purpose, Emergency, Access_policy)
Output : ``models/policy_extractor/``

Usage
-----
    python scripts/train_policy_extractor.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from transformers import (
    DataCollatorForLanguageModeling,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    TextDataset,
    Trainer,
    TrainingArguments,
)

from src.config import POLICY_MODEL_PATH, TRAINING_DATA_DIR

DATA_FILE = TRAINING_DATA_DIR / "access_policy_dataset.csv"
PROCESSED_FILE = TRAINING_DATA_DIR / "processed_dataset.txt"


def prepare_dataset():
    data = pd.read_csv(DATA_FILE)
    data["text"] = (
        "Data Attributes:\n"
        + "Data Type: " + data["Data_type"] + "\n"
        + "Sensitivity: " + data["Sensitivity"] + "\n"
        + "Department: " + data["Department"] + "\n"
        + "Purpose: " + data["Purpose"] + "\n"
        + "Emergency: " + data["Emergency"] + "\n"
        + "### Access Policy:\n"
        + data["Access_policy"]
    )
    PROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROCESSED_FILE.write_text("\n".join(data["text"].tolist()))
    print(f"Processed {len(data)} records → {PROCESSED_FILE}")
    return str(PROCESSED_FILE)


def main():
    print("Preparing dataset...")
    processed = prepare_dataset()

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    train_dataset = TextDataset(tokenizer=tokenizer, file_path=processed, block_size=128)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir=str(POLICY_MODEL_PATH),
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        save_steps=500,
        save_total_limit=2,
        prediction_loss_only=True,
        logging_dir=str(POLICY_MODEL_PATH / "logs"),
    )

    Trainer(model=model, args=args, data_collator=collator, train_dataset=train_dataset).train()

    model.save_pretrained(str(POLICY_MODEL_PATH))
    tokenizer.save_pretrained(str(POLICY_MODEL_PATH))
    print(f"\nGPT-2 policy model saved → {POLICY_MODEL_PATH}")


if __name__ == "__main__":
    main()
