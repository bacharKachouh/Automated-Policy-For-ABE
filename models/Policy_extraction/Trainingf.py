import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling

# Step 1: Load and preprocess the dataset
def load_and_preprocess_dataset(filepath):
    # Load the dataset
    data = pd.read_csv(filepath)
    
    # Combine the attributes to form a "prompt" and the access policy as the "completion"
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
    # Save the processed data to a text file for fine-tuning
    processed_data = data["text"].tolist()
    with open("processed_dataset.txt", "w") as f:
        f.write("\n".join(processed_data))
    return "processed_dataset.txt"

# Step 2: Fine-tune GPT-2
def fine_tune_gpt2(processed_file):
    # Load the tokenizer and model
    model_name = "gpt2"  # Or use a larger model like "gpt-neo-1.3B" if resources allow
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Set padding token
    
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # Load the processed dataset
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=processed_file,
        block_size=128,
    )
    
    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Masked language modeling is off
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./gpt2-finetuned-access-policy",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        save_steps=500,
        save_total_limit=2,
        prediction_loss_only=True,
        logging_dir='./logs',
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
    
    # Train the model
    trainer.train()

    # Save the fine-tuned model and tokenizer for future use
    model.save_pretrained("./gpt2-finetuned-access-policy")
    tokenizer.save_pretrained("./gpt2-finetuned-access-policy")
    
    print("Fine-tuned model and tokenizer saved successfully!")

    return model, tokenizer

# Example of how to use this function
if __name__ == "__main__":
    # Load and preprocess the dataset
    filepath = 'access_policy_dataset.csv'  # Specify the path to your CSV file
    processed_file = load_and_preprocess_dataset(filepath)
    
    # Fine-tune the GPT-2 model
    fine_tune_gpt2(processed_file)

