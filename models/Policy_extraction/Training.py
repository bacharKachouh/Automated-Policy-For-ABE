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
    return model, tokenizer

# Step 3: Generate access policies
def generate_access_policy(model, tokenizer, input_attributes):
    prompt = (
        "Data Attributes:\n"
        + "Data Type: " + input_attributes["Data_type"] + "\n"
        + "Sensitivity: " + input_attributes["Sensitivity"] + "\n"
        + "Department: " + input_attributes["Department"] + "\n"
        + "Purpose: " + input_attributes["Purpose"] + "\n"
        + "Emergency: " + input_attributes["Emergency"] + "\n"
        + "### Access Policy:\n"
    )
    
    # Tokenize the prompt and generate a response
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs["input_ids"],
        max_length=150,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Main workflow
if __name__ == "__main__":
    # Prepare dataset
    dataset_path = "access_policy_dataset.csv"  # Ensure your file is in the same directory
    processed_file = load_and_preprocess_dataset(dataset_path)
    
    # Fine-tune the model
    print("Fine-tuning GPT-2 on the dataset...")
    model, tokenizer = fine_tune_gpt2(processed_file)
    
    # Test the fine-tuned model
    print("Generating an access policy...")
    test_input = {
        "Data_type": "Medical Records",
        "Sensitivity": "Highly Confidential",
        "Department": "Cardiology",
        "Purpose": "Treatment",
        "Emergency": "Yes",
    }
    generated_policy = generate_access_policy(model, tokenizer, test_input)
    print("Generated Policy:")
    print(generated_policy)

