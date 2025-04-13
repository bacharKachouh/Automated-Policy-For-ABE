from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# Load the pre-trained model and tokenizer
model_name = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=4)  # 4 labels for security classification

# Example: Converting your data to Huggingface Dataset
def prepare_data(file_path):
    # Assuming CSV with columns 'Content', 'SecurityLabel'
    data = load_dataset('csv', data_files=file_path)
    label_map = {'Highly Confidential': 0, 'Confidential': 1, 'Restricted': 2, 'Public': 3}
    
    def tokenize_and_encode(example):
        return tokenizer(example['MedicalText'], padding='max_length', truncation=True, max_length=128)
    
    # Apply tokenization and encoding
    data = data.map(tokenize_and_encode, batched=True)
    data = data.map(lambda example: {'label': label_map[example['SecurityLabel']]})
    
    # Split the dataset into training and testing sets
    train_test_split = data['train'].train_test_split(test_size=0.2)
    return train_test_split

# Load and prepare the data
data_file = './medical_data.csv'
dataset = prepare_data(data_file)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',          # Output directory
    evaluation_strategy="epoch",     # Evaluate every epoch
    logging_dir='./logs',            # Directory for logging
    logging_steps=10,                # Log every 10 steps
    learning_rate=2e-5,              # Learning rate
    per_device_train_batch_size=4,   # Batch size for training
    num_train_epochs=3,              # Number of epochs
    weight_decay=0.01,               # Weight decay to prevent overfitting
    save_steps=100,                  # Save model every 100 steps
    save_total_limit=2,              # Limit the number of saved models
)

# Initialize the trainer
trainer = Trainer(
    model=model,                     # The model to train
    args=training_args,              # Training arguments
    train_dataset=dataset['train'],  # Training dataset
    eval_dataset=dataset['test']     # Validation dataset
)

# Train the model
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained('./security_classification_model')
tokenizer.save_pretrained('./security_classification_tokenizer')

