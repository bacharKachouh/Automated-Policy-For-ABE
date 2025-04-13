import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset

# Load the pre-trained tokenizer and model
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
    
    train_test_split = data['train'].train_test_split(test_size=0.2)
    return train_test_split

# Load and prepare the data
data_file = './medical_data.csv'
dataset = prepare_data(data_file)

training_args = TrainingArguments(
    output_dir='./results',         
    evaluation_strategy="epoch",    
    logging_dir='./logs',
    logging_steps=10,  # Adjust this to log more or less frequently
    learning_rate=2e-5,
    per_device_train_batch_size=4,  
    num_train_epochs=3,             
    weight_decay=0.01,
    save_steps=100,  # Save model every 100 steps
    save_total_limit=2,  # Limit the total number of saved models
)

# Initialize the trainer
trainer = Trainer(
    model=model,                         
    args=training_args,                  
    train_dataset=dataset['train'],         # Training data
    eval_dataset=dataset['test']            # Validation data (previously split)
)

# Fine-tuning
trainer.train()

# Saving the model
model.save_pretrained('./security_classification_model')
tokenizer.save_pretrained('./security_classification_tokenizer')
