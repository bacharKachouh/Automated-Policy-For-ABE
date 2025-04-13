from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os 
# Load the fine-tuned model and tokenizer
model = GPT2LMHeadModel.from_pretrained("./gpt2-finetuned-access-policy")
tokenizer = GPT2Tokenizer.from_pretrained("./gpt2-finetuned-access-policy")

# Use the model for inference as needed
#input_text = '/home/bachar/DCSM/Hospital1/Patients/DataAttributes'
path_text = input("Enter the path to the directory (or type 'exit' to quit): ").strip()
with open(path_text, 'r') as file:
            # Read the contents of the file
            input_text = file.read()
# Encode the input prompt
inputs = tokenizer.encode(input_text, return_tensors="pt")

# Generate the response (access policy)
outputs = model.generate(
    inputs,
    max_length=150,  # Adjust the length depending on your needs
    num_return_sequences=1,  # Number of sequences to generate
    temperature=0.7,  # Controls randomness: lower = more deterministic
    top_p=0.9,  # Top-p (nucleus sampling): consider only top 90% of probability mass
    top_k=50,  # Top-k sampling: limit to top 50 tokens
    no_repeat_ngram_size=2,  # Avoid repeating n-grams
    pad_token_id=tokenizer.eos_token_id  # Pad to the EOS token
)

# Decode and print the generated access policy
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
cleaned_text = generated_text.split("### Access Policy:")[-1]  # Keep only the access policy part
access_policy = cleaned_text.split('.')[0] + '.'
def save_to_text_file(formatted_text, output_file):
    with open(output_file, 'w') as file:
        file.write(formatted_text)
base_path = "./patients"
patient_id = input("Enter Patient ID: ").strip()
directory_name = f"Patient_{patient_id}"
patient_path = os.path.join(base_path, directory_name)
acess_path = os.path.join(patient_path,'Accesspolicy')
file_output = input("Enter the name of the access policy txt file (or type 'exit' to quit): ").strip()
output_file = os.path.join(access_path, file_output) 
save_to_text_file(access_poicy, output_file)
print(access_policy)

