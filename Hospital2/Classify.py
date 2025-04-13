import xml.etree.ElementTree as ET
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from collections import Counter
import os
# Function to parse the XML and extract content dynamically
def parse_xml_and_extract_content(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Extract all content sections dynamically
    content_data = {}
    content = root.find(".//Content")
    
    if content is not None:
        # Loop over all child elements in <Content> (sections could vary)
        for section in content:
            content_data[section.tag] = section.text
            
    print(content_data)

    return content_data

# Initialize model and tokenizer
model_path = '/home/bachar/DCSM/SecurityClassification_final/security_classification_model'  # Path to your saved model
tokenizer_path = '/home/bachar/DCSM/SecurityClassification_final/security_classification_tokenizer'  # Path to your saved tokenizer
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

# Function to classify extracted content using the trained model
def classify_data_with_model(data, model, tokenizer):
    security_labels = {}

    # Iterate over each section and classify it
    for section, text in data.items():
        # Prepare input text from each section
        input_text = f"{section}: {text}"

        # Tokenize the input text
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=128)

        # Get prediction from model
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = outputs.logits.argmax(dim=-1)  # Get the label with the highest score

        # Map prediction to the corresponding security label
        label_map = {0: 'Highly Confidential', 1: 'Confidential', 2: 'Restricted', 3: 'Public'}
        predicted_label = label_map[prediction.item()]
        security_labels[section] = predicted_label
    
    return security_labels

# Function to classify the entire document based on section labels and thresholds
def classify_entire_document(security_labels, thresholds=None):
    # Default thresholds if not provided
    if thresholds is None:
        thresholds = {
            'Highly Confidential': 0.3,
            'Confidential': 0.35,
            'Restricted': 0.4,
            'Public': 0.0
        }
    
    # Count the occurrences of each label
    label_counts = Counter(security_labels.values())

    # Total sections in the content
    total_sections = len(security_labels)

    # Check the percentage of each label
    for label, threshold in thresholds.items():
        # Calculate the percentage of sections with the current label
        percentage = label_counts[label] / total_sections

        if percentage >= threshold:
            return label  # If the percentage of this label exceeds the threshold, assign it to the entire document

    # If no label meets the threshold, return the lowest level (Public)
    return 'Public'

def add_security_label_to_xml(root, document_label):
    # Create the SecurityLabel element
    security_label_element = ET.Element("SecurityLabel")
    security_label_element.text = document_label

    # Insert SecurityLabel at the beginning of the XML document
    root.insert(0, security_label_element)

# Function to save updated XML
def save_updated_xml(tree, output_file):
    tree.write(output_file, encoding="utf-8", xml_declaration=True)
# Example usage
#xml_file = '/home/bachar/DCSM/Hospital1/Patients/Patient_Files/Patient_PT98765_1.xml'  # Provide the path to your XML file
base_path = "./patients"
patient_id = input("Enter Patient ID: ").strip()
directory_name = f"Patient_{patient_id}"
patient_path = os.path.join(base_path, directory_name)
patient_plain_path = os.path.join(patient_path,'Plaindata')
file_name = input("Enter the name of the patient XML file (or type 'exit' to quit): ").strip()       
#Construct the full file path
xml_file = os.path.join(patient_plain_path, file_name)
content_data = parse_xml_and_extract_content(xml_file)
tree = ET.parse(xml_file)
root = tree.getroot()
# Classify each section of the extracted content
security_labels = classify_data_with_model(content_data, model, tokenizer)

# Classify the entire document based on section classifications and thresholds
document_label = classify_entire_document(security_labels)

# Output the results
print(f"Predicted Security Labels for each section: {security_labels}")
print(f"Overall document security label: {document_label}")

add_security_label_to_xml(root, document_label)

# Save the updated XML to a new file
classified_plain_path = os.path.join(patient_path,'Classifieddata')
output_file = os.path.join(classified_plain_path, file_name)
save_updated_xml(tree, output_file)
