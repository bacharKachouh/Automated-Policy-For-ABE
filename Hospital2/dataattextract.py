import xml.etree.ElementTree as ET
import os
# Function to extract details and generate formatted text
def generate_formatted_text(xml_file):
    # Parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Extract required fields
    data_type = root.find(".//DataType").text
    sensitivity = "Highly Confidential"  # Placeholder, you would set it based on your classification logic
    department = root.find(".//Department").text
    purpose = root.find(".//Purpose").text
    emergency = root.find(".//Emergency").text

    # Format the output
    formatted_text =  f"""
Data Type: {data_type}  
Sensitivity: {sensitivity}  
Department: {department}  
Purpose: {purpose}  
Emergency: {emergency}  
### Access Policy:
    """
    return formatted_text

# Function to save the generated text to a file
def save_to_text_file(formatted_text, output_file):
    with open(output_file, 'w') as file:
        file.write(formatted_text)

# Example usage
base_path = "./patients"
patient_id = input("Enter Patient ID: ").strip()
directory_name = f"Patient_{patient_id}"
patient_path = os.path.join(base_path, directory_name)
classified_plain_path = os.path.join(patient_path,'Classifieddata')
file_name = input("Enter the name of the patient XML file (or type 'exit' to quit): ").strip()
xml_file =  os.path.join(classified_plain_path, file_name) # Path to your XML file
formatted_text = generate_formatted_text(xml_file)

# Specify the output file path (save in a new folder)
attribute_plain_path = os.path.join(patient_path,'DataAttribute')
file_output = input("Enter the name of the atribute patient txt file (or type 'exit' to quit): ").strip()
output_file = os.path.join(attribute_plain_path, file_output) # Path to your XML file  # Replace with your desired path
save_to_text_file(formatted_text, output_file)

print(f"Formatted text saved to {output_file}")
