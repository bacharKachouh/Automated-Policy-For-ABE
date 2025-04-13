# Automated Policy For ABE

This repository implements a **Data-Centric Security Model** for ensuring the privacy and security of healthcare data. The system includes the following key features:

- **Data Classification**: Classifies clinical data sections based on their sensitivity using BioClinicalBERT.
- **Automated Policy Extraction**: Uses a fine-tuned GPT-2 model to generate data access policies based on extracted features.
- **Hybrid Lewko-Waters Attribute-Based Encryption (ABE)**: Encrypts data based on policy enforcement using ABE.

## Project Overview

The project includes:

1. **Security Classification Model** (`securityclassification_final`): Classifies clinical data into security categories.
2. **Policy Extraction Model** (`Policy_extraction`): Generates access control policies based on clinical data and extracted features.
3. **Demo Script**: Allows users to choose a hospital, enter a client ID, and runs the entire flow: data classification, policy extraction, encryption, and decryption.

## Setup and Installation

To set up your project environment, create and activate a **virtual environment**:

### On Linux/macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

Once activated, install the required dependencies:
```bash
pip install -r requirements2.txt
```
#### Training the Models
1. **Security Classification Model**
To train the security classification model:
```bash
cd models/securityclassification_final
python training.py
```
2. **Policy Extraction Model**
To train the policy extraction model:
```bash
cd models/Policy_extraction
python trainingf.py
```
##### Running the Demo
The demo script allows you to choose a hospital, input a client ID, and the system will run the following:

Classify the data based on sensitivity.

Extract the policy based on classification.

Encrypt the data using the generated policy.

Decrypt the data based on the client's attributes.

To run the demo:
```bash
python demo.py
```
You will be prompted to:

Select a hospital (e.g., Hospital1 or Hospital2).

Enter the patient ID (make sure the patient ID exists in the relevant hospital's patients folder).

The system will then execute the full pipeline. Note: The patient ID will be validated against the patients folder of the chosen hospital to ensure it exists before proceeding with classification, policy extraction, encryption, and decryption.

Demo Instructions
The demo will ask for the patient file name, which should be in the format: Patient_{patient_id}.xml.

For testing, you can try using a file like Patient_PT22222_1.xml or any other valid patient ID. However, make sure to delete the content of each folder in the patients folder except the plaindata folder before running the demo.

Contact
Developed by Bachar KACHOUH. For inquiries or collaborations, reach out at bachar.kachouh@hotmail.com.

