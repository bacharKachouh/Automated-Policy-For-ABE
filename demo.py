import xml.etree.ElementTree as ET
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from collections import Counter
import os
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
    
    
current_directory = os.getcwd()    
hospital_choice = input("Enter the hospital (Hospital1 or Hospital2): ").strip()

# Validate the hospital choice
if hospital_choice not in ['Hospital1', 'Hospital2']:
    print("Invalid hospital choice. Please enter either 'Hospital1' or 'Hospital2'.")
else:
    # Define the base path relative to the current script location
    base_path = os.path.join(current_directory, hospital_choice, "patients")
    print(f"The base path for {hospital_choice} is set to: {base_path}")

    # Ask for the patient ID
    patient_id = input("Enter Patient ID: ").strip()
    patient_folder = f"Patient_{patient_id}"

    # Create the full path to the patient's folder
    patient_path = os.path.join(base_path, patient_folder)
    
    if not os.path.exists(patient_path):
        print(f"Patient folder {patient_folder} does not exist in {base_path}. Please check the patient ID.")
    else:
        print(f"Patient's path: {patient_path}")
hospital_prefix = 'hospitalA' if hospital_choice == 'Hospital1' else 'hospitalB'
print("test=",hospital_prefix)
def save_to_text_file(formatted_text, output_file):
    with open(output_file, 'w') as file:
        file.write(formatted_text)

from charm.core.math.pairing import hashPair as sha2
from charm.toolbox.ABEncMultiAuth import ABEncMultiAuth
from charm.toolbox.pairinggroup import PairingGroup,GT,G1,ZR, pair
from charm.core.math.pairing import pairing
from charm.toolbox.secretutil import SecretUtil
from charm.toolbox.symcrypto import AuthenticatedCryptoAbstraction
import os
import re
debug = False
def parse_policy_to_abe_format(policy_text, attributes_prefix):
    # Normalize
    policy_text = policy_text.lower()

    # Extract roles
    roles = []
    if "doctor" in policy_text:
        roles.append(f"{attributes_prefix}.doctor")
    if "researcher" in policy_text:
        roles.append(f"{attributes_prefix}.researcher")
    if "nurse" in policy_text:
        roles.append(f"{attributes_prefix}.nurse")
    if "admin" in policy_text:
        roles.append(f"{attributes_prefix}.admin")

    # Join roles with OR
    role_expr = " OR ".join(roles)
    if role_expr:
        role_expr = f"({role_expr})"

    # Extract department
    department = None
    for dept in ["cardiology", "oncology", "pharmacy", "emergency"]:
        if dept in policy_text:
            department = f"{attributes_prefix}.{dept}"
            break

    # Extract clearance level
    clearance = None
    match = re.search(r"clearance level (\d+)", policy_text)
    if match:
        level = match.group(1)
        clearance = f"{attributes_prefix}.clearance{level}"

    # Compose full policy string
    components = [expr for expr in [role_expr, department, clearance] if expr]
    return " AND ".join(components)
class Dabe(ABEncMultiAuth):
    def __init__(self, groupObj):
        ABEncMultiAuth.__init__(self)
        global util, group
        util = SecretUtil(groupObj, verbose=False)  #Create Secret Sharing Scheme
        group = groupObj    #:Prime order group        
	#Another comment
   
    def setup(self):
        '''Global Setup'''
        #:In global setup, a bilinear group G of prime order p is chosen
        #:The global public parameters, GP and p, and a generator g of G. A random oracle H maps global identities GID to elements of G
    
        #:group contains 
        #:the prime order p is contained somewhere within the group object
        g = group.random(G1)
        #: The oracle that maps global identities GID onto elements of G
        #:H = lambda str: g** group.hash(str)
        H = lambda x: group.hash(x, G1)
        GP = {'g':g, 'H': H}

        return GP

    def authsetup(self, GP, attributes):
        '''Authority Setup for a given set of attributes'''
        #For each attribute i belonging to the authority, the authority chooses two random exponents, 
        #alpha_i, y_i and publishes PK={e(g,g)^alpha_i, g^y_i} for each attribute 
        #it keeps SK = {alpha_i, y_i} as its secret key
        SK = {} #dictionary of {s: {alpha_i, y_i}} 
        PK = {} #dictionary of {s: {e(g,g)^alpha_i, g^y}}
        for i in attributes:
            #TODO: Is ZR an appropriate choice for a random element in Zp?
            alpha_i, y_i = group.random(), group.random()
            e_gg_alpha_i = pair(GP['g'],GP['g']) ** alpha_i
            g_y_i = GP['g'] ** y_i
            SK[i.upper()] = {'alpha_i': alpha_i, 'y_i': y_i}
            PK[i.upper()] = {'e(gg)^alpha_i': e_gg_alpha_i, 'g^y_i': g_y_i}
        
        if(debug):
            print("Authority Setup for %s" % attributes)
            print("SK = {alpha_i, y_i}")
            print(SK)
            print("PK = {e(g,g) ^ alpha_i, g ^ y_i}")
            print(PK)
             
        return (SK, PK)
        
    def keygen(self, gp, sk, i, gid, pkey):
        '''Create a key for GID on attribute i belonging to authority sk
        sk is the private key for the releveant authority
        i is the attribute to give bob
        pkey is bob's private key dictionary, to which the appropriate private key is added
        '''
        #To create a key for GID for attribute i belonging to an authority, the authority computes K_{i,GID} = g^alpha_i * H(GID)^y_
        h = gp['H'](gid) 
        K = (gp['g'] ** sk[i.upper()]['alpha_i']) * (h ** sk[i.upper()]['y_i'])
        
        pkey[i.upper()] = {'k': K}
        pkey['gid'] = gid
        
        if(debug):
            print("Key gen for %s on %s" % (gid, i))
            print("H(GID): '%s'" % h)
            print("K = g^alpha_i * H(GID) ^ y_i: %s" % K)
        return None

    def encrypt(self, gp, pk, M, policy_str):
        '''Encrypt'''
        #M is a group element
        #pk is a dictionary with all the attributes of all authorities put together.
        #This is legal because no attribute can be shared by more than one authority
        #{i: {'e(gg)^alpha_i: , 'g^y_i'}
        s = group.random()
        w = group.init(ZR, 0)
        egg_s = pair(gp['g'],gp['g']) ** s
        C0 = M * egg_s
        C1, C2, C3 = {}, {}, {}
        #Parse the policy string into a tree
        policy = util.createPolicy(policy_str)
        sshares = util.calculateSharesList(s, policy) #Shares of the secret 
        wshares = util.calculateSharesList(w, policy) #Shares of 0
        
    
        wshares = dict([(x[0].getAttributeAndIndex(), x[1]) for x in wshares])
        sshares = dict([(x[0].getAttributeAndIndex(), x[1]) for x in sshares])
        for attr, s_share in sshares.items():
            k_attr = util.strip_index(attr)
            w_share = wshares[attr]
            r_x = group.random()
            C1[attr] = (pair(gp['g'],gp['g']) ** s_share) * (pk[k_attr]['e(gg)^alpha_i'] ** r_x)
            C2[attr] = gp['g'] ** r_x
            C3[attr] = (pk[k_attr]['g^y_i'] ** r_x) * (gp['g'] ** w_share)
            
        return { 'C0':C0, 'C1':C1, 'C2':C2, 'C3':C3, 'policy':policy_str }

    def decrypt(self, gp, sk, ct):
        '''Decrypt a ciphertext
        SK is the user's private key dictionary {attr: { xxx , xxx }}
        ''' 
        usr_attribs = list(sk.keys())
        usr_attribs.remove('gid')
        policy = util.createPolicy(ct['policy'])
        pruned = util.prune(policy, usr_attribs)
        if pruned == False:
            raise Exception("Don't have the required attributes for decryption!")        
        coeffs = util.getCoefficients(policy)
    
        h_gid = gp['H'](sk['gid'])  #find H(GID)
        egg_s = 1
        for i in pruned:
            x = i.getAttributeAndIndex()
            y = i.getAttribute()
            num = ct['C1'][x] * pair(h_gid, ct['C3'][x])
            dem = pair(sk[y]['k'], ct['C2'][x])
            egg_s *= ( (num / dem) ** coeffs[x] )
   
        if(debug): print("e(gg)^s: %s" % egg_s)

        return ct['C0'] / egg_s

class HybridABEncMA(ABEncMultiAuth):
    def __init__(self, scheme, groupObj):
        global abencma, group
        # check properties (TODO)
        abencma = scheme
        group = groupObj

    def setup(self):
        return abencma.setup()
    
    def authsetup(self, gp, attributes):
        return abencma.authsetup(gp, attributes)
    
    def keygen(self, gp, sk, i, gid, pkey):
        return abencma.keygen(gp, sk, i, gid, pkey)

    def encrypt(self, gp, pk, M, policy_str):
        if type(M) != bytes and type(policy_str) != str:
            raise Exception("message and policy not right type!")
        key = group.random(GT)
        c1 = abencma.encrypt(gp, pk, key, policy_str)
        # instantiate a symmetric enc scheme from this key
        cipher = AuthenticatedCryptoAbstraction(sha2(key))
        c2 = cipher.encrypt(M)
        return { 'c1':c1, 'c2':c2 }
    
    def decrypt(self, gp, sk, ct):
        c1, c2 = ct['c1'], ct['c2']
        key = abencma.decrypt(gp, sk, c1)
        if key is False:
            raise Exception("failed to decrypt!")
        cipher = AuthenticatedCryptoAbstraction(sha2(key))
        return cipher.decrypt(c2)
    
    
    
    
    
#Begin Classification    
# Initialize model and tokenizer
model_path = '/home/bachar/DCSM/SecurityClassification_final/security_classification_model'  # Path to your saved model
tokenizer_path = '/home/bachar/DCSM/SecurityClassification_final/security_classification_tokenizer'  # Path to your saved tokenizer
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
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

#Extract Data Labels
classified_plain_path = os.path.join(patient_path,'Classifieddata')
file_name = input("Enter the name of the patient XML file (or type 'exit' to quit): ").strip()
xml_file =  os.path.join(classified_plain_path, file_name) # Path to your XML file
formatted_text = generate_formatted_text(xml_file)
print (" Extracting data labels are done")
# Specify the output file path (save in a new folder)
attribute_plain_path = os.path.join(patient_path,'DataAttribute')
file_output = input("Enter the name of the atribute patient txt file (or type 'exit' to quit): ").strip()
output_file = os.path.join(attribute_plain_path, file_output) # Path to your XML file  # Replace with your desired path
save_to_text_file(formatted_text, output_file)

print(f"Formatted text saved to {output_file}")
#Begin Policy Extraction
model = GPT2LMHeadModel.from_pretrained("/home/bachar/Desktop/Github/models/Policy_extraction/gpt2-finetuned-access-policy")
tokenizer = GPT2Tokenizer.from_pretrained("/home/bachar/Desktop/Github/models/Policy_extraction/gpt2-finetuned-access-policy")
filetext_name = input("Enter the name of  the data labels file (or type 'exit' to quit): ").strip()
path_text = os.path.join(attribute_plain_path, filetext_name)
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
access_path = os.path.join(patient_path,'Accesspolicy')
file_output = input("Enter the name of the access policy txt file (or type 'exit' to quit): ").strip()
output_file = os.path.join(access_path, file_output) 
save_to_text_file(access_policy, output_file)
print(access_policy)

#Hybrid Lewko-Water Algorithm
groupObj = PairingGroup('SS512')
dabe = Dabe(groupObj)

hyb_abema = HybridABEncMA(dabe, groupObj)

#Setup global parameters for all new authorities
gp = hyb_abema.setup()
#print ("Global params", gp)
#Instantiate a few authorities
#Attribute names must be globally unique.  HybridABEncMA
#Two authorities may not issue keys for the same attribute.
#Otherwise, the decryption algorithm will not know which private key to use
hospital_A_attributes = [
    'hospitalA.nurse', 'hospitalA.doctor', 'hospitalA.admin', 'hospitalA.researcher', 
    'hospitalA.cardiology', 'hospitalA.oncology', 'hospitalA.pharmacy', 'hospitalA.emergency',
    'hospitalA.clearance1', 'hospitalA.clearance2', 'hospitalA.clearance3'
]
hospital_B_attributes = [
    'hospitalB.nurse', 'hospitalB.doctor', 'hospitalB.admin', 'hospitalB.researcher', 
    'hospitalB.cardiology', 'hospitalB.oncology', 'hospitalB.pharmacy', 'hospitalB.emergency',
    'hospitalB.clearance_level1', 'hospitalB.clearance2', 'hospitalB.clearance3'
]
    
# Define insurance company attributes (with prefixes to avoid conflict)
insCoA_attributes = [
    'insCoA.underwriter', 'insCoA.claims_adjuster', 'insCoA.customer_service', 'insCoA.policy_admin',
    'insCoA.claims_processing', 'insCoA.policy_expertise_health', 'insCoA.policy_expertise_life', 
    'insCoA.clearance1', 'insCoA.clearance2', 'insCoA.clearance3'
]
insCoB_attributes = [
    'insCoB.underwriter', 'insCoB.claims_adjuster', 'insCoB.customer_service', 'insCoB.policy_admin',
    'insCoB.claims_processing', 'insCoB.policy_expertise_auto', 'insCoB.policy_expertise_property', 
    'insCoB.clearance1', 'insCoB.clearance2', 'insCoB.clearance3'
]
(hosASK, hosAPK) = hyb_abema.authsetup(gp, hospital_A_attributes)
(hosBSK, hosBPK) = hyb_abema.authsetup(gp, hospital_B_attributes)
(insASK, insAPK) = hyb_abema.authsetup(gp, insCoA_attributes)
(insBSK, insBPK) = hyb_abema.authsetup(gp, insCoB_attributes)
allAuthPK = {}; allAuthPK.update(hosAPK); allAuthPK.update(hosBPK); allAuthPK.update(insAPK); allAuthPK.update(insBPK)
#Setup a user with a few keys
bobs_gid = "user_12345"
K = {}
role_string = f"{hospital_prefix}.doctor"
department_string = f"{hospital_prefix}.oncology"
clearance_string = f"{hospital_prefix}.clearance2"
if hospital_choice == 'Hospital1':
    hospital_key = hosASK  # Use the correct key for Hospital1
elif hospital_choice == 'Hospital2':
    hospital_key = hosBSK  # Use the correct key for Hospital2
else:
    print("Invalid hospital choice.")
hyb_abema.keygen(gp, hospital_key,role_string, bobs_gid, K)
hyb_abema.keygen(gp, hospital_key,department_string, bobs_gid, K)
hyb_abema.keygen(gp, hospital_key,clearance_string, bobs_gid, K)
#file_path= '/home/bachar/DCSM/Hospital1/Patients/Patient_Files_Classified/Patient_PT98765_1.xml'

full_path = os.path.join(patient_path,"Plaindata",f"Patient_{patient_id}_1.xml")
print(full_path)
with open(full_path, 'rb') as file:
    xml_content = file.read()
print('Extracting the policy')
policy_path=os.path.join(base_path,f"Patient_{patient_id}", "Accesspolicy",f"Patient_{patient_id}_1.txt")
with open(policy_path, 'r') as file:
    policy=file.read()
print(policy)
    
msg = xml_content
size = len(msg)
print('test1')
policy_str = parse_policy_to_abe_format(policy,hospital_prefix)
print(policy_str)
#policy_str = "(hospitalA.doctor OR hospitalA.researcher) AND hospitalA.oncology"
ct = hyb_abema.encrypt(gp,allAuthPK, msg, policy_str)

if debug:
    print("Ciphertext")
    print("c1 =>", ct['c1'])
    print("c2 =>", ct['c2'])

orig_msg = hyb_abema.decrypt(gp, K, ct)
print("Result =>", orig_msg)
assert orig_msg == msg, "Failed Decryption!!!"
print("Successful Decryption!!!")
