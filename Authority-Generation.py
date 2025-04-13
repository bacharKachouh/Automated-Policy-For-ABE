import json
from charm.toolbox.pairinggroup import PairingGroup, G1, ZR, pair
from charm.core.math.pairing import hashPair as sha2
from charm.toolbox.ABEncMultiAuth import ABEncMultiAuth
from charm.core.math.pairing import pairing
from charm.toolbox.secretutil import SecretUtil
from charm.toolbox.symcrypto import AuthenticatedCryptoAbstraction
import base64
class AuthorityGeneration:
    def __init__(self, groupObj):
        self.group = groupObj
        self.util = SecretUtil(groupObj, verbose=False)  # Secret Sharing Scheme for policy parsing

    def setup(self):
        '''Global Setup'''
        g = self.group.random(G1)
        H = lambda x: self.group.hash(x, G1)  # Hash function for identities
        GP = {'g': g, 'H': H}
        return GP

    def authsetup(self, GP, entity, attributes):
        '''Authority Setup for a given set of attributes from hospitals or insurance companies'''
        SK = {}  # Dictionary of {attribute: {alpha_i, y_i}} 
        PK = {}  # Dictionary of {attribute: {e(g,g)^alpha_i, g^y_i}}
        
        for attr in attributes:
            alpha_i, y_i = self.group.random(), self.group.random()
            e_gg_alpha_i = pair(GP['g'], GP['g']) ** alpha_i
            g_y_i = GP['g'] ** y_i
            SK[attr] = {'alpha_i': alpha_i, 'y_i': y_i}
            PK[attr] = {'e(gg)^alpha_i': e_gg_alpha_i, 'g^y_i': g_y_i}
        
        return SK, PK

    def keygen(self, gp, sk, i, gid, pkey):
        '''Create a key for GID on attribute i belonging to authority sk'''
        h = gp['H'](gid)  # Get H(GID)
        K = (gp['g'] ** sk[i]['alpha_i']) * (h ** sk[i]['y_i'])
        pkey[i] = {'k': K}
        pkey['gid'] = gid

        return None

# Save public keys and GP to a single JSON file
def save_public_keys_and_GP_to_file(public_keys, gp, filename):
    # Convert the group element 'g' to a hexadecimal string before saving
    print(gp)
    g_str = str(gp['g'])  # Convert the pairing element to string
    H_str = str(gp['H'])  # You may need to customize this depending on how H is structured

    gp_serialized = {'g': g_str, 'H': H_str}
    # Now store both GP and PK
    data_to_save = {
        'GP': gp_serialized,
        'public_keys': public_keys
    }
    
    with open(filename, 'w') as file:
        json.dump(data_to_save, file, default=str)  # Default=str ensures we can store non-serializable objects

# Save each secret key to a separate JSON file
def save_secret_key_to_file(secret_key, filename):
    with open(filename, 'w') as file:
        json.dump(secret_key, file, default=str)

# Load keys from file
def load_keys_from_file(filename):
    with open(filename, 'r') as file:
        return json.load(file)

# Example Usage for Hospitals and Insurance Companies
def generate_authority_keys():
    groupObj = PairingGroup('SS512')
    authority_gen = AuthorityGeneration(groupObj)

    # Setup global parameters for both hospitals and insurance companies
    GP = authority_gen.setup()

    # Define hospital attributes (with prefixes to avoid conflict)
    hospital_A_attributes = [
        'hospitalA.nurse', 'hospitalA.doctor', 'hospitalA.admin', 'hospitalA.researcher', 
        'hospitalA.cardiology', 'hospitalA.oncology', 'hospitalA.pharmacy', 'hospitalA.emergency',
        'hospitalA.clearance_level_1', 'hospitalA.clearance_level_2', 'hospitalA.clearance_level_3'
    ]
    hospital_B_attributes = [
        'hospitalB.nurse', 'hospitalB.doctor', 'hospitalB.admin', 'hospitalB.researcher', 
        'hospitalB.cardiology', 'hospitalB.oncology', 'hospitalB.pharmacy', 'hospitalB.emergency',
        'hospitalB.clearance_level_1', 'hospitalB.clearance_level_2', 'hospitalB.clearance_level_3'
    ]
    
    # Define insurance company attributes (with prefixes to avoid conflict)
    insCoA_attributes = [
        'insCoA.underwriter', 'insCoA.claims_adjuster', 'insCoA.customer_service', 'insCoA.policy_admin',
        'insCoA.claims_processing', 'insCoA.policy_expertise_health', 'insCoA.policy_expertise_life', 
        'insCoA.clearance_level_1', 'insCoA.clearance_level_2', 'insCoA.clearance_level_3'
    ]
    insCoB_attributes = [
        'insCoB.underwriter', 'insCoB.claims_adjuster', 'insCoB.customer_service', 'insCoB.policy_admin',
        'insCoB.claims_processing', 'insCoB.policy_expertise_auto', 'insCoB.policy_expertise_property', 
        'insCoB.clearance_level_1', 'insCoB.clearance_level_2', 'insCoB.clearance_level_3'
    ]

    # Generate authority keys for hospitals
    hospital_A_SK, hospital_A_PK = authority_gen.authsetup(GP, "Hospital A", hospital_A_attributes)
    hospital_B_SK, hospital_B_PK = authority_gen.authsetup(GP, "Hospital B", hospital_B_attributes)

    # Generate authority keys for insurance companies
    insCoA_SK, insCoA_PK = authority_gen.authsetup(GP, "Insurance Co A", insCoA_attributes)
    insCoB_SK, insCoB_PK = authority_gen.authsetup(GP, "Insurance Co B", insCoB_attributes)

    # Save public keys and GP to a single file
    public_keys = {
        'hospital_A': hospital_A_PK,
        'hospital_B': hospital_B_PK,
        'insCoA': insCoA_PK,
        'insCoB': insCoB_PK
    }
    save_public_keys_and_GP_to_file(public_keys, GP, 'public_keys_and_GP.json')

    # Save secret keys for each authority in a separate file
    save_secret_key_to_file(hospital_A_SK, '/home/bachar/DCSM/Hospital1/hospital_A_SK.json')
    save_secret_key_to_file(hospital_B_SK, '/home/bachar/DCSM/Hospital2/hospital_A_SK.json')
    save_secret_key_to_file(insCoA_SK, '/home/bachar/DCSM/Insurance1/insCoA_SK.json')
    save_secret_key_to_file(insCoB_SK, '/home/bachar/DCSM/Insurance2/insCoB_SK.json')

    return public_keys

# Call the function to generate and store the keys
generated_public_keys = generate_authority_keys()

# Load the public keys and GP from the file
loaded_data = load_keys_from_file('public_keys_and_GP.json')
loaded_public_keys = loaded_data['public_keys']
loaded_GP = loaded_data['GP']

# Example to access hospital A's public keys
hospital_A_PK = loaded_public_keys['hospital_A']

# Example to load hospital A's secret key later
hospital_A_SK = load_keys_from_file('/home/bachar/DCSM/Hospital1/hospital_A_SK.json')

# Example to access Insurance Co A's public keys
insCoA_PK = loaded_public_keys['insCoA']

# Example to load Insurance Co A's secret key later
insCoA_SK = load_keys_from_file('/home/bachar/DCSM/Insurance1/insCoA_SK.json')

# You can now access all the keys from the loaded files
print("Hospital A's Public Key:", hospital_A_PK)
print("Hospital A's Secret Key:", hospital_A_SK)
print("Insurance Co A's Public Key:", insCoA_PK)
print("Insurance Co A's Secret Key:", insCoA_SK)

