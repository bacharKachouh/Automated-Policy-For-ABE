from charm.core.math.pairing import hashPair as sha2
from charm.toolbox.ABEncMultiAuth import ABEncMultiAuth
from charm.toolbox.pairinggroup import PairingGroup,GT,G1,ZR, pair
from charm.core.math.pairing import pairing
from charm.toolbox.secretutil import SecretUtil
from charm.toolbox.symcrypto import AuthenticatedCryptoAbstraction
import os
debug = False
class Dabe(ABEncMultiAuth):
    """
    Decentralized Attribute-Based Encryption by Lewko and Waters

    >>> group = PairingGroup('SS512')
    >>> dabe = Dabe(group)
    >>> public_parameters = dabe.setup()
    >>> auth_attrs= ['ONE', 'TWO', 'THREE', 'FOUR'] #setup an authority
    >>> (master_secret_key, master_public_key) = dabe.authsetup(public_parameters, auth_attrs)

        Setup a user and give him some keys
    >>> ID, secret_keys = "bob", {}
    >>> usr_attrs = ['THREE', 'ONE', 'TWO']
    >>> for i in usr_attrs:  dabe.keygen(public_parameters, master_secret_key, i, ID, secret_keys)
    >>> msg = group.random(GT)
    >>> policy = '((one or three) and (TWO or FOUR))'
    >>> cipher_text = dabe.encrypt(public_parameters, master_public_key, msg, policy)
    >>> decrypted_msg = dabe.decrypt(public_parameters, secret_keys, cipher_text)
    >>> decrypted_msg == msg
    True
    """
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
    """
    >>> from charm.toolbox.pairinggroup import PairingGroup,GT
    >>> group = PairingGroup('SS512')
    >>> dabe = Dabe(group)

        Setup master authority.
    >>> hyb_abema = HybridABEncMA(dabe, group)
    >>> global_parameters = hyb_abema.setup()

        Generate attributes for two different sub-authorities:
        Johns Hopkins University, and Johns Hopkins Medical Institutions.
    >>> jhu_attributes = ['jhu.professor', 'jhu.staff', 'jhu.student']
    >>> jhmi_attributes = ['jhmi.doctor', 'jhmi.nurse', 'jhmi.staff', 'jhmi.researcher']

        Johns Hopkins sub-authorities master key.
    >>> (jhu_secret_key, jhu_public_key) = hyb_abema.authsetup(global_parameters, jhu_attributes)

         JHMI sub-authorities master key
    >>> (jhmi_secret_key, jhmi_public_key) = hyb_abema.authsetup(global_parameters, jhmi_attributes)

        To encrypt messages we need all of the authorities' public keys.
    >>> allAuth_public_key = {}; 
    >>> allAuth_public_key.update(jhu_public_key); 
    >>> allAuth_public_key.update(jhmi_public_key)

        An example user, Bob, who is both a professor at JHU and a researcher at JHMI.
    >>> ID = "20110615 bob@gmail.com cryptokey"
    >>> secrets_keys = {}
    >>> hyb_abema.keygen(global_parameters, jhu_secret_key,'jhu.professor', ID, secrets_keys)
    >>> hyb_abema.keygen(global_parameters, jhmi_secret_key,'jhmi.researcher', ID, secrets_keys)

        Encrypt a message to anyone who is both a profesor at JHU and a researcher at JHMI.
    >>> msg = b'Hello World, I am a sensitive record!'
    >>> policy_str = "(jhmi.doctor or (jhmi.researcher and jhu.professor))"
    >>> cipher_text = hyb_abema.encrypt(global_parameters, allAuth_public_key, msg, policy_str)
    >>> hyb_abema.decrypt(global_parameters, secrets_keys, cipher_text)
    b'Hello World, I am a sensitive record!'
    """
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
        
def main():
    groupObj = PairingGroup('SS512')
    dabe = Dabe(groupObj)

    hyb_abema = HybridABEncMA(dabe, groupObj)

    #Setup global parameters for all new authorities
    gp = hyb_abema.setup()
    print ("Global params", gp)
    #Instantiate a few authorities
    #Attribute names must be globally unique.  HybridABEncMA
    #Two authorities may not issue keys for the same attribute.
    #Otherwise, the decryption algorithm will not know which private key to use
    jhu_attributes = ['jhu.professor', 'jhu.staff', 'jhu.student']
    jhmi_attributes = ['jhmi.doctor', 'jhmi.nurse', 'jhmi.staff', 'jhmi.researcher']
    (jhuSK, jhuPK) = hyb_abema.authsetup(gp, jhu_attributes)
    (jhmiSK, jhmiPK) = hyb_abema.authsetup(gp, jhmi_attributes)
    allAuthPK = {}; allAuthPK.update(jhuPK); allAuthPK.update(jhmiPK)

    #Setup a user with a few keys
    bobs_gid = "20110615 bob@gmail.com cryptokey"
    K = {}
    hyb_abema.keygen(gp, jhuSK,'jhu.professor', bobs_gid, K)
    hyb_abema.keygen(gp, jhmiSK,'jhmi.researcher', bobs_gid, K)
    file_path= '/home/bachar/DCSM/Hospital1/Patients/Patient_Files_Classified/Patient_PT98765_1.xml'
    with open(file_path, 'rb') as file:
        xml_content = file.read()

    msg = xml_content
    size = len(msg)
    policy_str = "(jhmi.doctor OR (jhmi.researcher AND jhu.professor))"
    ct = hyb_abema.encrypt(gp,allAuthPK, msg, policy_str)

    if debug:
        print("Ciphertext")
        print("c1 =>", ct['c1'])
        print("c2 =>", ct['c2'])

    orig_msg = hyb_abema.decrypt(gp, K, ct)
    if debug: print("Result =>", orig_msg)
    assert orig_msg == msg, "Failed Decryption!!!"
    if debug: print("Successful Decryption!!!")

if __name__ == "__main__":
    debug = True
    main()

