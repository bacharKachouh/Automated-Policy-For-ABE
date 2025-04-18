from charm.toolbox.pairinggroup import PairingGroup,ZR,G1,G2,GT,pair
from charm.toolbox.secretutil import SecretUtil
from charm.toolbox.ABEncMultiAuth import ABEncMultiAuth
from charm.core.math.pairing import pairing
import time

debug = True
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
        
#        if(debug):
#            print("Key gen for %s on %s" % (gid, i))
#            print("H(GID): '%s'" % h)
#            print("K = g^alpha_i * H(GID) ^ y_i: %s" % K)
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
   
        print("e(gg)^s: %s" % egg_s)

        return ct['C0'] / egg_s

groupObj = PairingGroup('SS512')

l=[]

dabe = Dabe(groupObj)
start1 = time.time()
GP = dabe.setup()
l.append(time.time() - start1)

#Setup an authority
auth_attrs= ['ONE', 'TWO', 'THREE', 'FOUR']
(SK, PK) = dabe.authsetup(GP, auth_attrs)
if debug: print("Authority SK")
if debug: print(SK)

#Setup a user and give him some keys
gid, K = "bob", {}
usr_attrs = ['THREE', 'ONE', 'TWO']

start2 = time.time()
for i in usr_attrs: dabe.keygen(GP, SK, i, gid, K)
l.append(time.time() - start2)

print('User credential list: %s' % usr_attrs)
print("\nSecret key:")
    #if debug: groupObj.debug(K)''


    #Encrypt a random element in GT
message= "hello"
message_bytes = message.encode('utf-8')
print( 'intital',message_bytes)
m_int=int.from_bytes(message_bytes,byteorder='big')
print('initial_int',m_int)
m= group.encode(m_int)
#m = groupObj.random(GT)
print("intial message",m)
policy = '((one or three) and (TWO or FOUR))'
print('Acces Policy: %s' % policy)
start3 = time.time()
CT = dabe.encrypt(GP, PK, m, policy)
l.append(time.time() - start3)

#   ''' if debug: print("\nCiphertext...")
#   if debug: groupObj.debug(CT)

start4 = time.time()
orig_m = dabe.decrypt(GP, K, CT)
print("cipher=",orig_m)
m_int = orig_m
m_bytes = m_int.to_bytes( (m_int.bit_length()+7)//8, byteorder= 'big')
#m_bytes = orig_m.getBytes()
#serialized_bytes=group.serialize(orig_m)
#m_0 = int.from_bytes(serialized_bytes, byteorder= 'big')
m_f=m_bytes.decode('utf-8')
l.append(time.time() - start4)
#print('test',serialized_bytes)
#print('test_int', m_0)
assert m == orig_m, 'FAILED Decryption!!!'
print('Successful Decryption!')
assert message == m_f, 'FAILED Encoding!!!'
print('Successful Encoding')
  
