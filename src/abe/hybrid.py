"""
Hybrid Multi-Authority ABE.

Wraps a Dabe scheme so that arbitrary byte payloads (e.g. XML files) can be
encrypted.  The ABE layer encapsulates a random symmetric key; the payload is
encrypted with AES via Charm-Crypto's AuthenticatedCryptoAbstraction.
"""
from charm.core.math.pairing import hashPair as sha2
from charm.toolbox.ABEncMultiAuth import ABEncMultiAuth
from charm.toolbox.pairinggroup import GT
from charm.toolbox.symcrypto import AuthenticatedCryptoAbstraction


class HybridABEncMA(ABEncMultiAuth):
    """
    >>> from charm.toolbox.pairinggroup import PairingGroup
    >>> from src.abe.dabe import Dabe
    >>> group = PairingGroup('SS512')
    >>> hyb = HybridABEncMA(Dabe(group), group)
    >>> gp = hyb.setup()
    >>> (sk, pk) = hyb.authsetup(gp, ['auth.doctor', 'auth.researcher'])
    >>> user_keys = {}
    >>> hyb.keygen(gp, sk, 'auth.doctor', 'alice@example.com', user_keys)
    >>> msg = b'Sensitive patient record'
    >>> ct = hyb.encrypt(gp, pk, msg, '(auth.doctor OR auth.researcher)')
    >>> hyb.decrypt(gp, user_keys, ct) == msg
    True
    """

    def __init__(self, scheme, groupObj):
        self._scheme = scheme
        self._group = groupObj

    def setup(self):
        return self._scheme.setup()

    def authsetup(self, gp, attributes):
        return self._scheme.authsetup(gp, attributes)

    def keygen(self, gp, sk, attr, gid, pkey):
        return self._scheme.keygen(gp, sk, attr, gid, pkey)

    def encrypt(self, gp, pk, M, policy_str):
        if not isinstance(M, bytes):
            raise TypeError("Plaintext M must be bytes.")
        if not isinstance(policy_str, str):
            raise TypeError("policy_str must be a str.")
        key = self._group.random(GT)
        c1 = self._scheme.encrypt(gp, pk, key, policy_str)
        c2 = AuthenticatedCryptoAbstraction(sha2(key)).encrypt(M)
        return {"c1": c1, "c2": c2}

    def decrypt(self, gp, sk, ct):
        key = self._scheme.decrypt(gp, sk, ct["c1"])
        if key is False:
            raise Exception("ABE decryption failed — attributes do not satisfy policy.")
        return AuthenticatedCryptoAbstraction(sha2(key)).decrypt(ct["c2"])
