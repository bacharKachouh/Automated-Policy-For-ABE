"""
Decentralised Attribute-Based Encryption — Lewko-Waters scheme.

Reference: "Decentralizing Attribute-Based Encryption" (Lewko & Waters, 2011).
Implemented on top of the Charm-Crypto pairing library.
"""
from charm.toolbox.ABEncMultiAuth import ABEncMultiAuth
from charm.toolbox.pairinggroup import G1, ZR, pair
from charm.toolbox.secretutil import SecretUtil


class Dabe(ABEncMultiAuth):
    """
    Multi-authority decentralised ABE.

    Attribute names are stored and matched case-insensitively (uppercased
    internally).  Policy strings use Charm-Crypto syntax, e.g.::

        "(hospitalA.doctor OR hospitalA.researcher) AND hospitalA.oncology"

    >>> from charm.toolbox.pairinggroup import PairingGroup, GT
    >>> group = PairingGroup('SS512')
    >>> dabe = Dabe(group)
    >>> gp = dabe.setup()
    >>> (sk, pk) = dabe.authsetup(gp, ['ONE', 'TWO', 'THREE', 'FOUR'])
    >>> user_keys = {}
    >>> for attr in ['THREE', 'ONE', 'TWO']:
    ...     dabe.keygen(gp, sk, attr, 'bob', user_keys)
    >>> msg = group.random(GT)
    >>> ct = dabe.encrypt(gp, pk, msg, '((one or three) and (TWO or FOUR))')
    >>> dabe.decrypt(gp, user_keys, ct) == msg
    True
    """

    def __init__(self, groupObj):
        ABEncMultiAuth.__init__(self)
        self._util = SecretUtil(groupObj, verbose=False)
        self._group = groupObj

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def setup(self):
        """Generate global public parameters."""
        g = self._group.random(G1)
        H = lambda x: self._group.hash(x, G1)  # noqa: E731  (hash oracle GID → G1)
        return {"g": g, "H": H}

    def authsetup(self, GP, attributes):
        """
        Generate secret key (SK) and public key (PK) for a set of attributes.

        Returns ``(SK, PK)`` where each is a dict keyed by uppercased attribute name.
        """
        SK, PK = {}, {}
        for attr in attributes:
            alpha_i, y_i = self._group.random(), self._group.random()
            SK[attr.upper()] = {"alpha_i": alpha_i, "y_i": y_i}
            PK[attr.upper()] = {
                "e(gg)^alpha_i": pair(GP["g"], GP["g"]) ** alpha_i,
                "g^y_i": GP["g"] ** y_i,
            }
        return SK, PK

    def keygen(self, gp, sk, attr, gid, pkey):
        """
        Issue an attribute key for *gid* on *attr* using authority secret key *sk*.
        The key is added in-place to *pkey*.
        """
        h = gp["H"](gid)
        K = (gp["g"] ** sk[attr.upper()]["alpha_i"]) * (h ** sk[attr.upper()]["y_i"])
        pkey[attr.upper()] = {"k": K}
        pkey["gid"] = gid
        return None

    def encrypt(self, gp, pk, M, policy_str):
        """
        Encrypt group element *M* under *policy_str*.

        *pk* must contain public keys for every attribute mentioned in the policy.
        """
        s = self._group.random()
        w = self._group.init(ZR, 0)
        egg_s = pair(gp["g"], gp["g"]) ** s
        C0 = M * egg_s
        C1, C2, C3 = {}, {}, {}

        policy = self._util.createPolicy(policy_str)
        sshares = dict(
            (x[0].getAttributeAndIndex(), x[1])
            for x in self._util.calculateSharesList(s, policy)
        )
        wshares = dict(
            (x[0].getAttributeAndIndex(), x[1])
            for x in self._util.calculateSharesList(w, policy)
        )

        for attr, s_share in sshares.items():
            k_attr = self._util.strip_index(attr)
            r_x = self._group.random()
            C1[attr] = (pair(gp["g"], gp["g"]) ** s_share) * (pk[k_attr]["e(gg)^alpha_i"] ** r_x)
            C2[attr] = gp["g"] ** r_x
            C3[attr] = (pk[k_attr]["g^y_i"] ** r_x) * (gp["g"] ** wshares[attr])

        return {"C0": C0, "C1": C1, "C2": C2, "C3": C3, "policy": policy_str}

    def decrypt(self, gp, sk, ct):
        """
        Decrypt ciphertext *ct* using user secret keys *sk*.

        Raises ``Exception`` if the user's attributes do not satisfy the policy.
        """
        usr_attribs = [k for k in sk if k != "gid"]
        policy = self._util.createPolicy(ct["policy"])
        pruned = self._util.prune(policy, usr_attribs)
        if pruned is False:
            raise Exception("Attributes do not satisfy the access policy.")
        coeffs = self._util.getCoefficients(policy)

        h_gid = gp["H"](sk["gid"])
        egg_s = 1
        for i in pruned:
            x = i.getAttributeAndIndex()
            y = i.getAttribute()
            num = ct["C1"][x] * pair(h_gid, ct["C3"][x])
            dem = pair(sk[y]["k"], ct["C2"][x])
            egg_s *= (num / dem) ** coeffs[x]

        return ct["C0"] / egg_s
