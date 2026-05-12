"""
Authority key generation (decoupled from persistence).

Persistence of the resulting SK/PK is handled by src.db.repo.
"""
from charm.toolbox.pairinggroup import G1, pair


class AuthorityGeneration:
    """Generates global parameters and per-authority key pairs."""

    def __init__(self, groupObj):
        self._group = groupObj

    def setup(self):
        g = self._group.random(G1)
        H = lambda x: self._group.hash(x, G1)  # noqa: E731
        return {"g": g, "H": H}

    def authsetup(self, GP, attributes):
        SK, PK = {}, {}
        for attr in attributes:
            alpha_i, y_i = self._group.random(), self._group.random()
            SK[attr] = {"alpha_i": alpha_i, "y_i": y_i}
            PK[attr] = {
                "e(gg)^alpha_i": pair(GP["g"], GP["g"]) ** alpha_i,
                "g^y_i": GP["g"] ** y_i,
            }
        return SK, PK
