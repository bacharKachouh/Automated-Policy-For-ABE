"""
Authority key generation and persistence helpers.

NOTE on key serialization: Charm-Crypto pairing group elements cannot be
trivially round-tripped through JSON.  The save functions below serialise
elements as strings for inspection/auditing purposes only.  The live pipeline
regenerates keys in memory on each run.  A production deployment should use
Charm-Crypto's native serialisation (group.serialize / group.deserialize) and
store the raw bytes securely (e.g. in a HSM or encrypted key store).
"""
import json
from pathlib import Path

from charm.toolbox.pairinggroup import G1, pair
from charm.toolbox.secretutil import SecretUtil
from charm.toolbox.ABEncMultiAuth import ABEncMultiAuth


class AuthorityGeneration:
    """Generates global parameters and per-authority key pairs."""

    def __init__(self, groupObj):
        self._group = groupObj
        self._util = SecretUtil(groupObj, verbose=False)

    def setup(self):
        """Return global public parameters ``{'g': ..., 'H': ...}``."""
        g = self._group.random(G1)
        H = lambda x: self._group.hash(x, G1)  # noqa: E731
        return {"g": g, "H": H}

    def authsetup(self, GP, attributes):
        """
        Generate SK and PK for *attributes*.  Attribute names are stored as-is
        (no uppercasing) so that persisted keys remain human-readable.
        """
        SK, PK = {}, {}
        for attr in attributes:
            alpha_i, y_i = self._group.random(), self._group.random()
            SK[attr] = {"alpha_i": alpha_i, "y_i": y_i}
            PK[attr] = {
                "e(gg)^alpha_i": pair(GP["g"], GP["g"]) ** alpha_i,
                "g^y_i": GP["g"] ** y_i,
            }
        return SK, PK

    def keygen(self, gp, sk, attr, gid, pkey):
        h = gp["H"](gid)
        K = (gp["g"] ** sk[attr]["alpha_i"]) * (h ** sk[attr]["y_i"])
        pkey[attr] = {"k": K}
        pkey["gid"] = gid


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_secret_key(secret_key, path):
    """Serialise *secret_key* to *path* (string representation, audit only)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(secret_key, f, default=str, indent=2)
    print(f"Secret key saved → {path}")


def save_public_keys_and_gp(public_keys, gp, path):
    """Serialise public keys and global params to *path* (audit only)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "GP": {"g": str(gp["g"]), "H": str(gp["H"])},
        "public_keys": public_keys,
    }
    with open(path, "w") as f:
        json.dump(data, f, default=str, indent=2)
    print(f"Public keys + GP saved → {path}")


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)
