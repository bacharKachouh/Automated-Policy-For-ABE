#!/usr/bin/env python3
"""
One-time authority key generation for all hospitals and insurance companies.

Outputs
-------
- ``keys/public_keys_and_GP.json``       — global parameters + all public keys (audit)
- ``data/Hospital1/keys/hospital_A_SK.json``  — Hospital A secret key (audit)
- ``data/Hospital2/keys/hospital_B_SK.json``  — Hospital B secret key (audit)
- ``keys/InsCoA_SK.json``, ``keys/InsCoB_SK.json``  — insurance company secret keys

NOTE: The JSON files store string representations of Charm-Crypto group elements
and are for auditing / inspection only.  The live pipeline regenerates keys in
memory.  See ``src/abe/authority.py`` for the serialisation caveat.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from charm.toolbox.pairinggroup import PairingGroup

from src import config
from src.abe.authority import AuthorityGeneration, save_public_keys_and_gp, save_secret_key


def main():
    print("Generating ABE authority keys...")
    group = PairingGroup(config.PAIRING_GROUP)
    auth = AuthorityGeneration(group)
    gp = auth.setup()

    all_pk = {}

    for name, cfg in config.HOSPITAL_CONFIGS.items():
        sk, pk = auth.authsetup(gp, cfg["attributes"])
        save_secret_key(sk, cfg["sk_file"])
        all_pk[name] = pk

    for name, cfg in config.INSURANCE_CONFIGS.items():
        sk, pk = auth.authsetup(gp, cfg["attributes"])
        save_secret_key(sk, config.KEYS_DIR / f"{name}_SK.json")
        all_pk[name] = pk

    save_public_keys_and_gp(all_pk, gp, config.PUBLIC_KEYS_FILE)
    print("\nAuthority key generation complete.")


if __name__ == "__main__":
    main()
