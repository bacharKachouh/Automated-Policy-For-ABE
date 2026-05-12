#!/usr/bin/env python3
"""
Generate the global pairing parameters and per-authority key pairs and
persist them to the SQLite database. Idempotent: re-running replaces
existing rows.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from charm.toolbox.pairinggroup import PairingGroup

from src import config
from src.abe.dabe import Dabe
from src.db import repo
from src.db.connection import init_db


def main():
    init_db()  # idempotent

    print(f"Initialising pairing group ({config.PAIRING_GROUP})...")
    group = PairingGroup(config.PAIRING_GROUP)
    dabe = Dabe(group)
    gp = dabe.setup()

    repo.save_global_params(group, gp, config.PAIRING_GROUP)
    print("Global params saved.")

    for name, cfg in config.HOSPITAL_CONFIGS.items():
        sk, pk = dabe.authsetup(gp, cfg["attributes"])
        repo.save_authority(group, name, "hospital", cfg["prefix"], sk, pk)
        print(f"  Authority saved: {name} ({len(cfg['attributes'])} attributes)")

    for name, cfg in config.INSURANCE_CONFIGS.items():
        sk, pk = dabe.authsetup(gp, cfg["attributes"])
        repo.save_authority(group, name, "insurance", cfg["prefix"], sk, pk)
        print(f"  Authority saved: {name} ({len(cfg['attributes'])} attributes)")


if __name__ == "__main__":
    main()
