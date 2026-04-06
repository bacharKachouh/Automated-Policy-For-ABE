"""
End-to-end pipeline: classification → attribute extraction →
policy generation → ABE encryption / decryption.
"""
from pathlib import Path

from charm.toolbox.pairinggroup import PairingGroup

from src import config
from src.abe.dabe import Dabe
from src.abe.hybrid import HybridABEncMA
from src.abe.policy_parser import parse_policy_to_abe_format
from src.classification.classifier import SecurityClassifier
from src.hospital.patient import get_patient_dir
from src.policy.extractor import PolicyExtractor
from src.utils.xml_utils import (
    add_security_label,
    extract_data_attributes,
    format_attribute_prompt,
    parse_content_sections,
)


def _build_abe_system():
    """
    Instantiate the Hybrid ABE system and generate authority key pairs.

    NOTE: Keys are generated fresh on every call.  This is acceptable for a
    demo but a production system should generate keys once, persist them with
    Charm-Crypto's native serialisation, and load them here.

    Returns
    -------
    tuple
        ``(hyb, gp, authority_keys, all_pk)``
    """
    group = PairingGroup(config.PAIRING_GROUP)
    hyb = HybridABEncMA(Dabe(group), group)
    gp = hyb.setup()

    all_pk = {}
    authority_keys = {}

    for name, cfg in config.HOSPITAL_CONFIGS.items():
        sk, pk = hyb.authsetup(gp, cfg["attributes"])
        authority_keys[name] = sk
        all_pk.update(pk)

    for name, cfg in config.INSURANCE_CONFIGS.items():
        sk, pk = hyb.authsetup(gp, cfg["attributes"])
        authority_keys[name] = sk
        all_pk.update(pk)

    return hyb, gp, authority_keys, all_pk


def run_pipeline(hospital_name, patient_id, xml_filename, user_gid, user_attributes):
    """
    Execute the full four-stage pipeline for a single patient record.

    Stages
    ------
    1. Security classification (BioClinicalBERT)
    2. Data attribute extraction
    3. Access policy generation (GPT-2)
    4. Hybrid ABE encryption + decryption verification

    Parameters
    ----------
    hospital_name : str
        ``"Hospital1"`` or ``"Hospital2"``.
    patient_id : str
        Patient ID, e.g. ``"PT12345"``.
    xml_filename : str
        XML filename inside ``Plaindata/``, e.g. ``"Patient_PT12345_1.xml"``.
    user_gid : str
        Global identity of the requesting user.
    user_attributes : list[str]
        Full attribute names the user holds, e.g.
        ``["hospitalA.doctor", "hospitalA.oncology", "hospitalA.clearance2"]``.

    Returns
    -------
    bool
        ``True`` if decryption succeeds and the round-trip is verified.
    """
    hospital_cfg = config.HOSPITAL_CONFIGS[hospital_name]
    prefix = hospital_cfg["prefix"]
    patient_dir = get_patient_dir(hospital_cfg["patients_dir"], patient_id)

    # -----------------------------------------------------------------------
    # Stage 1 — Security Classification
    # -----------------------------------------------------------------------
    print("\n[1/4] Classifying patient data...")
    plain_xml = patient_dir / "Plaindata" / xml_filename

    classifier = SecurityClassifier()
    sections = parse_content_sections(plain_xml)
    section_labels = classifier.classify_sections(sections)
    document_label = classifier.classify_document(section_labels)

    print(f"  Section labels : {section_labels}")
    print(f"  Document label : {document_label}")

    classified_xml = patient_dir / "Classifieddata" / xml_filename
    add_security_label(plain_xml, document_label, classified_xml)
    print(f"  Saved           → {classified_xml}")

    # -----------------------------------------------------------------------
    # Stage 2 — Data Attribute Extraction
    # -----------------------------------------------------------------------
    print("\n[2/4] Extracting data attributes...")
    attributes = extract_data_attributes(classified_xml)
    prompt = format_attribute_prompt(attributes)

    attr_file = patient_dir / "DataAttribute" / xml_filename.replace(".xml", ".txt")
    attr_file.parent.mkdir(parents=True, exist_ok=True)
    attr_file.write_text(prompt)
    print(f"  Saved → {attr_file}")

    # -----------------------------------------------------------------------
    # Stage 3 — Access Policy Generation
    # -----------------------------------------------------------------------
    print("\n[3/4] Generating access policy...")
    extractor = PolicyExtractor()
    policy_text = extractor.extract(prompt)
    print(f"  Policy text : {policy_text}")

    policy_file = patient_dir / "Accesspolicy" / xml_filename.replace(".xml", ".txt")
    policy_file.parent.mkdir(parents=True, exist_ok=True)
    policy_file.write_text(policy_text)
    print(f"  Saved → {policy_file}")

    # -----------------------------------------------------------------------
    # Stage 4 — ABE Encryption + Decryption
    # -----------------------------------------------------------------------
    print("\n[4/4] Running ABE encryption...")
    hyb, gp, authority_keys, all_pk = _build_abe_system()

    # Issue keys for the requesting user
    hospital_sk = authority_keys[hospital_name]
    user_keys = {}
    for attr in user_attributes:
        hyb.keygen(gp, hospital_sk, attr, user_gid, user_keys)

    # Encrypt the plain XML bytes
    xml_bytes = plain_xml.read_bytes()
    policy_str = parse_policy_to_abe_format(policy_text, prefix)
    if not policy_str:
        print("  WARNING: Could not derive an ABE policy string from the generated text.")
        return False

    print(f"  ABE policy : {policy_str}")
    ct = hyb.encrypt(gp, all_pk, xml_bytes, policy_str)

    # Verify round-trip
    try:
        decrypted = hyb.decrypt(gp, user_keys, ct)
        assert decrypted == xml_bytes, "Decrypted content does not match original!"
        print("  Decryption successful. Round-trip verified.")
        return True
    except Exception as exc:
        print(f"  Decryption failed: {exc}")
        return False
