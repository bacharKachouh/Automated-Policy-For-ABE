import pytest

from src.hospital.pipeline import resolve_authority


def test_resolves_hospital_a():
    assert resolve_authority("hospitalA.doctor") == "Hospital1"


def test_resolves_hospital_b():
    assert resolve_authority("hospitalB.nurse") == "Hospital2"


def test_resolves_insurance_a():
    assert resolve_authority("insCoA.underwriter") == "InsCoA"


def test_resolves_insurance_b():
    assert resolve_authority("insCoB.claims_adjuster") == "InsCoB"


def test_unknown_prefix_raises():
    with pytest.raises(KeyError):
        resolve_authority("unknownAuthority.doctor")


def test_malformed_attribute_raises():
    with pytest.raises(ValueError):
        resolve_authority("no_dot_here")
