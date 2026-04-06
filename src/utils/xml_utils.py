"""
XML parsing and writing utilities for patient records.
"""
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_content_sections(xml_path):
    """
    Return ``{tag: text}`` for every child element inside ``<Content>``.

    Parameters
    ----------
    xml_path : str or Path
        Path to the patient XML file.

    Returns
    -------
    dict
        Empty dict if no ``<Content>`` element is found.
    """
    root = ET.parse(xml_path).getroot()
    content = root.find(".//Content")
    if content is None:
        return {}
    return {el.tag: el.text for el in content}


def add_security_label(xml_path, label, output_path):
    """
    Insert ``<SecurityLabel>{label}</SecurityLabel>`` as the first child of the
    root element and write the result to *output_path*.

    Parameters
    ----------
    xml_path : str or Path
        Source XML (plain patient data).
    label : str
        Security label to embed.
    output_path : str or Path
        Destination path (created if necessary).
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    elem = ET.Element("SecurityLabel")
    elem.text = label
    root.insert(0, elem)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(output_path), encoding="utf-8", xml_declaration=True)


def extract_data_attributes(xml_path):
    """
    Extract metadata fields from a *classified* XML file.

    Returns
    -------
    dict
        Keys: ``DataType``, ``Sensitivity``, ``Department``, ``Purpose``, ``Emergency``.
    """
    root = ET.parse(xml_path).getroot()

    def _text(tag):
        el = root.find(f".//{tag}")
        return el.text if el is not None else ""

    return {
        "DataType": _text("DataType"),
        "Sensitivity": _text("SecurityLabel"),
        "Department": _text("Department"),
        "Purpose": _text("Purpose"),
        "Emergency": _text("Emergency"),
    }


def format_attribute_prompt(attributes):
    """
    Format *attributes* dict into the GPT-2 input prompt template.

    Parameters
    ----------
    attributes : dict
        Output of :func:`extract_data_attributes`.

    Returns
    -------
    str
        Multi-line prompt ending with ``### Access Policy:\\n``.
    """
    return (
        f"Data Type: {attributes['DataType']}  \n"
        f"Sensitivity: {attributes['Sensitivity']}  \n"
        f"Department: {attributes['Department']}  \n"
        f"Purpose: {attributes['Purpose']}  \n"
        f"Emergency: {attributes['Emergency']}  \n"
        "### Access Policy:\n"
    )
