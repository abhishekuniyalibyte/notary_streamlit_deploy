import json
import re
import unicodedata
from collections import defaultdict

INPUT_FILE = "customers_index.json"
OUTPUT_FILE = "certificate_types.json"

# Base legal certificate categories
BASE_KEYWORDS = [
    "firma",
    "firmas",
    "personeria",
    "representacion",
    "representación",
]

# Purposes (used as certificate types when no base keywords found)
PURPOSE_KEYWORDS = [
    "bps",
    "dgi",
    "bcu",
    "registro",
    "comercio",
    "zona franca",
]

ATTRIBUTE_KEYWORDS = [
    "domicilio",
    "domicilios",
    "objeto",
    "giro",
    "leyes",
    "estatutos",
    "poder",
    "poderes",
]


def normalize_text(text: str) -> str:
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def extract_base_type(name: str):
    found = []
    for key in BASE_KEYWORDS:
        base = normalize_text(key)
        if base in name:
            found.append(base)

    if found:
        return "_".join(sorted(found))

    return None


def extract_purposes(name: str):
    purposes = []

    # case 1: "para BPS", "para DGI" etc.
    if "para" in name:
        for key in PURPOSE_KEYWORDS:
            base = normalize_text(key)
            if base in name:
                purposes.append(base.replace(" ", "_"))

    # case 2: filenames like "Certificado BPS", "Certif.Anual DGI"
    for key in PURPOSE_KEYWORDS:
        base = normalize_text(key)
        if base in name and base not in purposes:
            purposes.append(base.replace(" ", "_"))

    return purposes


def extract_attributes(name: str):
    attrs = []
    for key in ATTRIBUTE_KEYWORDS:
        base = normalize_text(key)
        if base in name:
            attrs.append(base)
    return attrs


def determine_type(name: str):
    base = extract_base_type(name)
    purposes = extract_purposes(name)

    # Priority order:
    # 1. Base legal type (firma, personeria...)
    if base:
        return base

    # 2. Purpose-only type
    if purposes:
        # If multiple purposes appear, choose the first one
        return purposes[0]

    # 3. Unknown category
    return "otros"


def build_certificate_types(data):
    result = defaultdict(lambda: {
        "count": 0,
        "purposes": defaultdict(int),
        "attributes": set(),
        "examples": []
    })

    for customer in data.values():
        for cert in customer["files"]["certificates"]:
            filename = cert["filename"]
            name = normalize_text(filename)

            cert_type = determine_type(name)
            purposes = extract_purposes(name)
            attributes = extract_attributes(name)

            entry = result[cert_type]
            entry["count"] += 1

            for p in purposes:
                entry["purposes"][p] += 1

            for attr in attributes:
                entry["attributes"].add(attr)

            if len(entry["examples"]) < 5:
                entry["examples"].append(filename)

    # Clean up result
    final = {}
    for t, info in result.items():
        final[t] = {
            "count": info["count"],
            "purposes": dict(info["purposes"]),
            "attributes": list(info["attributes"]),
            "examples": info["examples"]
        }

    return final


if __name__ == "__main__":
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    result = build_certificate_types(data)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("Normalization complete.")
    print(f"Output written to {OUTPUT_FILE}")
