import os
import json
from typing import Dict


BASE_DIR = "Notaria"
OUTPUT_FILE = "customers_index.json"


def classify_file(filename: str):
    name = filename.lower().strip()

    if name.startswith("error"):
        return "certificates", True

    if "certif" in name:
        return "certificates", False

    return "non_certificates", False


def scan_all_files(folder_path):
    """
    Recursively walks through ALL subfolders.
    Returns a flat list of file paths (relative).
    """
    collected = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            rel_path = os.path.relpath(os.path.join(root, file), folder_path)
            collected.append(rel_path)

    return collected


def index_notaria_folders(base_dir: str) -> Dict:
    data = {}

    for customer_name in sorted(os.listdir(base_dir)):
        customer_path = os.path.join(base_dir, customer_name)

        if not os.path.isdir(customer_path):
            continue

        data[customer_name] = {
            "customer_type": "unknown",
            "files": {
                "certificates": [],
                "non_certificates": []
            }
        }

        all_files = scan_all_files(customer_path)

        for rel_path in all_files:
            filename = os.path.basename(rel_path)

            category, error_flag = classify_file(filename)

            if category == "certificates":
                data[customer_name]["files"]["certificates"].append({
                    "filename": filename,
                    "relative_path": rel_path,
                    "error_flag": error_flag
                })
            else:
                data[customer_name]["files"]["non_certificates"].append({
                    "filename": filename,
                    "relative_path": rel_path
                })

    return data


if __name__ == "__main__":
    if not os.path.exists(BASE_DIR):
        raise FileNotFoundError(f"Base directory not found: {BASE_DIR}")

    index = index_notaria_folders(BASE_DIR)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    print("Indexing complete")
    print(f"Output written to: {OUTPUT_FILE}")
    print(f"Customers indexed: {len(index)}")
