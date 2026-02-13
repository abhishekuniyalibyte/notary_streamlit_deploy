import argparse
import json
import os
import re
import unicodedata
import zipfile
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher
from typing import Dict, List, Tuple


def col_to_index(col: str) -> int:
    index = 0
    for ch in col:
        index = index * 26 + (ord(ch.upper()) - ord("A") + 1)
    return index


def read_xlsx_rows(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing XLSX file: {path}")

    with zipfile.ZipFile(path) as workbook:
        shared = []
        if "xl/sharedStrings.xml" in workbook.namelist():
            root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
            ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
            for si in root.findall("a:si", ns):
                texts = [t.text or "" for t in si.findall(".//a:t", ns)]
                shared.append("".join(texts))

        sheet_name = "xl/worksheets/sheet1.xml"
        root = ET.fromstring(workbook.read(sheet_name))
        ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

        rows = []
        for row in root.findall(".//a:sheetData/a:row", ns):
            cells = {}
            for cell in row.findall("a:c", ns):
                ref = cell.attrib.get("r")
                if not ref:
                    continue
                col = "".join(ch for ch in ref if ch.isalpha())
                cell_type = cell.attrib.get("t")
                value_node = cell.find("a:v", ns)
                value = ""
                if value_node is not None and value_node.text is not None:
                    value = value_node.text
                    if cell_type == "s":
                        try:
                            value = shared[int(value)]
                        except (ValueError, IndexError):
                            pass
                elif cell_type == "inlineStr":
                    text_node = cell.find("a:is/a:t", ns)
                    value = text_node.text if text_node is not None else ""

                cells[col] = value

            rows.append(cells)

    if not rows:
        return []

    header_row = rows[0]
    header_cols = sorted(header_row.keys(), key=col_to_index)
    headers = [header_row.get(c, "") for c in header_cols]
    header_norm = [h.strip().lower() for h in headers if h]

    result = []
    for row in rows[1:]:
        values = [row.get(c, "") for c in header_cols]
        if not any(str(v).strip() for v in values):
            continue
        row_norm = {headers[i]: str(values[i]).strip() for i in range(len(headers))}
        if header_norm and all(
            str(row_norm.get(h, "")).strip().lower() == h.strip().lower()
            for h in headers
            if h
        ):
            continue
        result.append(row_norm)

    return result


def normalize_for_match(value: str) -> str:
    value = value or ""
    value = unicodedata.normalize("NFD", value)
    value = "".join(ch for ch in value if unicodedata.category(ch) != "Mn")
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def list_files(root: str) -> List[str]:
    collected = []
    for base, _, files in os.walk(root):
        for filename in files:
            full_path = os.path.join(base, filename)
            rel_path = os.path.relpath(full_path, root)
            collected.append(rel_path)
    return collected


def build_file_index(paths: List[str]) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    basename_map: Dict[str, List[str]] = {}
    normalized_map: Dict[str, List[str]] = {}

    for rel_path in paths:
        basename = os.path.basename(rel_path)
        basename_lower = basename.lower()
        basename_map.setdefault(basename_lower, []).append(rel_path)

        base_no_ext = os.path.splitext(basename)[0]
        normalized = normalize_for_match(base_no_ext)
        normalized_map.setdefault(normalized, []).append(rel_path)

    return basename_map, normalized_map


def build_candidates(name: str, type_raw: str) -> List[str]:
    name = (name or "").strip().rstrip(" .")
    if not name:
        return []

    type_raw = (type_raw or "").strip().lower()
    ext = os.path.splitext(name)[1].lower().lstrip(".")
    candidates = [name]

    if type_raw == "word":
        for word_ext in ("docx", "doc", "pages"):
            if ext != word_ext:
                candidates.append(f"{name}.{word_ext}")
    elif type_raw:
        if ext != type_raw:
            candidates.append(f"{name}.{type_raw}")

    deduped = []
    seen = set()
    for candidate in candidates:
        if candidate.lower() in seen:
            continue
        seen.add(candidate.lower())
        deduped.append(candidate)
    return deduped


def score_match(candidate: str, basename_map: Dict[str, List[str]], normalized_map: Dict[str, List[str]]):
    candidate_base = os.path.basename(candidate)
    candidate_lower = candidate_base.lower()

    if candidate_lower in basename_map:
        return {
            "status": "exact",
            "score": 1.0,
            "matched_path": basename_map[candidate_lower][0],
            "method": "exact",
        }

    candidate_norm = normalize_for_match(os.path.splitext(candidate_base)[0])
    if candidate_norm in normalized_map:
        return {
            "status": "normalized",
            "score": 0.95,
            "matched_path": normalized_map[candidate_norm][0],
            "method": "normalized",
        }

    best_score = 0.0
    best_key = ""
    best_overlap_ratio = 0.0
    candidate_tokens = set(candidate_norm.split())

    for key in normalized_map.keys():
        key_tokens = set(key.split())
        overlap_ratio = 0.0
        if candidate_tokens:
            overlap_ratio = len(candidate_tokens & key_tokens) / len(candidate_tokens)
        score = SequenceMatcher(None, candidate_norm, key).ratio()
        if (overlap_ratio, score) > (best_overlap_ratio, best_score):
            best_overlap_ratio = overlap_ratio
            best_score = score
            best_key = key

    if best_key in normalized_map and best_score >= 0.7 and best_overlap_ratio >= 0.6:
        return {
            "status": "fuzzy",
            "score": round(best_score, 3),
            "matched_path": normalized_map[best_key][0],
            "method": "fuzzy",
        }

    return {
        "status": "missing",
        "score": 0.0,
        "matched_path": "",
        "method": "missing",
    }


def expected_extensions(type_raw: str) -> List[str]:
    type_raw = (type_raw or "").strip().lower()
    if not type_raw:
        return []
    if type_raw == "word":
        return ["doc", "docx", "pages"]
    return [type_raw]


def build_catalog_entries(rows: List[Dict[str, str]], file_paths: List[str]) -> List[Dict[str, str]]:
    basename_map, normalized_map = build_file_index(file_paths)
    entries = []

    for row in rows:
        row_norm = {k.strip().lower(): (v or "").strip() for k, v in row.items()}
        name = row_norm.get("file name", "")
        description = row_norm.get("file content", "")
        type_raw = row_norm.get("type", "")

        if not name:
            continue

        candidates = build_candidates(name, type_raw)
        best = {"status": "missing", "score": 0.0, "matched_path": "", "method": "missing"}
        best_candidate = ""
        for candidate in candidates:
            result = score_match(candidate, basename_map, normalized_map)
            if result["score"] > best["score"]:
                best = result
                best_candidate = candidate

        matched_extension = ""
        matched_filename = ""
        if best["matched_path"]:
            matched_filename = os.path.basename(best["matched_path"])
            matched_extension = os.path.splitext(matched_filename)[1].lstrip(".")

        expected_exts = expected_extensions(type_raw)
        type_mismatch = bool(
            matched_extension
            and expected_exts
            and matched_extension.lower() not in expected_exts
        )

        entries.append(
            {
                "raw_name": name,
                "description": description,
                "type_raw": type_raw,
                "expected_extensions": expected_exts,
                "candidate_filenames": candidates,
                "matched_path": best["matched_path"],
                "matched_filename": matched_filename,
                "matched_extension": matched_extension,
                "match_status": best["status"],
                "match_method": best["method"],
                "match_score": best["score"],
                "candidate_used": best_candidate,
                "type_mismatch": type_mismatch,
            }
        )

    return entries


def update_catalog_file(catalog_path: str, customer: str, source_file: str, entries: List[Dict[str, str]]):
    catalogs = {}
    if os.path.exists(catalog_path):
        with open(catalog_path, "r", encoding="utf-8") as handle:
            catalogs = json.load(handle)

    catalogs[customer] = {
        "source_file": source_file,
        "entry_count": len(entries),
        "entries": entries,
    }

    with open(catalog_path, "w", encoding="utf-8") as handle:
        json.dump(catalogs, handle, indent=2, ensure_ascii=False)


def update_customers_index(customers_path: str, customer: str, source_file: str, catalog_path: str, entry_count: int):
    with open(customers_path, "r", encoding="utf-8") as handle:
        customers = json.load(handle)

    if customer not in customers:
        raise KeyError(f"Customer '{customer}' not found in {customers_path}")

    catalogs = customers[customer].get("catalogs", [])
    if not isinstance(catalogs, list):
        catalogs = []

    updated = False
    for item in catalogs:
        if item.get("source_file") == source_file:
            item.update(
                {
                    "catalog_path": catalog_path,
                    "entry_count": entry_count,
                }
            )
            updated = True
            break

    if not updated:
        catalogs.append(
            {
                "source_file": source_file,
                "catalog_path": catalog_path,
                "entry_count": entry_count,
            }
        )

    customers[customer]["catalogs"] = catalogs

    with open(customers_path, "w", encoding="utf-8") as handle:
        json.dump(customers, handle, indent=2, ensure_ascii=False)


def update_summary(summary_path: str, customer: str, source_file: str, catalog_path: str, entry_count: int):
    with open(summary_path, "r", encoding="utf-8") as handle:
        summary = json.load(handle)

    external = summary.get("external_file_catalogs", {})
    external[customer] = {
        "source_file": source_file,
        "catalog_path": catalog_path,
        "entry_count": entry_count,
    }
    summary["external_file_catalogs"] = external

    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import client XLSX file catalog.")
    parser.add_argument("--xlsx", required=True, help="Path to the XLSX file.")
    parser.add_argument("--customer", required=True, help="Customer name.")
    parser.add_argument("--data-root", required=True, help="Customer data root folder.")
    parser.add_argument(
        "--catalog-path",
        default="client_file_catalogs.json",
        help="Output JSON file for catalogs.",
    )
    parser.add_argument(
        "--customers-index",
        default="customers_index.json",
        help="Path to customers_index.json.",
    )
    parser.add_argument(
        "--summary",
        default="certificate_summary.json",
        help="Path to certificate_summary.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rows = read_xlsx_rows(args.xlsx)
    if not rows:
        raise ValueError(f"No rows found in {args.xlsx}")

    customer_folder = os.path.join(args.data_root, args.customer)
    if not os.path.isdir(customer_folder):
        raise FileNotFoundError(f"Missing customer folder: {customer_folder}")

    file_paths = list_files(customer_folder)
    entries = build_catalog_entries(rows, file_paths)

    update_catalog_file(args.catalog_path, args.customer, os.path.basename(args.xlsx), entries)
    update_customers_index(
        args.customers_index,
        args.customer,
        os.path.basename(args.xlsx),
        args.catalog_path,
        len(entries),
    )
    update_summary(
        args.summary,
        args.customer,
        os.path.basename(args.xlsx),
        args.catalog_path,
        len(entries),
    )

    print(f"Catalog entries: {len(entries)}")
    print(f"Catalog file updated: {args.catalog_path}")


if __name__ == "__main__":
    main()
