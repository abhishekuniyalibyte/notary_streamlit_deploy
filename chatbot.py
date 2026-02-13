import csv
import html
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import unicodedata
import difflib
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import streamlit as st

from dotenv import load_dotenv
from groq import Groq

from src.phase1_certificate_intent import CertificateIntentCapture
from src.phase2_legal_requirements import LegalRequirementsEngine
from src.phase3_document_intake import DocumentIntake, DocumentTypeDetector, FileFormat
from src.phase4_text_extraction import (
    CollectionExtractionResult,
    DataExtractor,
    DocumentExtractionResult,
    ExtractedData,
    TextExtractor,
    TextNormalizer,
)
from src.phase5_legal_validation import LegalValidator
from src.phase6_gap_detection import ActionPriority, Gap, GapDetector, GapType
from src.error_pattern_analyzer import ErrorPatternAnalyzer
from src.document_writer import build_fixed_outputs_for_results, collect_fixed_downloads
from src.phase7_data_update import DataUpdater
from src.phase8_final_confirmation import FinalConfirmationEngine
from src.phase9_certificate_generation import CertificateGenerator
from src.phase10_notary_review import NotaryReviewSystem, ReviewStatus
from src.phase11_final_output import FinalOutputGenerator
from src.smart_auto_fix import SmartAutoFix, generate_correction_report


DEFAULT_SUMMARY_PATH = "certificate from dataset/certificate_summary.json"
DEFAULT_CATALOG_PATH = "certificate from dataset/client_file_catalogs.json"
DEFAULT_CERT_TYPE = "certificacion_de_firmas"
DEFAULT_PURPOSE = "para_bps"
DEFAULT_EXTRACTION_MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct"
DEFAULT_ANALYSIS_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
DEFAULT_QA_MODEL = DEFAULT_ANALYSIS_MODEL
MAX_LLM_CHARS = 3000
MAX_QA_CONTEXT_CHARS = 8000
MAX_QA_DOC_CHARS = 4000




def compute_case_status(results: Dict[str, Any]) -> Dict[str, Any]:
    file_results = results.get("file_results") or []
    error_count = sum(1 for fr in file_results if fr.get("has_error"))
    missing_docs = summarize_missing_documents(results.get("gap_structure") or {})

    # Check for semantic validation errors (consistency + expiry issues)
    gap_structure = results.get("gap_structure") or {}
    all_gaps = gap_structure.get("gaps", []) or []

    semantic_errors = [
        gap for gap in all_gaps
        if gap.get("gap_type") in ("semantic_error", "SEMANTIC_ERROR") and
        gap.get("priority") in ("urgent", "high", "URGENT", "HIGH")
    ]

    status_files = "ATTENTION REQUIRED" if error_count > 0 else "VALID"
    status_case = "ATTENTION REQUIRED" if (error_count > 0 or bool(missing_docs) or bool(semantic_errors)) else "VALID"

    return {
        "status_files": status_files,
        "status_case": status_case,
        "error_count": error_count,
        "missing_docs": missing_docs,
        "semantic_errors": semantic_errors,
        "semantic_error_count": len(semantic_errors),
        "total_files": len(file_results),
        "doc_types": sorted({fr.get("document_type", "unknown") for fr in file_results}),
    }


def render_case_status_panel(results: Dict[str, Any]) -> None:
    status = compute_case_status(results)
    status_files = status["status_files"]
    status_case = status["status_case"]
    missing_docs = status["missing_docs"] or []
    operation = results.get("operation", "create")

    st.subheader("Document analysis")

    # Show validation verdict for validation mode
    if operation == "validate":
        if status_case == "VALID":
            st.success("✅ **VALIDATION RESULT: CORRECT** - All documents are valid and complete")
        else:
            # Build specific error message
            issues = []
            if status["error_count"] > 0:
                issues.append(f"{status['error_count']} file error(s)")
            if status.get("semantic_error_count", 0) > 0:
                issues.append(f"{status['semantic_error_count']} data consistency/validity issue(s)")
            if missing_docs:
                issues.append(f"{len(missing_docs)} missing document(s)")

            issue_text = ", ".join(issues) if issues else "issues"
            st.error(f"❌ **VALIDATION RESULT: ISSUES FOUND** - {issue_text}")

    # Show detailed status
    if status_case == "VALID":
        if operation != "validate":
            st.success("No blocking issues found. Case looks complete for the selected request.")
    else:
        # Build dynamic message based on actual issues
        error_count = status["error_count"]
        total_files = status["total_files"]

        message_parts = []

        # What's working
        if total_files > 0 and error_count == 0:
            message_parts.append(f"✅ Your {total_files} uploaded file(s) are readable")
        elif total_files > 0:
            valid_count = total_files - error_count
            if valid_count > 0:
                message_parts.append(f"✅ {valid_count} of {total_files} files are readable")

        # What's wrong
        issues = []
        if error_count > 0:
            issues.append(f"{error_count} file(s) have processing errors")
        if missing_docs:
            issues.append(f"{len(missing_docs)} required document(s) are missing")
        if status.get("semantic_error_count", 0) > 0:
            issues.append(f"{status['semantic_error_count']} data consistency/validity issue(s)")

        if issues:
            message_parts.append(f"❌ {' and '.join(issues)}")

        # What to do (only for create mode, not validation)
        if operation != "validate":
            if missing_docs:
                message_parts.append("📤 Upload the missing documents listed below")
            if error_count > 0:
                message_parts.append("🔧 Fix or replace the files with errors (check details below)")
        else:
            # Validation mode - optional fix
            message_parts.append("💡 You can click 'Fix issues' below to attempt auto-correction (optional)")

        st.warning("\n\n".join(message_parts))

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Uploaded files status", status_files)
    col_b.metric("Case completeness status", status_case)
    col_c.metric("Files processed", str(status["total_files"]))

    if status["doc_types"]:
        st.caption(f"Document types found: {', '.join(status['doc_types'])}")

    if status["error_count"] > 0:
        st.error(f"File errors found: {status['error_count']} file(s) had processing errors.")

    if missing_docs:
        st.warning("Missing required documents (urgent):")
        for doc in missing_docs[:25]:
            st.write(f"- {doc}")

    # Show semantic validation issues (Phase 1: consistency + expiry)
    gap_structure = results.get("gap_structure") or {}
    all_gaps = gap_structure.get("gaps", []) or []

    consistency_issues = [
        gap for gap in all_gaps
        if "Inconsistencia" in gap.get("title", "") or "inconsistente" in gap.get("title", "").lower()
    ]
    expiry_issues = [
        gap for gap in all_gaps
        if "vencido" in gap.get("title", "").lower() or "vencer" in gap.get("title", "").lower()
    ]

    if consistency_issues:
        st.error("⚠️ **Data Consistency Issues Found:**")
        for gap in consistency_issues[:5]:
            st.write(f"**{gap.get('title')}**")
            st.caption(gap.get('description', ''))

    if expiry_issues:
        urgent_expiry = [g for g in expiry_issues if g.get("priority") == "urgent"]
        warning_expiry = [g for g in expiry_issues if g.get("priority") != "urgent"]

        if urgent_expiry:
            st.error("⏰ **Expired Documents Found:**")
            for gap in urgent_expiry[:5]:
                st.write(f"**{gap.get('title')}**")
                st.caption(gap.get('description', ''))

        if warning_expiry:
            st.warning("⚠️ **Documents Expiring Soon:**")
            for gap in warning_expiry[:5]:
                st.write(f"**{gap.get('title')}**")
                st.caption(gap.get('description', ''))

    file_results = results.get("file_results") or []
    if file_results:
        rows: List[Dict[str, Any]] = []
        for fr in file_results[:200]:
            rows.append(
                {
                    "filename": fr.get("filename") or "",
                    "type": fr.get("document_type") or "",
                    "validation": str(fr.get("validation", {}).get("status") or "").upper(),
                    "has_error": bool(fr.get("has_error")),
                }
            )
        st.dataframe(rows, width="stretch", hide_index=True)


def save_text_output(output_dir: Path, filename: str, content: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = (output_dir / filename).resolve()
    if content and not content.endswith("\n"):
        content = content + "\n"
    out_path.write_text(content or "", encoding="utf-8")
    return out_path


def get_default_option(options: List[Dict[str, str]], value: str) -> Dict[str, str]:
    for option in options:
        if option["value"] == value:
            return option
    return options[0] if options else {"value": value, "label": value}


def normalize_text(value: str) -> str:
    if not value:
        return ""
    value = unicodedata.normalize("NFKD", value)
    value = value.encode("ascii", "ignore").decode("ascii")
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def normalize_customer_key(value: str) -> str:
    value_norm = normalize_text(value)
    if not value_norm:
        return ""
    suffixes = {"sa", "srl", "ltda", "sas"}
    tokens = [token for token in value_norm.split() if token not in suffixes]
    return " ".join(tokens).strip()


def infer_catalog_customer(subject_name: str, catalog_customers: List[str]) -> Optional[str]:
    if not subject_name or not catalog_customers:
        return None
    target = normalize_customer_key(subject_name)
    for customer in catalog_customers:
        if normalize_customer_key(customer) == target:
            return customer
    return None


def parse_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    text = text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    candidate = match.group(0) if match else text
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if isinstance(data, dict) and "status" not in data:
        data["status"] = "ok"
    return data


@st.cache_data(show_spinner=False)
def load_client_catalog(path: str) -> Dict[str, Any]:
    if not path:
        return {}
    catalog_path = Path(path)
    if not catalog_path.exists():
        return {}
    with open(catalog_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_text_for_llm(file_path: str) -> str:
    try:
        document = DocumentIntake.process_file(file_path)
        text, _, _ = extract_text_without_ocr(document)
        return text
    except Exception:
        return ""


def truncate_text(value: str, limit: int) -> str:
    if not value:
        return ""
    if len(value) <= limit:
        return value
    truncated = value[:limit]
    if " " in truncated:
        trimmed = truncated.rsplit(" ", 1)[0]
        if trimmed:
            truncated = trimmed
    return f"{truncated}..."


def coerce_optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned or cleaned.lower() in ("none", "null"):
            return None
        return cleaned
    cleaned = str(value).strip()
    if not cleaned or cleaned.lower() in ("none", "null"):
        return None
    return cleaned


def ensure_list(value: Any) -> List[str]:
    if not value:
        return []
    if isinstance(value, list):
        cleaned_items = []
        for item in value:
            cleaned = str(item).strip()
            if not cleaned or cleaned.lower() in ("none", "null"):
                continue
            cleaned_items.append(cleaned)
        return cleaned_items
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned or cleaned.lower() in ("none", "null"):
            return []
        return [cleaned]
    cleaned = str(value).strip()
    if not cleaned or cleaned.lower() in ("none", "null"):
        return []
    return [cleaned]


def format_confidence(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{value:.2f}"
    return str(value)


def build_document_type_detail(
    selected_type: str,
    type_source: str,
    detected_type: Optional[str],
    llm_result: Optional[Dict[str, Any]],
    keyword_result: Optional[Dict[str, Any]],
) -> str:
    llm_type = llm_result.get("certificate_type") if llm_result else None
    llm_conf = llm_result.get("confidence") if llm_result else None
    llm_status = llm_result.get("status") if llm_result else None
    keyword_type = keyword_result.get("certificate_type") if keyword_result else None
    keyword_conf = keyword_result.get("confidence") if keyword_result else None
    keyword_status = keyword_result.get("status") if keyword_result else None

    return (
        f"selected={selected_type}; source={type_source}; "
        f"detected={detected_type or 'n/a'}; "
        f"llm={llm_type or 'n/a'} ({format_confidence(llm_conf)}, {llm_status or 'n/a'}); "
        f"keyword={keyword_type or 'n/a'} ({format_confidence(keyword_conf)}, {keyword_status or 'n/a'})"
    )


def build_detected_type_detail(
    detected_type: Optional[str],
    catalog_info: Optional[Dict[str, Any]],
) -> str:
    parts = []
    if detected_type:
        parts.append(detected_type)
    if catalog_info:
        description = catalog_info.get("description")
        if description:
            parts.append(description)
    return " - ".join(parts)


def collect_error_reasons(
    validation_status: str,
    extraction_success: Optional[bool],
    extraction_error: Optional[str],
    llm_extraction_error: Optional[str],
    text_extraction_error: Optional[str],
    ocr_error: Optional[str],
    processing_status: Optional[str],
) -> List[str]:
    reasons = []
    if processing_status == "error":
        reasons.append("processing_status=error")
    if extraction_success is False:
        reasons.append("extraction_success=false")
    if extraction_error:
        reasons.append(f"extraction_error: {extraction_error}")
    if llm_extraction_error:
        reasons.append(f"llm_extraction_error: {llm_extraction_error}")
    if text_extraction_error:
        reasons.append(f"text_extraction_error: {text_extraction_error}")
    if ocr_error:
        reasons.append(f"ocr_error: {ocr_error}")
    if validation_status == "invalid":
        reasons.append("validation_status=invalid")
    return reasons


def format_has_error_flag(has_error: Optional[bool]) -> str:
    return "yes" if has_error else "no"


def format_has_error_detail(has_error: Optional[bool], error_reasons: List[str]) -> str:
    if not has_error:
        return "no"
    if error_reasons:
        return f"yes: {' | '.join(error_reasons)}"
    return "yes"


def is_zip_file(file_path: Path) -> bool:
    try:
        with open(file_path, "rb") as handle:
            return handle.read(4) == b"PK\x03\x04"
    except OSError:
        return False


def extract_text_from_doc_via_libreoffice(file_path: Path) -> Tuple[str, Optional[str]]:
    soffice_path = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice_path:
        return "", "LibreOffice is required to extract legacy .doc files."

    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            soffice_path,
            "--headless",
            "--convert-to",
            "txt:Text",
            "--outdir",
            tmpdir,
            str(file_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            stderr = result.stderr.strip() or result.stdout.strip()
            return "", f"LibreOffice conversion failed: {stderr or 'unknown error'}"

        output_path = Path(tmpdir) / f"{file_path.stem}.txt"
        if not output_path.exists():
            txt_files = list(Path(tmpdir).glob("*.txt"))
            if not txt_files:
                return "", "LibreOffice conversion succeeded but no output file found."
            output_path = txt_files[0]

        try:
            return output_path.read_text(encoding="utf-8", errors="ignore"), None
        except OSError as exc:
            return "", f"Failed to read converted text: {exc}"


def extract_text_without_ocr(document) -> Tuple[str, str, Optional[str]]:
    file_path = Path(document.file_path)
    file_format = getattr(document, "file_format", None)
    format_value = file_format.value if file_format else file_path.suffix.lower().lstrip(".")

    if format_value == "txt":
        try:
            return TextExtractor.extract_from_text_file(file_path), "text", None
        except Exception as exc:
            return "", "text", f"Text extraction failed: {exc}"
    if format_value == "pdf":
        try:
            from PyPDF2 import PdfReader
        except Exception as exc:
            return "", "text", f"PyPDF2 is required for PDF extraction: {exc}"
        try:
            reader = PdfReader(str(file_path))
            text_chunks = [page.extract_text() or "" for page in reader.pages]
            return "\n".join(text_chunks).strip(), "text", None
        except Exception as exc:
            return "", "text", f"PDF extraction failed: {exc}"
    if format_value == "docx":
        try:
            return TextExtractor.extract_from_docx(file_path), "text", None
        except Exception as exc:
            return "", "text", f"DOCX extraction failed: {exc}"
    if format_value == "doc":
        if is_zip_file(file_path):
            try:
                return TextExtractor.extract_from_docx(file_path), "text", None
            except Exception as exc:
                return "", "text", f"DOCX extraction failed: {exc}"

        text, error = extract_text_from_doc_via_libreoffice(file_path)
        if text:
            return text, "text", None
        return "", "text", (error or "Legacy .doc extraction failed; convert to .docx or enable LibreOffice.")

    return "", "none", f"Unsupported file format: {format_value or 'unknown'}"


def call_groq_extraction(
    model: str,
    api_key: str,
    doc_text: str,
    filename: str,
) -> Dict[str, Any]:
    if not api_key:
        return {"status": "error", "message": "Missing GROQ_API_KEY."}
    if not doc_text.strip():
        return {"status": "error", "message": "No text provided for LLM extraction."}

    client = Groq(api_key=api_key)
    prompt = (
        "You extract structured data from Uruguayan notarial documents.\n"
        "Reply with JSON only (no Markdown).\n"
        "Keys:\n"
        "company_name, rut, ci, registro_comercio, acta_number, padron_bps, dates, emails,\n"
        "company_constitution_date (fecha de constitución de la empresa),\n"
        "statute_approval_date (fecha de aprobación del estatuto),\n"
        "registration_date (fecha de registro),\n"
        "acta_date (fecha del acta de directorio),\n"
        "registry_number (número de registro).\n"
        "Use null when missing and [] for lists.\n"
        "For dates, use format YYYY-MM-DD or the format found in the document.\n\n"
        f"Filename: {filename}\n\n"
        "Document text:\n"
        f"{doc_text[:MAX_LLM_CHARS]}\n"
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a precise data extraction engine."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )
        content = response.choices[0].message.content or ""
        parsed = parse_json_from_text(content)
        if parsed is not None:
            return parsed
        return {
            "status": "error",
            "message": "LLM did not return valid JSON.",
            "raw": content,
        }
    except Exception as exc:
        return {"status": "error", "message": f"Groq request failed: {exc}"}


def apply_llm_fields(extracted_data: ExtractedData, llm_payload: Dict[str, Any]) -> None:
    extracted_data.company_name = coerce_optional_str(llm_payload.get("company_name")) or extracted_data.company_name
    extracted_data.rut = coerce_optional_str(llm_payload.get("rut")) or extracted_data.rut
    extracted_data.ci = coerce_optional_str(llm_payload.get("ci")) or extracted_data.ci
    extracted_data.registro_comercio = (
        coerce_optional_str(llm_payload.get("registro_comercio"))
        or extracted_data.registro_comercio
    )
    extracted_data.acta_number = coerce_optional_str(llm_payload.get("acta_number")) or extracted_data.acta_number
    extracted_data.padron_bps = coerce_optional_str(llm_payload.get("padron_bps")) or extracted_data.padron_bps

    # Specific date fields for validation
    extracted_data.company_constitution_date = coerce_optional_str(llm_payload.get("company_constitution_date")) or extracted_data.company_constitution_date
    extracted_data.statute_approval_date = coerce_optional_str(llm_payload.get("statute_approval_date")) or extracted_data.statute_approval_date
    extracted_data.registration_date = coerce_optional_str(llm_payload.get("registration_date")) or extracted_data.registration_date
    extracted_data.acta_date = coerce_optional_str(llm_payload.get("acta_date")) or extracted_data.acta_date
    extracted_data.registry_number = coerce_optional_str(llm_payload.get("registry_number")) or extracted_data.registry_number

    dates = ensure_list(llm_payload.get("dates"))
    if dates:
        extracted_data.dates = dates
    emails = ensure_list(llm_payload.get("emails"))
    if emails:
        extracted_data.emails = emails


def apply_regex_fallback(extracted_data: ExtractedData, normalized_text: str) -> None:
    if not normalized_text:
        return
    if not extracted_data.company_name:
        extracted_data.company_name = DataExtractor.extract_company_name(normalized_text)
    if not extracted_data.rut:
        extracted_data.rut = DataExtractor.extract_rut(normalized_text)
    if not extracted_data.ci:
        extracted_data.ci = DataExtractor.extract_ci(normalized_text)
    if not extracted_data.registro_comercio:
        extracted_data.registro_comercio = DataExtractor.extract_registro_comercio(normalized_text)
    if not extracted_data.acta_number:
        extracted_data.acta_number = DataExtractor.extract_acta_number(normalized_text)
    if not extracted_data.padron_bps:
        extracted_data.padron_bps = DataExtractor.extract_padron_bps(normalized_text)
    if not extracted_data.dates:
        extracted_data.dates = DataExtractor.extract_dates(normalized_text)
    if not extracted_data.emails:
        extracted_data.emails = DataExtractor.extract_emails(normalized_text)


def process_collection_with_llm(
    collection,
    llm_settings: Dict[str, str],
) -> CollectionExtractionResult:
    result = CollectionExtractionResult(collection=collection)

    for document in collection.documents:
        raw_text = ""
        base_method = "none"
        base_error = None
        ocr_used = False
        ocr_error = None

        try:
            raw_text, base_method, base_error = extract_text_without_ocr(document)
        except Exception as exc:
            base_error = str(exc)

        if not raw_text and llm_settings.get("ocr_fallback"):
            file_format = getattr(document, "file_format", None)
            if file_format in (FileFormat.PDF, FileFormat.JPG, FileFormat.JPEG, FileFormat.PNG):
                try:
                    raw_text, extraction_method = TextExtractor.extract_text(document)
                    base_method = extraction_method
                    ocr_used = extraction_method == "ocr"
                except Exception as exc:
                    ocr_error = str(exc)

        normalized_text = TextNormalizer.normalize_text(raw_text or "")

        # If filename-based detection failed, try content-based detection so users can name files anything.
        if not document.detected_type and normalized_text:
            inferred_type = DocumentTypeDetector.detect_from_text(normalized_text)
            if inferred_type:
                document.detected_type = inferred_type
                if isinstance(document.metadata, dict):
                    document.metadata["detected_type_source"] = "content"

        extraction_method = base_method if base_method in ("text", "ocr") else "none"
        extracted_data = ExtractedData(
            document_type=document.detected_type,
            raw_text=raw_text or "",
            normalized_text=normalized_text,
            extraction_method=extraction_method,
        )

        if not raw_text:
            extracted_data.additional_fields["extraction_warning"] = (
                "No text extracted. Enable OCR fallback to read scanned documents."
            )
            extracted_data.confidence = 0.2
        else:
            extracted_data.confidence = 0.7 if llm_settings.get("enabled") else 1.0
        if base_error:
            extracted_data.additional_fields["text_extraction_error"] = base_error
        if ocr_error:
            extracted_data.additional_fields["ocr_error"] = ocr_error
        if ocr_used:
            extracted_data.additional_fields["ocr_used"] = True

        llm_payload = None
        if llm_settings.get("enabled") and raw_text:
            llm_payload = call_groq_extraction(
                model=llm_settings.get("extraction_model", DEFAULT_EXTRACTION_MODEL),
                api_key=llm_settings.get("api_key", ""),
                doc_text=raw_text,
                filename=document.file_name,
            )
            if llm_payload.get("status") == "error":
                extracted_data.additional_fields["llm_extraction_error"] = llm_payload.get("message")
            else:
                apply_llm_fields(extracted_data, llm_payload)
                extracted_data.additional_fields["llm_extraction"] = llm_payload

        apply_regex_fallback(extracted_data, normalized_text)

        result.extraction_results.append(
            DocumentExtractionResult(
                document=document,
                extracted_data=extracted_data,
                success=True,
            )
        )

    return result


def normalize_purpose(value: str) -> str:
    if not value:
        return ""
    value = value.lower()
    value = value.replace("para_", "").replace("_", " ")
    return normalize_text(value)


def make_filename_keys(filename: str) -> List[str]:
    if not filename:
        return []
    path = Path(filename)
    keys = {
        normalize_text(path.name),
        normalize_text(path.stem),
    }
    return [key for key in keys if key]


@st.cache_data(show_spinner=False)
def build_llm_reference(summary: Dict[str, Any]) -> Dict[str, Any]:
    reference = {}
    for cert_type, info in summary.get("identified_certificate_types", {}).items():
        reference[cert_type] = {
            "count": info.get("count", 0),
            "purposes": list(info.get("purposes", {}).keys()),
            "attributes": info.get("attributes", []),
            "examples": info.get("examples", [])[:3],
        }
    return reference


def keyword_classification(doc_text: str, summary_reference: Dict[str, Any]) -> Dict[str, Any]:
    text_norm = normalize_text(doc_text)
    if not text_norm:
        return {"status": "error", "message": "No text for keyword classification."}

    best_type = None
    best_score = 0
    matched_attrs: List[str] = []

    for cert_type, info in summary_reference.items():
        attrs = [normalize_text(attr) for attr in info.get("attributes", [])]
        score = 0
        hits = []
        for attr in attrs:
            if attr and attr in text_norm:
                score += 1
                hits.append(attr)
        if score > best_score:
            best_score = score
            best_type = cert_type
            matched_attrs = hits

    detected_purpose = ""
    for cert_type, info in summary_reference.items():
        for purpose in info.get("purposes", []):
            purpose_norm = normalize_purpose(purpose)
            if purpose_norm and purpose_norm in text_norm:
                detected_purpose = purpose
                break
        if detected_purpose:
            break

    if best_type and best_score > 0:
        confidence = min(0.8, 0.2 + 0.1 * best_score)
        return {
            "status": "ok",
            "is_certificate": True,
            "certificate_type": best_type,
            "purpose": detected_purpose or "",
            "confidence": confidence,
            "reason": f"Matched attributes: {', '.join(matched_attrs)}" if matched_attrs else "Matched attributes.",
        }

    return {"status": "error", "message": "No keyword match found."}


def map_summary_type_to_intent(cert_type: str) -> str:
    cert_norm = normalize_text(cert_type)
    if "firma" in cert_norm:
        return "certificacion_de_firmas"
    if "personeria" in cert_norm:
        return "certificado_de_personeria"
    if "representacion" in cert_norm:
        return "certificado_de_representacion"
    if "vigencia" in cert_norm:
        return "certificado_de_vigencia"
    if "situacion" in cert_norm or "juridica" in cert_norm:
        return "certificado_de_situacion_juridica"
    if "poder" in cert_norm:
        return "poder_general"
    return "otros"


def map_summary_purpose_to_intent(purpose: str) -> str:
    if not purpose:
        return "otros"
    purpose_norm = normalize_text(purpose)
    if purpose.lower().startswith("para_"):
        return purpose.lower()

    mapping = {
        "bps": "para_bps",
        "abitab": "para_abitab",
        "dgi": "para_dgi",
        "zona franca": "para_zona_franca",
        "zona_franca": "para_zona_franca",
        "zonafranca": "para_zona_franca",
        "msp": "para_msp",
        "ute": "para_ute",
        "antel": "para_antel",
        "rupe": "para_rupe",
        "mef": "para_mef",
        "imm": "para_imm",
        "mtop": "para_mtop",
        "migraciones": "para_migraciones",
        "banco": "para_banco",
        "compraventa": "para_compraventa",
        "base de datos": "para_base_datos",
        "base datos": "para_base_datos",
    }

    for key, value in mapping.items():
        if key in purpose_norm:
            return value
    return "otros"


def is_positive_classification(result: Optional[Dict[str, Any]]) -> bool:
    if not result or result.get("status") == "error":
        return False
    if result.get("is_certificate") is False:
        return False
    cert_type = result.get("certificate_type")
    if not cert_type or cert_type == "non_certificate":
        return False
    return True


def choose_classification(
    llm_result: Optional[Dict[str, Any]],
    keyword_result: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if is_positive_classification(llm_result):
        return llm_result
    if is_positive_classification(keyword_result):
        return keyword_result
    return None


def is_result_ok(result: Optional[Dict[str, Any]]) -> bool:
    return bool(result) and result.get("status") != "error"


def is_keyword_ok(result: Optional[Dict[str, Any]]) -> bool:
    return bool(result) and result.get("status") == "ok"


def dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    ordered = []
    for item in items:
        if not item or item in seen:
            continue
        ordered.append(item)
        seen.add(item)
    return ordered


def detect_classification_conflict(
    llm_result: Optional[Dict[str, Any]],
    keyword_result: Optional[Dict[str, Any]],
) -> Optional[str]:
    if not is_result_ok(llm_result) or not is_keyword_ok(keyword_result):
        return None
    llm_is_cert = llm_result.get("is_certificate")
    keyword_is_cert = keyword_result.get("is_certificate")
    if llm_is_cert is False and keyword_is_cert is True:
        return "Conflicto: LLM indica no certificado, keywords indican certificado."
    if llm_is_cert is True and keyword_is_cert is True:
        llm_type = llm_result.get("certificate_type")
        keyword_type = keyword_result.get("certificate_type")
        if llm_type and keyword_type and llm_type != keyword_type:
            return "Conflicto: tipos de certificado difieren entre LLM y keywords."
    return None


def build_review_reasons(
    llm_result: Optional[Dict[str, Any]],
    keyword_result: Optional[Dict[str, Any]],
    match_result: Optional[Dict[str, Any]],
) -> List[str]:
    reasons: List[str] = []

    conflict = detect_classification_conflict(llm_result, keyword_result)
    if conflict:
        reasons.append(conflict)

    if not is_result_ok(llm_result) and not is_keyword_ok(keyword_result):
        reasons.append("No se pudo clasificar por contenido.")

    if match_result:
        status = match_result.get("status")
        if status == "needs_review":
            reasons.append(match_result.get("reason") or "Requiere verificación manual.")
        elif status == "not_found":
            reasons.append("No se encontró coincidencia en certificate_summary.json.")

    return dedupe_preserve_order(reasons)


def build_review_queue(per_file_data: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    queue = []
    for item in per_file_data.values():
        if not item.get("review_required"):
            continue
        queue.append(
            {
                "filename": item.get("filename", "documento"),
                "reasons": item.get("review_reasons", []) or [],
            }
        )
    return queue


def build_review_gaps(review_queue: List[Dict[str, Any]]) -> List[Gap]:
    gaps = []
    for entry in review_queue:
        filename = entry.get("filename", "documento")
        reasons = entry.get("reasons", []) or []
        description = "; ".join(reasons) if reasons else "Documento requiere verificación manual."
        gaps.append(
            Gap(
                gap_type=GapType.REVIEW_REQUIRED,
                priority=ActionPriority.URGENT,
                title=f"Revisar documento: {filename}",
                description=description,
                current_state="No verificado",
                required_state="Verificado por notario o fuente oficial",
                action_required="Verificar manualmente el documento",
            )
        )
    return gaps


def derive_intent_override(classification: Dict[str, Any]) -> Optional[Dict[str, str]]:
    if not is_positive_classification(classification):
        return None
    cert_type = map_summary_type_to_intent(classification.get("certificate_type", ""))
    if cert_type == "otros":
        return None
    purpose = map_summary_purpose_to_intent(classification.get("purpose", ""))
    return {"certificate_type": cert_type, "purpose": purpose}


def choose_intent_override(
    candidates: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    overrides = []
    for candidate in candidates:
        classification = candidate.get("classification")
        if not classification:
            continue
        override = derive_intent_override(classification)
        if override:
            overrides.append((override, candidate))

    if not overrides:
        return None

    cert_types = {item[0]["certificate_type"] for item in overrides}
    if len(cert_types) != 1:
        return None

    best_override, best_candidate = max(
        overrides,
        key=lambda item: float(item[1].get("confidence", 0.0)),
    )
    result = dict(best_override)
    result["source"] = best_candidate.get("source") or "unknown"
    result["confidence"] = float(best_candidate.get("confidence", 0.0))
    result["filename"] = best_candidate.get("filename")
    return result


def call_groq_classification(
    model: str,
    api_key: str,
    doc_text: str,
    summary_reference: Dict[str, Any],
) -> Dict[str, Any]:
    if not api_key:
        return {"status": "error", "message": "Missing GROQ_API_KEY."}

    client = Groq(api_key=api_key)
    context_lines = []
    for cert_type, info in summary_reference.items():
        purposes = ", ".join(info.get("purposes", [])) or "none"
        examples = "; ".join(info.get("examples", [])) or "none"
        context_lines.append(
            f"- {cert_type}: purposes={purposes}; examples={examples}"
        )
    context_text = "\n".join(context_lines)

    prompt = (
        "You classify Uruguayan notarial documents.\n"
        "Use ONLY the categories provided. Reply with JSON only.\n"
        "Do not use Markdown or code fences.\n\n"
        "Categories (from certificate_summary.json):\n"
        f"{context_text}\n\n"
        "Return JSON with keys:\n"
        "is_certificate (true/false), certificate_type, purpose, confidence (0-1), reason.\n"
        "If non-certificate, set certificate_type='non_certificate'.\n\n"
        "Document text:\n"
        f"{doc_text[:MAX_LLM_CHARS]}\n"
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a precise document classifier."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )
        content = response.choices[0].message.content or ""
        parsed = parse_json_from_text(content)
        if parsed is not None:
            return parsed
        return {
            "status": "error",
            "message": "LLM did not return valid JSON.",
            "raw": content,
        }
    except Exception as exc:
        return {
            "status": "error",
            "message": f"Groq request failed: {exc}",
        }


def build_file_context(file_result: Dict[str, Any], include_text: bool = True) -> str:
    filename = file_result.get("filename", "document")
    document_type = file_result.get("document_type") or "unknown"
    validation = file_result.get("validation", {})
    match_result = file_result.get("match") or {}
    doc_text = ""
    if include_text:
        doc_text = truncate_text(file_result.get("doc_text", ""), MAX_QA_DOC_CHARS)

    lines = [
        f"Filename: {filename}",
        f"Document type: {document_type}",
        f"Validation: {validation.get('status')} - {validation.get('reason')}",
    ]
    if match_result:
        lines.append(
            f"Dataset match: {match_result.get('status')} - {match_result.get('reason')}"
        )
    if doc_text:
        lines.append("Content:")
        lines.append(doc_text)
    return "\n".join(lines).strip()


def build_qa_context(
    file_results: List[Dict[str, Any]],
    certificate_text: str,
    scope_key: str,
) -> str:
    if scope_key == "certificate":
        if not certificate_text:
            return ""
        content = truncate_text(certificate_text, MAX_QA_CONTEXT_CHARS)
        return f"Generated certificate text:\n{content}"
    if scope_key.startswith("file:"):
        try:
            index = int(scope_key.split(":", 1)[1])
        except ValueError:
            return ""
        if 0 <= index < len(file_results):
            content = build_file_context(file_results[index], include_text=True)
            return truncate_text(content, MAX_QA_CONTEXT_CHARS)
        return ""

    total = len(file_results)
    type_counts: Dict[str, int] = {}
    doc_lines = []
    for file_result in file_results:
        doc_type = file_result.get("document_type") or "unknown"
        type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
        filename = file_result.get("filename", "document")
        validation = file_result.get("validation", {})
        doc_lines.append(
            f"- {filename} | type={doc_type} | validation={validation.get('status')}"
        )

    type_summary = ", ".join(f"{doc_type} ({count})" for doc_type, count in type_counts.items())
    header_lines = [
        f"Total documents: {total}",
        f"Document types: {type_summary or 'unknown'}",
        "Documents:",
    ]
    combined = "\n".join(header_lines + doc_lines)
    return truncate_text(combined, MAX_QA_CONTEXT_CHARS)


def call_groq_document_qa(
    model: str,
    api_key: str,
    question: str,
    context: str,
) -> Dict[str, Any]:
    if not api_key:
        return {"status": "error", "message": "Missing GROQ_API_KEY."}
    if not question or not question.strip():
        return {"status": "error", "message": "Question is empty."}
    if not context or not context.strip():
        return {"status": "error", "message": "No document context available for Q&A."}

    client = Groq(api_key=api_key)
    prompt = (
        "You answer questions about processed notarial documents.\n"
        "Use ONLY the context below. If the answer is not in the context, say so.\n\n"
        "Context:\n"
        f"{context}\n\n"
        f"Question: {question}\n"
        "Answer briefly and clearly."
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a careful document Q&A assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        content = response.choices[0].message.content or ""
        return {"status": "ok", "answer": content.strip()}
    except Exception as exc:
        return {"status": "error", "message": f"Groq request failed: {exc}"}


@st.cache_data(show_spinner=False)
def load_summary(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data(show_spinner=False)
def build_summary_index(summary: Dict[str, Any]) -> Dict[str, Any]:
    entries: List[Dict[str, Any]] = []

    for group, group_entries in summary.get("certificate_file_mapping", {}).items():
        for entry in group_entries:
            item = dict(entry)
            item["group"] = group
            item["entry_type"] = "certificate"
            entries.append(item)

    for entry in summary.get("non_certificate_documents", []):
        item = dict(entry)
        item["group"] = "non_certificate"
        item["entry_type"] = "non_certificate"
        entries.append(item)

    filename_index: Dict[str, List[Dict[str, Any]]] = {}
    customer_index: Dict[str, List[Dict[str, Any]]] = {}
    all_filenames_display: List[str] = []
    all_customers_display: List[str] = []

    for entry in entries:
        filename = entry.get("filename") or entry.get("path") or ""
        if filename:
            all_filenames_display.append(filename)
        for key in make_filename_keys(filename):
            filename_index.setdefault(key, []).append(entry)

        customer = entry.get("customer") or ""
        if customer:
            all_customers_display.append(customer)
        customer_key = normalize_text(customer)
        if customer_key:
            customer_index.setdefault(customer_key, []).append(entry)

    return {
        "entries": entries,
        "filename_index": filename_index,
        "customer_index": customer_index,
        "all_filenames_display": sorted(set(all_filenames_display)),
        "all_customers_display": sorted(set(all_customers_display)),
    }


def is_certificate_entry(entry: Dict[str, Any]) -> bool:
    return entry.get("entry_type") == "certificate"


def entry_has_error(entry: Dict[str, Any]) -> bool:
    return bool(entry.get("error_flag"))


def purpose_matches(entry_purpose: str, user_purpose: str) -> bool:
    if not entry_purpose or not user_purpose:
        return False
    entry_norm = normalize_purpose(entry_purpose)
    user_norm = normalize_purpose(user_purpose)
    if not entry_norm or not user_norm:
        return False
    return entry_norm == user_norm or entry_norm in user_norm or user_norm in entry_norm


def top_fuzzy_matches(query: str, candidates: List[str], limit: int = 5) -> List[Tuple[str, float]]:
    query_norm = normalize_text(query)
    if not query_norm:
        return []
    scored = []
    for candidate in candidates:
        candidate_norm = normalize_text(candidate)
        ratio = difflib.SequenceMatcher(None, query_norm, candidate_norm).ratio()
        scored.append((candidate, ratio))
    scored.sort(key=lambda item: item[1], reverse=True)
    return [(cand, score) for cand, score in scored[:limit] if score > 0]


def dedupe_entries(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    unique = []
    for entry in entries:
        key = (entry.get("customer"), entry.get("filename"), entry.get("path"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(entry)
    return unique


def match_document(
    filename: str,
    subject_name: str,
    extracted_company: Optional[str],
    purpose_value: str,
    summary_index: Dict[str, Any],
    llm_result: Optional[Dict[str, Any]] = None,
    keyword_result: Optional[Dict[str, Any]] = None,
    content_text: str = "",
    content_only: bool = False,
) -> Dict[str, Any]:
    llm_ok = is_result_ok(llm_result)
    keyword_ok = is_keyword_ok(keyword_result)
    llm_non_certificate = llm_ok and llm_result.get("is_certificate") is False
    keyword_certificate = keyword_ok and keyword_result.get("is_certificate") is True
    if llm_non_certificate and keyword_certificate:
        return {
            "status": "needs_review",
            "match_type": "llm_keyword_conflict",
            "confidence": float(llm_result.get("confidence", 0.0)) if llm_ok else 0.0,
            "reason": "Conflicto entre LLM (no certificado) y keywords (certificado).",
            "matches": [],
            "llm_result": llm_result,
            "keyword_result": keyword_result,
        }
    if llm_non_certificate and not keyword_certificate:
        return {
            "status": "not_applicable",
            "match_type": "non_certificate",
            "confidence": float(llm_result.get("confidence", 0.0)) if llm_ok else 0.0,
            "reason": "Documento clasificado como no certificado; match del dataset omitido.",
            "matches": [],
            "llm_result": llm_result,
            "keyword_result": keyword_result,
        }

    if not content_only:
        filename_keys = make_filename_keys(filename)
        matched_entries: List[Dict[str, Any]] = []

        for key in filename_keys:
            matched_entries.extend(summary_index["filename_index"].get(key, []))

        matched_entries = dedupe_entries(matched_entries)

        if matched_entries:
            cert_entries = [e for e in matched_entries if is_certificate_entry(e)]
            non_cert_entries = [e for e in matched_entries if not is_certificate_entry(e)]
            if cert_entries:
                if any(entry_has_error(entry) for entry in cert_entries):
                    return {
                        "status": "not_found",
                        "match_type": "filename_error",
                        "confidence": 0.6,
                        "reason": "Filename matched, but entry is flagged as error in dataset.",
                        "matches": cert_entries,
                    }
                return {
                    "status": "correct",
                    "match_type": "filename",
                    "confidence": 1.0,
                    "reason": "Exact filename match found in certificate dataset.",
                    "matches": cert_entries,
                }
            if non_cert_entries:
                if llm_result and llm_result.get("is_certificate") is True:
                    return {
                        "status": "needs_review",
                        "match_type": "filename_non_certificate_llm_conflict",
                        "confidence": 0.6,
                        "reason": "Filename is non-certificate, but LLM classified as certificate.",
                        "matches": non_cert_entries,
                        "llm_result": llm_result,
                    }
                return {
                    "status": "not_found",
                    "match_type": "filename_non_certificate",
                    "confidence": 0.8,
                    "reason": "Filename matched, but document is classified as non-certificate.",
                    "matches": non_cert_entries,
                }

    customer_keys = []
    if subject_name:
        customer_keys.append(normalize_text(subject_name))
    if extracted_company and normalize_text(extracted_company) not in customer_keys:
        customer_keys.append(normalize_text(extracted_company))

    customer_matches: List[Dict[str, Any]] = []
    for key in customer_keys:
        customer_matches.extend(summary_index["customer_index"].get(key, []))
    customer_matches = dedupe_entries(customer_matches)

    if customer_matches:
        purpose_matches_entries = [
            entry for entry in customer_matches
            if purpose_matches(entry.get("purpose", ""), purpose_value)
        ]
        cert_entries = [e for e in purpose_matches_entries if is_certificate_entry(e)]
        if cert_entries:
            return {
                "status": "correct",
                "match_type": "customer_purpose",
                "confidence": 0.7,
                "reason": "Customer and purpose match found in certificate dataset.",
                "matches": cert_entries,
            }
        return {
            "status": "not_found",
            "match_type": "customer_only",
            "confidence": 0.5,
            "reason": "Customer match found, but no purpose match for this document.",
            "matches": customer_matches[:5],
        }

    if llm_result and llm_result.get("is_certificate") is True:
        llm_type = llm_result.get("certificate_type", "")
        llm_purpose = llm_result.get("purpose", "")
        summary_types = summary_index.get("summary_reference", {})
        if llm_type in summary_types:
            purpose_list = summary_types[llm_type].get("purposes", [])
            if not purpose_list or normalize_purpose(llm_purpose) in [
                normalize_purpose(p) for p in purpose_list
            ]:
                return {
                    "status": "correct",
                    "match_type": "llm_only",
                    "confidence": float(llm_result.get("confidence", 0.5)),
                    "reason": "LLM classified document type matches summary taxonomy.",
                    "matches": [],
                    "llm_result": llm_result,
                }

    if content_text:
        if not keyword_result:
            keyword_result = keyword_classification(content_text, summary_index.get("summary_reference", {}))
        if keyword_result.get("status") == "ok":
            return {
                "status": "correct",
                "match_type": "content_keywords",
                "confidence": float(keyword_result.get("confidence", 0.5)),
                "reason": keyword_result.get("reason", "Keyword match from content."),
                "matches": [],
                "llm_result": llm_result,
                "keyword_result": keyword_result,
            }

    filename_suggestions = top_fuzzy_matches(filename, summary_index["all_filenames_display"])
    customer_suggestions = top_fuzzy_matches(subject_name, summary_index["all_customers_display"])

    has_certificate_signal = (
        (llm_ok and llm_result.get("is_certificate") is True)
        or (keyword_ok and keyword_result.get("is_certificate") is True)
        or (not llm_ok and not keyword_ok)
    )
    status = "needs_review" if has_certificate_signal else "not_found"
    reason = "No strong match found in certificate_summary.json."
    if status == "needs_review":
        reason = f"{reason} Requiere verificación manual."

    return {
        "status": status,
        "match_type": "none",
        "confidence": 0.0,
        "reason": reason,
        "matches": [],
        "suggestions": {
            "filename": filename_suggestions,
            "customer": customer_suggestions,
        },
        "llm_result": llm_result,
        "keyword_result": keyword_result,
    }


def perform_web_search(query: str, provider: str, api_key: str) -> Dict[str, Any]:
    if not query:
        return {"status": "skipped", "message": "Empty query."}
    if not provider or provider == "none":
        return {"status": "skipped", "message": "Search provider not configured."}
    if not api_key:
        return {"status": "skipped", "message": "API key not provided."}
    return {
        "status": "not_implemented",
        "message": "Web search is not implemented yet. Add provider integration here.",
        "query": query,
    }


def extract_company_name(extraction_result) -> Optional[str]:
    for result in extraction_result.extraction_results:
        if result.success and result.extracted_data and result.extracted_data.company_name:
            return result.extracted_data.company_name
    return None


def validate_cross_document_consistency(extraction_result) -> List[Gap]:
    """
    Validate that key data (company name, RUT, CI) is consistent across all documents.
    Returns list of Gap objects for any inconsistencies found.
    """
    gaps = []

    # Collect all extracted data
    company_names = []
    ruts = []
    cis = []

    for result in extraction_result.extraction_results:
        if not result.success or not result.extracted_data:
            continue

        doc_name = result.document.file_name if result.document else "documento"
        extracted = result.extracted_data

        if extracted.company_name:
            company_names.append((extracted.company_name.strip(), doc_name))
        if extracted.rut:
            ruts.append((extracted.rut.strip(), doc_name))
        if extracted.ci:
            cis.append((extracted.ci.strip(), doc_name))

    # Check company name consistency
    if len(company_names) > 1:
        unique_names = {}
        for name, doc in company_names:
            name_normalized = normalize_text(name)
            if name_normalized not in unique_names:
                unique_names[name_normalized] = []
            unique_names[name_normalized].append((name, doc))

        if len(unique_names) > 1:
            all_variants = []
            for variants in unique_names.values():
                all_variants.extend(variants)

            description = "Nombres de empresa inconsistentes encontrados:\n"
            for name, doc in all_variants[:5]:
                description += f"  • '{name}' en {doc}\n"

            gaps.append(Gap(
                gap_type=GapType.SEMANTIC_ERROR,
                priority=ActionPriority.URGENT,
                title="Inconsistencia en nombre de empresa",
                description=description.strip(),
                legal_basis="Los datos deben ser consistentes en todos los documentos",
                current_state=f"Se encontraron {len(unique_names)} variantes del nombre",
                required_state="Un único nombre de empresa consistente en todos los documentos",
                action_required="Verificar y corregir el nombre de la empresa en todos los documentos"
            ))

    # Check RUT consistency
    if len(ruts) > 1:
        unique_ruts = set(rut for rut, _ in ruts)
        if len(unique_ruts) > 1:
            description = "RUTs inconsistentes encontrados:\n"
            for rut, doc in ruts[:5]:
                description += f"  • {rut} en {doc}\n"

            gaps.append(Gap(
                gap_type=GapType.SEMANTIC_ERROR,
                priority=ActionPriority.URGENT,
                title="Inconsistencia en RUT",
                description=description.strip(),
                legal_basis="El RUT debe ser único y consistente",
                current_state=f"Se encontraron {len(unique_ruts)} RUTs diferentes",
                required_state="Un único RUT consistente en todos los documentos",
                action_required="Verificar y corregir el RUT en todos los documentos"
            ))

    # Check CI consistency
    if len(cis) > 1:
        unique_cis = set(ci for ci, _ in cis)
        if len(unique_cis) > 1:
            description = "Cédulas de identidad inconsistentes encontradas:\n"
            for ci, doc in cis[:5]:
                description += f"  • {ci} en {doc}\n"

            gaps.append(Gap(
                gap_type=GapType.SEMANTIC_ERROR,
                priority=ActionPriority.HIGH,
                title="Inconsistencia en Cédula de Identidad",
                description=description.strip(),
                legal_basis="La CI del representante debe ser consistente",
                current_state=f"Se encontraron {len(unique_cis)} CIs diferentes",
                required_state="Una única CI consistente en todos los documentos",
                action_required="Verificar y corregir la CI en todos los documentos"
            ))

    return gaps


def validate_document_expiry(extraction_result) -> List[Gap]:
    """
    Validate that documents are not expired based on legal requirements.
    Returns list of Gap objects for any expired documents.
    """
    from datetime import datetime, timedelta

    gaps = []
    current_date = datetime.now()

    # Define document expiry rules (max age in days)
    expiry_rules = {
        "certificado_dgi": 180,  # 6 months
        "certificado_bps": 30,   # 30 days
        "certificado_situacion_tributaria": 180,  # 6 months
        "padron_bps": 30,  # 30 days
        "estado_cuenta": 90,  # 3 months
        "balance": 365,  # 1 year
    }

    for result in extraction_result.extraction_results:
        if not result.success or not result.extracted_data:
            continue

        doc = result.document
        doc_name = doc.file_name if doc else "documento"
        doc_type = str(doc.detected_type.value) if doc and doc.detected_type else None
        extracted = result.extracted_data

        # Check if this document type has expiry rules
        max_age_days = None
        for doc_pattern, max_days in expiry_rules.items():
            if doc_type and doc_pattern in doc_type.lower():
                max_age_days = max_days
                break

        if not max_age_days:
            continue  # No expiry rule for this document type

        # Try to find document date
        doc_date = None

        # Check specific date fields first
        if extracted.registration_date:
            try:
                doc_date = datetime.strptime(extracted.registration_date, "%Y-%m-%d")
            except:
                pass

        # Fallback to general dates list
        if not doc_date and extracted.dates:
            for date_str in extracted.dates:
                try:
                    # Try various date formats
                    for fmt in ["%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"]:
                        try:
                            doc_date = datetime.strptime(date_str, fmt)
                            break
                        except:
                            continue
                    if doc_date:
                        break
                except:
                    continue

        if not doc_date:
            # No date found - flag as warning
            gaps.append(Gap(
                gap_type=GapType.SEMANTIC_ERROR,
                priority=ActionPriority.MEDIUM,
                title=f"Fecha no encontrada en {doc_name}",
                description=f"No se pudo extraer la fecha de emisión del documento {doc_name}",
                legal_basis=f"Documentos de tipo {doc_type} deben tener fecha de emisión",
                current_state="Fecha no encontrada",
                required_state="Fecha de emisión clara y extraíble",
                action_required="Verificar que el documento contenga una fecha legible"
            ))
            continue

        # Check if document is expired
        age_days = (current_date - doc_date).days

        if age_days > max_age_days:
            gaps.append(Gap(
                gap_type=GapType.SEMANTIC_ERROR,
                priority=ActionPriority.URGENT,
                title=f"Documento vencido: {doc_name}",
                description=f"El documento {doc_name} tiene {age_days} días de antigüedad (máximo permitido: {max_age_days} días)\nFecha del documento: {doc_date.strftime('%d/%m/%Y')}",
                legal_basis=f"Documentos de tipo {doc_type} no pueden tener más de {max_age_days} días",
                current_state=f"Documento con {age_days} días de antigüedad",
                required_state=f"Documento con menos de {max_age_days} días",
                action_required="Obtener una versión actualizada del documento"
            ))
        elif age_days > (max_age_days * 0.8):  # Warning at 80% of expiry
            days_until_expiry = max_age_days - age_days
            gaps.append(Gap(
                gap_type=GapType.SEMANTIC_ERROR,
                priority=ActionPriority.MEDIUM,
                title=f"Documento próximo a vencer: {doc_name}",
                description=f"El documento {doc_name} vencerá en {days_until_expiry} días\nFecha del documento: {doc_date.strftime('%d/%m/%Y')}",
                legal_basis=f"Documentos de tipo {doc_type} no pueden tener más de {max_age_days} días",
                current_state=f"Documento con {age_days} días de antigüedad",
                required_state=f"Documento con menos de {max_age_days} días",
                action_required="Considerar obtener una versión actualizada del documento"
            ))

    return gaps


# ============================================================================
# CONVERSATIONAL CERTIFICATE ASSISTANT FUNCTIONS
# ============================================================================

def load_certificate_requirements() -> Dict[str, Any]:
    """Load certificate requirements from JSON file"""
    requirements_path = Path("config/certificate_requirements.json")
    if requirements_path.exists():
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def detect_user_intent(user_message: str, groq_api_key: str) -> Dict[str, Any]:
    """
    Detect user intent using LLM
    Returns: {"intent": "create|validate|add_docs|other", "certificate_type": str, "company_name": str}
    """
    try:
        from groq import Groq
        client = Groq(api_key=groq_api_key)

        prompt = f"""You are a Uruguayan notarial assistant. Analyze this user message and detect their intent.

User message: "{user_message}"

Possible intents:
1. CREATE - User wants to create a new certificate (e.g., "create personeria for Adecco", "I need a firma certificate")
2. VALIDATE - User wants to validate an existing certificate (e.g., "validate this certificate", "check if this is correct")
3. ADD_DOCS - User is uploading/adding documents (e.g., "I uploaded the cedula", "here's the DGI certificate")
4. OTHER - General questions or other requests

Certificate types in Uruguay:
- personeria (legal status)
- firma (signature certification)
- representacion (representation powers)
- personeria_representacion_representacion (combined)
- firma_personeria_representacion_representacion (complete)

Respond ONLY with valid JSON:
{{
    "intent": "create|validate|add_docs|other",
    "certificate_type": "personeria" or null,
    "company_name": "Company Name" or null,
    "confidence": 0.0-1.0
}}"""

        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0.1,
            max_tokens=200
        )

        result_text = response.choices[0].message.content.strip()

        # Extract JSON from response
        import re
        json_match = re.search(r'\{[^}]+\}', result_text)
        if json_match:
            result = json.loads(json_match.group())
            return result

        return {"intent": "other", "certificate_type": None, "company_name": None, "confidence": 0.0}

    except Exception as e:
        print(f"Intent detection error: {e}")
        return {"intent": "other", "certificate_type": None, "company_name": None, "confidence": 0.0}


def check_document_requirements(
    certificate_type: str,
    purpose: Optional[str],
    uploaded_docs: List[Dict[str, str]],
    requirements_db: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Check if uploaded documents meet requirements for certificate type
    Returns: {"found": [...], "missing": [...], "expired": [...], "ready": bool}
    """
    if certificate_type not in requirements_db:
        return {
            "found": [],
            "missing": [],
            "expired": [],
            "ready": False,
            "error": f"Unknown certificate type: {certificate_type}"
        }

    cert_reqs = requirements_db[certificate_type]
    required_docs = cert_reqs["required_documents"]

    # Add purpose-specific requirements
    if purpose and purpose in requirements_db.get("purpose_specific_requirements", {}):
        purpose_reqs = requirements_db["purpose_specific_requirements"][purpose]
        required_docs = required_docs + purpose_reqs.get("additional_documents", [])

    found = []
    missing = []
    expired = []

    # Check each required document
    for req_doc in required_docs:
        doc_name = req_doc["name"]
        is_mandatory = req_doc.get("mandatory", True)
        max_age = req_doc.get("max_age_days")

        # Try to find this document in uploaded files
        matched = False
        for uploaded in uploaded_docs:
            filename = uploaded.get("filename", "").lower()

            # Simple keyword matching (can be improved with LLM)
            doc_keywords = doc_name.lower().split()
            if any(keyword in filename for keyword in doc_keywords):
                matched = True

                # Check age if applicable
                if max_age:
                    # Would need to extract document date - placeholder for now
                    found.append({
                        "name": doc_name,
                        "file": uploaded["filename"],
                        "status": "valid"
                    })
                else:
                    found.append({
                        "name": doc_name,
                        "file": uploaded["filename"],
                        "status": "valid"
                    })
                break

        if not matched and is_mandatory:
            missing.append({
                "name": doc_name,
                "description": req_doc.get("description", ""),
                "mandatory": is_mandatory
            })

    ready = len(missing) == 0 and len(expired) == 0

    return {
        "found": found,
        "missing": missing,
        "expired": expired,
        "ready": ready,
        "certificate_info": {
            "name": cert_reqs["name"],
            "articles": cert_reqs["articles"]
        }
    }


def generate_certificate_text(
    certificate_type: str,
    company_name: str,
    extracted_data: Dict[str, Any],
    requirements_db: Dict[str, Any]
) -> str:
    """
    Generate certificate text based on type and extracted data
    """
    if certificate_type not in requirements_db:
        return f"Error: Unknown certificate type {certificate_type}"

    cert_info = requirements_db[certificate_type]
    articles = ", ".join(cert_info["articles"])

    # Basic template - can be enhanced with actual templates
    template = f"""CERTIFICO QUE:

De acuerdo a lo dispuesto en los artículos {articles} del Reglamento Notarial:

I) {company_name} es una sociedad anónima con plazo vigente.

RUT: {extracted_data.get('rut', 'N/A')}
Fecha de constitución: {extracted_data.get('constitution_date', 'N/A')}
Registro de Comercio: {extracted_data.get('registry_number', 'N/A')}

II) Representación legal: {extracted_data.get('representative', 'N/A')}
Cédula de Identidad: {extracted_data.get('representative_id', 'N/A')}

EN FE DE ELLO expido el presente certificado.

Montevideo, {datetime.now().strftime('%d de %B de %Y')}
"""

    return template


def run_flow(
    uploaded_files: List[Dict[str, str]],
    intent_inputs: Dict[str, str],
    summary_index: Dict[str, Any],
    catalog_settings: Dict[str, Any],
    notary_inputs: Dict[str, str],
    search_settings: Dict[str, str],
    llm_settings: Dict[str, str],
    content_only: bool,
    operation: str = "create",
    verification_sources: Optional[List[str]] = None,
    auto_fix: bool = False,
) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    if verification_sources is None:
        verification_sources = []
    operation = (operation or "create").strip().lower()
    if operation not in {"create", "validate"}:
        operation = "create"
    results["operation"] = operation

    intent = CertificateIntentCapture.capture_intent_from_params(
        certificate_type=intent_inputs["certificate_type"],
        purpose=intent_inputs["purpose"],
        subject_name=intent_inputs["subject_name"],
        subject_type=intent_inputs["subject_type"],
        additional_notes=intent_inputs.get("additional_notes") or None,
    )

    requirements = LegalRequirementsEngine.resolve_requirements(intent)

    catalog_path = catalog_settings.get("path")
    catalog_customer = catalog_settings.get("customer")
    catalog_customers = catalog_settings.get("customers", [])
    if not catalog_customer or catalog_customer == "auto":
        catalog_customer = infer_catalog_customer(intent.subject_name, catalog_customers)

    collection = DocumentIntake.create_collection(
        intent,
        requirements,
        catalog_path=catalog_path,
        catalog_customer=catalog_customer,
    )
    file_paths = [item["path"] for item in uploaded_files]
    file_name_overrides = {item["path"]: item["filename"] for item in uploaded_files}
    collection = DocumentIntake.add_files_to_collection(
        collection,
        file_paths,
        file_name_overrides=file_name_overrides,
    )

    extraction = process_collection_with_llm(collection, llm_settings)
    results["phase4"] = extraction.get_summary()

    # Run error pattern analysis on extracted data
    try:
        pattern_analyzer = ErrorPatternAnalyzer()
        pattern_matches = pattern_analyzer.analyze_collection(extraction)
        results["pattern_matches"] = len(pattern_matches)
    except Exception as e:
        print(f"⚠️  Error pattern analysis failed: {e}")
        pattern_matches = []
        results["pattern_matches"] = 0

    extracted_company = extract_company_name(extraction)
    subject_name = intent.subject_name.strip() or extracted_company or intent.subject_name
    if subject_name != intent.subject_name:
        intent.subject_name = subject_name

    documents_by_path = {str(doc.file_path): doc for doc in collection.documents}
    extraction_by_path = {
        str(result.document.file_path): result
        for result in extraction.extraction_results
    }

    per_file_data: Dict[str, Dict[str, Any]] = {}
    intent_candidates: List[Dict[str, Any]] = []
    summary_reference = summary_index.get("summary_reference", {})

    for file_info in uploaded_files:
        path = file_info["path"]
        original_filename = file_info["filename"]
        extraction_result = extraction_by_path.get(path)

        doc_text = ""
        if content_only or llm_settings.get("enabled"):
            if (
                extraction_result
                and extraction_result.success
                and extraction_result.extracted_data
            ):
                doc_text = (
                    extraction_result.extracted_data.normalized_text
                    or extraction_result.extracted_data.raw_text
                    or ""
                )

        qa_text = ""
        if extraction_result and extraction_result.success and extraction_result.extracted_data:
            qa_text = extraction_result.extracted_data.raw_text or ""
        if not qa_text:
            qa_text = doc_text
        qa_text = truncate_text(qa_text, MAX_QA_DOC_CHARS)

        llm_result = None
        if llm_settings.get("enabled"):
            if doc_text:
                llm_result = call_groq_classification(
                    model=llm_settings.get("analysis_model", DEFAULT_ANALYSIS_MODEL),
                    api_key=llm_settings.get("api_key", ""),
                    doc_text=doc_text,
                    summary_reference=summary_reference,
                )
            else:
                llm_result = {"status": "error", "message": "No text extracted for LLM."}

        keyword_result = None
        if doc_text:
            keyword_result = keyword_classification(doc_text, summary_reference)

        chosen_classification = choose_classification(llm_result, keyword_result)
        chosen_source = None
        if chosen_classification:
            if llm_result and chosen_classification == llm_result:
                chosen_source = "llm"
            elif keyword_result and chosen_classification == keyword_result:
                chosen_source = "keywords"

        per_file_data[path] = {
            "filename": original_filename,
            "doc_text": doc_text,
            "qa_text": qa_text,
            "llm_result": llm_result,
            "keyword_result": keyword_result,
            "chosen_classification": chosen_classification,
            "chosen_source": chosen_source,
        }

        if chosen_classification:
            intent_candidates.append(
                {
                    "classification": chosen_classification,
                    "source": chosen_source,
                    "confidence": float(chosen_classification.get("confidence", 0.0)),
                    "filename": original_filename,
                }
            )

    intent_override = choose_intent_override(intent_candidates)
    if intent_override:
        intent = CertificateIntentCapture.capture_intent_from_params(
            certificate_type=intent_override["certificate_type"],
            purpose=intent_override["purpose"],
            subject_name=subject_name,
            subject_type=intent.subject_type,
            additional_notes=intent.additional_notes,
        )
        requirements = LegalRequirementsEngine.resolve_requirements(intent)
        collection.certificate_intent = intent
        collection.legal_requirements = requirements
        results["intent_override"] = intent_override

    results["final_subject_name"] = intent.subject_name

    for file_info in uploaded_files:
        path = file_info["path"]
        per_file = per_file_data.get(path, {})
        doc_text = per_file.get("doc_text", "")
        llm_result = per_file.get("llm_result")
        keyword_result = per_file.get("keyword_result")

        match_result = match_document(
            filename=file_info["filename"],
            subject_name=intent.subject_name,
            extracted_company=extracted_company,
            purpose_value=intent.purpose.value,
            summary_index=summary_index,
            llm_result=llm_result,
            keyword_result=keyword_result,
            content_text=doc_text,
            content_only=content_only,
        )
        review_reasons = build_review_reasons(llm_result, keyword_result, match_result)
        per_file["match_result"] = match_result
        per_file["review_required"] = bool(review_reasons)
        per_file["review_reasons"] = review_reasons

    results["phase1"] = intent.get_display_summary()
    results["phase2"] = requirements.get_summary()
    results["phase3"] = collection.get_summary()

    validation = LegalValidator.validate(requirements, extraction)
    gap_report = GapDetector.analyze(validation)

    # In VALIDATE mode, filter out "missing_document" gaps
    # We only want to validate what was uploaded, not check for missing docs
    if operation == "validate":
        gap_report.gaps = [
            gap for gap in gap_report.gaps
            if gap.gap_type != GapType.MISSING_DOCUMENT
        ]
        gap_report.calculate_summary()

    # PHASE 1 SEMANTIC VALIDATION: Cross-document consistency + Date expiry
    consistency_gaps = validate_cross_document_consistency(extraction)
    expiry_gaps = validate_document_expiry(extraction)

    if consistency_gaps:
        gap_report.gaps.extend(consistency_gaps)
        gap_report.calculate_summary()
        results["consistency_gaps_added"] = len(consistency_gaps)

    if expiry_gaps:
        gap_report.gaps.extend(expiry_gaps)
        gap_report.calculate_summary()
        results["expiry_gaps_added"] = len(expiry_gaps)

    review_queue = build_review_queue(per_file_data)
    review_gaps = build_review_gaps(review_queue)
    if review_gaps:
        gap_report.gaps.extend(review_gaps)
        gap_report.calculate_summary()

    # Add pattern-based gaps from error pattern analysis
    if pattern_matches:
        pattern_gaps = []
        for match in pattern_matches:
            # Map pattern severity to gap priority
            if match.pattern.severity.value == "critical":
                priority = ActionPriority.URGENT
            elif match.pattern.severity.value == "error":
                priority = ActionPriority.HIGH
            elif match.pattern.severity.value == "warning":
                priority = ActionPriority.MEDIUM
            else:
                priority = ActionPriority.LOW

            gap = Gap(
                gap_type=GapType.SEMANTIC_ERROR,
                priority=priority,
                title=f"Patrón detectado: {match.pattern.description}",
                description=f"{match.evidence} (Confianza: {match.confidence:.0%})",
                legal_basis="Aprendizaje de errores históricos",
                current_state=f"Patrón de error encontrado en {match.matched_field}",
                required_state="Datos correctos sin errores históricos conocidos",
                action_required=match.recommendation
            )
            pattern_gaps.append(gap)

        if pattern_gaps:
            gap_report.gaps.extend(pattern_gaps)
            gap_report.calculate_summary()
            results["pattern_gaps_added"] = len(pattern_gaps)

    update_result = DataUpdater.create_update_session(gap_report, collection)
    update_result.updated_extraction_result = extraction
    update_result.review_required = review_queue
    if not verification_sources and (gap_report.gaps or review_queue):
        update_result.system_note = (
            "Fuentes oficiales no configuradas; se requiere verificación manual o carga de documentos."
        )

    if auto_fix:
        results["phase5_before_fix"] = validation.get_summary()
        results["phase6_before_fix"] = gap_report.get_summary()
        update_result = DataUpdater.attempt_system_corrections(update_result, intent=intent)
        # Re-run validation & gap detection after system corrections.
        validation = LegalValidator.validate(requirements, update_result.updated_extraction_result)
        gap_report = GapDetector.analyze(validation)
        if review_gaps:
            gap_report.gaps.extend(review_gaps)
            gap_report.calculate_summary()
        update_result.original_gap_report = gap_report

    results["phase5"] = validation.get_summary()
    results["phase6"] = gap_report.get_summary()
    results["gap_structure"] = gap_report.to_dict()
    results["phase7"] = update_result.get_summary()

    confirmation = FinalConfirmationEngine.confirm(requirements, update_result)
    results["phase8"] = confirmation.get_summary()
    results["confirmation_report"] = confirmation

    validation_by_type = {
        doc_validation.document_type: doc_validation
        for doc_validation in validation.document_validations
        if doc_validation.document_type
    }

    file_results = []
    for file_info in uploaded_files:
        path = file_info["path"]
        original_filename = file_info["filename"]
        doc = documents_by_path.get(path)
        extraction_result = extraction_by_path.get(path)
        per_file = per_file_data.get(path, {})
        catalog_info = doc.metadata.get("catalog") if doc else None
        doc_text = per_file.get("doc_text", "")
        qa_text = per_file.get("qa_text", "")
        llm_result = per_file.get("llm_result")
        keyword_result = per_file.get("keyword_result")
        chosen_classification = per_file.get("chosen_classification")
        match_result = per_file.get("match_result") or {}
        review_required = per_file.get("review_required", False)
        review_reasons = per_file.get("review_reasons", [])

        validation_status = "unknown"
        validation_reason = "No validation available."
        validation_issues = []
        validation_required = None
        detected_doc_type = doc.detected_type if doc else None
        if detected_doc_type and detected_doc_type in validation_by_type:
            doc_validation = validation_by_type[detected_doc_type]
            validation_required = doc_validation.required
            validation_issues = [issue.to_dict() for issue in doc_validation.issues]
            blocking = any(
                issue.severity.value in ("critical", "error")
                for issue in doc_validation.issues
            )
            if not doc_validation.present:
                validation_status = "invalid" if doc_validation.required else "missing_optional"
                validation_reason = "Required document missing." if doc_validation.required else "Optional document missing."
            elif blocking:
                validation_status = "invalid"
                validation_reason = doc_validation.issues[0].description if doc_validation.issues else "Validation errors found."
            else:
                validation_status = "valid"
                validation_reason = "Document passes validation checks."
        elif detected_doc_type:
            validation_status = "not_required"
            validation_reason = "Document type is not required for the selected certificate."

        extraction_error = None
        extraction_warning = None
        llm_extraction_error = None
        text_extraction_error = None
        ocr_error = None
        ocr_used = False
        extraction_success = None
        if extraction_result:
            extraction_success = extraction_result.success
            if not extraction_result.success:
                extraction_error = extraction_result.error
                if validation_status in ("unknown", "not_required"):
                    validation_status = "invalid"
                    validation_reason = f"Extraction failed: {extraction_error}"
            if extraction_result.extracted_data:
                extraction_warning = extraction_result.extracted_data.additional_fields.get(
                    "extraction_warning"
                )
                llm_extraction_error = extraction_result.extracted_data.additional_fields.get(
                    "llm_extraction_error"
                )
                text_extraction_error = extraction_result.extracted_data.additional_fields.get(
                    "text_extraction_error"
                )
                ocr_error = extraction_result.extracted_data.additional_fields.get(
                    "ocr_error"
                )
                ocr_used = bool(
                    extraction_result.extracted_data.additional_fields.get("ocr_used")
                )

        doc_type_label = "unknown"
        doc_type_source = "content_missing"
        if llm_result and llm_result.get("status") != "error":
            llm_type = llm_result.get("certificate_type")
            if llm_type:
                doc_type_label = llm_type
                doc_type_source = "content"
        elif keyword_result and keyword_result.get("status") == "ok":
            keyword_type = keyword_result.get("certificate_type")
            if keyword_type:
                doc_type_label = keyword_type
                doc_type_source = "content"

        detected_doc_type = doc.detected_type.value if doc and doc.detected_type else None
        detected_type_detail = build_detected_type_detail(detected_doc_type, catalog_info)
        file_size_bytes = doc.file_size_bytes if doc else None
        processing_status = doc.processing_status.value if doc else None
        is_scanned = doc.is_scanned if doc else None
        document_type_detail = build_document_type_detail(
            doc_type_label,
            doc_type_source,
            detected_doc_type,
            llm_result,
            keyword_result,
        )
        error_reasons = collect_error_reasons(
            validation_status,
            extraction_success,
            extraction_error,
            llm_extraction_error,
            text_extraction_error,
            ocr_error,
            processing_status,
        )
        has_error = bool(error_reasons)

        file_results.append(
            {
                "path": path,
                "filename": original_filename,
                "document_type": doc_type_label,
                "document_type_detected": detected_type_detail,
                "document_type_detail": document_type_detail,
                "type_source": doc_type_source,
                "file_format": doc.file_format.value if doc else "unknown",
                "file_size_bytes": file_size_bytes,
                "processing_status": processing_status,
                "is_scanned": is_scanned,
                "catalog": catalog_info,
                "validation": {
                    "status": validation_status,
                    "reason": validation_reason,
                    "required": validation_required,
                    "issues": validation_issues,
                },
                "match": match_result,
                "llm_result": llm_result,
                "keyword_result": keyword_result,
                "review_required": review_required,
                "review_reasons": review_reasons,
                "doc_text": qa_text,
                "extraction_success": extraction_success,
                "extraction_error": extraction_error,
                "extraction_warning": extraction_warning,
                "llm_extraction_error": llm_extraction_error,
                "text_extraction_error": text_extraction_error,
                "ocr_error": ocr_error,
                "ocr_used": ocr_used,
                "has_error": has_error,
                "error_reasons": error_reasons,
            }
        )

    results["file_results"] = file_results
    # Store internal objects so UI actions (fix/download) can run without re-analyzing.
    results["_intent_obj"] = intent
    results["_extraction_result"] = extraction
    results["_extraction_by_path"] = extraction_by_path

    # In create mode: attempt certificate generation if Phase 8 approved,
    # OR if auto_fix is on (meaning we already tried to fix errors) and the
    # only remaining blocking issues are missing documents (not hard data errors).
    _phase8_approved = confirmation.can_proceed_to_phase9()
    _create_force_generate = (
        operation == "create"
        and auto_fix
        and not _phase8_approved
        and confirmation.critical_issues == 0  # no unfixable data errors
    )

    if operation == "create" and (_phase8_approved or _create_force_generate):
        try:
            certificate = CertificateGenerator.generate(
                certificate_intent=intent,
                legal_requirements=requirements,
                extraction_result=update_result.updated_extraction_result,
                confirmation_report=confirmation,
                notary_name=notary_inputs.get("notary_name"),
                notary_office=notary_inputs.get("notary_office"),
            )
            results["phase9"] = certificate.get_summary()
            results["certificate_text"] = certificate.get_formatted_text()
            if _create_force_generate:
                results["certificate_warning"] = (
                    "Certificate generated with available data. "
                    "Some required documents were missing — review before issuing."
                )

            review_session = NotaryReviewSystem.start_review(
                certificate=certificate,
                reviewer_name=notary_inputs.get("reviewer_name") or "Notary",
            )
            review_session = NotaryReviewSystem.approve_certificate(
                session=review_session,
                notes=notary_inputs.get("review_notes", ""),
            )
            results["phase10"] = review_session.get_summary()

            if review_session.status in [ReviewStatus.APPROVED, ReviewStatus.APPROVED_WITH_CHANGES]:
                final_cert = FinalOutputGenerator.generate_final_certificate(
                    certificate=certificate,
                    review_session=review_session,
                    certificate_number=notary_inputs.get("certificate_number") or "AUTO-0001",
                    issuing_notary=notary_inputs.get("notary_name") or "Notary",
                    notary_office=notary_inputs.get("notary_office") or "Notary Office",
                )
                results["phase11"] = final_cert.get_summary()
        except Exception as _cert_exc:
            results["phase9"] = f"Certificate generation failed: {_cert_exc}"
            results["phase10"] = "Skipped: Phase 9 failed."
            results["phase11"] = "Skipped: Phase 10 was not approved."
    else:
        if operation != "create":
            results["phase9"] = "Skipped: Validation-only run (no certificate generation)."
            results["phase10"] = "Skipped: Validation-only run (no notary review)."
            results["phase11"] = "Skipped: Validation-only run (no final certificate)."
        else:
            results["phase9"] = "Skipped: Phase 8 did not approve certificate generation."
            results["phase10"] = "Skipped: Phase 9 was not generated."
            results["phase11"] = "Skipped: Phase 10 was not approved."

    if search_settings.get("enabled") and file_results:
        has_not_found = any(
            file_result.get("match", {}).get("status") == "not_found"
            for file_result in file_results
        )
        if has_not_found:
            query = f"{intent.subject_name} {intent.purpose.value.replace('para_', '')}"
            search_result = perform_web_search(
                query=query,
                provider=search_settings.get("provider", "none"),
                api_key=search_settings.get("api_key", ""),
            )
            results["web_search"] = search_result

    return results


def list_mandatory_required_documents(intent_inputs: Dict[str, str]) -> List[str]:
    try:
        intent = CertificateIntentCapture.capture_intent_from_params(
            certificate_type=intent_inputs.get("certificate_type") or DEFAULT_CERT_TYPE,
            purpose=intent_inputs.get("purpose") or DEFAULT_PURPOSE,
            subject_name=intent_inputs.get("subject_name") or "",
            subject_type=intent_inputs.get("subject_type") or "company",
            additional_notes=intent_inputs.get("additional_notes") or None,
        )
        requirements = LegalRequirementsEngine.resolve_requirements(intent)
        docs = [
            doc.description
            for doc in requirements.required_documents
            if getattr(doc, "mandatory", True)
        ]
        return dedupe_preserve_order([d.strip() for d in docs if d and d.strip()])
    except Exception:
        return []


def summarize_file_errors(file_results: List[Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    for fr in file_results or []:
        if not fr.get("has_error"):
            continue
        filename = fr.get("filename") or "document"
        reasons = fr.get("error_reasons") or []
        if isinstance(reasons, list) and reasons:
            reason_text = "; ".join([str(r).strip() for r in reasons if str(r).strip()])
            reason_text = truncate_text(reason_text, 300)
            if reason_text:
                lines.append(f"- {filename}: {reason_text}")
                continue
        lines.append(f"- {filename}: error detected")
    return lines


def parse_chat_operation_message(message: str) -> Optional[str]:
    text = normalize_text(message)
    if not text:
        return None

    create_triggers = ["create", "crear", "generate", "make", "hacer"]
    validate_triggers = ["validate", "validar", "revisar", "verify", "verificar"]
    mentions_create = any(token in text for token in create_triggers)
    mentions_validate = any(token in text for token in validate_triggers)

    if mentions_validate and not mentions_create:
        return "validate"
    if mentions_create:
        return "create"
    return None


def is_basic_status_question(message: str) -> bool:
    text = normalize_text(message)
    if not text:
        return False
    triggers = [
        "is my document",
        "are my documents",
        "my document is",
        "my documents are",
        "enough",
        "sufficient",
        "suficiente",
        "correct",
        "incorrect",
        "complete",
        "completed",
        "completo",
        "completa",
        "correcto",
        "correcta",
        "error",
        "errores",
        "missing",
        "falta",
    ]
    return any(token in text for token in triggers)


def build_basic_status_answer(results: Dict[str, Any]) -> str:
    file_errors = summarize_file_errors(results.get("file_results") or [])
    missing_docs = summarize_missing_documents(results.get("gap_structure") or {})
    status_files = "ATTENTION REQUIRED" if file_errors else "VALID"
    status_case = "ATTENTION REQUIRED" if (file_errors or missing_docs) else "VALID"
    is_valid = status_case == "VALID"

    verdict = "CORRECT / COMPLETE" if is_valid else "NOT COMPLETE / NEEDS FIXES"
    lines: List[str] = [f"Verdict: {verdict}", f"Uploaded files status: {status_files}", f"Case completeness status: {status_case}"]
    if file_errors:
        lines.append("")
        lines.append("File errors:")
        lines.extend(file_errors)
    if missing_docs:
        lines.append("")
        lines.append("Missing required documents:")
        lines.extend([f"- {doc}" for doc in missing_docs])
    if is_valid:
        op = (results.get("operation") or "validate").strip().lower()
        lines.append("")
        if op == "create":
            lines.append("Your documents are correct/complete for this request. Generate/download the certificate below.")
        else:
            lines.append("Your documents are correct/complete for validation. Download the report below.")
    else:
        lines.append("")
        lines.append("Your documents are not enough yet; upload the missing/corrected document(s), then say “check” (or click “Analyze / check now”).")
    return "\n".join(lines).strip()


def render_match_result(match_result: Dict[str, Any]) -> None:
    status = match_result.get("status")
    reason = match_result.get("reason", "")
    confidence = match_result.get("confidence", 0.0)

    if status == "correct":
        st.success(f"Correct: {reason} (confidence {confidence:.2f})")
    elif status == "not_applicable":
        st.info(f"Not applicable: {reason} (confidence {confidence:.2f})")
    elif status == "needs_review":
        st.warning(f"Needs review: {reason} (confidence {confidence:.2f})")
    else:
        st.error(f"Not found: {reason} (confidence {confidence:.2f})")

    matches = match_result.get("matches", [])
    if matches:
        st.write("Matched entries:")
        st.dataframe(matches)

    suggestions = match_result.get("suggestions", {})
    if suggestions:
        if suggestions.get("filename"):
            st.write("Top filename suggestions:")
            st.dataframe(suggestions["filename"], width="stretch")
        if suggestions.get("customer"):
            st.write("Top customer suggestions:")
            st.dataframe(suggestions["customer"], width="stretch")

    llm_result = match_result.get("llm_result")
    if llm_result:
        st.write("LLM classification:")
        st.json(llm_result)
    keyword_result = match_result.get("keyword_result")
    if keyword_result:
        st.write("Keyword classification:")
        st.json(keyword_result)


def format_report_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def format_as_text(data: Any, indent: int = 0, seen=None) -> str:
    """Formats complex data structures into readable text efficiently."""
    if seen is None:
        seen = set()
    
    if not data:
        return ""
    
    # Simple recursion guard (id-based)
    if id(data) in seen:
        return "<circular reference>"
    if isinstance(data, (dict, list)):
        seen.add(id(data))

    prefix = " " * indent
    if isinstance(data, list):
        lines = []
        for item in data:
            if isinstance(item, (dict, list)):
                formatted_item = format_as_text(item, indent + 2, seen)
                if "\n" in formatted_item:
                    lines.append(f"{prefix}- Item:")
                    lines.append(formatted_item)
                else:
                    lines.append(f"{prefix}- {formatted_item.strip()}")
            else:
                lines.append(f"{prefix}- {item}")
        return "\n".join(lines)
    
    if isinstance(data, dict):
        lines = []
        for k, v in data.items():
            label = str(k).replace("_", " ").title()
            if isinstance(v, (dict, list)):
                lines.append(f"{prefix}{label}:")
                lines.append(format_as_text(v, indent + 2, seen))
            else:
                # Truncate very long strings in the report for performance
                val_str = str(v)
                if len(val_str) > 1000:
                    val_str = val_str[:1000] + "... [truncated]"
                lines.append(f"{prefix}{label}: {val_str}")
        return "\n".join(lines)
    
    return f"{prefix}{str(data)}"


def parse_chat_intent_message(message: str) -> Dict[str, str]:
    """Parse a user chat message into Phase 1 intent fields (best-effort)."""
    message_norm = normalize_text(message)
    updates: Dict[str, str] = {}

    if "personeria" in message_norm:
        updates["certificate_type"] = "certificado_de_personeria"
    elif "firma" in message_norm or "signature" in message_norm:
        updates["certificate_type"] = "certificacion_de_firmas"
    elif "representacion" in message_norm or "representation" in message_norm:
        updates["certificate_type"] = "certificado_de_representacion"
    elif "vigencia" in message_norm or "validity" in message_norm:
        updates["certificate_type"] = "certificado_de_vigencia"
    elif "situacion juridica" in message_norm or "legal situation" in message_norm:
        updates["certificate_type"] = "certificado_de_situacion_juridica"
    elif "poder" in message_norm or "power of attorney" in message_norm:
        updates["certificate_type"] = "poder_general"

    purpose_tokens = [
        "bps",
        "abitab",
        "bse",
        "bcu",
        "dgi",
        "msp",
        "ute",
        "antel",
        "rupe",
        "mef",
        "imm",
        "zona franca",
        "zona_franca",
        "zonafranca",
    ]
    for token in purpose_tokens:
        token_norm = normalize_text(token)
        if token_norm and token_norm in message_norm:
            updates["purpose"] = map_summary_purpose_to_intent(token)
            break

    if "person" in message_norm or "persona" in message_norm:
        updates["subject_type"] = "person"
    elif "company" in message_norm or "empresa" in message_norm or "sociedad" in message_norm:
        updates["subject_type"] = "company"

    raw = (message or "").strip()
    patterns = [
        r"(?i)\b(?:company|empresa|sociedad)\b\s*[:\-]?\s*([A-Za-z0-9ÁÉÍÓÚÜÑáéíóúüñ .,&()'\\-]{2,80})",
        r"(?i)\bfor\s+company\s+([A-Za-z0-9ÁÉÍÓÚÜÑáéíóúüñ .,&()'\\-]{2,80})",
        r"(?i)\bpara\s+(?:la\s+)?empresa\s+([A-Za-z0-9ÁÉÍÓÚÜÑáéíóúüñ .,&()'\\-]{2,80})",
    ]
    for pattern in patterns:
        match = re.search(pattern, raw)
        if match:
            candidate = match.group(1).strip().strip(" .,-")
            if candidate:
                updates["subject_name"] = candidate
                break

    return updates


def is_chat_check_message(message: str) -> bool:
    text = normalize_text(message)
    triggers = [
        "check",
        "start process again",
        "run again",
        "rerun",
        "re run",
        "validate",
        "validar",
        "analysing",
        "analyzing",
        "analyse",
        "analizar",
    ]
    return any(normalize_text(trigger) in text for trigger in triggers)


def is_chat_fix_message(message: str) -> bool:
    text = normalize_text(message)
    triggers = [
        "fix", "fix it", "fix this", "fixme",
        "correct", "correct it", "corregir",
        "arreglar", "repair", "please fix",
        "auto fix", "auto-fix",
    ]
    return any(normalize_text(trigger) in text for trigger in triggers)


def summarize_missing_documents(gap_structure: Dict[str, Any]) -> List[str]:
    gaps = (gap_structure or {}).get("gaps", []) or []
    missing = []
    for gap in gaps:
        if gap.get("gap_type") != "missing_document":
            continue
        if (gap.get("priority") or "").lower() != "urgent":
            continue
        title = (gap.get("title") or "").strip()
        if not title:
            continue
        missing.append(title.replace("Falta ", "").strip())
    return dedupe_preserve_order(missing)


def build_reports_from_results(results: Dict[str, Any]) -> Tuple[str, str]:
    file_results = results.get("file_results", []) or []
    subject_final = results.get("final_subject_name") or "Unknown Subject"
    gap_struct = results.get("gap_structure") or {}
    gaps_list = gap_struct.get("gaps", []) or []
    total_files = len(file_results)
    error_count = sum(1 for f in file_results if f.get("has_error"))
    doc_types_found = sorted(list({f.get("document_type", "unknown") for f in file_results}))
    doc_types_str = ", ".join(doc_types_found) if doc_types_found else "None"
    missing_docs = [
        gap.get("title", "").replace("Falta ", "")
        for gap in gaps_list
        if gap.get("gap_type") == "missing_document" and gap.get("priority") == "urgent"
    ]
    missing_docs_str = ", ".join([m for m in missing_docs if m]) if missing_docs else "None"
    status_files = "ATTENTION REQUIRED" if (error_count > 0) else "VALID"
    status_case = "ATTENTION REQUIRED" if (error_count > 0 or missing_docs) else "VALID"
    attention_reasons = ", ".join(
        [reason for reason in [("File errors" if error_count > 0 else ""), ("Missing required documents" if missing_docs else "")] if reason]
    ) or "None"

    summary_header = [
        "SUMMARY OF ANALYSIS",
        "-" * 20,
        f"Subject: {subject_final}",
        f"Total Files Processed: {total_files}",
        f"Documents Found: {doc_types_str}",
        f"Uploaded Files Status: {status_files}",
        f"Case Completeness Status: {status_case}",
        "",
        f"File Errors Found: {'Yes' if error_count > 0 else 'No'} ({error_count} files affected)",
        f"Missing Required Documents: {missing_docs_str}",
        f"Attention Reasons: {attention_reasons}",
        "=" * 60,
        "",
    ]

    detailed_lines = [
        "Notarial Chatbot Flow - Detailed Analysis Report",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "=" * 60,
        *summary_header,
    ]
    short_lines = [
        "Notarial Chatbot Flow - Short Summary",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "=" * 60,
        *summary_header,
    ]

    for fr in file_results:
        short_lines.extend(
            [
                f"File: {fr.get('filename')}",
                f"Subject: {subject_final}",
                f"Type: {fr.get('document_type')}",
                f"Status: {str(fr.get('validation', {}).get('status', 'unknown')).upper()}",
                f"Errors: {'Yes' if fr.get('has_error') else 'No'}",
                "-" * 20,
            ]
        )
        detailed_lines.append(f"Details: {fr.get('filename')}")
        detailed_lines.append(f"Type: {fr.get('document_type')} (source: {fr.get('type_source')})")
        detailed_lines.append(
            f"Validation: {fr.get('validation', {}).get('status')} - {fr.get('validation', {}).get('reason')}"
        )
        if fr.get("has_error"):
            detailed_lines.append(f"Errors: {format_as_text(fr.get('error_reasons'), 2)}")
        if fr.get("validation", {}).get("issues"):
            detailed_lines.append(f"Issues: {format_as_text(fr.get('validation', {}).get('issues'), 2)}")
        detailed_lines.append("-" * 80)

    return "\n".join(short_lines), "\n".join(detailed_lines)


def render_chat_mode(
    *,
    output_dir: Path,
    summary_index: Dict[str, Any],
    catalog_settings: Dict[str, Any],
    search_settings: Dict[str, str],
    llm_settings: Dict[str, str],
    content_only: bool,
) -> None:
    st.subheader("Chat mode (guided)")

    if "chat_case_id" not in st.session_state:
        st.session_state["chat_case_id"] = uuid4().hex
    case_dir = Path(".tmp_uploads") / f"chat_case_{st.session_state['chat_case_id']}"
    case_dir.mkdir(parents=True, exist_ok=True)

    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = [
            {
                "role": "assistant",
                "content": (
                    "Hi! How can I help you today?\n\n"
                    "You can say things like:\n"
                    "- **\"I want to validate my documents\"**\n"
                    "- **\"Create a personería certificate for company Adecco\"**"
                ),
            }
        ]
    if "chat_case" not in st.session_state:
        st.session_state["chat_case"] = {
            "operation": "create",
            "intent_inputs": {
                "certificate_type": DEFAULT_CERT_TYPE,
                "purpose": DEFAULT_PURPOSE,
                "subject_name": "",
                "subject_type": "company",
                "additional_notes": "",
            },
            "uploaded_items": [],
            "notary_inputs": {
                "notary_name": "Dr. Juan Perez",
                "notary_office": "Notary Office",
                "reviewer_name": "Dr. Juan Perez",
                "certificate_number": "AUTO-0001",
                "review_notes": "Auto-approved (demo)",
            },
        }
    if "chat_last_results" not in st.session_state:
        st.session_state["chat_last_results"] = None
    if "chat_flow_phase" not in st.session_state:
        # 0 = chat only, 1 = upload visible, 2 = results shown, 3 = download ready
        st.session_state["chat_flow_phase"] = 0

    # Session state for file uploads
    if "chat_uploaded_keys" not in st.session_state:
        st.session_state["chat_uploaded_keys"] = set()
    if "chat_last_loaded_folder" not in st.session_state:
        st.session_state["chat_last_loaded_folder"] = None

    # ── Minimal top bar ──────────────────────────────────────────────────────
    _tb_info, _tb_reset = st.columns([0.8, 0.2])
    with _tb_info:
        _all_items = st.session_state["chat_case"]["uploaded_items"]
        n_files = len(_all_items)
        op = st.session_state["chat_case"].get("operation", "create")
        if n_files:
            _fnames = [item["filename"] for item in _all_items[:3]]
            _fnames_str = ", ".join(_fnames)
            if n_files > 3:
                _fnames_str += f" +{n_files - 3} more"
            st.caption(f"Mode: **{op}** · 📎 {_fnames_str}")
    with _tb_reset:
        if st.button("Reset", help="Start a new case"):
            st.session_state.pop("chat_last_results", None)
            st.session_state.pop("chat_case", None)
            st.session_state.pop("chat_case_id", None)
            st.session_state.pop("chat_messages", None)
            st.session_state.pop("chat_uploaded_keys", None)
            st.session_state.pop("chat_flow_phase", None)
            st.rerun()

    # Initialize action tracking in session state
    if "action_analyze_run" not in st.session_state:
        st.session_state["action_analyze_run"] = False
    if "action_fix_run" not in st.session_state:
        st.session_state["action_fix_run"] = False
    if "action_download_run" not in st.session_state:
        st.session_state["action_download_run"] = False

    # Auto-analyze when files are uploaded (both create and validate modes)
    if st.session_state.get("auto_analyze_pending"):
        st.session_state["auto_analyze_pending"] = False  # Reset flag

        uploaded_items = st.session_state["chat_case"]["uploaded_items"]
        if uploaded_items:
            verdict = None
            operation_now = st.session_state["chat_case"].get("operation", "create")
            is_create = operation_now == "create"
            try:
                spinner_msg = "🔍 Scanning documents and generating certificate..." if is_create else "🔍 Auto-analyzing your documents..."
                with st.spinner(spinner_msg):
                    st.session_state["chat_last_results"] = run_flow(
                        uploaded_files=uploaded_items,
                        intent_inputs=st.session_state["chat_case"]["intent_inputs"],
                        summary_index=summary_index,
                        catalog_settings=catalog_settings,
                        notary_inputs=st.session_state["chat_case"]["notary_inputs"],
                        search_settings=search_settings,
                        llm_settings=llm_settings,
                        content_only=content_only,
                        operation=operation_now,
                        auto_fix=is_create,  # auto-fix errors for create mode
                    )
                    st.session_state["action_analyze_run"] = True
                    st.session_state["action_fix_run"] = False
                    st.session_state["action_download_run"] = False

                    results = st.session_state["chat_last_results"]

                    if is_create:
                        # ── CREATE mode: attempt auto-fix + certificate generation ──
                        certificate_text = results.get("certificate_text")

                        if certificate_text:
                            # Certificate was successfully generated
                            warning = results.get("certificate_warning")
                            if warning:
                                verdict = (
                                    "⚠️ **Certificate generated with available data.**\n\n"
                                    f"{warning}\n\n"
                                    "Say **\"generate files\"** to download the draft certificate."
                                )
                            else:
                                verdict = (
                                    "✅ **Certificate generated successfully!**\n\n"
                                    "Document errors were auto-corrected and the certificate has been created.\n\n"
                                    "Say **\"generate files\"** to download the certificate."
                                )
                            st.session_state["chat_flow_phase"] = 3
                        else:
                            # Phase 8 blocked — show what's still missing/wrong
                            gap_structure = results.get("gap_structure") or {}
                            all_gaps = gap_structure.get("gaps", []) or []
                            semantic_errors = [
                                gap for gap in all_gaps
                                if gap.get("gap_type") in ("semantic_error", "SEMANTIC_ERROR") and
                                gap.get("priority") in ("urgent", "high", "URGENT", "HIGH")
                            ]
                            missing_docs = summarize_missing_documents(results.get("gap_structure") or {})
                            file_errors = summarize_file_errors(results.get("file_results") or [])

                            issues_parts = []
                            if file_errors:
                                issues_parts.append(f"**File Issues ({len(file_errors)}):**")
                                issues_parts.extend([f"- {e}" for e in file_errors[:3]])
                            if semantic_errors:
                                issues_parts.append(f"\n**Data Errors ({len(semantic_errors)}) — auto-fix attempted:**")
                                for err in semantic_errors[:3]:
                                    issues_parts.append(f"- {err.get('title', 'Consistency error')}")
                            if missing_docs:
                                issues_parts.append(f"\n**Missing Required Documents ({len(missing_docs)}):**")
                                issues_parts.extend([f"- {d}" for d in missing_docs[:5]])
                                issues_parts.append("\nUpload these documents and the certificate will be generated automatically.")

                            verdict = (
                                "⚠️ **Scanned your documents — could not generate certificate yet.**\n\n"
                                + "\n".join(issues_parts)
                            )
                            st.session_state["chat_flow_phase"] = 2
                    else:
                        # ── VALIDATE mode: show issues as before ──
                        file_errors = summarize_file_errors(results.get("file_results") or [])
                        gap_structure = results.get("gap_structure") or {}
                        all_gaps = gap_structure.get("gaps", []) or []
                        semantic_errors = [
                            gap for gap in all_gaps
                            if gap.get("gap_type") in ("semantic_error", "SEMANTIC_ERROR") and
                            gap.get("priority") in ("urgent", "high", "URGENT", "HIGH")
                        ]

                        if not file_errors and not semantic_errors:
                            verdict = "✅ **Your documents look correct!**\n\nAll files are valid with no consistency issues.\n\nSay **\"generate files\"** when you're ready to download the report."
                        else:
                            issues_summary = []
                            if file_errors:
                                issues_summary.append(f"**File Errors ({len(file_errors)} files):**")
                                issues_summary.extend(file_errors[:5])
                            if semantic_errors:
                                issues_summary.append(f"\n**Data Consistency Issues ({len(semantic_errors)}):**")
                                for err in semantic_errors[:3]:
                                    issues_summary.append(f"- {err.get('title', 'Consistency error')}")
                            verdict = f"❌ **Issues found in your documents.**\n\n{chr(10).join(issues_summary)}\n\nSay **\"fix it\"** to auto-correct these issues, or **\"generate files\"** to download the validation report as-is."
                        st.session_state["chat_flow_phase"] = 2

            except Exception as _exc:
                verdict = f"❌ **Analysis failed:** {_exc}\n\nPlease try uploading your documents again."
                st.session_state["chat_flow_phase"] = 1

            if verdict:
                st.session_state["chat_messages"].append({
                    "role": "assistant",
                    "content": verdict
                })
            st.rerun()

    # ── Chat ──────────────────────────────────────────────────────────────────
    for message in st.session_state["chat_messages"][-60:]:
        if message["role"] == "user":
            _safe = html.escape(message["content"]).replace("\n", "<br>")
            _pad, _msg_col = st.columns([0.15, 0.85])
            with _msg_col:
                st.markdown(
                    f'<div style="display:flex;justify-content:flex-end;margin:0.15rem 0;">'
                    f'<div style="background:#1565c0;color:#fff;padding:0.65rem 0.95rem;'
                    f'border-radius:1rem 1rem 0.2rem 1rem;max-width:82%;font-size:0.9rem;line-height:1.55;">'
                    f'{_safe}</div></div>',
                    unsafe_allow_html=True,
                )
        else:
            with st.chat_message("assistant"):
                st.markdown(message["content"])

    # ── Inline file upload (shown only after user states intent) ─────────────
    if st.session_state.get("chat_flow_phase", 0) >= 1:
        new_files = st.file_uploader(
            "Attach documents",
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="chat_file_uploader",
        )
        if new_files:
            added = 0
            added_names = []
            for uploaded_file in new_files:
                file_key = f"{uploaded_file.name}:{getattr(uploaded_file, 'size', None)}"
                if file_key in st.session_state["chat_uploaded_keys"]:
                    continue
                tmp_filename = f"{uuid4().hex}_{uploaded_file.name}"
                tmp_path = (case_dir / tmp_filename).resolve()
                tmp_path.write_bytes(uploaded_file.getbuffer())
                st.session_state["chat_case"]["uploaded_items"].append(
                    {"path": str(tmp_path), "filename": uploaded_file.name}
                )
                st.session_state["chat_uploaded_keys"].add(file_key)
                added += 1
                added_names.append(uploaded_file.name)
            if added:
                st.session_state["chat_messages"].append({
                    "role": "user",
                    "content": f"📎 Uploaded {added} file(s):\n" + "\n".join(f"• {n}" for n in added_names),
                })
                st.session_state["auto_analyze_pending"] = True
                st.rerun()

    # ── Download button (shown when files are ready) ────────────────────────
    if st.session_state.get("chat_flow_phase", 0) >= 3:
        last_results = st.session_state.get("chat_last_results") or {}
        fixed_downloads = collect_fixed_downloads(last_results) if last_results else []
        if fixed_downloads:
            st.success("Your files are ready to download!")
            with st.expander("View saved file paths", expanded=False):
                for item in fixed_downloads[:50]:
                    st.write(f"- {item.get('source')}: {item.get('path')}")
        else:
            if st.button("Generate & save files", type="primary"):
                with st.spinner("Generating files..."):
                    updated = build_fixed_outputs_for_results(last_results)
                    st.session_state["chat_last_results"] = updated
                    fixed = collect_fixed_downloads(updated)
                    if fixed:
                        st.session_state["chat_messages"].append({
                            "role": "assistant",
                            "content": f"✅ **{len(fixed)} file(s) saved** to the output/ folder.\n\n"
                            + "\n".join(f"- `{item.get('path')}`" for item in fixed[:10]),
                        })
                    st.rerun()

    user_message = st.chat_input("Message")
    if user_message:
        user_message = user_message.strip()
    if user_message:
        st.session_state["chat_messages"].append({"role": "user", "content": user_message})

        # ========================================================================
        # CONVERSATIONAL ASSISTANT - Intent Detection
        # ========================================================================
        try:
            groq_key = os.getenv("GROQ_API_KEY", "")
            if groq_key:
                intent_result = detect_user_intent(user_message, groq_key)

                # Handle CREATE intent
                if intent_result.get("intent") == "create" and intent_result.get("certificate_type"):
                    cert_type = intent_result["certificate_type"]
                    company = intent_result.get("company_name", "the company")

                    # Load requirements
                    requirements_db = load_certificate_requirements()

                    if cert_type in requirements_db:
                        # Auto-update intent inputs
                        st.session_state["chat_case"]["intent_inputs"]["certificate_type"] = cert_type
                        if company and company != "the company":
                            st.session_state["chat_case"]["intent_inputs"]["subject"] = company

                        # Check current documents
                        uploaded_docs = st.session_state["chat_case"]["uploaded_items"]
                        purpose = st.session_state["chat_case"]["intent_inputs"].get("purpose")

                        check_result = check_document_requirements(
                            cert_type, purpose, uploaded_docs, requirements_db
                        )

                        # Build conversational response
                        response_parts = [
                            f"📋 Creating **{check_result['certificate_info']['name']}** for **{company}**",
                            f"",
                            f"Legal requirements (Articles {', '.join(check_result['certificate_info']['articles'])}):"
                        ]

                        if check_result["found"]:
                            response_parts.append("\n✅ **Found documents:**")
                            for doc in check_result["found"]:
                                response_parts.append(f"  • {doc['name']}: {doc['file']}")

                        if check_result["missing"]:
                            response_parts.append("\n❌ **Missing documents:**")
                            for doc in check_result["missing"]:
                                desc = f" - {doc['description']}" if doc.get('description') else ""
                                response_parts.append(f"  • {doc['name']}{desc}")

                        if check_result["expired"]:
                            response_parts.append("\n⚠️ **Expired documents:**")
                            for doc in check_result["expired"]:
                                response_parts.append(f"  • {doc['name']}: {doc['file']}")

                        if check_result["ready"]:
                            response_parts.append("\n✅ **All requirements met! Say 'check' to analyze and generate the certificate.**")
                        else:
                            response_parts.append("\n📤 **Upload the missing documents above, then say 'check'.**")

                        # Show upload area
                        st.session_state["chat_flow_phase"] = 1
                        # Add response to chat
                        st.session_state["chat_messages"].append({
                            "role": "assistant",
                            "content": "\n".join(response_parts)
                        })
                        st.rerun()
        except Exception as e:
            print(f"Conversational assistant error: {e}")

        # ========================================================================
        # ORIGINAL CHAT LOGIC
        # ========================================================================
        op_update = parse_chat_operation_message(user_message)
        if op_update:
            st.session_state["chat_case"]["operation"] = op_update
            if st.session_state.get("chat_flow_phase", 0) == 0:
                st.session_state["chat_flow_phase"] = 1  # show upload area

        updates = parse_chat_intent_message(user_message)
        intent_inputs = st.session_state["chat_case"]["intent_inputs"]
        intent_inputs.update(updates)
        if updates and st.session_state.get("chat_flow_phase", 0) == 0:
            st.session_state["chat_flow_phase"] = 1  # show upload area

        # ── "Generate files" detection ───────────────────────────────────────
        _gen_triggers = ["generate", "download", "descargar", "save files", "generate files", "get files", "export"]
        _msg_lower = user_message.lower()
        if any(t in _msg_lower for t in _gen_triggers) and st.session_state.get("chat_last_results"):
            with st.spinner("Generating files..."):
                updated = build_fixed_outputs_for_results(st.session_state["chat_last_results"])
                st.session_state["chat_last_results"] = updated
                fixed = collect_fixed_downloads(updated)
            st.session_state["chat_flow_phase"] = 3
            dl_msg = (
                f"✅ **{len(fixed)} file(s) saved** to the output/ folder.\n\n"
                + ("\n".join(f"- `{item.get('path')}`" for item in fixed[:10]) if fixed else "No files generated.")
            )
            st.session_state["chat_messages"].append({"role": "assistant", "content": dl_msg})
            st.session_state["chat_messages"] = st.session_state["chat_messages"][-60:]
            st.rerun()

        # ── Fix message detection ────────────────────────────────────────────
        if is_chat_fix_message(user_message) and st.session_state.get("chat_last_results"):
            last_results = st.session_state["chat_last_results"]
            extraction = last_results.get("_extraction_result")
            if extraction:
                with st.spinner("🔧 Auto-fixing issues..."):
                    all_corrections, applied_count = SmartAutoFix.auto_fix_all(extraction)
                    correction_report = generate_correction_report(all_corrections)
                    last_results["smart_fix_corrections"] = [
                        {"field": c.field_name, "old": c.old_value, "new": c.new_value,
                         "reason": c.reason, "confidence": c.confidence, "file": c.document_file}
                        for c in all_corrections
                    ]
                    last_results["smart_fix_report"] = correction_report
                    st.session_state["chat_last_results"] = last_results

                if applied_count > 0:
                    fix_msg = (
                        f"🔧 **Fixed {applied_count} issue(s)** in your documents.\n\n"
                        + "\n".join(
                            f"- `{c.field_name}` in {c.document_file}: `{c.old_value}` → `{c.new_value}`"
                            for c in all_corrections[:5]
                        )
                        + "\n\nSay **\"generate files\"** to download the corrected documents."
                    )
                else:
                    fix_msg = "ℹ️ No auto-fixable issues found. Say **\"generate files\"** to download the validation report."

                st.session_state["chat_flow_phase"] = 3
                st.session_state["chat_messages"].append({"role": "assistant", "content": fix_msg})
                st.session_state["chat_messages"] = st.session_state["chat_messages"][-60:]
                st.rerun()

        assistant_parts: List[str] = []
        basic_status_handled = False
        if is_basic_status_question(user_message):
            last_results = st.session_state.get("chat_last_results")
            if last_results:
                assistant_parts.append(build_basic_status_answer(last_results))
                basic_status_handled = True
            else:
                uploaded_items = st.session_state["chat_case"]["uploaded_items"]
                if uploaded_items:
                    assistant_parts.append(
                        "I have your files, but I haven't analyzed them yet. Say \"check\" to start."
                    )
                else:
                    assistant_parts.append("Upload documents first, then say \"check\".")
        if not basic_status_handled and (updates or op_update):
            operation_now = st.session_state["chat_case"].get("operation", "create")
            if operation_now == "validate":
                assistant_parts.append("Ok — I'll validate your documents.")
                assistant_parts.append("Upload whatever documents you have. I'll check if each one is valid and correct.")
                assistant_parts.append("(I won't complain about missing documents - just validate what you upload)")
            else:
                assistant_parts.append("I'll do it...checking if I have everything to create the certificate.")
                required_docs = list_mandatory_required_documents(intent_inputs)
                if required_docs:
                    assistant_parts.append("I'm missing (based on the requirements):")
                    assistant_parts.extend([f"- {doc}" for doc in required_docs])
                assistant_parts.append("Upload documents, then say “check” (or “start process again”).")
                st.session_state["chat_case"]["_last_requirements_listed"] = True

        if not basic_status_handled and is_chat_check_message(user_message):
            uploaded_items = st.session_state["chat_case"]["uploaded_items"]
            operation_now = st.session_state["chat_case"].get("operation", "create")

            if not uploaded_items:
                if operation_now == "validate":
                    # For validation, just ask for documents - don't list requirements
                    assistant_parts.append(
                        "Upload the documents you want to validate, then I'll analyze them automatically."
                    )
                else:
                    # For creation, list missing requirements
                    already_listed = st.session_state["chat_case"].pop("_last_requirements_listed", False)
                    if not already_listed:
                        required_docs = list_mandatory_required_documents(intent_inputs)
                        if required_docs:
                            assistant_parts.append("I'll do it...checking if I have everything to proceed.")
                            assistant_parts.append("I'm missing (based on requirements):")
                            assistant_parts.extend([f"- {doc}" for doc in required_docs])
                        else:
                            assistant_parts.append(
                                "I can't check requirements yet. Tell me the certificate type/purpose/subject first."
                            )
                    assistant_parts.append("Upload the missing doc(s) and say 'check' again.")
            else:
                with st.spinner("Analyzing..."):
	                        results = run_flow(
	                            uploaded_files=uploaded_items,
	                            intent_inputs=intent_inputs,
	                            summary_index=summary_index,
	                            catalog_settings=catalog_settings,
	                            notary_inputs=st.session_state["chat_case"]["notary_inputs"],
	                            search_settings=search_settings,
	                            llm_settings=llm_settings,
	                            content_only=content_only,
	                            operation=st.session_state["chat_case"].get("operation", "create"),
	                            auto_fix=False,
	                        )
                st.session_state["chat_last_results"] = results
                st.session_state["chat_flow_phase"] = 2  # results ready

                file_errors = summarize_file_errors(results.get("file_results") or [])
                missing_docs = summarize_missing_documents(results.get("gap_structure") or {})

                # ========================================================================
                # CONVERSATIONAL ASSISTANT - Generate Certificate if Ready
                # ========================================================================
                if not file_errors and not missing_docs and st.session_state["chat_case"].get("operation") == "create":
                    try:
                        cert_type = intent_inputs.get("certificate_type")
                        company = intent_inputs.get("subject", "Unknown Company")

                        if cert_type:
                            requirements_db = load_certificate_requirements()

                            # Extract data from results
                            extracted_data = {}
                            if results.get("extraction_results"):
                                for file_result in results["extraction_results"]:
                                    if hasattr(file_result, 'extracted_data') and file_result.extracted_data:
                                        ed = file_result.extracted_data
                                        if ed.company_name:
                                            extracted_data['company_name'] = ed.company_name
                                        if ed.rut:
                                            extracted_data['rut'] = ed.rut
                                        if ed.constitution_date:
                                            extracted_data['constitution_date'] = ed.constitution_date
                                        if ed.registry_number:
                                            extracted_data['registry_number'] = ed.registry_number
                                        if ed.representative_name:
                                            extracted_data['representative'] = ed.representative_name
                                        if ed.representative_id:
                                            extracted_data['representative_id'] = ed.representative_id

                            # Generate certificate text
                            cert_text = generate_certificate_text(
                                cert_type, company, extracted_data, requirements_db
                            )

                            # Save to file
                            output_dir = Path("output/certificates")
                            output_dir.mkdir(parents=True, exist_ok=True)

                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"{company.replace(' ', '_')}_{cert_type}_{timestamp}.txt"
                            output_path = output_dir / filename

                            with open(output_path, 'w', encoding='utf-8') as f:
                                f.write(cert_text)

                            assistant_parts.append("✅ **All requirements met! Certificate generated successfully!**")
                            assistant_parts.append(f"\n📄 **Download**: {filename}")
                            assistant_parts.append(f"\n```\n{cert_text[:500]}...\n```")
                            assistant_parts.append(f"\n💾 Saved to: `{output_path}`")

                    except Exception as e:
                        assistant_parts.append(f"⚠️ Certificate generation error: {str(e)}")

                # ========================================================================

                if file_errors:
                    assistant_parts.append("I found errors while reading some files:")
                    assistant_parts.extend(file_errors)
                    assistant_parts.append("\nSay **\"fix it\"** to auto-correct, or **\"generate files\"** to get the report as-is.")
                if missing_docs:
                    assistant_parts.append("I'm missing (based on requirements):")
                    assistant_parts.extend([f"- {doc}" for doc in missing_docs])
                    assistant_parts.append("Upload the missing doc(s) and say **\"check\"** again.")
                else:
                    operation_now = results.get("operation") or st.session_state["chat_case"].get("operation", "create")
                    if operation_now == "validate":
                        assistant_parts.append("Validation complete. Say **\"generate files\"** to save the report.")
                    elif results.get("certificate_text"):
                        assistant_parts.append("Certificate ready. Say **\"generate files\"** to save it.")
                    else:
                        assistant_parts.append("Analysis done. Say **\"fix it\"** to correct any issues, or **\"generate files\"** to save the report.")

        if not assistant_parts:
            assistant_parts.append(
                "Tell me what you want to do — for example: **\"I want to validate my documents\"** or **\"create a personería for company X\"**."
            )

        st.session_state["chat_messages"].append(
            {"role": "assistant", "content": "\n".join(assistant_parts).strip()}
        )
        st.session_state["chat_messages"] = st.session_state["chat_messages"][-60:]
        st.rerun()

    results = st.session_state.get("chat_last_results")
    if not results:
        return

    if st.session_state.get("chat_flow_phase", 0) < 3:
        return

    st.divider()
    st.subheader("Latest run outputs")

    short_text, detailed_text = build_reports_from_results(results)
    case_id = results.get("_case_id") or st.session_state.get("chat_case_id")
    case_suffix = f"chat_case_{case_id}" if case_id else datetime.now().strftime("case_%Y%m%d_%H%M%S")
    case_output_dir = (Path("output").resolve() / "runs" / case_suffix).resolve()
    summary_path = save_text_output(case_output_dir, "notary_summary.txt", short_text)
    detailed_path = save_text_output(case_output_dir, "notary_detailed_report.txt", detailed_text)

    st.write("Saved outputs:")
    st.write(f"- {summary_path}")
    st.write(f"- {detailed_path}")

    phase_sections = [
        ("Phase 1", "phase1"),
        ("Phase 2", "phase2"),
        ("Phase 3", "phase3"),
        ("Phase 4", "phase4"),
        ("Phase 5", "phase5"),
        ("Phase 6", "phase6"),
        ("Phase 7", "phase7"),
        ("Phase 8", "phase8"),
        ("Phase 9", "phase9"),
        ("Phase 10", "phase10"),
        ("Phase 11", "phase11"),
    ]
    phase_lines = [
        "Notarial Chatbot Flow - Phase Outputs",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
    ]
    for label, key in phase_sections:
        phase_lines.append(label)
        phase_lines.append(str(results.get(key, "No output")))
        phase_lines.append("-" * 80)
    phase_output_text = "\n".join(phase_lines)
    phases_path = save_text_output(case_output_dir, "notary_phase_outputs.txt", phase_output_text)
    st.write(f"- {phases_path}")

    if results.get("certificate_text"):
        cert_path = save_text_output(case_output_dir, "generated_certificate.txt", str(results["certificate_text"]))
        st.write(f"- {cert_path}")

    if results.get("certificate_text"):
        st.subheader("Generated certificate text")
        st.code(results["certificate_text"], language="text")


def main() -> None:
    output_dir = Path("output").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    st.set_page_config(page_title="Notarial", layout="wide")
    st.title("Notarial")
    st.write("Streamlit UI for Notarial, dataset matching, and optional web search stub.")

    st.sidebar.header("Settings")
    with st.sidebar.expander("Dataset settings", expanded=False):
        summary_path = st.text_input("certificate_summary.json path", DEFAULT_SUMMARY_PATH)
        catalog_path = st.text_input("client_file_catalogs.json path", DEFAULT_CATALOG_PATH)
    enable_llm = st.sidebar.checkbox("Enable LLM extraction + classification (Groq)", value=True)
    enable_ocr_fallback = st.sidebar.checkbox(
        "Enable OCR fallback when no text is found",
        value=True,
    )
    content_only = st.sidebar.checkbox("Match by content only", value=True)
    if enable_llm:
        load_dotenv()
        groq_api_key = os.getenv("GROQ_API_KEY", "")
        extraction_model = st.sidebar.text_input(
            "Groq extraction model",
            value=DEFAULT_EXTRACTION_MODEL,
        )
        analysis_model = st.sidebar.text_input(
            "Groq analysis model",
            value=DEFAULT_ANALYSIS_MODEL,
        )
        qa_model = st.sidebar.text_input(
            "Groq Q&A model",
            value=DEFAULT_QA_MODEL,
        )
        if groq_api_key:
            st.sidebar.caption("GROQ API key loaded from .env")
        else:
            st.sidebar.warning("GROQ API key not found in .env")
    else:
        groq_api_key = ""
        extraction_model = DEFAULT_EXTRACTION_MODEL
        analysis_model = DEFAULT_ANALYSIS_MODEL
        qa_model = DEFAULT_QA_MODEL
    enable_search = st.sidebar.checkbox("Enable web search fallback (stub)", value=False)
    if enable_search:
        search_provider = st.sidebar.selectbox("Search provider", ["none", "serpapi", "bing"])
        search_api_key = st.sidebar.text_input("API key", type="password")
    else:
        search_provider = "none"
        search_api_key = ""

    summary_path_obj = Path(summary_path)
    if not summary_path_obj.exists():
        st.error(f"Summary file not found: {summary_path}")
        st.stop()

    summary_data = load_summary(summary_path)
    summary_index = build_summary_index(summary_data)
    summary_index["summary_reference"] = build_llm_reference(summary_data)

    catalog_data = load_client_catalog(catalog_path)
    catalog_customers = sorted(catalog_data.keys())
    if catalog_customers:
        catalog_customer = st.sidebar.selectbox(
            "Catalog customer (optional)",
            options=["auto"] + catalog_customers,
            index=0,
        )
    else:
        catalog_customer = "auto"
        if catalog_path:
            st.sidebar.info("Catalog file not found or empty.")

    st.sidebar.markdown("### Summary stats")
    st.sidebar.write(f"Total entries: {len(summary_index['entries'])}")
    st.sidebar.write(f"Certificates: {len([e for e in summary_index['entries'] if e['entry_type'] == 'certificate'])}")
    st.sidebar.write(f"Non-certificates: {len([e for e in summary_index['entries'] if e['entry_type'] == 'non_certificate'])}")
    if catalog_customers:
        st.sidebar.write(f"Catalog customers: {len(catalog_customers)}")

    ui_mode = st.sidebar.radio("UI mode", ["Form (current)", "Chat (guided)"], index=1)
    if ui_mode == "Chat (guided)":
        catalog_settings = {
            "path": catalog_path,
            "customer": catalog_customer,
            "customers": catalog_customers,
            "data": catalog_data,
        }
        search_settings = {
            "enabled": enable_search,
            "provider": search_provider,
            "api_key": search_api_key,
        }
        llm_settings = {
            "enabled": enable_llm,
            "extraction_model": extraction_model,
            "analysis_model": analysis_model,
            "api_key": groq_api_key,
            "ocr_fallback": enable_ocr_fallback,
        }
        render_chat_mode(
            output_dir=output_dir,
            summary_index=summary_index,
            catalog_settings=catalog_settings,
            search_settings=search_settings,
            llm_settings=llm_settings,
            content_only=content_only,
        )
        return

    st.subheader("Inputs")
    cert_types = CertificateIntentCapture.get_available_certificate_types()
    purposes = CertificateIntentCapture.get_available_purposes()
    default_cert_type = get_default_option(cert_types, DEFAULT_CERT_TYPE)
    default_purpose = get_default_option(purposes, DEFAULT_PURPOSE)

    simple_mode = st.checkbox("Simple mode (use defaults)", value=True)
    if simple_mode:
        st.caption(
            f"Defaults: {default_cert_type['label']} / {default_purpose['label']} / subject type = company."
        )

    cert_type = default_cert_type
    purpose = default_purpose
    subject_type = "company"
    additional_notes = ""
    notary_name = "Dr. Juan Perez"
    notary_office = "Notary Office"
    reviewer_name = "Dr. Juan Perez"
    certificate_number = "AUTO-0001"
    review_notes = "Auto-approved (demo)"

    input_method = st.radio(
        "Input method",
        [
            "Select from client folders",
            "Upload files (select multiple)",
            "Manual folder path"
        ]
    )

    with st.form("flow_form"):
        uploaded_files = []
        folder_path = ""
        selected_client_folder = None

        if input_method == "Select from client folders":
            # Get list of folders in data for testing/
            client_data_path = Path("data for testing")
            if client_data_path.exists():
                client_folders = sorted([
                    f.name for f in client_data_path.iterdir()
                    if f.is_dir() and not f.name.startswith(".")
                ])

                if client_folders:
                    selected_client_folder = st.selectbox(
                        "Select client folder",
                        options=client_folders,
                        help="Choose a client folder from data for testing/"
                    )

                    # Show folder contents preview
                    if selected_client_folder:
                        preview_path = client_data_path / selected_client_folder
                        all_files = [
                            f for f in preview_path.rglob("*")
                            if f.is_file()
                            and not f.name.startswith(".")
                            and not any(part in ['Data', 'Index', 'Metadata'] for part in f.parts)
                        ]
                        file_count = len(all_files)

                        st.info(f"📁 **{selected_client_folder}** contains {file_count} file(s)")

                        # Optional: Show file tree preview
                        with st.expander("Preview files in this folder"):
                            files_to_show = all_files[:20]
                            for f in files_to_show:
                                relative = f.relative_to(preview_path)
                                st.text(f"📄 {relative}")
                            if file_count > 20:
                                st.text(f"... and {file_count - 20} more files")
                else:
                    st.warning("No client folders found in data for testing/")
            else:
                st.error("data for testing/ directory not found")

        elif input_method == "Upload files (select multiple)":
            uploaded_files = st.file_uploader(
                "Upload files (select multiple - including files from subdirectories)",
                type=None,
                accept_multiple_files=True,
                help="Select all files you want to process. You can select files from different folders."
            )
            st.caption("💡 Tip: In the file picker, you can navigate into subfolders and select all files")

        else:  # Manual folder path
            folder_path = st.text_input(
                "Local Folder Path (Absolute path)",
                value="",
                help="Enter the full path to a folder on your computer"
            )
            st.caption("Example: /home/user/documents/client_files")

        subject_name = st.text_input("Subject name (optional)", value="")

        if not simple_mode:
            with st.expander("Certificate details", expanded=True):
                cert_type_index = next(
                    (idx for idx, item in enumerate(cert_types) if item["value"] == default_cert_type["value"]),
                    0,
                )
                purpose_index = next(
                    (idx for idx, item in enumerate(purposes) if item["value"] == default_purpose["value"]),
                    0,
                )
                col1, col2 = st.columns(2)
                with col1:
                    cert_type = st.selectbox(
                        "Certificate type",
                        options=cert_types,
                        index=cert_type_index,
                        format_func=lambda x: x["label"],
                    )
                with col2:
                    purpose = st.selectbox(
                        "Purpose",
                        options=purposes,
                        index=purpose_index,
                        format_func=lambda x: x["label"],
                    )
                subject_type = st.selectbox("Subject type", ["company", "person"])
                additional_notes = st.text_input("Additional notes (optional)", value="")

            with st.expander("Notary info", expanded=False):
                notary_name = st.text_input("Notary name", value=notary_name)
                notary_office = st.text_input("Notary office", value=notary_office)
                reviewer_name = st.text_input("Reviewer name", value=reviewer_name)
                certificate_number = st.text_input("Certificate number", value=certificate_number)
                review_notes = st.text_input("Review notes", value=review_notes)

        submit = st.form_submit_button("Run flow")

    results = st.session_state.get("last_results")
    if submit:
        tmp_dir = Path(".tmp_uploads")
        tmp_dir.mkdir(exist_ok=True)
        tmp_paths = []
        uploaded_items = []

        # Process based on input method
        if input_method == "Select from client folders":
            if not selected_client_folder:
                st.error("Please select a client folder")
                return

            client_folder_path = Path("data for testing") / selected_client_folder

            # Recursively find all files
            files_found = []
            skip_folders = {'.git', '__pycache__', '.pytest_cache', '.DS_Store', 'venv', 'node_modules'}

            for file_path in client_folder_path.rglob("*"):
                # Skip system/hidden folders and files
                if any(skip in file_path.parts for skip in skip_folders):
                    continue
                # Skip files inside unzipped .pages folders (Apple Pages documents)
                # These folders contain Data/, Index/, Metadata/ subfolders
                if any(part in ['Data', 'Index', 'Metadata'] for part in file_path.parts):
                    continue
                if file_path.is_file() and not file_path.name.startswith("."):
                    files_found.append(file_path)

            if not files_found:
                st.error(f"No files found in {selected_client_folder}")
                return

            st.success(f"Found {len(files_found)} files in {selected_client_folder} (including subdirectories)")

            for file_path in files_found:
                # Preserve relative path from client folder root
                relative_path = file_path.relative_to(client_folder_path)
                uploaded_items.append({
                    "path": str(file_path.absolute()),
                    "filename": f"{selected_client_folder}/{relative_path}",
                })

        elif input_method == "Upload files (select multiple)":
            if not uploaded_files:
                st.error("Please upload files before running the flow.")
                return

            for uploaded_file in uploaded_files:
                tmp_filename = f"{uuid4().hex}_{uploaded_file.name}"
                tmp_path = tmp_dir / tmp_filename
                tmp_path.write_bytes(uploaded_file.getbuffer())
                tmp_paths.append(tmp_path)
                uploaded_items.append({
                    "path": str(tmp_path),
                    "filename": uploaded_file.name,
                })

        else:  # Manual folder path
            if not folder_path:
                st.error("Please provide a local folder path.")
                return

            # Sanitize path: remove potential terminal prompt copy-paste artifacts
            sanitized_path = folder_path.strip()
            # Remove trailing $ or %
            if sanitized_path.endswith("$") or sanitized_path.endswith("%"):
                sanitized_path = sanitized_path[:-1].strip()
            # Remove user@host: prefix if present
            if "@" in sanitized_path and ":" in sanitized_path:
                parts = sanitized_path.split(":", 1)
                # Only take the right side if it looks like a path (starts with / or ~)
                if len(parts) > 1 and (parts[1].strip().startswith("/") or parts[1].strip().startswith("~")):
                    sanitized_path = parts[1].strip()

            p = Path(sanitized_path).expanduser()

            if not p.exists() or not p.is_dir():
                st.error(f"Invalid folder path: {sanitized_path}")
                return

            # Recursively find all files
            files_found = []
            skip_folders = {'.git', '__pycache__', '.pytest_cache', '.DS_Store', 'venv', 'node_modules'}

            for file_path in p.rglob("*"):
                # Skip system/hidden folders and files
                if any(skip in file_path.parts for skip in skip_folders):
                    continue
                # Skip files inside unzipped .pages folders (Apple Pages documents)
                # These folders contain Data/, Index/, Metadata/ subfolders
                if any(part in ['Data', 'Index', 'Metadata'] for part in file_path.parts):
                    continue
                if file_path.is_file() and not file_path.name.startswith("."):
                    files_found.append(file_path)

            if not files_found:
                st.error(f"No files found in {folder_path} (including subdirectories)")
                return

            st.success(f"Found {len(files_found)} files in {folder_path} (including subdirectories)")

            for file_path in files_found:
                relative_path = file_path.relative_to(p)
                uploaded_items.append({
                    "path": str(file_path.absolute()),
                    "filename": str(relative_path),
                })

        intent_inputs = {
            "certificate_type": cert_type["value"],
            "purpose": purpose["value"],
            "subject_name": subject_name.strip(),
            "subject_type": subject_type,
            "additional_notes": additional_notes.strip(),
        }
        notary_inputs = {
            "notary_name": notary_name.strip(),
            "notary_office": notary_office.strip(),
            "reviewer_name": reviewer_name.strip(),
            "certificate_number": certificate_number.strip(),
            "review_notes": review_notes.strip(),
        }
        search_settings = {
            "enabled": enable_search,
            "provider": search_provider,
            "api_key": search_api_key,
        }
        llm_settings = {
            "enabled": enable_llm,
            "extraction_model": extraction_model,
            "analysis_model": analysis_model,
            "api_key": groq_api_key,
            "ocr_fallback": enable_ocr_fallback,
        }
        catalog_settings = {
            "path": catalog_path,
            "customer": catalog_customer,
            "customers": catalog_customers,
            "data": catalog_data,
        }

        try:
            with st.spinner("Analyzing documents and generating reports..."):
                results = run_flow(
                    uploaded_files=uploaded_items,
                    intent_inputs=intent_inputs,
                    summary_index=summary_index,
                    catalog_settings=catalog_settings,
                    notary_inputs=notary_inputs,
                    search_settings=search_settings,
                    llm_settings=llm_settings,
                    content_only=content_only,
                )
                st.session_state["last_results"] = results
                st.session_state["qa_history"] = []
                
                # Pre-generate reports immediately after flow completes
                if results and "file_results" in results:
                     # Calculate Summary Header
                    subject_final = results.get("final_subject_name") or subject_name or "Unknown Subject"
                    gap_struct = results.get("gap_structure") or {}
                    gaps_list = gap_struct.get("gaps", [])
                    total_files = len(results["file_results"])
                    error_count = sum(1 for f in results["file_results"] if f.get("has_error"))
                    doc_types_found = sorted(list({f.get("document_type", "unknown") for f in results["file_results"]}))
                    doc_types_str = ", ".join(doc_types_found) if doc_types_found else "None"
                    missing_docs = [gap.get("title", "").replace("Falta ", "") for gap in gaps_list if gap.get("gap_type") == "missing_document" and gap.get("priority") == "urgent"]
                    missing_docs_str = ", ".join(missing_docs) if missing_docs else "None"
                    status_files = "ATTENTION REQUIRED" if (error_count > 0) else "VALID"
                    status_case = "ATTENTION REQUIRED" if (error_count > 0 or missing_docs) else "VALID"
                    attention_reasons = ", ".join(
                        [reason for reason in [("File errors" if error_count > 0 else ""), ("Missing required documents" if missing_docs else "")] if reason]
                    ) or "None"

                    summary_header = [
                        "SUMMARY OF ANALYSIS",
                        "-" * 20,
                        f"Subject: {subject_final}",
                        f"Total Files Processed: {total_files}",
                        f"Documents Found: {doc_types_str}",
                        f"Uploaded Files Status: {status_files}",
                        f"Case Completeness Status: {status_case}",
                        "",
                        f"File Errors Found: {'Yes' if error_count > 0 else 'No'} ({error_count} files affected)",
                        f"Missing Required Documents: {missing_docs_str}",
                        f"Attention Reasons: {attention_reasons}",
                        "=" * 60,
                        "",
                    ]

                    detailed_lines = ["Notarial Chatbot Flow - Detailed Analysis Report", f"Generated: {datetime.now().isoformat(timespec='seconds')}", "=" * 60] + summary_header
                    short_lines = ["Notarial Chatbot Flow - Short Summary", f"Generated: {datetime.now().isoformat(timespec='seconds')}", "=" * 60] + summary_header
                    
                    for fr in results["file_results"]:
                        # Short entry
                        short_lines.extend([
                            f"File: {fr.get('filename')}",
                            f"Subject: {subject_final}",
                            f"Type: {fr.get('document_type')}",
                            f"Status: {str(fr.get('validation', {}).get('status', 'unknown')).upper()}",
                            f"Errors: {'Yes' if fr.get('has_error') else 'No'}",
                            "-" * 20
                        ])
                        # Detailed entry
                        detailed_lines.append(f"Details: {fr.get('filename')}")
                        detailed_lines.append(f"Type: {fr.get('document_type')} (source: {fr.get('type_source')})")
                        detailed_lines.append(f"Validation: {fr.get('validation', {}).get('status')} - {fr.get('validation', {}).get('reason')}")
                        if fr.get("has_error"):
                             detailed_lines.append(f"Errors: {format_as_text(fr.get('error_reasons'), 2)}")
                        if fr.get("validation", {}).get("issues"):
                             detailed_lines.append(f"Issues: {format_as_text(fr.get('validation', {}).get('issues'), 2)}")
                        detailed_lines.append("-" * 80)
                    
                    st.session_state["detailed_report_text"] = "\n".join(detailed_lines)
                    st.session_state["short_summary_text"] = "\n".join(short_lines)
                    
                    # Also write to local files once
                    (output_dir / "notary_summary.txt").write_text(st.session_state["short_summary_text"], encoding="utf-8")
                    (output_dir / "notary_detailed_report.txt").write_text(st.session_state["detailed_report_text"], encoding="utf-8")

        except Exception as exc:
            st.exception(exc)
            return
        finally:
            for tmp_path in tmp_paths:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            try:
                if tmp_dir.exists() and not any(tmp_dir.iterdir()):
                    tmp_dir.rmdir()
            except OSError:
                pass
    if results is None:
        return

    st.subheader("Document analysis")
    file_results = results.get("file_results", [])
    if file_results:
        problem_files = []
        for fr in file_results:
            validation = fr.get("validation") or {}
            issues = validation.get("issues") or []
            if fr.get("has_error") or validation.get("status") in ("invalid", "missing_optional"):
                problem_files.append((fr.get("filename") or "document", issues))
            elif issues:
                problem_files.append((fr.get("filename") or "document", issues))
        if problem_files:
            st.warning("Documents with detected issues (see details below):")
            for fname, issues in problem_files[:20]:
                issue_summaries = []
                for issue in issues[:3]:
                    desc = issue.get("description") or issue.get("issue_type") or "issue"
                    issue_summaries.append(desc)
                suffix = f" — {', '.join(issue_summaries)}" if issue_summaries else ""
                st.write(f"- {fname}{suffix}")

        summary_rows = []
        report_rows = []
        review_files = [fr for fr in file_results if fr.get("review_required")]
        if review_files:
            st.warning(f"{len(review_files)} documento(s) requieren revisión manual.")
        fixed_downloads: List[Dict[str, str]] = []
        for file_result in file_results:
            validation = file_result.get("validation", {})
            match_result = file_result.get("match") or {}
            llm_result = file_result.get("llm_result") or {}
            keyword_result = file_result.get("keyword_result") or {}
            catalog = file_result.get("catalog") or {}
            for item in file_result.get("fixed_outputs") or []:
                fixed_downloads.append(
                    {
                        "label": item.get("label") or "Download fixed output",
                        "path": item.get("path") or "",
                        "mime": item.get("mime") or "application/octet-stream",
                        "source": file_result.get("filename") or "document",
                    }
                )
            summary_rows.append(
                {
                    "file": file_result.get("filename"),
                    "document_type": file_result.get("document_type"),
                    "document_type_detected": file_result.get("document_type_detected"),
                    "type_source": file_result.get("type_source"),
                    "catalog_match": catalog.get("match_status"),
                    "has_error": format_has_error_flag(file_result.get("has_error")),
                }
            )
            report_rows.append(
                {
                    "file": file_result.get("filename"),
                    "document_type": file_result.get("document_type"),
                    "document_type_detected": file_result.get("document_type_detected"),
                    "document_type_detail": file_result.get("document_type_detail"),
                    "type_source": file_result.get("type_source"),
                    "file_format": file_result.get("file_format"),
                    "file_size_bytes": file_result.get("file_size_bytes"),
                    "processing_status": file_result.get("processing_status"),
                    "is_scanned": file_result.get("is_scanned"),
                    "extraction_warning": file_result.get("extraction_warning"),
                    "extraction_error": file_result.get("extraction_error"),
                    "text_extraction_error": file_result.get("text_extraction_error"),
                    "ocr_error": file_result.get("ocr_error"),
                    "ocr_used": file_result.get("ocr_used"),
                    "has_error": format_has_error_detail(
                        file_result.get("has_error"),
                        file_result.get("error_reasons") or [],
                    ),
                    "match_status": match_result.get("status"),
                    "match_type": match_result.get("match_type"),
                    "match_reason": match_result.get("reason"),
                    "llm_status": llm_result.get("status"),
                    "llm_is_certificate": llm_result.get("is_certificate"),
                    "llm_certificate_type": llm_result.get("certificate_type"),
                    "llm_purpose": llm_result.get("purpose"),
                    "llm_confidence": llm_result.get("confidence"),
                    "keyword_status": keyword_result.get("status"),
                    "keyword_is_certificate": keyword_result.get("is_certificate"),
                    "keyword_certificate_type": keyword_result.get("certificate_type"),
                    "keyword_purpose": keyword_result.get("purpose"),
                    "keyword_confidence": keyword_result.get("confidence"),
                    "catalog_customer": catalog.get("customer"),
                    "catalog_source": catalog.get("source_file"),
                    "catalog_description": catalog.get("description"),
                    "catalog_expected_extensions": ",".join(catalog.get("expected_extensions", []) or []),
                    "catalog_match_status": catalog.get("match_status"),
                    "catalog_match_score": catalog.get("match_score"),
                    "catalog_type_mismatch": catalog.get("type_mismatch"),
                    "review_required": file_result.get("review_required"),
                    "review_reasons": "; ".join(file_result.get("review_reasons") or []),
                }
            )
        st.dataframe(summary_rows, width="stretch")
        if report_rows:
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=list(report_rows[0].keys()))
            writer.writeheader()
            writer.writerows(report_rows)
            csv_text = output.getvalue()
            (output_dir / "notary_file_report.csv").write_text(
                csv_text,
                encoding="utf-8",
            )
            st.caption(f"Saved: {(output_dir / 'notary_file_report.csv').resolve()}")
        short_summary_report = st.session_state.get("short_summary_text", "")
        detailed_report = st.session_state.get("detailed_report_text", "")

        if short_summary_report and detailed_report:
            st.caption(f"Saved: {(output_dir / 'notary_summary.txt').resolve()}")
            st.caption(f"Saved: {(output_dir / 'notary_detailed_report.txt').resolve()}")

        if fixed_downloads:
            st.subheader("Fixed outputs saved")
            st.caption("Files are saved under output/. No browser downloads are generated.")
            for fixed in fixed_downloads[:50]:
                st.write(f"- {fixed.get('source')}: {fixed.get('path')}")

        for file_idx, file_result in enumerate(file_results):
            filename = file_result.get("filename", "document")
            with st.expander(f"Details: {filename}", expanded=False):
                st.write(
                    "Type: "
                    f"{file_result.get('document_type')} "
                    f"(source: {file_result.get('type_source')})"
                )
                st.write(
                    "Detected type (auto): "
                    f"{file_result.get('document_type_detected')}"
                )
                st.write(
                    "File info: "
                    f"format={file_result.get('file_format')}, "
                    f"size_bytes={file_result.get('file_size_bytes')}, "
                    f"processing_status={file_result.get('processing_status')}, "
                    f"is_scanned={file_result.get('is_scanned')}"
                )
                if file_result.get("has_error") is not None:
                    st.write(
                        f"Error flag: {'yes' if file_result.get('has_error') else 'no'}"
                    )
                    if file_result.get("error_reasons"):
                        st.write("Error reasons:")
                        st.write("; ".join(file_result.get("error_reasons") or []))
                validation = file_result.get("validation", {})
                st.write(
                    "Validation: "
                    f"{validation.get('status')} - {validation.get('reason')}"
                )
                if file_result.get("extraction_error"):
                    st.warning(f"Extraction error: {file_result['extraction_error']}")
                if file_result.get("extraction_warning"):
                    st.warning(f"Extraction warning: {file_result['extraction_warning']}")
                if file_result.get("llm_extraction_error"):
                    st.warning(f"LLM extraction error: {file_result['llm_extraction_error']}")
                if file_result.get("text_extraction_error"):
                    st.warning(f"Text extraction error: {file_result['text_extraction_error']}")
                if file_result.get("ocr_error"):
                    st.warning(f"OCR error: {file_result['ocr_error']}")
                if file_result.get("ocr_used"):
                    st.info("OCR was used to extract text for this file.")
                if validation.get("issues"):
                    st.write("Validation issues:")
                    st.dataframe(validation["issues"], width="stretch")

                fixed_outputs = file_result.get("fixed_outputs") or []
                if fixed_outputs:
                    st.write("Fixed outputs saved:")
                    for fixed in fixed_outputs:
                        st.write(f"- {fixed.get('path')}")
                match_result = file_result.get("match")
                if match_result:
                    st.write("Dataset match:")
                    render_match_result(match_result)
                if file_result.get("llm_result"):
                    st.write("LLM classification:")
                    st.json(file_result["llm_result"])
                if file_result.get("keyword_result"):
                    st.write("Keyword classification:")
                    st.json(file_result["keyword_result"])
    else:
        st.info("No documents were processed.")

    if results.get("intent_override"):
        override = results["intent_override"]
        file_label = f", file: {override.get('filename')}" if override.get("filename") else ""
        st.info(
            "Intent overridden from content: "
            f"{override['certificate_type']} / {override['purpose']} "
            f"(source: {override['source']}{file_label}, confidence: {override['confidence']:.2f})"
        )

    if results.get("web_search"):
        st.subheader("Web search fallback")
        st.write(results["web_search"])

    st.subheader("Phase outputs")
    phase_sections = [
        ("Phase 1", "phase1"),
        ("Phase 2", "phase2"),
        ("Phase 3", "phase3"),
        ("Phase 4", "phase4"),
        ("Phase 5", "phase5"),
        ("Phase 6", "phase6"),
        ("Phase 7", "phase7"),
        ("Phase 8", "phase8"),
        ("Phase 9", "phase9"),
        ("Phase 10", "phase10"),
        ("Phase 11", "phase11"),
    ]

    phase_lines = [
        "Notarial Chatbot Flow - Phase Outputs",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
    ]
    for label, key in phase_sections:
        phase_lines.append(label)
        phase_lines.append(str(results.get(key, "No output")))
        phase_lines.append("-" * 80)
    phase_output_text = "\n".join(phase_lines)
    (output_dir / "notary_phase_outputs.txt").write_text(
        phase_output_text,
        encoding="utf-8",
    )
    st.caption(f"Saved: {(output_dir / 'notary_phase_outputs.txt').resolve()}")

    for label, key in phase_sections:
        with st.expander(label, expanded=False):
            st.code(str(results.get(key, "No output")), language="text")

    if results.get("certificate_text"):
        st.subheader("Generated certificate text")
        st.code(results["certificate_text"], language="text")

    st.subheader("Document Q&A")
    if "qa_history" not in st.session_state:
        st.session_state["qa_history"] = []

    file_results = results.get("file_results", [])
    certificate_text = results.get("certificate_text", "")
    if not file_results and not certificate_text:
        st.info("No document content available for Q&A.")
    elif not enable_llm:
        st.info("Enable LLM extraction + classification to use Q&A.")
    elif not groq_api_key:
        st.warning("GROQ API key not found in .env.")
    else:
        scope_options = ["All documents"]
        scope_map = {"All documents": "all"}
        if certificate_text:
            scope_options.append("Generated certificate")
            scope_map["Generated certificate"] = "certificate"
        for idx, file_result in enumerate(file_results):
            label = f"File {idx + 1}: {file_result.get('filename', 'document')}"
            scope_options.append(label)
            scope_map[label] = f"file:{idx}"

        with st.form("qa_form"):
            scope_label = st.selectbox("Question scope", options=scope_options)
            question = st.text_area("Question", value="")
            submitted = st.form_submit_button("Ask")

        if submitted:
            context = build_qa_context(
                file_results=file_results,
                certificate_text=certificate_text,
                scope_key=scope_map.get(scope_label, "all"),
            )
            qa_result = call_groq_document_qa(
                model=qa_model,
                api_key=groq_api_key,
                question=question,
                context=context,
            )
            if qa_result.get("status") == "ok":
                st.session_state["qa_history"].append(
                    {
                        "question": question.strip(),
                        "answer": qa_result.get("answer", ""),
                        "scope": scope_label,
                    }
                )
            else:
                st.error(qa_result.get("message", "Q&A failed."))

        if st.session_state["qa_history"]:
            st.write("Conversation")
            for entry in st.session_state["qa_history"][-5:]:
                st.write(f"Q ({entry.get('scope')}): {entry.get('question')}")
                st.write(f"A: {entry.get('answer')}")

        if st.button("Clear Q&A history"):
            st.session_state["qa_history"] = []


if __name__ == "__main__":
    main()
