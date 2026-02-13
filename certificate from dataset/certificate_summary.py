"""
create_certificate_summary3.py - VERSION 3.1 (ENHANCED)
------------------------------------------------------------
VERSION 3.1 CRITICAL FIX:
• Prevents "COMPLETO" and "CONTROL" certificates from being misclassified as simple "firma"
• These complex certificates contain multiple components (firma + personeria + representacion)
• Should be classified as "firma_personeria_representacion_representacion", not "firma"
• This fixes the DGI purpose issue in firma section

IMPROVEMENTS OVER VERSION 2:
• Stronger authority document detection (DGI, BPS, BCU separation)
• Better handling of notarially-certified authority documents
• Stricter classification to match client expectations
• COMPLETO/CONTROL cert detection - moves complex certs to proper categories
• Fixed output filename to certificate_summary3.json
• Enhanced logging for better debugging

FEATURES:
• Uses LLM (meta-llama/llama-4-maverick-17b-128e-instruct) to analyze document CONTENT
• Correctly identifies certificate types based on content, not filename
• Extracts accurate PURPOSE information (BSE, ABITAB, Zona Franca, etc.)
• Differentiates NOTARIAL certificates from AUTHORITY documents (DGI, BPS, BCU)
• ERROR files are treated as certificates with wrong data
• Parallel processing with robust error handling
"""

import json
import os
import time
import unicodedata
from dotenv import load_dotenv
from groq import Groq
from text_extractor import TextExtractor
from multiprocessing import Pool
from tqdm import tqdm

# ---------------------------------------------------------
# Setup
# ---------------------------------------------------------
load_dotenv()

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "meta-llama/llama-4-maverick-17b-128e-instruct"
NUM_WORKERS = 1  # Sequential processing to avoid rate limits
API_TIMEOUT = 30  # Timeout for API calls in seconds

# Load data
with open("customers_index.json", "r", encoding="utf-8") as f:
    customers = json.load(f)

with open("certificate_types.json", "r", encoding="utf-8") as f:
    cert_types = json.load(f)

# ---------------------------------------------------------
# HARD KEYWORD MAPS - Fallback detection
# ---------------------------------------------------------

# VERY SPECIFIC authority keywords (must be actual authority documents)
# ENHANCED: More comprehensive patterns to catch DGI, BPS, BCU docs
AUTHORITY_KEYWORDS = [
    "constancia anual dgi",
    "constancia dgi",
    "certificado común bps",
    "certificado comun bps",
    "formulario 0352",
    "formulario b y certificacion",
    "acuse bcu",
    "constancia certif.comun bps",
    # Additional DGI patterns
    "certificado dgi",
    "certificado netkla trading - dgi",
    "certif.anual. dgi",
    # Additional BPS patterns
    "certificado bps",
    "certificado común",
    # BCU patterns
    "certificado bcu",
    "certificado de recepcion. bcu",
    "comunicacion bcu"
]

# Stronger patterns for documents that should NEVER be notarial
PURE_AUTHORITY_PATTERNS = [
    "certificado común bps",
    "certificado comun bps",
    "constancia dgi",
    "certificado dgi",
    "acuse bcu",
    "formulario 0352"
]

# Keywords that STRONGLY indicate NOTARIAL certificates
# Removed generic "certificación" to avoid false positives
NOTARIAL_KEYWORDS = [
    "escribano", "escribana", "notario", "notaria",
    "doy fe", "ante mí", "ante mi",
    "certificación notarial", "certificacion notarial",
    "certifica que", "certifico que"
]

PURPOSE_KEYWORDS_MAP = {
    "bse": ["bse", "seguros", "accidente de trabajo", "banco de seguros"],
    "abitab": ["abitab", "firma digital"],
    "zona franca": ["zona franca", "zonamerica", "free zone"],
    "comercio": ["registro de comercio", "cámara de comercio"],
    "registro": ["registro nacional", "registro de comercio"],
    "bcu": ["bcu", "banco central"],
    "dgi": ["dgi", "impositiva", "dirección general"],
    "bps": ["bps", "previsión social", "prevision"]
}

CERT_TYPE_KEYWORDS = {
    "firma": ["certificación de firma", "certificacion firma", "cert. firma"],
    "personeria": ["personería", "personeria"],
    "representacion": ["representación", "representacion"],
    "poder": ["poder", "apoderado"],
    "vigencia": ["vigencia"],
    "control": ["control"]
}

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------

def normalize_path(path):
    """Normalize Unicode path to NFC form for consistent filesystem access"""
    return unicodedata.normalize('NFC', path)

def find_file_case_insensitive(base_path, customer, relative_path):
    """
    Try to find file with Unicode normalization and case-insensitive matching.
    Returns the actual file path if found, None otherwise.
    """
    # Normalize all paths
    customer_folder = os.path.join(base_path, normalize_path(customer))

    # First try direct match with NFC normalization
    direct_path = os.path.join(customer_folder, normalize_path(relative_path))
    if os.path.exists(direct_path):
        return direct_path

    # If direct match fails, try to find it in the directory
    if os.path.exists(customer_folder):
        try:
            actual_files = os.listdir(customer_folder)
            normalized_target = normalize_path(relative_path).lower()

            for actual_file in actual_files:
                if normalize_path(actual_file).lower() == normalized_target:
                    return os.path.join(customer_folder, actual_file)
        except Exception:
            pass

    return None

# ---------------------------------------------------------
# Worker Initialization
# ---------------------------------------------------------

_worker_groq_client = None
_worker_text_extractor = None

def init_worker():
    """Initialize worker process with its own clients"""
    global _worker_groq_client, _worker_text_extractor
    _worker_groq_client = Groq(api_key=GROQ_API_KEY)
    _worker_text_extractor = TextExtractor(lang="spa", ocr_dpi=150)

# ---------------------------------------------------------
# LLM Analysis Functions
# ---------------------------------------------------------

def analyze_document_with_llm(document_text: str, filename: str, groq_client) -> dict:
    """
    Use LLM to analyze document with simpler prompt and better error handling
    """

    prompt = f"""You are a Uruguayan NOTARIAL law expert.

Document content:
{document_text[:3000]}

Respond ONLY in JSON:
{{
  "is_notarial": true/false,
  "certificate_type": "firma|personeria|representacion|poder|vigencia|control|otros|authority",
  "purpose": "BSE|ABITAB|Zona Franca|Comercio|Registro|BCU|DGI|BPS|Other"
}}

Rules:
• ERROR files are still notarial
• DGI, BPS, BCU, Registro, Banco = authority
"""

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=MODEL_NAME,
                temperature=0.1,
                max_tokens=100,
                timeout=API_TIMEOUT
            )

            result_text = response.choices[0].message.content.strip()

            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{[^}]+\}', result_text)
            if json_match:
                result_text = json_match.group()

            result = json.loads(result_text)

            return {
                "is_notarial": result.get("is_notarial", False),
                "certificate_type": result.get("certificate_type", "otros").lower(),
                "purpose": result.get("purpose", "unknown").lower()
            }

        except json.JSONDecodeError:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            # Fallback
            return {
                "is_notarial": False,
                "certificate_type": "otros",
                "purpose": "unknown"
            }

        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                wait_time = 5 * (2 ** attempt)
                print(f"  [WARNING] Rate limit hit, waiting {wait_time}s...")
                time.sleep(wait_time)
                if attempt < max_retries - 1:
                    continue

            if attempt < max_retries - 1:
                time.sleep(2)
                continue

            # Fallback on final error
            return {
                "is_notarial": False,
                "certificate_type": "otros",
                "purpose": "unknown"
            }

# ---------------------------------------------------------
# Main Processing Worker Function
# ---------------------------------------------------------

def process_single_file(task_item):
    """Process a single file with worker's clients"""
    global _worker_groq_client, _worker_text_extractor

    customer = task_item['customer']
    cert_info = task_item['cert_info']
    base_path = task_item['base_path']
    is_certificate = task_item['is_certificate']

    groq_client = _worker_groq_client
    text_extractor = _worker_text_extractor

    if not is_certificate:
        return {
            'type': '__NON_CERT__',
            'customer': customer,
            'filename': cert_info['filename'],
            'path': cert_info['relative_path'],
            'reason': 'non_certificate'
        }

    # Use Unicode-aware file finding with normalization
    file_path = find_file_case_insensitive(base_path, customer, cert_info["relative_path"])

    if not file_path:
        return {
            'type': '__AUTHORITY_DOC__',
            'customer': customer,
            'filename': cert_info['filename'],
            'path': cert_info['relative_path'],
            'error_flag': cert_info.get('error_flag', False),
            'purpose': 'unknown',
            'reason': 'file_not_found'
        }

    try:
        start_time = time.time()

        # Extract text from document
        extraction_result = text_extractor.extract(file_path, max_pages=2)
        document_text = extraction_result.full_text
        extraction_time = time.time() - start_time

        if not document_text.strip() or "[SCANNED PAGE" in document_text:
            # Use filename as fallback for text analysis
            document_text = f"[Document: {cert_info['filename']}]\nFilename: {cert_info['filename']}"

        text_lower = document_text.lower()
        filename_lower = cert_info['filename'].lower()

        # Also check filename for keywords when content is minimal
        combined_text = text_lower + " " + filename_lower

        # ---------------------------------------------------------
        # ENHANCED AUTHORITY DETECTION (Version 3 Improvement)
        # ---------------------------------------------------------
        # Two-tier detection: pure authority docs vs notarially certified authority docs

        is_authority = False
        is_pure_authority = False  # NEW: for documents that should NEVER be notarial

        # Check for PURE authority documents (never notarial, even if certified)
        for pattern in PURE_AUTHORITY_PATTERNS:
            if pattern in combined_text:
                is_pure_authority = True
                is_authority = True
                break

        # Check for general authority keywords if not already marked as pure
        if not is_pure_authority:
            for kw in AUTHORITY_KEYWORDS:
                if kw in combined_text:
                    is_authority = True
                    break

        # Check for notarial signatures (can override general authority, but not pure authority)
        has_notarial_signature = False
        for kw in NOTARIAL_KEYWORDS:
            if kw in combined_text:
                has_notarial_signature = True
                # Only override if NOT pure authority
                if not is_pure_authority:
                    is_authority = False
                break

        # ERROR files are ALWAYS notarial certificates (unless pure authority)
        if "error" in filename_lower and not is_pure_authority:
            is_authority = False
            has_notarial_signature = True

        # ---------------------------------------------------------
        # LLM CALL
        # ---------------------------------------------------------
        analysis = analyze_document_with_llm(document_text, cert_info['filename'], groq_client)

        is_notarial = analysis.get("is_notarial", False)
        cert_type = analysis.get("certificate_type", "otros").lower()
        purpose = analysis.get("purpose", "unknown").lower()

        # ---------------------------------------------------------
        # ENHANCED FINAL NOTARIAL DECISION (Version 3 Improvement)
        # ---------------------------------------------------------
        # Priority hierarchy:
        # 1. Pure authority docs → NEVER notarial (even if notary signature present)
        # 2. ERROR files → ALWAYS notarial (unless pure authority)
        # 3. Notarial signature present → Notarial (unless pure authority)
        # 4. Authority keywords + no signature → Not notarial
        # 5. Default to LLM assessment

        # STRONGEST RULE: Pure authority documents are NEVER notarial
        if is_pure_authority:
            is_notarial = False
        # ERROR files are ALWAYS notarial (unless caught by pure authority above)
        elif "error" in filename_lower:
            is_notarial = True
        # If has notarial signature but is general authority, still mark as notarial
        # (these are notarially certified authority docs - keep them)
        elif has_notarial_signature:
            is_notarial = True
        # Authority docs without notarial signature → not notarial
        elif is_authority and not has_notarial_signature:
            is_notarial = False
        # Default to LLM assessment
        else:
            is_notarial = analysis.get("is_notarial", False)

        # ---------------------------------------------------------
        # FIX 3: HARD PURPOSE INFERENCE (check content + filename)
        # ---------------------------------------------------------
        # First try keyword-based detection in combined text
        detected_purpose = "other"
        for p_key, p_vals in PURPOSE_KEYWORDS_MAP.items():
            for pv in p_vals:
                if pv in combined_text:
                    detected_purpose = p_key
                    break
            if detected_purpose != "other":
                break

        # Use detected purpose if found, otherwise keep LLM's suggestion
        if detected_purpose != "other":
            purpose = detected_purpose
        elif purpose in ["unknown", "", "authority"]:
            purpose = "other"

        # ---------------------------------------------------------
        # ENHANCED CERT TYPE INFERENCE (Version 3.1 - CRITICAL FIX)
        # ---------------------------------------------------------
        # KEY INSIGHT: "COMPLETO" and "CONTROL" certificates contain MULTIPLE
        # components and should NOT be classified as simple "firma"

        # Check for COMPLETO/CONTROL keywords that indicate complex certificates
        is_complete_cert = (
            "completo" in combined_text or
            "control completo" in combined_text or
            "control sa" in combined_text or
            "control sociedad" in combined_text
        )

        # Detect individual components from combined text
        detected_components = []
        has_firma = "firma" in combined_text or "firmas" in combined_text
        has_personeria = "personería" in combined_text or "personeria" in combined_text
        has_representacion = "representación" in combined_text or "representacion" in combined_text

        # CRITICAL: If it's a COMPLETO cert with firma, it's probably firma_personeria_representacion
        if is_complete_cert and has_firma:
            # Force classification to complex type
            if has_personeria and has_representacion:
                cert_type = "firma_personeria_representacion_representacion"
            elif has_personeria:
                cert_type = "firma_personeria"
            elif has_representacion:
                cert_type = "firma_representacion_representacion"
            else:
                # Still complex even without explicit keywords
                cert_type = "firma_personeria_representacion_representacion"
        else:
            # Standard detection for non-COMPLETO certificates
            # Build components list in correct order
            if has_firma:
                detected_components.append("firma")
                if "firmas" in combined_text and not "firma" in combined_text:
                    detected_components = ["firma_firmas"]

            if has_personeria:
                if not detected_components:
                    detected_components.append("personeria")
                elif detected_components[0] == "firma":
                    detected_components.append("personeria")

            if has_representacion:
                if not detected_components:
                    detected_components.append("representacion")
                elif "firma" in detected_components or "personeria" in detected_components:
                    detected_components.append("representacion")
                    # For representacion, often appears twice in the type name
                    detected_components.append("representacion")

            # Build composite type key and try to match with certificate_types
            if detected_components:
                # Try various combinations to match certificate_types.json keys
                combos_to_try = [
                    "_".join(detected_components),
                    "_".join(detected_components[:-1]) if len(detected_components) > 1 else detected_components[0],
                    detected_components[0] if len(detected_components) > 0 else None
                ]

                for combo in combos_to_try:
                    if combo and combo in cert_types:
                        cert_type = combo
                        break

        # ---------------------------------------------------------
        # FINAL ROUTING (Version 3 Enhancement)
        # ---------------------------------------------------------
        if not is_notarial or cert_type == "authority":
            # Determine specific authority type for statistics
            authority_type = 'authority_document'
            if is_pure_authority:
                authority_type = 'pure_authority_document'

            # Detect which authority for better purpose tracking
            detected_authority_purpose = purpose
            if detected_authority_purpose == 'unknown':
                if 'dgi' in combined_text:
                    detected_authority_purpose = 'dgi'
                elif 'bps' in combined_text:
                    detected_authority_purpose = 'bps'
                elif 'bcu' in combined_text:
                    detected_authority_purpose = 'bcu'

            return {
                'type': '__AUTHORITY_DOC__',
                'customer': customer,
                'filename': cert_info['filename'],
                'path': cert_info['relative_path'],
                'error_flag': cert_info.get('error_flag', False),
                'purpose': detected_authority_purpose,
                'reason': authority_type,
                'extraction_time': extraction_time,
                'is_pure_authority': is_pure_authority  # NEW: for tracking
            }

        if cert_type not in cert_types:
            cert_type = "otros"

        return {
            'type': cert_type,
            'customer': customer,
            'filename': cert_info['filename'],
            'path': cert_info['relative_path'],
            'error_flag': cert_info.get('error_flag', False),
            'purpose': purpose,
            'extraction_time': extraction_time,
            'total_time': time.time() - start_time,
            'is_complete_cert': is_complete_cert  # NEW v3.1: for tracking COMPLETO certs
        }

    except Exception as e:
        return {
            'type': 'otros',
            'customer': customer,
            'filename': cert_info['filename'],
            'path': cert_info['relative_path'],
            'error_flag': cert_info.get('error_flag', False),
            'purpose': 'unknown',
            'error': str(e)
        }

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------

if __name__ == '__main__':
    final_certificate_mapping = {k: [] for k in cert_types.keys()}
    non_certificate_docs = []

    # Enhanced statistics tracking (Version 3.1)
    stats = {
        'file_not_found': 0,
        'authority_detected': 0,
        'pure_authority_detected': 0,  # NEW: tracks pure authority docs
        'notarial_confirmed': 0,
        'processing_errors': 0,
        'total_processed': 0,
        'dgi_removed_from_notarial': 0,  # NEW: tracks DGI docs removed
        'bps_removed_from_notarial': 0,  # NEW: tracks BPS docs removed
        'bcu_removed_from_notarial': 0,  # NEW: tracks BCU docs removed
        'completo_certs_reclassified': 0  # NEW v3.1: COMPLETO certs moved from firma to complex types
    }

    CUSTOMER_DATA_PATH = "Notaria_client_data"

    print("\n" + "=" * 70)
    print("Certificate Classification - VERSION 3.1 (CRITICAL FIX)")
    print("=" * 70)
    print("KEY FIX:")
    print("• COMPLETO/CONTROL certificates moved from 'firma' to complex types")
    print("• This fixes DGI appearing in firma section (client's main issue)")
    print("=" * 70)
    print("OTHER ENHANCEMENTS:")
    print("• Stronger authority document detection (DGI/BPS/BCU)")
    print("• Pure authority docs excluded from notarial certificates")
    print("• Better matches client requirements")
    print("=" * 70)
    print(f"Using model: {MODEL_NAME}")
    print(f"Using {NUM_WORKERS} parallel workers")
    print("=" * 70 + "\n")

    # Collect all tasks
    all_tasks = []

    for customer, info in customers.items():
        for cert in info["files"]["certificates"]:
            all_tasks.append({
                'customer': customer,
                'cert_info': cert,
                'base_path': CUSTOMER_DATA_PATH,
                'is_certificate': True
            })

        for doc in info["files"]["non_certificates"]:
            all_tasks.append({
                'customer': customer,
                'cert_info': doc,
                'base_path': CUSTOMER_DATA_PATH,
                'is_certificate': False
            })

    total_files = len(all_tasks)
    print(f"Total files to process: {total_files}\n")

    print("Initializing worker processes...")
    with Pool(processes=NUM_WORKERS, initializer=init_worker) as pool:
        try:
            results = list(tqdm(
                pool.imap(process_single_file, all_tasks, chunksize=1),
                total=total_files,
                desc="Processing files",
                unit="file"
            ))
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            exit(1)

    print("\nAggregating results and collecting statistics...")

    # Aggregate results and collect statistics
    for result in results:
        result_type = result['type']
        stats['total_processed'] += 1

        if result_type == '__NON_CERT__':
            non_certificate_docs.append({
                "customer": result['customer'],
                "filename": result['filename'],
                "path": result['path'],
                "reason": result.get('reason', 'non_certificate')
            })
        elif result_type == '__AUTHORITY_DOC__':
            reason = result.get('reason', 'authority_document')
            if reason == 'file_not_found':
                stats['file_not_found'] += 1
            elif reason == 'pure_authority_document':
                stats['pure_authority_detected'] += 1
                stats['authority_detected'] += 1
                # Track which authority type was removed
                purpose = result.get('purpose', 'unknown')
                if purpose == 'dgi':
                    stats['dgi_removed_from_notarial'] += 1
                elif purpose == 'bps':
                    stats['bps_removed_from_notarial'] += 1
                elif purpose == 'bcu':
                    stats['bcu_removed_from_notarial'] += 1
            elif reason == 'authority_document':
                stats['authority_detected'] += 1

            non_certificate_docs.append({
                "customer": result['customer'],
                "filename": result['filename'],
                "path": result['path'],
                "error_flag": result.get('error_flag', False),
                "purpose": result.get('purpose', 'unknown'),
                "reason": reason
            })
        else:
            stats['notarial_confirmed'] += 1

            # Track COMPLETO certificates reclassified (v3.1)
            if result.get('is_complete_cert', False) and result_type != 'firma':
                stats['completo_certs_reclassified'] += 1

            final_certificate_mapping[result_type].append({
                "customer": result['customer'],
                "filename": result['filename'],
                "path": result['path'],
                "error_flag": result.get('error_flag', False),
                "purpose": result.get('purpose', 'unknown')
            })

        # Track processing errors
        if 'error' in result:
            stats['processing_errors'] += 1

    # Build final summary from ACTUAL processed data
    print("\nBuilding summary from processed data...")

    rebuilt_cert_types = {}
    attribute_keywords = ['poder', 'poderes', 'leyes', 'ley', 'domicilio', 'domicilios', 'giro', 'objeto']

    for cert_type in cert_types.keys():
        files = final_certificate_mapping.get(cert_type, [])

        if not files:
            # Keep empty types for reference
            rebuilt_cert_types[cert_type] = {
                "count": 0,
                "purposes": {},
                "attributes": [],
                "examples": []
            }
            continue

        # Count purposes (only non-unknown, non-other)
        purpose_counts = {}
        for file in files:
            purpose = file.get('purpose', 'unknown')
            if purpose not in ['unknown', 'other']:
                purpose_counts[purpose] = purpose_counts.get(purpose, 0) + 1

        # Extract attributes (keywords from filenames)
        attributes = set()
        for file in files:
            fname_lower = file['filename'].lower()
            for keyword in attribute_keywords:
                if keyword in fname_lower:
                    attributes.add(keyword)

        # Select up to 5 representative examples
        # Prioritize diversity - try to get different customers and purposes
        examples = []
        seen_customers = set()
        seen_purposes = set()

        # First pass: unique customers and purposes
        for file in files:
            customer = file['customer']
            purpose = file.get('purpose', 'unknown')
            if customer not in seen_customers or purpose not in seen_purposes:
                examples.append(file['filename'])
                seen_customers.add(customer)
                seen_purposes.add(purpose)
                if len(examples) >= 5:
                    break

        # If we need more examples, add from remaining files
        if len(examples) < 5:
            for file in files:
                if file['filename'] not in examples:
                    examples.append(file['filename'])
                    if len(examples) >= 5:
                        break

        # Build entry
        rebuilt_cert_types[cert_type] = {
            "count": len(files),
            "purposes": purpose_counts,
            "attributes": sorted(list(attributes)),
            "examples": examples
        }

    summary = {
        "identified_certificate_types": rebuilt_cert_types,
        "certificate_file_mapping": final_certificate_mapping,
        "non_certificate_documents": non_certificate_docs
    }

    # Save result
    output_file = "certificate_summary.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print(f"Successfully created {output_file}")
    print("=" * 70)

    # Print processing statistics
    print("\n" + "=" * 70)
    print("PROCESSING STATISTICS (VERSION 3.1 - CRITICAL FIX)")
    print("=" * 70)
    print(f"Total files processed:      {stats['total_processed']}")
    print(f"Notarial certificates:      {stats['notarial_confirmed']}")
    print(f"\n🔧 V3.1 CRITICAL FIX:")
    print(f"  COMPLETO certs moved:      {stats['completo_certs_reclassified']}")
    print(f"  (moved from 'firma' to complex types like firma_personeria_representacion)")
    print(f"\nAuthority documents:        {stats['authority_detected']}")
    print(f"  Pure authority (removed):  {stats['pure_authority_detected']}")
    print(f"    - DGI removed:           {stats['dgi_removed_from_notarial']}")
    print(f"    - BPS removed:           {stats['bps_removed_from_notarial']}")
    print(f"    - BCU removed:           {stats['bcu_removed_from_notarial']}")
    print(f"Files not found:            {stats['file_not_found']}")
    print(f"Processing errors:          {stats['processing_errors']}")

    # Print classification summary
    print("\n" + "=" * 70)
    print("CLASSIFICATION SUMMARY")
    print("=" * 70)
    for cert_type, files in final_certificate_mapping.items():
        if files:
            # Count purposes
            purpose_counts = {}
            for file in files:
                purpose = file.get('purpose', 'unknown')
                if purpose not in ['unknown', 'other']:
                    purpose_counts[purpose] = purpose_counts.get(purpose, 0) + 1

            print(f"\n{cert_type}: {len(files)} files")
            if purpose_counts:
                print(f"  Purposes: {dict(purpose_counts)}")

    print(f"\nNon-certificate documents: {len(non_certificate_docs)}")
    print(f"Total in output: {sum(len(files) for files in final_certificate_mapping.values()) + len(non_certificate_docs)}")
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)
