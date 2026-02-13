# NOTARIA Project

## What This Project Does

This project helps a notary:

- Upload a set of documents
- Extract important information (company name, RUT, CI, dates, registry numbers, etc.)
- Validate the set against legal + institutional requirements
- Generate a certificate draft **only when** the requirements are satisfied

The goal is to make it clear what is missing, inconsistent, expired, or needs manual verification before proceeding.

## Quick Start

### Local

```bash
pip install -r requirements.txt
streamlit run chatbot.py
```

Optional:

- Set `GROQ_API_KEY` in `.env` if you want LLM extraction/classification (Groq).
- Install system tools for OCR + scanned PDF support (see `Dockerfile` for the typical packages).

### Docker

```bash
docker compose up --build
```

Open `http://localhost:8501`.

## UI (How You Use It)

The Streamlit app has two styles:

- **Chat (guided) (default):** guided workflow with a right-side “Upload & controls” panel and a left-side chat panel.
- **Form:** original form-based workflow where you pick inputs and run the full pipeline.

### Chat (Guided) Workflow

1. Say what you want to do:
   - **Create:** `create a certificate type personería for company Adecco`
   - **Validate:** `validate this certificate …`
2. Upload documents in the right panel.
3. Click **Analyze / check now** (or type `check`) to run extraction + validation.
4. (Optional) Click **Fix issues (no re-analyze)** to apply safe fixes **without** rerunning extraction:
   - Does **not** fetch anything online.
   - Only applies conservative fixes when the correct value already exists in another uploaded document (e.g. filling missing RUT/CI/company name/registry number, normalizing formatting differences).
5. Click **Generate downloads** to produce “fixed” versions of uploaded files (saved to disk):
   - Outputs are written under `output/fixed/<chat_case_id>/`
   - Files keep the **same filename and extension** as the upload (`.pdf` stays `.pdf`, `.doc` stays `.doc`, `.docx` stays `.docx`)
   - Original uploaded files are **not modified**
6. Review the verdict and action items:
   - **CORRECT / COMPLETE** → the case can proceed
   - **NOT COMPLETE / NEEDS FIXES** → missing/expired/review-needed items block proceeding

## Inputs (What The Notary Provides)

- **Certificate type** and **purpose/destination** (e.g., BPS). These decide what the system will require and validate.
- **Subject name** (company/person). If not provided, the system may infer it from extracted text, which can introduce inconsistencies if multiple name variations exist.
- **Documents** (upload multiple files or provide a local folder path). If you mix different companies in the same folder, consistency warnings may block generation.
- Optional:
  - **Catalog** built from an Excel file (e.g. `ACRISOUND.xlsx`) to improve matching/reporting. It does **not** define legal requirements.
  - **OCR fallback** and **LLM extraction/classification (Groq)**. OCR helps scanned documents; LLM helps extraction/classification but low confidence still requires manual review.

## End-to-End Workflow

Input → Processing → Output

1. Inputs collected (certificate type, purpose, subject, files or folder path).
2. Files normalized into a list of `{path, filename}` items:
   - Uploads are saved to a temp directory with stable local paths.
   - Folder mode adds every file in the folder as part of the same request.
3. `run_flow(...)` executes the pipeline and produces:
   - Per-file analysis (`file_results`)
   - Phase summaries (`phase1`…`phase11`)
   - Optional reports saved to disk (e.g. `notary_summary.txt`, `notary_detailed_report.txt`, `notary_file_report.csv`)

## Pipeline Phases (What Happens In Each Phase)

This is what `notary_phase_outputs.txt` prints in order:

1. **Phase 1 — Intent**
   - Captures the request (certificate type + purpose/destination + subject).
2. **Phase 2 — Requirements (Legal + Institutional)**
   - Loads legal articles and institutional rules (e.g. BPS expiry windows).
   - Builds the required-documents checklist (mandatory/optional, expiry rules, legal basis labels).
3. **Phase 3 — Document Collection & Coverage**
   - Indexes files and assigns metadata (format, size, scanned/digital flags when available).
   - Reports missing required documents early.
4. **Phase 4 — Extraction**
   - Extracts text (with OCR fallback if enabled) and extracts key fields.
   - Produces extraction warnings (low-quality scans, empty text, OCR-only reads).
5. **Phase 5 — Validation Matrix**
   - Checks presence of required documents.
   - Validates required elements/fields and article compliance where mapped.
   - Checks cross-document consistency (e.g. company name and RUT consistency).
6. **Phase 6 — Gap Analysis**
   - Converts validation output into actionable gaps (missing documents, expired documents, missing data, inconsistencies, format issues, catalog mismatches).
7. **Phase 7 — Update Attempt**
   - Tracks updates and re-extraction after changes.
   - Includes a conservative “system correction” step (local-only) that can fill missing extracted fields from other uploaded documents.
8. **Phase 8 — Final Confirmation**
   - Re-validates and makes the final go/no-go decision.
9. **Phase 9 — Certificate Generation**
   - Generates the certificate draft when Phase 8 allows proceeding.
10. **Phase 10 — Notary Review**
   - Captures notary edits and feedback.
11. **Phase 11 — Final Output**
   - Produces the final outputs for delivery.

### How This Maps To The Chat Buttons

- **Analyze / check now**: runs extraction + validation (Phases 1–8), then proceeds to certificate generation only if allowed.
- **Fix issues (no re-analyze)**: applies the conservative “system correction” step on the latest extracted data **without** rerunning Phase 4 extraction.
- **Generate downloads**: regenerates “fixed” output files from the latest extracted text and saves them under `output/fixed/<chat_case_id>/` (same filename + extension).

## Outputs

- `notary_summary.txt`: short overview of the run (saved under `output/runs/<chat_case_id>/` in Chat mode).
- `notary_detailed_report.txt`: detailed per-file explanation (saved under `output/runs/<chat_case_id>/` in Chat mode).
- `notary_file_report.csv`: spreadsheet of file-level classification/validation signals (saved under `output/` in Form mode).
- `notary_phase_outputs.txt`: combined run log across phases (saved under `output/runs/<chat_case_id>/` in Chat mode and under `output/` in Form mode).
- `generated_certificate.txt`: appears only when Phase 8 approves certificate generation (saved under `output/runs/<chat_case_id>/` in Chat mode).

### Fixed Outputs (Chat Mode)

After clicking **Generate downloads**, fixed files are written to:

- `output/fixed/<chat_case_id>/`

Notes:

- Fixed files keep the **same filename + extension** as the original upload.
- Fixed PDF/DOC/DOCX files are regenerated from extracted text: they preserve the file type/name, but they do **not** preserve original layout, stamps, signatures, or embedded images.
- PDF generation uses `reportlab` when available; otherwise it falls back to a minimal built-in PDF writer.

## How To Interpret Results (Common Confusion)

**Per-file “VALID” does NOT mean the case is complete.**

- Per-file status (“VALID/INVALID/NOT_REQUIRED”) is about that single file.
- Overall status (“VALID” vs “ATTENTION REQUIRED”) is about the entire case (all mandatory docs present, not expired, and no blocking gaps).

If the summary says “Missing Required Documents …” or “Errors Found: Yes”, Phase 8 will reject generation.

## Limitations / What “Fix” Means

- The “Fix issues” step is **conservative** and **local-only**:
  - It can fill missing extracted fields when the correct value exists in another uploaded document.
  - It can normalize minor formatting differences (e.g. RUT formatting, name punctuation/case/accents).
  - It does **not** fetch anything online.
- Expired documents cannot be truly fixed automatically. If a required document is expired (e.g. BPS/DGI freshness rules), you must upload a newer official document.

## Where The Rules Come From

The mapping is defined in `src/phase2_legal_requirements.py`. It takes certificate type + purpose and returns a structured checklist:

- Mandatory vs optional documents
- Expiry rules (freshness windows)
- Legal basis labels used in reporting

Additional configuration lives in `config/`:

- `config/certificate_requirements.json`: certificate-type specific requirements/checklists used by the app.
- `config/error_patterns.json`: known error patterns used for gap/error grouping.

## Dataset / Catalog Files

### `certificate from dataset/certificate_summary.json`

Taxonomy/reference used for classification and dataset-style matching. It does not decide what is legally required.

### `certificate from dataset/customers_index.json`

Index of historical files grouped by customer. Used to build derived summaries; not used directly at runtime.

### Catalog JSON file

Catalogs created from Excel (e.g. `ACRISOUND.xlsx`) help match uploaded filenames to expected documents and warn on format mismatches.

## Which Folders/Files To Upload (Important)

- Client input documents should come from: `Notaria_client_data/<Client Name>/`
  - Example: `Notaria_client_data/DATYLCO/`
  - Example: `Notaria_client_data/Inversora Rinlen SA/`
- Do **not** upload/use files from: `certificate from dataset/`
  - Those files are reference/taxonomy and catalog data, not source documents for a case.

## Limitations

- Extraction can miss fields on low-quality scans. When required fields are missing, the pipeline may block until the notary uploads clearer documents or manually reviews.
- **Expired documents cannot be auto-fixed.** If BPS/DGI freshness rules flag a doc as expired, you must upload a newer official document.
- Some checks are conservative and may require manual review (destination entity, document source, signature presence).
- Web search fallback exists but is not implemented for real results.

## Repo Hygiene (Recommended `.gitignore`)

Do not commit runtime or client-document folders. Typical ignores:

- `.tmp_uploads/` (uploaded case files; can be large and sensitive)
- `**/fixed/` (generated fixed outputs)
- `.env` (secrets)
- `venv/`, `__pycache__/`, `.pytest_cache/`

## How To Run (Local)

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Add `GROQ_API_KEY` to `.env` if you want LLM extraction/classification.
4. Run the app:

```bash
streamlit run chatbot.py
```

If you want legacy DOC extraction and OCR, install the required system tools (LibreOffice, OCR tools) and ensure they are available on PATH.

## How To Run With Docker

Prerequisites:

- Docker + Docker Compose
- (Optional) `.env` with `GROQ_API_KEY`
- (Optional) client folders on host under `Notaria_client_data/`

Run:

```bash
docker compose up --build
```

Open:

- `http://localhost:8501`

Notes:

- `docker-compose.yml` mounts `Notaria_client_data/` read-only at `/app/Notaria_client_data`.
- `docker-compose.yml` mounts `.tmp_uploads/` so uploaded files persist between restarts.
