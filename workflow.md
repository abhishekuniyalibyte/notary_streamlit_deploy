# 🔷 PROJECT WORKFLOW

**AI-Powered Uruguayan Notarial Certificate Automation System**

---

## 0️⃣ Core Idea (1-minute mental model)

> The system is **not** “just generating certificates”.

It is a **legal validation engine** that:

* Knows **Uruguayan notarial law**
* Knows **what documents are required per certificate type**
* Knows **how to verify freshness and correctness**
* Guides the notary step-by-step
* Generates a **legally compliant certificate only when everything is valid**

Think of it as:

> **Legal Rules Engine + Document Intelligence + Guided Notary Assistant**

---

## 1️⃣ SYSTEM ACTORS

### Human

* **Notary (primary user)**

### System

* **AI Agent**
* **Legal Knowledge Base**
* **Document Store (Drive / Uploads)**
* **Validation Engine**
* **Certificate Generator**

---

## 2️⃣ HIGH-LEVEL PHASES (Bird’s-Eye View)

1. **Notary selects certificate type**
2. **System loads legal requirements**
3. **Notary uploads documents**
4. **System extracts + validates data**
5. **Gap & error detection**
6. **Optional data update attempts**
7. **Final validation**
8. **Certificate generation**
9. **Notary feedback & learning**

Every certificate — no matter how simple — must go through **Articles 248–255**.

---

## 3️⃣ PHASE-BY-PHASE DETAILED WORKFLOW

---

## 🟦 PHASE 1 — Certificate Intent Definition

### Goal

Understand **what legal instrument** is being created and **for what purpose**.

### Steps

1. Notary selects:

   * **Certificate type**

     * e.g.:

       * Certificación de firmas
       * Certificado de personería
       * Certificado de representación
       * Certificado de situación jurídica
       * Certificado para BPS / MSP / Abitab
   * **Purpose**

     * e.g.:

       * “Para BPS”
       * “Para Abitab”
       * “Para compraventa”
   * **Subject**

     * Person / Company name

### Output

```json
{
  "certificate_type": "certificado_de_firmas",
  "purpose": "para_abitab",
  "subject": "INVERSORA RINLEN S.A."
}
```

This is the **trigger** for the entire legal pipeline.

---

## 🟦 PHASE 2 — Legal Requirement Resolution (Rules Engine)

### Goal

Determine **exactly which laws apply**.

### What the system does

1. Loads **Articles 248–255** (mandatory for all)
2. Adds **cross-references**

   * Example:

     * Art. 250 → Art. 130 (identification rules)
3. Adds **institution-specific rules**

   * Abitab → representation + 30-day validity
   * BPS → registry inscriptions, dates, padrones

### Output (internal)

A **structured checklist**, NOT text.

```json
{
  "mandatory_articles": [248, 249, 250, 255],
  "cross_references": [130],
  "required_elements": [
    "identity_verification",
    "document_source",
    "signature_presence",
    "destination_entity",
    "validity_dates"
  ]
}
```

This prevents hallucinations.
The AI is **not guessing** — it is following rules.

---

## 🟦 PHASE 3 — Document Intake

### Goal

Collect **evidence**.

### Sources

* Direct upload (PDF, DOCX, JPG)
* Google Drive (future)
* Previously stored documents (dataset of 911 items)

### System actions

* Index documents by:

  * Client
  * Type
  * Date
  * Institution
* Detect:

  * Scanned vs digital
  * Language (Spanish expected)

### Output

```json
{
  "uploaded_documents": [
    "estatuto.pdf",
    "acta_directorio.pdf",
    "certificado_bps.pdf"
  ]
}
```

---

## 🟦 PHASE 4 — Text Extraction & Structuring

### Goal

Turn raw files into **structured facts**.

### Pipeline

1. **PDF text extraction**
2. **OCR fallback** if needed
3. Normalize:

   * Names
   * Dates
   * Registry numbers
4. Produce **raw extracted JSON**

Example:

```json
{
  "company_name": "INVERSORA RINLEN S.A.",
  "rut": "21XXXXXXX",
  "representative": "Juan Pérez",
  "acta_date": "2022-06-14",
  "document_source": "Registro de Comercio"
}
```

This step is **pure extraction**, no legal judgment yet.

---

## 🟦 PHASE 5 — Legal Validation Engine

### Goal

Check extracted data **against law**.

### What is validated

* Identity completeness (Art. 130)
* Signature presence (Art. 250 / 251)
* Document source legitimacy (Art. 249)
* Required mentions (Art. 255)
* Destination correctness
* Expiration dates (institutional)

### Output

A **validation matrix**:

```json
{
  "identity_verified": true,
  "signature_valid": true,
  "document_source_valid": true,
  "destination_present": false,
  "expiration_ok": false
}
```

---

## 🟦 PHASE 6 — Gap & Error Detection

### Goal

Tell the notary **what is wrong and why**.

### Types of issues

* ❌ Missing document
* ❌ Expired certificate
* ❌ Wrong institution format
* ❌ Missing destination
* ❌ Missing registry reference

### UI-level output

```json
{
  "errors": [
    {
      "field": "destination",
      "reason": "Artículo 255 requiere destinatario explícito"
    },
    {
      "field": "certificado_bps",
      "reason": "Documento vencido (más de 30 días)"
    }
  ]
}
```

---

## 🟦 PHASE 7 — Data Update Attempt (Optional)

### Goal

Reduce manual effort.

### Notary chooses:

* Upload updated document
* OR allow system to attempt update

### System may:

* Fetch public registry info
* Validate online records
* Replace outdated data

### Output

```json
{
  "updates_attempted": true,
  "updates_successful": false,
  "updated_fields": ["registry_status"]
}
```

Updated fields are **highlighted** for the notary.

---

## 🟦 PHASE 8 — Final Legal Confirmation

### Goal

Ensure **zero legal defects**.

### System re-runs:

* Articles 248–255
* All institution rules
* Cross-references

Only if **100% valid** → proceed.

```json
{
  "legal_status": "fully_compliant"
}
```

---

## 🟦 PHASE 9 — Certificate Generation

### Goal

Create the **official instrument**.

### Inputs

* Verified structured data
* Notary-provided template
* Institution formatting rules

### Output

* Draft certificate (Spanish)
* Correct legal language
* Proper structure
* Explicit legal references

Example:

```text
CERTIFICO: Que conforme a lo dispuesto en los artículos 248 a 255 del Reglamento Notarial...
```

---

## 🟦 PHASE 10 — Notary Review & Learning

### Goal

Human-in-the-loop correction.

### Notary can:

* Edit wording
* Adjust format
* Reject a paragraph

### System learns:

* Preferred phrasing
* Template corrections
* Institution nuances

This improves **future certificates**.

---

## 🟦 PHASE 11 — Final Output

### Formats

* PDF
* DOCX
* (Later) digitally signed certificate

### Stored with:

* Audit trail
* Source documents
* Validation report

---

## 4️⃣ HOW YOUR DATASET (911 FILES) FITS

Your dataset is:

* **Training reference**
* **Template source**
* **Historical validation examples**

It is **not training an ML model**, but:

* Teaching structure
* Teaching formatting
* Teaching institutional differences

---

## 5️⃣ WHAT THIS SYSTEM IS (AND IS NOT)

### ✅ IS

* Legal decision support system
* Rule-driven
* Auditable
* Explainable

### ❌ IS NOT

* Chatbot guessing law
* Pure OCR tool
* One-click generator without validation

---

## 6️⃣ NEXT STEP (IMPORTANT)

If you want, next I can:

1. Convert this into a **technical architecture diagram**
2. Define **exact folder structure**
3. Define **JSON schemas for rules**
4. Define **milestone-wise implementation plan**
5. Start **Phase 1 coding plan only**

Just tell me **what you want next**.

You are now **fully aligned with the project**.
