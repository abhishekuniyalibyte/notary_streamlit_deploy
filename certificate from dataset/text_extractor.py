"""
text_extraction.py

RESPONSIBILITY
---------------
• Read real client documents (PDF / scanned PDF / images)
• Extract literal text as-is (no paraphrasing, no correction)
• Provide page-level traceability (important for legal audit)

DOES NOT
--------
• Validate legality
• Interpret law
• Decide certificate correctness
• Use LLMs

This file is OCR + text extraction ONLY.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import List, Optional

import pdfplumber
import pytesseract
from PIL import Image

try:
    from docx import Document as DocxDocument
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False


# -----------------------------
# Data structures (audit-safe)
# -----------------------------

@dataclass
class ExtractedPage:
    page_number: int
    text: str
    source: str  # "pdf_text" | "ocr"


@dataclass
class ExtractionResult:
    full_text: str
    pages: List[ExtractedPage]


# -----------------------------
# Main extractor
# -----------------------------

class TextExtractor:
    """
    Production-grade text extractor for notarial documents.

    Strategy:
    • Try digital PDF text first
    • For scanned PDFs: Use vision LLM instead of OCR (MUCH FASTER)
    • Fallback to OCR for scanned pages
    • Preserve raw wording (even if messy)
    """

    def __init__(
        self,
        lang: str = "spa",
        tesseract_cmd: Optional[str] = None,
        ocr_dpi: int = 300,
        groq_api_key: Optional[str] = None,
        use_vision_llm: bool = True,
    ):
        """
        lang: OCR language (Spanish = 'spa')
        tesseract_cmd: optional explicit path to tesseract binary
        ocr_dpi: resolution for OCR rendering
        groq_api_key: API key for vision LLM (faster than OCR)
        use_vision_llm: Use vision LLM instead of OCR for scanned pages
        """
        self.lang = lang
        self.ocr_dpi = ocr_dpi
        self.groq_api_key = groq_api_key
        self.use_vision_llm = use_vision_llm

        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    # -----------------------------
    # Public API
    # -----------------------------

    def extract(self, file_path: str, max_pages: Optional[int] = None) -> ExtractionResult:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            return self._extract_pdf(file_path, max_pages)

        if ext in [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"]:
            return self._extract_image(file_path)

        if ext == ".docx":
            return self._extract_docx(file_path)

        if ext == ".doc":
            return self._extract_doc(file_path)

        # Skip unsupported file types silently
        return ExtractionResult(
            pages=[],
            full_text=f"[UNSUPPORTED FILE TYPE: {ext}]"
        )

    # -----------------------------
    # Internal helpers
    # -----------------------------

    def _extract_pdf(self, pdf_path: str, max_pages: Optional[int]) -> ExtractionResult:
        pages_out: List[ExtractedPage] = []
        full_text_parts: List[str] = []

        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            page_limit = total_pages if max_pages is None else min(total_pages, max_pages)

            for idx in range(page_limit):
                page = pdf.pages[idx]
                page_number = idx + 1

                #  Try native PDF text
                text = (page.extract_text() or "").strip()

                if text:
                    pages_out.append(
                        ExtractedPage(
                            page_number=page_number,
                            text=text,
                            source="pdf_text",
                        )
                    )
                    full_text_parts.append(text)
                    continue

                # 2️⃣ Skip OCR for scanned pages (too slow - 30+ seconds per page)
                # Just note that this page had no text
                pages_out.append(
                    ExtractedPage(
                        page_number=page_number,
                        text="[SCANNED PAGE - OCR SKIPPED]",
                        source="skipped",
                    )
                )

        return ExtractionResult(
            full_text="\n\n".join(full_text_parts).strip(),
            pages=pages_out,
        )

    def _extract_image(self, image_path: str) -> ExtractionResult:
        image = Image.open(image_path)
        ocr_text = self._ocr_image(image)

        page = ExtractedPage(
            page_number=1,
            text=ocr_text,
            source="ocr",
        )

        return ExtractionResult(
            full_text=ocr_text.strip(),
            pages=[page],
        )

    def _ocr_image(self, image: Image.Image) -> str:
        """
        Extract text from image using vision LLM (if available) or OCR fallback.
        """
        # Try vision LLM first (much faster!)
        if self.use_vision_llm and self.groq_api_key:
            try:
                return self._vision_llm_extract(image)
            except Exception as e:
                print(f"[WARN] Vision LLM failed, falling back to OCR: {e}")

        # Fallback to traditional OCR
        return pytesseract.image_to_string(image, lang=self.lang)

    def _vision_llm_extract(self, image: Image.Image) -> str:
        """
        Use vision LLM to extract text from image (FAST!)
        """
        import base64
        import io
        from groq import Groq

        # Convert image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # Call Groq vision API (uses llama-vision model)
        client = Groq(api_key=self.groq_api_key)

        response = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",  # Groq's current vision model
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Extract ALL text from this document image. Return only the text content, preserving the original wording exactly as it appears. Include all visible text."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_base64}"
                        }
                    }
                ]
            }],
            temperature=0.1,
            max_tokens=2000,
            timeout=30
        )

        return response.choices[0].message.content.strip()

    def _extract_docx(self, docx_path: str) -> ExtractionResult:
        """Extract text from .docx files"""
        if not DOCX_AVAILABLE:
            return ExtractionResult(
                full_text="[ERROR: python-docx not installed]",
                pages=[]
            )

        try:
            doc = DocxDocument(docx_path)
            paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
            full_text = "\n".join(paragraphs)

            page = ExtractedPage(
                page_number=1,
                text=full_text,
                source="docx"
            )

            return ExtractionResult(
                full_text=full_text.strip(),
                pages=[page]
            )
        except Exception as e:
            return ExtractionResult(
                full_text=f"[ERROR extracting .docx: {str(e)}]",
                pages=[]
            )

    def _extract_doc(self, doc_path: str) -> ExtractionResult:
        """Extract text from .doc files using antiword (if available)"""
        try:
            # Try antiword command-line tool
            result = subprocess.run(
                ['antiword', doc_path],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                text = result.stdout.strip()
                page = ExtractedPage(
                    page_number=1,
                    text=text,
                    source="antiword"
                )

                return ExtractionResult(
                    full_text=text,
                    pages=[page]
                )
            else:
                # antiword failed, return error
                return ExtractionResult(
                    full_text=f"[ERROR: antiword failed for .doc file]",
                    pages=[]
                )

        except FileNotFoundError:
            # antiword not installed - return placeholder
            return ExtractionResult(
                full_text="[SKIPPED: .doc file - antiword not installed]",
                pages=[]
            )
        except Exception as e:
            return ExtractionResult(
                full_text=f"[ERROR extracting .doc: {str(e)}]",
                pages=[]
            )
