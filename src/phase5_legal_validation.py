"""
Phase 5: Legal Validation Engine

This module validates extracted data against legal requirements:
- Checks if all required documents are present
- Validates document expiration dates
- Verifies data consistency across documents
- Checks compliance with Articles 248-255
- Adds article-based compliance checks using local article texts
- Generates validation matrix

This is the core validation engine that determines if a certificate can be issued.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from datetime import datetime, timedelta
from enum import Enum
import json
import re
import unicodedata

from src.phase2_legal_requirements import (
    LegalRequirements,
    DocumentType,
    DocumentRequirement,
    RequiredElement,
    ArticleReference
)
from src.phase3_document_intake import FileFormat
from src.phase4_text_extraction import (
    CollectionExtractionResult,
    ExtractedData,
    DocumentExtractionResult
)


class ValidationStatus(Enum):
    """Status of validation checks"""
    VALID = "valid"
    INVALID = "invalid"
    WARNING = "warning"
    MISSING = "missing"
    EXPIRED = "expired"
    PENDING = "pending"


class ValidationSeverity(Enum):
    """Severity level of validation issues"""
    CRITICAL = "critical"  # Blocks certificate issuance
    ERROR = "error"  # Should be fixed
    WARNING = "warning"  # Recommended to fix
    INFO = "info"  # Informational only


@dataclass
class ValidationIssue:
    """Represents a single validation issue"""
    field: str
    issue_type: str
    severity: ValidationSeverity
    description: str
    legal_basis: Optional[str] = None  # Which article requires this
    recommendation: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "field": self.field,
            "issue_type": self.issue_type,
            "severity": self.severity.value,
            "description": self.description,
            "legal_basis": self.legal_basis,
            "recommendation": self.recommendation
        }

    def get_display(self) -> str:
        """Get formatted display string"""
        severity_icon = {
            ValidationSeverity.CRITICAL: "🔴",
            ValidationSeverity.ERROR: "🟠",
            ValidationSeverity.WARNING: "🟡",
            ValidationSeverity.INFO: "🔵"
        }

        icon = severity_icon.get(self.severity, "⚪")
        display = f"{icon} {self.field}: {self.description}"

        if self.legal_basis:
            display += f"\n      Base legal: {self.legal_basis}"
        if self.recommendation:
            display += f"\n      Recomendación: {self.recommendation}"

        return display


@dataclass
class ArticleComplianceCheck:
    """Represents compliance status for a legal article."""
    article: str
    status: ValidationStatus
    details: str
    required_elements: List[str] = field(default_factory=list)
    excerpt: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "article": self.article,
            "status": self.status.value,
            "details": self.details,
            "required_elements": self.required_elements,
            "excerpt": self.excerpt
        }


@dataclass
class DocumentValidation:
    """Validation result for a single document"""
    document_type: DocumentType
    required: bool
    present: bool
    status: ValidationStatus
    issues: List[ValidationIssue] = field(default_factory=list)
    extracted_data: Optional[ExtractedData] = None
    file_name: Optional[str] = None
    file_format: Optional[str] = None
    catalog_info: Optional[Dict[str, any]] = None

    def is_valid(self) -> bool:
        """Check if document passes validation"""
        if self.required and not self.present:
            return False
        if self.status in [ValidationStatus.INVALID, ValidationStatus.EXPIRED, ValidationStatus.MISSING]:
            return False
        return not any(
            issue.severity in (ValidationSeverity.CRITICAL, ValidationSeverity.ERROR)
            for issue in self.issues
        )

    def to_dict(self) -> dict:
        return {
            "document_type": self.document_type.value if self.document_type else None,
            "required": self.required,
            "present": self.present,
            "status": self.status.value,
            "is_valid": self.is_valid(),
            "issues": [issue.to_dict() for issue in self.issues],
            "file_name": self.file_name,
            "file_format": self.file_format,
            "catalog_info": self.catalog_info
        }


@dataclass
class ElementValidation:
    """Validation result for a required element"""
    element: RequiredElement
    status: ValidationStatus
    value_found: Optional[str] = None
    issues: List[ValidationIssue] = field(default_factory=list)

    def is_valid(self) -> bool:
        """Check if element passes validation"""
        return self.status == ValidationStatus.VALID

    def to_dict(self) -> dict:
        return {
            "element": self.element.value,
            "status": self.status.value,
            "value_found": self.value_found,
            "is_valid": self.is_valid(),
            "issues": [issue.to_dict() for issue in self.issues]
        }


@dataclass
class ValidationMatrix:
    """
    Complete validation matrix for a certificate request.
    This is the output of Phase 5.
    """
    legal_requirements: LegalRequirements
    extraction_result: CollectionExtractionResult

    # Validation results
    document_validations: List[DocumentValidation] = field(default_factory=list)
    element_validations: List[ElementValidation] = field(default_factory=list)
    cross_document_issues: List[ValidationIssue] = field(default_factory=list)
    article_checks: List[ArticleComplianceCheck] = field(default_factory=list)

    # Summary
    validation_timestamp: datetime = field(default_factory=datetime.now)
    overall_status: ValidationStatus = ValidationStatus.PENDING
    can_issue_certificate: bool = False

    def get_all_issues(self) -> List[ValidationIssue]:
        """Get all validation issues"""
        all_issues = []

        for doc_val in self.document_validations:
            all_issues.extend(doc_val.issues)

        for elem_val in self.element_validations:
            all_issues.extend(elem_val.issues)

        all_issues.extend(self.cross_document_issues)

        return all_issues

    def get_critical_issues(self) -> List[ValidationIssue]:
        """Get only critical issues that block certificate issuance"""
        return [issue for issue in self.get_all_issues()
                if issue.severity == ValidationSeverity.CRITICAL]

    def get_issue_count_by_severity(self) -> Dict[ValidationSeverity, int]:
        """Get count of issues by severity"""
        counts = {severity: 0 for severity in ValidationSeverity}
        for issue in self.get_all_issues():
            counts[issue.severity] += 1
        return counts

    def to_dict(self) -> dict:
        issue_counts = self.get_issue_count_by_severity()

        return {
            "validation_timestamp": self.validation_timestamp.isoformat(),
            "overall_status": self.overall_status.value,
            "can_issue_certificate": self.can_issue_certificate,
            "issue_summary": {
                "critical": issue_counts[ValidationSeverity.CRITICAL],
                "error": issue_counts[ValidationSeverity.ERROR],
                "warning": issue_counts[ValidationSeverity.WARNING],
                "info": issue_counts[ValidationSeverity.INFO]
            },
            "document_validations": [dv.to_dict() for dv in self.document_validations],
            "element_validations": [ev.to_dict() for ev in self.element_validations],
            "cross_document_issues": [issue.to_dict() for issue in self.cross_document_issues],
            "article_checks": [check.to_dict() for check in self.article_checks]
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def get_summary(self) -> str:
        """Get human-readable summary in Spanish"""
        issue_counts = self.get_issue_count_by_severity()

        status_icon = {
            ValidationStatus.VALID: "✅",
            ValidationStatus.INVALID: "❌",
            ValidationStatus.WARNING: "⚠️",
            ValidationStatus.PENDING: "⏳"
        }

        icon = status_icon.get(self.overall_status, "❓")

        summary = f"""
╔══════════════════════════════════════════════════════════════╗
║              MATRIZ DE VALIDACIÓN - FASE 5                   ║
╚══════════════════════════════════════════════════════════════╝

{icon} ESTADO GENERAL: {self.overall_status.value.upper()}
   ¿Puede emitir certificado?: {"✅ SÍ" if self.can_issue_certificate else "❌ NO"}

📊 RESUMEN DE PROBLEMAS:
   🔴 Críticos: {issue_counts[ValidationSeverity.CRITICAL]}
   🟠 Errores: {issue_counts[ValidationSeverity.ERROR]}
   🟡 Advertencias: {issue_counts[ValidationSeverity.WARNING]}
   🔵 Info: {issue_counts[ValidationSeverity.INFO]}

📄 VALIDACIÓN DE DOCUMENTOS ({len(self.document_validations)} total):
"""

        for doc_val in self.document_validations:
            doc_icon = "✅" if doc_val.is_valid() else "❌"
            doc_type = doc_val.document_type.value if doc_val.document_type else "desconocido"
            status_text = "REQUERIDO" if doc_val.required else "OPCIONAL"
            present_text = "PRESENTE" if doc_val.present else "FALTANTE"

            summary += f"\n   {doc_icon} {doc_type.upper()} [{status_text}] - {present_text}"

            if doc_val.issues:
                for issue in doc_val.issues:
                    summary += f"\n      {issue.get_display()}"

        summary += f"\n\n🔍 VALIDACIÓN DE ELEMENTOS ({len(self.element_validations)} total):\n"

        for elem_val in self.element_validations:
            elem_icon = "✅" if elem_val.is_valid() else "❌"
            elem_name = elem_val.element.value.replace('_', ' ').title()

            summary += f"\n   {elem_icon} {elem_name}: {elem_val.status.value.upper()}"
            if elem_val.value_found:
                summary += f" (Valor: {elem_val.value_found})"

            if elem_val.issues:
                for issue in elem_val.issues:
                    summary += f"\n      {issue.get_display()}"

        if self.cross_document_issues:
            summary += f"\n\n⚠️ PROBLEMAS DE CONSISTENCIA ENTRE DOCUMENTOS:\n"
            for issue in self.cross_document_issues:
                summary += f"\n   {issue.get_display()}"

        if self.article_checks:
            summary += f"\n\n📚 VERIFICACIÓN DE ARTÍCULOS:\n"
            status_icon = {
                ValidationStatus.VALID: "✅",
                ValidationStatus.INVALID: "❌",
                ValidationStatus.WARNING: "⚠️",
                ValidationStatus.PENDING: "⏳",
                ValidationStatus.MISSING: "❓",
                ValidationStatus.EXPIRED: "⚠️"
            }
            for check in self.article_checks:
                icon = status_icon.get(check.status, "❓")
                summary += f"\n   {icon} Art. {check.article} - {check.status.value.upper()}: {check.details}"
                if check.required_elements:
                    summary += f"\n      Elementos: {', '.join(check.required_elements)}"
                if check.excerpt:
                    summary += f"\n      Extracto: {check.excerpt}"

        if self.can_issue_certificate:
            summary += "\n\n✅ TODOS LOS REQUISITOS CUMPLIDOS - LISTO PARA GENERAR CERTIFICADO"
        else:
            critical = self.get_critical_issues()
            if critical:
                summary += f"\n\n❌ NO PUEDE EMITIR CERTIFICADO - {len(critical)} PROBLEMAS CRÍTICOS:\n"
                for issue in critical:
                    summary += f"\n   {issue.get_display()}"

        return summary


class LegalValidator:
    """
    Main validation engine.
    Validates documents and data against legal requirements.
    """

    MIN_TEXT_CHARS = 40

    SPANISH_MONTHS = {
        "enero": 1,
        "febrero": 2,
        "marzo": 3,
        "abril": 4,
        "mayo": 5,
        "junio": 6,
        "julio": 7,
        "agosto": 8,
        "septiembre": 9,
        "setiembre": 9,
        "octubre": 10,
        "noviembre": 11,
        "diciembre": 12,
    }

    DOCUMENT_FIELD_REQUIREMENTS: Dict[DocumentType, List[Tuple[str, ValidationSeverity, str]]] = {
        DocumentType.CEDULA_IDENTIDAD: [
            ("ci", ValidationSeverity.ERROR, "No se encontró CI en la cédula de identidad."),
        ],
        DocumentType.REGISTRO_COMERCIO: [
            ("registro_comercio", ValidationSeverity.ERROR, "No se encontró inscripción en Registro de Comercio."),
        ],
        DocumentType.PADRON_BPS: [
            ("padron_bps", ValidationSeverity.ERROR, "No se encontró padrón BPS."),
        ],
        DocumentType.CERTIFICADO_BPS: [
            ("padron_bps", ValidationSeverity.WARNING, "No se encontró padrón BPS en el certificado BPS."),
        ],
        DocumentType.CERTIFICADO_DGI: [
            ("rut", ValidationSeverity.ERROR, "No se encontró RUT en el certificado DGI."),
        ],
        DocumentType.ESTATUTO: [
            ("company_name", ValidationSeverity.ERROR, "No se encontró el nombre de la empresa en el estatuto."),
        ],
        DocumentType.CONTRATO_SOCIAL: [
            ("company_name", ValidationSeverity.ERROR, "No se encontró el nombre de la empresa en el contrato social."),
        ],
        DocumentType.ACTA_DIRECTORIO: [
            ("acta_number", ValidationSeverity.WARNING, "No se encontró número de acta en el acta de directorio."),
        ],
        DocumentType.CERTIFICADO_VIGENCIA: [
            ("company_name", ValidationSeverity.WARNING, "No se encontró el nombre de la empresa en el certificado de vigencia."),
        ],
        DocumentType.PODER: [
            ("company_name", ValidationSeverity.WARNING, "No se encontró el nombre de la empresa en el poder."),
        ],
    }

    ARTICLE_FILES = {
        ArticleReference.ART_130: "articles/article_130.txt",
        ArticleReference.ART_248: "articles/articles_248_255.txt",
        ArticleReference.ART_249: "articles/articles_248_255.txt",
        ArticleReference.ART_250: "articles/articles_248_255.txt",
        ArticleReference.ART_251: "articles/articles_248_255.txt",
        ArticleReference.ART_252: "articles/articles_248_255.txt",
        ArticleReference.ART_253: "articles/articles_248_255.txt",
        ArticleReference.ART_254: "articles/articles_248_255.txt",
        ArticleReference.ART_255: "articles/articles_248_255.txt",
    }

    ARTICLE_ELEMENT_MAP = {
        ArticleReference.ART_130: [
            RequiredElement.IDENTITY_VERIFICATION
        ],
        ArticleReference.ART_248: [
            RequiredElement.COMPANY_NAME,
            RequiredElement.RUT_NUMBER,
            RequiredElement.LEGAL_REPRESENTATIVE
        ],
        ArticleReference.ART_249: [
            RequiredElement.DOCUMENT_SOURCE
        ],
        ArticleReference.ART_250: [
            RequiredElement.SIGNATURE_PRESENCE
        ],
        ArticleReference.ART_251: [
            RequiredElement.SIGNATURE_PRESENCE
        ],
        ArticleReference.ART_252: [
            RequiredElement.DESTINATION_ENTITY
        ],
        ArticleReference.ART_253: [
            RequiredElement.DESTINATION_ENTITY
        ],
        ArticleReference.ART_254: [
            RequiredElement.DESTINATION_ENTITY
        ],
        ArticleReference.ART_255: [
            RequiredElement.DESTINATION_ENTITY,
            RequiredElement.VALIDITY_DATES
        ],
    }

    @staticmethod
    def _normalize_text(value: str) -> str:
        if not value:
            return ""
        normalized = unicodedata.normalize("NFKD", value)
        return normalized.encode("ascii", "ignore").decode("ascii").lower()

    @staticmethod
    def _parse_date_string(value: str) -> Optional[datetime]:
        if not value:
            return None
        value = value.strip()

        match = re.search(r"(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})", value)
        if match:
            day, month, year = match.groups()
            return LegalValidator._safe_date(day, month, year)

        match = re.search(r"(\d{4})[./-](\d{1,2})[./-](\d{1,2})", value)
        if match:
            year, month, day = match.groups()
            return LegalValidator._safe_date(day, month, year)

        match = re.search(r"(\d{1,2})\s+de\s+([A-Za-zÁÉÍÓÚÑáéíóúñ]+)\s+de\s+(\d{4})", value)
        if match:
            day, month_name, year = match.groups()
            month_key = LegalValidator._normalize_text(month_name)
            month = LegalValidator.SPANISH_MONTHS.get(month_key)
            if month:
                return LegalValidator._safe_date(day, str(month), year)

        return None

    @staticmethod
    def _safe_date(day: str, month: str, year: str) -> Optional[datetime]:
        try:
            day_val = int(day)
            month_val = int(month)
            year_val = int(year)
        except ValueError:
            return None

        if year_val < 100:
            year_val += 2000 if year_val < 50 else 1900

        if not (1 <= month_val <= 12 and 1 <= day_val <= 31):
            return None

        try:
            return datetime(year_val, month_val, day_val)
        except ValueError:
            return None

    @staticmethod
    def _collect_dates(extracted_data: ExtractedData) -> List[datetime]:
        dates = []
        for raw in extracted_data.dates:
            parsed = LegalValidator._parse_date_string(raw)
            if parsed:
                dates.append(parsed)

        if not dates and extracted_data.raw_text:
            for match in re.finditer(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b", extracted_data.raw_text):
                parsed = LegalValidator._parse_date_string(match.group(0))
                if parsed:
                    dates.append(parsed)
            for match in re.finditer(
                r"\b\d{1,2}\s+de\s+[A-Za-zÁÉÍÓÚÑáéíóúñ]+\s+de\s+\d{4}\b",
                extracted_data.raw_text,
            ):
                parsed = LegalValidator._parse_date_string(match.group(0))
                if parsed:
                    dates.append(parsed)

        return dates

    @staticmethod
    def _contains_law_reference(text: str, law_number: str) -> bool:
        normalized = LegalValidator._normalize_text(text)
        if not normalized:
            return False
        return bool(re.search(rf"\b{re.escape(law_number)}\b", normalized))

    @staticmethod
    def validate_document_quality(
        doc_validation: DocumentValidation,
        req_doc: DocumentRequirement,
    ) -> None:
        extracted_data = doc_validation.extracted_data
        if not doc_validation.present:
            return
        if not extracted_data:
            severity = ValidationSeverity.ERROR if req_doc.mandatory else ValidationSeverity.WARNING
            doc_validation.issues.append(
                ValidationIssue(
                    field=req_doc.document_type.value,
                    issue_type="missing_extraction",
                    severity=severity,
                    description="El documento está presente pero no se pudo extraer contenido.",
                    legal_basis=req_doc.legal_basis,
                    recommendation="Revisar el archivo o subir una versión legible.",
                )
            )
            return

        if not extracted_data.raw_text or len(extracted_data.raw_text.strip()) < LegalValidator.MIN_TEXT_CHARS:
            severity = ValidationSeverity.ERROR if req_doc.mandatory else ValidationSeverity.WARNING
            doc_validation.issues.append(
                ValidationIssue(
                    field=req_doc.document_type.value,
                    issue_type="empty_text",
                    severity=severity,
                    description="El texto extraído es insuficiente para validar el documento.",
                    legal_basis=req_doc.legal_basis,
                    recommendation="Subir un documento legible o versión digital.",
                )
            )

        if doc_validation.file_format == FileFormat.UNKNOWN.value:
            doc_validation.issues.append(
                ValidationIssue(
                    field=req_doc.document_type.value,
                    issue_type="unsupported_format",
                    severity=ValidationSeverity.ERROR,
                    description="Formato de archivo no reconocido para validación.",
                    legal_basis=req_doc.legal_basis,
                    recommendation="Usar PDF, DOCX, DOC o imagen legible.",
                )
            )

        if extracted_data.confidence < 0.4:
            doc_validation.issues.append(
                ValidationIssue(
                    field=req_doc.document_type.value,
                    issue_type="low_confidence",
                    severity=ValidationSeverity.WARNING,
                    description="La extracción tiene baja confianza, requiere revisión manual.",
                    recommendation="Revisar manualmente el documento.",
                )
            )

        if extracted_data.extraction_method in ("ocr", "vision"):
            doc_validation.issues.append(
                ValidationIssue(
                    field=req_doc.document_type.value,
                    issue_type="manual_review_needed",
                    severity=ValidationSeverity.WARNING,
                    description="El documento fue leído por OCR/visión y requiere verificación manual.",
                    recommendation="Confirmar manualmente los datos críticos.",
                )
            )

    @staticmethod
    def validate_document_fields(
        doc_validation: DocumentValidation,
        req_doc: DocumentRequirement,
    ) -> None:
        extracted_data = doc_validation.extracted_data
        if not doc_validation.present or not extracted_data:
            return
        requirements = LegalValidator.DOCUMENT_FIELD_REQUIREMENTS.get(req_doc.document_type, [])
        for field_name, severity, description in requirements:
            if not getattr(extracted_data, field_name, None):
                doc_validation.issues.append(
                    ValidationIssue(
                        field=field_name,
                        issue_type="missing_field",
                        severity=severity,
                        description=description,
                        legal_basis=req_doc.legal_basis,
                        recommendation="Verificar que el documento incluya este dato.",
                    )
                )

    @staticmethod
    def validate_catalog_alignment(
        doc_validation: DocumentValidation,
    ) -> None:
        catalog_info = doc_validation.catalog_info or {}
        if not doc_validation.present or not catalog_info:
            return

        expected_extensions = catalog_info.get("expected_extensions") or []
        file_format = (doc_validation.file_format or "").lower()
        if expected_extensions and file_format:
            normalized_expected = [ext.lower().lstrip(".") for ext in expected_extensions]
            if file_format not in normalized_expected:
                doc_validation.issues.append(
                    ValidationIssue(
                        field="file_format",
                        issue_type="catalog_type_mismatch",
                        severity=ValidationSeverity.WARNING,
                        description=(
                            "El catalogo esperaba formato "
                            f"{', '.join(expected_extensions)} pero se subio {file_format}."
                        ),
                        legal_basis="Catalogo del cliente",
                        recommendation="Verificar si el archivo corresponde al documento esperado.",
                    )
                )

        match_status = catalog_info.get("match_status")
        if match_status == "fuzzy":
            doc_validation.issues.append(
                ValidationIssue(
                    field="file_name",
                    issue_type="catalog_fuzzy_match",
                    severity=ValidationSeverity.WARNING,
                    description=(
                        "El catalogo se vinculo con coincidencia aproximada. "
                        "Revisar que el archivo sea el correcto."
                    ),
                    legal_basis="Catalogo del cliente",
                    recommendation="Confirmar manualmente el archivo y su descripcion.",
                )
            )
        elif match_status == "normalized":
            doc_validation.issues.append(
                ValidationIssue(
                    field="file_name",
                    issue_type="catalog_normalized_match",
                    severity=ValidationSeverity.INFO,
                    description=(
                        "El catalogo se vinculo usando normalizacion de nombre. "
                        "Se recomienda una verificacion rapida."
                    ),
                    legal_basis="Catalogo del cliente",
                    recommendation="Confirmar que el nombre corresponde al documento.",
                )
            )

    @staticmethod
    def validate_format_rules(
        requirements: LegalRequirements,
        extraction_result: CollectionExtractionResult,
    ) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        institution_rules = requirements.institution_rules
        if not institution_rules or not institution_rules.format_rules:
            return issues

        combined_texts = []
        for result in extraction_result.extraction_results:
            if result.success and result.extracted_data and result.extracted_data.normalized_text:
                combined_texts.append(result.extracted_data.normalized_text)

        for rule_key, rule_value in institution_rules.format_rules.items():
            if rule_key.startswith("include_law_"):
                law_number = rule_key.replace("include_law_", "")
                has_reference = any(
                    LegalValidator._contains_law_reference(text, law_number) for text in combined_texts
                )
                if not has_reference:
                    issues.append(
                        ValidationIssue(
                            field=f"ley_{law_number}",
                            issue_type="missing_law_reference",
                            severity=ValidationSeverity.ERROR,
                            description=f"No se encontró referencia a la Ley {law_number}.",
                            legal_basis=f"Requisito {institution_rules.institution}",
                            recommendation=f"Incluir mención a la Ley {law_number} en el certificado.",
                        )
                    )
            else:
                issues.append(
                    ValidationIssue(
                        field=rule_key,
                        issue_type="format_rule_check",
                        severity=ValidationSeverity.WARNING,
                        description=f"Verificar manualmente el requisito de formato: {rule_key}",
                        legal_basis=f"Requisito {institution_rules.institution}",
                        recommendation=f"Asegurar que {rule_key} = {rule_value}.",
                    )
                )

        return issues

    @staticmethod
    def _load_article_excerpt(article: ArticleReference, max_chars: int = 160) -> Optional[str]:
        rel_path = LegalValidator.ARTICLE_FILES.get(article)
        if not rel_path:
            return None
        root = Path(__file__).resolve().parent.parent
        file_path = root / rel_path
        if not file_path.exists():
            return None
        text = file_path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            return None
        excerpt = " ".join(text.split())[:max_chars]
        return excerpt

    @staticmethod
    def validate_article_compliance(
        requirements: LegalRequirements,
        matrix: ValidationMatrix
    ) -> List[ArticleComplianceCheck]:
        """
        Check article compliance based on required elements and article texts.
        """
        checks: List[ArticleComplianceCheck] = []
        elements_by_type = {ev.element: ev for ev in matrix.element_validations}
        articles = requirements.mandatory_articles + requirements.cross_references
        seen = set()

        for article in articles:
            if article in seen:
                continue
            seen.add(article)

            mapped_elements = LegalValidator.ARTICLE_ELEMENT_MAP.get(article, [])
            element_names = [elem.value for elem in mapped_elements]
            excerpt = LegalValidator._load_article_excerpt(article)

            if not mapped_elements:
                checks.append(ArticleComplianceCheck(
                    article=article.value,
                    status=ValidationStatus.WARNING,
                    details="No hay verificación automática configurada para este artículo.",
                    required_elements=[],
                    excerpt=excerpt
                ))
                continue

            missing = [
                elem.value for elem in mapped_elements
                if elements_by_type.get(elem, None) is None
                or elements_by_type[elem].status != ValidationStatus.VALID
            ]

            if missing:
                checks.append(ArticleComplianceCheck(
                    article=article.value,
                    status=ValidationStatus.INVALID,
                    details=f"Faltan elementos requeridos: {', '.join(missing)}",
                    required_elements=element_names,
                    excerpt=excerpt
                ))
            else:
                checks.append(ArticleComplianceCheck(
                    article=article.value,
                    status=ValidationStatus.VALID,
                    details="Elementos requeridos presentes.",
                    required_elements=element_names,
                    excerpt=excerpt
                ))

        return checks

    @staticmethod
    def validate_document_presence(
        requirements: LegalRequirements,
        extraction_result: CollectionExtractionResult
    ) -> List[DocumentValidation]:
        """
        Validate that all required documents are present.
        This is the first validation step.
        """
        validations = []

        # Get all extracted document types
        present_types = set()
        extraction_map: Dict[DocumentType, DocumentExtractionResult] = {}

        for result in extraction_result.extraction_results:
            if result.success and result.extracted_data:
                doc_type = result.extracted_data.document_type
                if doc_type:
                    present_types.add(doc_type)
                    if doc_type not in extraction_map:
                        extraction_map[doc_type] = result
                    else:
                        existing = extraction_map[doc_type]
                        existing_text = (
                            existing.extracted_data.raw_text.strip()
                            if existing.extracted_data and existing.extracted_data.raw_text
                            else ""
                        )
                        current_text = (
                            result.extracted_data.raw_text.strip()
                            if result.extracted_data and result.extracted_data.raw_text
                            else ""
                        )
                        if current_text and not existing_text:
                            extraction_map[doc_type] = result

        # Check each required document
        for req_doc in requirements.required_documents:
            doc_type = req_doc.document_type
            is_present = doc_type in present_types
            extraction_result_for_doc = extraction_map.get(doc_type)
            catalog_info = None
            if extraction_result_for_doc and extraction_result_for_doc.document.metadata:
                catalog_info = extraction_result_for_doc.document.metadata.get("catalog")

            validation = DocumentValidation(
                document_type=doc_type,
                required=req_doc.mandatory,
                present=is_present,
                status=ValidationStatus.VALID if is_present else ValidationStatus.MISSING,
                extracted_data=extraction_result_for_doc.extracted_data if extraction_result_for_doc else None,
                file_name=extraction_result_for_doc.document.file_name if extraction_result_for_doc else None,
                file_format=(
                    extraction_result_for_doc.document.file_format.value
                    if extraction_result_for_doc
                    else None
                ),
                catalog_info=catalog_info,
            )

            # If required but missing, add critical issue
            if req_doc.mandatory and not is_present:
                validation.issues.append(ValidationIssue(
                    field=doc_type.value,
                    issue_type="missing_document",
                    severity=ValidationSeverity.CRITICAL,
                    description=f"Falta documento obligatorio: {req_doc.description}",
                    legal_basis=req_doc.legal_basis,
                    recommendation=f"Cargar {req_doc.description}"
                ))

            validations.append(validation)

        return validations

    @staticmethod
    def validate_document_expiry(
        doc_validation: DocumentValidation,
        req_doc: DocumentRequirement,
        extracted_data: ExtractedData
    ) -> None:
        """
        Validate document expiry dates.
        Adds issues to the DocumentValidation object.
        """
        if not req_doc.expires or not req_doc.expiry_days:
            return

        dates = LegalValidator._collect_dates(extracted_data)
        if not dates:
            severity = ValidationSeverity.ERROR if req_doc.mandatory else ValidationSeverity.WARNING
            doc_validation.issues.append(
                ValidationIssue(
                    field=f"{req_doc.document_type.value}_date",
                    issue_type="missing_date",
                    severity=severity,
                    description=f"No se pudo encontrar fecha en {req_doc.description}",
                    legal_basis=req_doc.legal_basis,
                    recommendation="Verificar que el documento incluya fecha de emisión",
                )
            )
            return

        latest_date = max(dates)
        expiry_threshold = datetime.now() - timedelta(days=req_doc.expiry_days)

        if latest_date < expiry_threshold:
            doc_validation.status = ValidationStatus.EXPIRED
            severity = ValidationSeverity.ERROR if req_doc.mandatory else ValidationSeverity.WARNING
            doc_validation.issues.append(
                ValidationIssue(
                    field=f"{req_doc.document_type.value}_expiry",
                    issue_type="expired_document",
                    severity=severity,
                    description=f"{req_doc.description} está vencido (fecha {latest_date.date()}).",
                    legal_basis=req_doc.legal_basis,
                    recommendation=f"Actualizar documento con antigüedad menor a {req_doc.expiry_days} días.",
                )
            )

    @staticmethod
    def validate_required_elements(
        requirements: LegalRequirements,
        extraction_result: CollectionExtractionResult
    ) -> List[ElementValidation]:
        """
        Validate that all required elements are present in the extracted data.
        """
        validations = []

        # Aggregate all extracted data
        all_extracted_data = [
            result.extracted_data
            for result in extraction_result.extraction_results
            if result.success and result.extracted_data
        ]

        # Check each required element
        for element in requirements.required_elements:
            validation = LegalValidator._validate_single_element(element, all_extracted_data)
            validations.append(validation)

        return validations

    @staticmethod
    def _validate_single_element(
        element: RequiredElement,
        all_extracted_data: List[ExtractedData]
    ) -> ElementValidation:
        """Validate a single required element"""

        validation = ElementValidation(
            element=element,
            status=ValidationStatus.MISSING
        )

        # Check different element types
        if element == RequiredElement.COMPANY_NAME:
            # Look for company name in any document
            for data in all_extracted_data:
                if data.company_name:
                    validation.status = ValidationStatus.VALID
                    validation.value_found = data.company_name
                    return validation

            validation.issues.append(ValidationIssue(
                field="company_name",
                issue_type="missing_element",
                severity=ValidationSeverity.CRITICAL,
                description="No se encontró nombre de la empresa",
                legal_basis="Art. 248",
                recommendation="Verificar estatuto o documentos societarios"
            ))

        elif element == RequiredElement.RUT_NUMBER:
            for data in all_extracted_data:
                if data.rut:
                    validation.status = ValidationStatus.VALID
                    validation.value_found = data.rut
                    return validation

            validation.issues.append(ValidationIssue(
                field="rut",
                issue_type="missing_element",
                severity=ValidationSeverity.CRITICAL,
                description="No se encontró RUT",
                legal_basis="Art. 248",
                recommendation="Verificar documentos tributarios"
            ))

        elif element == RequiredElement.REGISTRY_INSCRIPTION:
            for data in all_extracted_data:
                if data.registro_comercio:
                    validation.status = ValidationStatus.VALID
                    validation.value_found = data.registro_comercio
                    return validation

            validation.issues.append(ValidationIssue(
                field="registro_comercio",
                issue_type="missing_element",
                severity=ValidationSeverity.CRITICAL,
                description="No se encontró inscripción en Registro de Comercio",
                legal_basis="Art. 249",
                recommendation="Cargar certificado de Registro de Comercio"
            ))

        elif element == RequiredElement.LEGAL_REPRESENTATIVE:
            # This would require more sophisticated name extraction
            validation.status = ValidationStatus.WARNING
            validation.issues.append(ValidationIssue(
                field="legal_representative",
                issue_type="verification_needed",
                severity=ValidationSeverity.WARNING,
                description="Verificar que se identifiquen los representantes legales",
                legal_basis="Art. 248",
                recommendation="Revisar acta de directorio"
            ))

        else:
            # Default: mark as needing verification
            validation.status = ValidationStatus.WARNING
            validation.issues.append(ValidationIssue(
                field=element.value,
                issue_type="manual_verification_needed",
                severity=ValidationSeverity.WARNING,
                description=f"Verificar manualmente: {element.value.replace('_', ' ')}",
                recommendation="Revisar documentos manualmente"
            ))

        return validation

    @staticmethod
    def validate_cross_document_consistency(
        extraction_result: CollectionExtractionResult
    ) -> List[ValidationIssue]:
        """
        Validate consistency across multiple documents.
        E.g., company name should be the same in all documents.
        """
        issues = []

        # Collect all company names
        company_names = set()
        rut_numbers = set()

        for result in extraction_result.extraction_results:
            if result.success and result.extracted_data:
                if result.extracted_data.company_name:
                    company_names.add(result.extracted_data.company_name)
                if result.extracted_data.rut:
                    rut_numbers.add(result.extracted_data.rut)

        # Check for inconsistencies
        if len(company_names) > 1:
            issues.append(ValidationIssue(
                field="company_name_consistency",
                issue_type="inconsistent_data",
                severity=ValidationSeverity.ERROR,
                description=f"Nombre de empresa inconsistente entre documentos: {', '.join(company_names)}",
                recommendation="Verificar que todos los documentos correspondan a la misma empresa"
            ))

        if len(rut_numbers) > 1:
            issues.append(ValidationIssue(
                field="rut_consistency",
                issue_type="inconsistent_data",
                severity=ValidationSeverity.ERROR,
                description=f"RUT inconsistente entre documentos: {', '.join(rut_numbers)}",
                recommendation="Verificar que todos los documentos correspondan al mismo RUT"
            ))

        return issues

    @staticmethod
    def validate_date_consistency(
        extraction_result: CollectionExtractionResult
    ) -> List[ValidationIssue]:
        """
        Validate that dates are logically consistent across documents.
        Catches impossible chronologies like statute approved before company constitution.
        """
        issues = []

        # Collect all relevant dates from all documents
        constitution_dates = []
        statute_approval_dates = []
        registration_dates = []
        acta_dates = []

        for result in extraction_result.extraction_results:
            if not result.success or not result.extracted_data:
                continue

            data = result.extracted_data
            doc_name = result.document.file_name if result.document else "unknown"

            # Collect constitution dates
            if data.company_constitution_date:
                try:
                    parsed_date = LegalValidator._parse_date_string(data.company_constitution_date)
                    if parsed_date:
                        constitution_dates.append((parsed_date, data.company_constitution_date, doc_name))
                except:
                    pass

            # Collect statute approval dates
            if data.statute_approval_date:
                try:
                    parsed_date = LegalValidator._parse_date_string(data.statute_approval_date)
                    if parsed_date:
                        statute_approval_dates.append((parsed_date, data.statute_approval_date, doc_name))
                except:
                    pass

            # Collect registration dates
            if data.registration_date:
                try:
                    parsed_date = LegalValidator._parse_date_string(data.registration_date)
                    if parsed_date:
                        registration_dates.append((parsed_date, data.registration_date, doc_name))
                except:
                    pass

            # Collect acta dates
            if data.acta_date:
                try:
                    parsed_date = LegalValidator._parse_date_string(data.acta_date)
                    if parsed_date:
                        acta_dates.append((parsed_date, data.acta_date, doc_name))
                except:
                    pass

        # Rule 1: Constitution date must come BEFORE or equal to statute approval date
        # This is CRITICAL - statute cannot be approved before company exists!
        for const_date, const_str, const_doc in constitution_dates:
            for statute_date, statute_str, statute_doc in statute_approval_dates:
                if statute_date < const_date:
                    issues.append(ValidationIssue(
                        field="date_consistency_statute_constitution",
                        issue_type="impossible_chronology",
                        severity=ValidationSeverity.CRITICAL,
                        description=(
                            f"IMPOSIBLE: Estatuto aprobado ({statute_str} en {statute_doc}) "
                            f"ANTES de la constitución de la empresa ({const_str} en {const_doc}). "
                            f"Una empresa no puede tener estatutos aprobados antes de ser constituida."
                        ),
                        legal_basis="Lógica jurídica básica - Art. 248",
                        recommendation=(
                            "Verificar las fechas en los documentos originales. "
                            "Posible error de transcripción o documento incorrecto."
                        )
                    ))

        # Rule 2: Constitution date should come BEFORE or equal to registration date
        for const_date, const_str, const_doc in constitution_dates:
            for reg_date, reg_str, reg_doc in registration_dates:
                if reg_date < const_date:
                    issues.append(ValidationIssue(
                        field="date_consistency_registration_constitution",
                        issue_type="impossible_chronology",
                        severity=ValidationSeverity.CRITICAL,
                        description=(
                            f"IMPOSIBLE: Registro ({reg_str} en {reg_doc}) "
                            f"ANTES de la constitución de la empresa ({const_str} en {const_doc}). "
                            f"Una empresa no puede estar registrada antes de ser constituida."
                        ),
                        legal_basis="Lógica jurídica básica - Art. 249",
                        recommendation=(
                            "Verificar las fechas en los documentos originales. "
                            "Posible error de transcripción o documento incorrecto."
                        )
                    ))

        # Rule 3: Statute approval should come BEFORE or equal to registration
        # (statute is approved first, then registered)
        for statute_date, statute_str, statute_doc in statute_approval_dates:
            for reg_date, reg_str, reg_doc in registration_dates:
                if reg_date < statute_date:
                    issues.append(ValidationIssue(
                        field="date_consistency_registration_statute",
                        issue_type="suspicious_chronology",
                        severity=ValidationSeverity.WARNING,
                        description=(
                            f"SOSPECHOSO: Registro ({reg_str} en {reg_doc}) "
                            f"ANTES de aprobación de estatuto ({statute_str} en {statute_doc}). "
                            f"Normalmente el estatuto se aprueba antes del registro."
                        ),
                        legal_basis="Práctica notarial - Art. 249",
                        recommendation="Verificar el orden cronológico de los eventos societarios."
                    ))

        # Rule 4: Acta dates should be recent relative to constitution
        # (actas are usually generated after the company exists)
        for acta_date, acta_str, acta_doc in acta_dates:
            for const_date, const_str, const_doc in constitution_dates:
                if acta_date < const_date:
                    issues.append(ValidationIssue(
                        field="date_consistency_acta_constitution",
                        issue_type="impossible_chronology",
                        severity=ValidationSeverity.ERROR,
                        description=(
                            f"IMPOSIBLE: Acta de directorio ({acta_str} en {acta_doc}) "
                            f"fechada ANTES de la constitución de la empresa ({const_str} en {const_doc}). "
                            f"No puede haber actas de directorio antes de que exista la empresa."
                        ),
                        legal_basis="Lógica jurídica básica - Art. 248",
                        recommendation="Verificar la fecha del acta de directorio."
                    ))

        return issues

    @staticmethod
    def validate(
        requirements: LegalRequirements,
        extraction_result: CollectionExtractionResult
    ) -> ValidationMatrix:
        """
        Main validation method.
        Runs all validation checks and returns complete validation matrix.
        """
        matrix = ValidationMatrix(
            legal_requirements=requirements,
            extraction_result=extraction_result
        )

        # Step 1: Validate document presence
        matrix.document_validations = LegalValidator.validate_document_presence(
            requirements, extraction_result
        )

        # Step 2: Validate document expiry
        for doc_val in matrix.document_validations:
            if doc_val.present:
                # Find the requirement
                req_doc = next(
                    (req for req in requirements.required_documents
                     if req.document_type == doc_val.document_type),
                    None
                )
                if req_doc:
                    if doc_val.extracted_data:
                        LegalValidator.validate_document_expiry(
                            doc_val, req_doc, doc_val.extracted_data
                        )
                    LegalValidator.validate_document_quality(doc_val, req_doc)
                    LegalValidator.validate_document_fields(doc_val, req_doc)
                    LegalValidator.validate_catalog_alignment(doc_val)

        # Step 3: Validate required elements
        matrix.element_validations = LegalValidator.validate_required_elements(
            requirements, extraction_result
        )

        # Step 4: Article compliance checks
        matrix.article_checks = LegalValidator.validate_article_compliance(
            requirements, matrix
        )

        # Step 5: Validate cross-document consistency
        matrix.cross_document_issues = LegalValidator.validate_cross_document_consistency(
            extraction_result
        )

        # Step 5.5: Validate date consistency (detect impossible chronologies)
        matrix.cross_document_issues.extend(
            LegalValidator.validate_date_consistency(extraction_result)
        )

        # Step 6: Validate institution format rules (laws, mentions, etc.)
        matrix.cross_document_issues.extend(
            LegalValidator.validate_format_rules(requirements, extraction_result)
        )

        # Step 7: Determine overall status
        critical_issues = matrix.get_critical_issues()

        if critical_issues:
            matrix.overall_status = ValidationStatus.INVALID
            matrix.can_issue_certificate = False
        else:
            # Check if there are any errors
            all_issues = matrix.get_all_issues()
            has_errors = any(issue.severity == ValidationSeverity.ERROR for issue in all_issues)

            if has_errors:
                matrix.overall_status = ValidationStatus.WARNING
                matrix.can_issue_certificate = False  # Don't issue with errors
            else:
                matrix.overall_status = ValidationStatus.VALID
                matrix.can_issue_certificate = True

        return matrix

    @staticmethod
    def save_validation_matrix(matrix: ValidationMatrix, output_path: str) -> None:
        """Save validation matrix to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(matrix.to_json())
        print(f"\n✅ Matriz de validación guardada en: {output_path}")


def example_usage():
    """Example usage of Phase 5"""

    print("\n" + "="*70)
    print("  EJEMPLOS DE USO - FASE 5: VALIDACIÓN LEGAL")
    print("="*70)

    print("\n📌 Ejemplo 1: Crear problemas de validación")
    print("-" * 70)

    issue1 = ValidationIssue(
        field="estatuto",
        issue_type="missing_document",
        severity=ValidationSeverity.CRITICAL,
        description="Falta estatuto social",
        legal_basis="Art. 248",
        recommendation="Cargar estatuto de la empresa"
    )

    print(issue1.get_display())

    issue2 = ValidationIssue(
        field="certificado_bps",
        issue_type="expired_document",
        severity=ValidationSeverity.ERROR,
        description="Certificado BPS vencido (más de 30 días)",
        legal_basis="Requisito BPS",
        recommendation="Obtener certificado BPS actualizado"
    )

    print(issue2.get_display())

    print("\n\n📌 Ejemplo 2: Flujo completo (requiere Fases 1-4)")
    print("-" * 70)
    print("Para ejecutar validación completa:")
    print("""
    # Fases 1-2: Intent y Requirements
    intent = CertificateIntentCapture.capture_intent_from_params(...)
    requirements = LegalRequirementsEngine.resolve_requirements(intent)

    # Fase 3: Document Collection
    collection = DocumentIntake.create_collection(intent, requirements)
    collection = DocumentIntake.add_files_to_collection(collection, file_paths)

    # Fase 4: Text Extraction
    extraction_result = TextExtractor.process_collection(collection)

    # Fase 5: Validation
    validation_matrix = LegalValidator.validate(requirements, extraction_result)
    print(validation_matrix.get_summary())

    if validation_matrix.can_issue_certificate:
        print("✅ Listo para generar certificado!")
    else:
        print("❌ Corregir problemas antes de emitir certificado")
    """)


if __name__ == "__main__":
    example_usage()
