"""
Phase 6: Gap & Error Detection

This module handles:
- Identifying missing documents
- Detecting expired certificates
- Finding inconsistencies across documents
- Generating detailed error reports
- Providing actionable recommendations

This phase processes validation results from Phase 5 and presents them
in a user-friendly format with clear guidance on what to fix.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from datetime import datetime, timedelta
from enum import Enum
import json

from src.phase2_legal_requirements import (
    LegalRequirements,
    DocumentType,
    DocumentRequirement,
    InstitutionRule
)
from src.phase4_text_extraction import CollectionExtractionResult, ExtractedData
from src.phase5_legal_validation import (
    ValidationMatrix,
    ValidationIssue,
    ValidationSeverity,
    ValidationStatus,
    DocumentValidation
)


class GapType(Enum):
    """Types of gaps that can be detected"""
    MISSING_DOCUMENT = "missing_document"
    EXPIRED_DOCUMENT = "expired_document"
    INVALID_DOCUMENT = "invalid_document"
    MISSING_DATA = "missing_data"
    INCONSISTENT_DATA = "inconsistent_data"
    INCORRECT_FORMAT = "incorrect_format"
    LEGAL_NONCOMPLIANCE = "legal_noncompliance"
    CATALOG_MISMATCH = "catalog_mismatch"
    REVIEW_REQUIRED = "review_required"
    SEMANTIC_ERROR = "semantic_error"  # Logical impossibilities like date inconsistencies


class ActionPriority(Enum):
    """Priority levels for required actions"""
    URGENT = "urgent"  # Must fix before proceeding
    HIGH = "high"  # Should fix soon
    MEDIUM = "medium"  # Recommended to fix
    LOW = "low"  # Optional improvement


@dataclass
class Gap:
    """
    Represents a single gap or error that needs to be addressed.
    """
    gap_type: GapType
    priority: ActionPriority
    title: str
    description: str
    affected_document: Optional[DocumentType] = None
    legal_basis: Optional[str] = None
    current_state: Optional[str] = None
    required_state: str = ""
    action_required: str = ""
    deadline: Optional[datetime] = None  # When this must be fixed by

    def to_dict(self) -> dict:
        return {
            "gap_type": self.gap_type.value,
            "priority": self.priority.value,
            "title": self.title,
            "description": self.description,
            "affected_document": self.affected_document.value if self.affected_document else None,
            "legal_basis": self.legal_basis,
            "current_state": self.current_state,
            "required_state": self.required_state,
            "action_required": self.action_required,
            "deadline": self.deadline.isoformat() if self.deadline else None
        }

    def get_priority_icon(self) -> str:
        """Get icon for priority level"""
        icons = {
            ActionPriority.URGENT: "🔴",
            ActionPriority.HIGH: "🟠",
            ActionPriority.MEDIUM: "🟡",
            ActionPriority.LOW: "🟢"
        }
        return icons.get(self.priority, "⚪")

    def get_display(self) -> str:
        """Get formatted display string"""
        icon = self.get_priority_icon()

        display = f"""
{icon} {self.title} [{self.priority.value.upper()}]
   {self.description}
"""
        if self.affected_document:
            display += f"   📄 Documento afectado: {self.affected_document.value.replace('_', ' ').title()}\n"

        if self.legal_basis:
            display += f"   ⚖️  Base legal: {self.legal_basis}\n"

        if self.current_state:
            display += f"   📊 Estado actual: {self.current_state}\n"

        if self.required_state:
            display += f"   ✅ Estado requerido: {self.required_state}\n"

        if self.action_required:
            display += f"   🔧 Acción requerida: {self.action_required}\n"

        if self.deadline:
            display += f"   ⏰ Plazo: {self.deadline.strftime('%d/%m/%Y')}\n"

        return display


@dataclass
class DocumentGapReport:
    """Detailed gap report for a specific document"""
    document_type: DocumentType
    is_present: bool
    is_required: bool
    is_valid: bool
    gaps: List[Gap] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def has_critical_gaps(self) -> bool:
        """Check if document has critical gaps"""
        return any(gap.priority == ActionPriority.URGENT for gap in self.gaps)

    def to_dict(self) -> dict:
        return {
            "document_type": self.document_type.value,
            "is_present": self.is_present,
            "is_required": self.is_required,
            "is_valid": self.is_valid,
            "has_critical_gaps": self.has_critical_gaps(),
            "gaps": [gap.to_dict() for gap in self.gaps],
            "warnings": self.warnings,
            "recommendations": self.recommendations
        }


@dataclass
class GapAnalysisReport:
    """
    Complete gap analysis report.
    This is the main output of Phase 6.
    """
    validation_matrix: ValidationMatrix
    gaps: List[Gap] = field(default_factory=list)
    document_reports: List[DocumentGapReport] = field(default_factory=list)

    # Summary statistics
    total_gaps: int = 0
    urgent_gaps: int = 0
    high_priority_gaps: int = 0
    medium_priority_gaps: int = 0
    low_priority_gaps: int = 0

    # Status flags
    ready_for_certificate: bool = False
    blocking_issues_count: int = 0

    analysis_timestamp: datetime = field(default_factory=datetime.now)

    def calculate_summary(self) -> None:
        """Calculate summary statistics"""
        self.total_gaps = len(self.gaps)
        self.urgent_gaps = sum(1 for gap in self.gaps if gap.priority == ActionPriority.URGENT)
        self.high_priority_gaps = sum(1 for gap in self.gaps if gap.priority == ActionPriority.HIGH)
        self.medium_priority_gaps = sum(1 for gap in self.gaps if gap.priority == ActionPriority.MEDIUM)
        self.low_priority_gaps = sum(1 for gap in self.gaps if gap.priority == ActionPriority.LOW)

        self.blocking_issues_count = self.urgent_gaps + self.high_priority_gaps
        self.ready_for_certificate = self.urgent_gaps == 0

    def get_gaps_by_priority(self, priority: ActionPriority) -> List[Gap]:
        """Get all gaps of a specific priority"""
        return [gap for gap in self.gaps if gap.priority == priority]

    def get_gaps_by_type(self, gap_type: GapType) -> List[Gap]:
        """Get all gaps of a specific type"""
        return [gap for gap in self.gaps if gap.gap_type == gap_type]

    def to_dict(self) -> dict:
        return {
            "analysis_timestamp": self.analysis_timestamp.isoformat(),
            "summary": {
                "total_gaps": self.total_gaps,
                "urgent": self.urgent_gaps,
                "high": self.high_priority_gaps,
                "medium": self.medium_priority_gaps,
                "low": self.low_priority_gaps,
                "blocking_issues": self.blocking_issues_count,
                "ready_for_certificate": self.ready_for_certificate
            },
            "gaps": [gap.to_dict() for gap in self.gaps],
            "document_reports": [dr.to_dict() for dr in self.document_reports]
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def get_summary(self) -> str:
        """Get human-readable summary in Spanish"""

        status_icon = "✅" if self.ready_for_certificate else "❌"
        status_text = "LISTO PARA CERTIFICADO" if self.ready_for_certificate else "REQUIERE CORRECCIONES"

        summary = f"""
╔══════════════════════════════════════════════════════════════╗
║              ANÁLISIS DE BRECHAS - FASE 6                    ║
╚══════════════════════════════════════════════════════════════╝

{status_icon} ESTADO: {status_text}

📊 RESUMEN DE PROBLEMAS:
   Total de brechas: {self.total_gaps}
   🔴 Urgentes (bloquean emisión): {self.urgent_gaps}
   🟠 Prioridad alta: {self.high_priority_gaps}
   🟡 Prioridad media: {self.medium_priority_gaps}
   🟢 Prioridad baja: {self.low_priority_gaps}

   ⚠️  Problemas bloqueantes: {self.blocking_issues_count}
"""

        # Urgent gaps
        if self.urgent_gaps > 0:
            summary += f"\n🔴 PROBLEMAS URGENTES ({self.urgent_gaps}):\n"
            summary += "   DEBE CORREGIR ESTOS ANTES DE EMITIR CERTIFICADO\n"
            for gap in self.get_gaps_by_priority(ActionPriority.URGENT):
                summary += gap.get_display()

        # High priority gaps
        if self.high_priority_gaps > 0:
            summary += f"\n🟠 PRIORIDAD ALTA ({self.high_priority_gaps}):\n"
            for gap in self.get_gaps_by_priority(ActionPriority.HIGH):
                summary += gap.get_display()

        # Medium priority gaps
        if self.medium_priority_gaps > 0:
            summary += f"\n🟡 PRIORIDAD MEDIA ({self.medium_priority_gaps}):\n"
            for gap in self.get_gaps_by_priority(ActionPriority.MEDIUM):
                summary += gap.get_display()

        # Document-specific reports
        summary += f"\n\n📄 REPORTE POR DOCUMENTO ({len(self.document_reports)}):\n"
        for doc_report in self.document_reports:
            doc_name = doc_report.document_type.value.replace('_', ' ').title()
            status = "✅" if doc_report.is_valid else "❌"
            presence = "PRESENTE" if doc_report.is_present else "FALTANTE"

            summary += f"\n   {status} {doc_name} - {presence}"

            if doc_report.gaps:
                summary += f" ({len(doc_report.gaps)} problemas)"

            if doc_report.warnings:
                for warning in doc_report.warnings:
                    summary += f"\n      ⚠️  {warning}"

            if doc_report.recommendations:
                for rec in doc_report.recommendations:
                    summary += f"\n      💡 {rec}"

        # Next steps
        if self.ready_for_certificate:
            summary += "\n\n✅ PRÓXIMO PASO: Proceder a Fase 7 (Confirmación Legal Final)"
        else:
            summary += "\n\n❌ PRÓXIMOS PASOS:"
            summary += "\n   1. Corregir problemas urgentes listados arriba"
            summary += "\n   2. Re-ejecutar validación (Fase 5)"
            summary += "\n   3. Verificar que no hayan problemas bloqueantes"
            summary += "\n   4. Proceder a Fase 7"

        return summary

    def get_action_plan(self) -> str:
        """Get prioritized action plan"""
        plan = """
╔══════════════════════════════════════════════════════════════╗
║              PLAN DE ACCIÓN - FASE 6                         ║
╚══════════════════════════════════════════════════════════════╝

Este plan indica las acciones en orden de prioridad.
"""

        # Group by priority
        urgent = self.get_gaps_by_priority(ActionPriority.URGENT)
        high = self.get_gaps_by_priority(ActionPriority.HIGH)
        medium = self.get_gaps_by_priority(ActionPriority.MEDIUM)

        if urgent:
            plan += f"\n🔴 PASO 1: CORREGIR URGENTE ({len(urgent)} items)\n"
            for i, gap in enumerate(urgent, 1):
                plan += f"\n   {i}. {gap.title}"
                plan += f"\n      → {gap.action_required}"

        if high:
            plan += f"\n\n🟠 PASO 2: CORREGIR PRIORIDAD ALTA ({len(high)} items)\n"
            for i, gap in enumerate(high, 1):
                plan += f"\n   {i}. {gap.title}"
                plan += f"\n      → {gap.action_required}"

        if medium:
            plan += f"\n\n🟡 PASO 3: CORREGIR PRIORIDAD MEDIA ({len(medium)} items)\n"
            for i, gap in enumerate(medium, 1):
                plan += f"\n   {i}. {gap.title}"
                plan += f"\n      → {gap.action_required}"

        plan += "\n\n✅ DESPUÉS DE COMPLETAR: Re-ejecutar validación completa (Fases 3-5)"

        return plan


class GapDetector:
    """
    Main service class for gap detection.
    Analyzes validation results and generates detailed gap reports.
    """

    @staticmethod
    def detect_missing_documents(validation_matrix: ValidationMatrix) -> List[Gap]:
        """Detect missing required documents"""
        gaps = []

        for doc_val in validation_matrix.document_validations:
            if doc_val.required and not doc_val.present:
                # Find the requirement details
                req = next(
                    (r for r in validation_matrix.legal_requirements.required_documents
                     if r.document_type == doc_val.document_type),
                    None
                )

                if req:
                    gap = Gap(
                        gap_type=GapType.MISSING_DOCUMENT,
                        priority=ActionPriority.URGENT,
                        title=f"Falta {req.description}",
                        description=f"Documento obligatorio no encontrado: {req.description}",
                        affected_document=doc_val.document_type,
                        legal_basis=req.legal_basis,
                        current_state="Documento no cargado",
                        required_state="Documento presente y válido",
                        action_required=f"Cargar {req.description} al sistema"
                    )

                    # Set deadline if institution has validity period
                    institution_rules = validation_matrix.legal_requirements.institution_rules
                    if institution_rules and institution_rules.validity_days:
                        gap.deadline = datetime.now() + timedelta(days=institution_rules.validity_days)

                    gaps.append(gap)

        return gaps

    @staticmethod
    def detect_expired_documents(validation_matrix: ValidationMatrix) -> List[Gap]:
        """Detect expired documents"""
        gaps = []

        for doc_val in validation_matrix.document_validations:
            # Check for expiry issues in validation issues
            expiry_issues = [
                issue for issue in doc_val.issues
                if "expir" in issue.issue_type.lower() or "venc" in issue.description.lower()
            ]

            for issue in expiry_issues:
                # Find the requirement to get expiry days
                req = next(
                    (r for r in validation_matrix.legal_requirements.required_documents
                     if r.document_type == doc_val.document_type),
                    None
                )

                expiry_days = req.expiry_days if req and req.expiry_days else "desconocidos"

                gap = Gap(
                    gap_type=GapType.EXPIRED_DOCUMENT,
                    priority=ActionPriority.URGENT if issue.severity == ValidationSeverity.CRITICAL else ActionPriority.HIGH,
                    title=f"Documento vencido: {doc_val.document_type.value.replace('_', ' ').title()}",
                    description=issue.description,
                    affected_document=doc_val.document_type,
                    legal_basis=issue.legal_basis,
                    current_state="Documento vencido o con más días de antigüedad de los permitidos",
                    required_state=f"Documento con menos de {expiry_days} días de antigüedad",
                    action_required=f"Obtener versión actualizada del documento",
                    deadline=datetime.now() + timedelta(days=7)  # 1 week to fix
                )

                gaps.append(gap)

        return gaps

    @staticmethod
    def detect_missing_data(validation_matrix: ValidationMatrix) -> List[Gap]:
        """Detect missing required data elements"""
        gaps = []

        for elem_val in validation_matrix.element_validations:
            if elem_val.status == ValidationStatus.MISSING:
                element_name = elem_val.element.value.replace('_', ' ').title()

                # Find related issues
                related_issues = [issue for issue in elem_val.issues]
                description = related_issues[0].description if related_issues else f"No se encontró {element_name}"
                legal_basis = related_issues[0].legal_basis if related_issues else None

                gap = Gap(
                    gap_type=GapType.MISSING_DATA,
                    priority=ActionPriority.URGENT,
                    title=f"Falta información: {element_name}",
                    description=description,
                    legal_basis=legal_basis,
                    current_state=f"{element_name} no encontrado en documentos",
                    required_state=f"{element_name} debe estar presente en documentos",
                    action_required=f"Verificar que documentos contengan {element_name} o cargar documento adicional"
                )

                gaps.append(gap)

        return gaps

    @staticmethod
    def detect_inconsistencies(validation_matrix: ValidationMatrix) -> List[Gap]:
        """Detect data inconsistencies across documents"""
        gaps = []

        for issue in validation_matrix.cross_document_issues:
            # Skip semantic errors (handled by detect_semantic_errors)
            if "chronology" in issue.issue_type or "consistency" in issue.field:
                continue

            gap = Gap(
                gap_type=GapType.INCONSISTENT_DATA,
                priority=ActionPriority.HIGH if issue.severity == ValidationSeverity.ERROR else ActionPriority.MEDIUM,
                title=f"Inconsistencia: {issue.field.replace('_', ' ').title()}",
                description=issue.description,
                legal_basis=issue.legal_basis,
                current_state="Datos inconsistentes entre documentos",
                required_state="Datos consistentes en todos los documentos",
                action_required=issue.recommendation or "Verificar y corregir información en documentos"
            )

            gaps.append(gap)

        return gaps

    @staticmethod
    def detect_semantic_errors(validation_matrix: ValidationMatrix) -> List[Gap]:
        """
        Detect semantic errors - logical impossibilities in the data.
        This includes date consistency issues, impossible chronologies, etc.
        """
        gaps = []

        for issue in validation_matrix.cross_document_issues:
            # Only process semantic/chronology errors
            if "chronology" not in issue.issue_type and "consistency" not in issue.field:
                continue

            # Determine priority based on severity
            if issue.severity == ValidationSeverity.CRITICAL:
                priority = ActionPriority.URGENT
            elif issue.severity == ValidationSeverity.ERROR:
                priority = ActionPriority.HIGH
            else:
                priority = ActionPriority.MEDIUM

            # Determine current and required state from issue type
            current_state = "Fechas en orden cronológico imposible"
            required_state = "Fechas en orden cronológico correcto"

            if "statute" in issue.field and "constitution" in issue.field:
                current_state = "Estatuto aprobado ANTES de constitución de empresa"
                required_state = "Fecha de constitución ANTES O IGUAL a fecha de aprobación de estatuto"
            elif "registration" in issue.field and "constitution" in issue.field:
                current_state = "Registro ANTES de constitución de empresa"
                required_state = "Fecha de constitución ANTES O IGUAL a fecha de registro"
            elif "acta" in issue.field and "constitution" in issue.field:
                current_state = "Acta de directorio ANTES de constitución de empresa"
                required_state = "Fecha de constitución ANTES de fecha de acta"

            gap = Gap(
                gap_type=GapType.SEMANTIC_ERROR,
                priority=priority,
                title=f"ERROR LÓGICO: {issue.field.replace('_', ' ').replace('date consistency', 'Fechas inconsistentes').title()}",
                description=issue.description,
                legal_basis=issue.legal_basis,
                current_state=current_state,
                required_state=required_state,
                action_required=issue.recommendation or "Verificar fechas en documentos originales - posible error de transcripción"
            )

            gaps.append(gap)

        return gaps

    @staticmethod
    def detect_format_issues(validation_matrix: ValidationMatrix) -> List[Gap]:
        """Detect format-related issues"""
        gaps = []

        # Check institution-specific format requirements
        institution_rules = validation_matrix.legal_requirements.institution_rules

        if institution_rules and institution_rules.format_rules:
            for rule_key, rule_value in institution_rules.format_rules.items():
                # This is a simple check - in production, verify actual compliance
                gap = Gap(
                    gap_type=GapType.INCORRECT_FORMAT,
                    priority=ActionPriority.MEDIUM,
                    title=f"Verificar formato: {rule_key.replace('_', ' ').title()}",
                    description=f"Verificar cumplimiento de requisito de formato: {rule_key}",
                    legal_basis=f"Requisito {institution_rules.institution}",
                    current_state="No verificado",
                    required_state=f"{rule_key} = {rule_value}",
                    action_required="Verificar manualmente el cumplimiento de este requisito de formato"
                )

                gaps.append(gap)

        return gaps

    @staticmethod
    def detect_catalog_issues(validation_matrix: ValidationMatrix) -> List[Gap]:
        """Detect catalog mismatches or naming issues from validation notes."""
        gaps = []

        for doc_val in validation_matrix.document_validations:
            if not doc_val.present or not doc_val.issues:
                continue

            for issue in doc_val.issues:
                if not issue.issue_type.startswith("catalog_"):
                    continue

                priority = ActionPriority.LOW
                current_state = "Catalogo y archivo no coinciden"
                if issue.issue_type == "catalog_type_mismatch":
                    priority = ActionPriority.MEDIUM
                    current_state = "Formato distinto al esperado por el catalogo"

                gap = Gap(
                    gap_type=GapType.CATALOG_MISMATCH,
                    priority=priority,
                    title=f"Revisar catalogo: {doc_val.document_type.value.replace('_', ' ').title()}",
                    description=issue.description,
                    affected_document=doc_val.document_type,
                    legal_basis=issue.legal_basis,
                    current_state=current_state,
                    required_state="Catalogo y documento verificados",
                    action_required=issue.recommendation or "Revisar catalogo y archivo cargado",
                )

                gaps.append(gap)

        return gaps

    @staticmethod
    def create_document_reports(validation_matrix: ValidationMatrix, all_gaps: List[Gap]) -> List[DocumentGapReport]:
        """Create detailed reports for each document"""
        reports = []

        for doc_val in validation_matrix.document_validations:
            # Get gaps for this document
            doc_gaps = [gap for gap in all_gaps if gap.affected_document == doc_val.document_type]

            # Create warnings and recommendations
            warnings = []
            recommendations = []

            if doc_val.required and not doc_val.present:
                warnings.append("Documento obligatorio faltante")
                recommendations.append(f"Cargar {doc_val.document_type.value.replace('_', ' ')}")

            if doc_val.present and not doc_val.is_valid():
                warnings.append("Documento presente pero con problemas de validación")
                recommendations.append("Revisar y corregir problemas listados")

            for issue in doc_val.issues:
                if issue.issue_type.startswith("catalog_"):
                    warnings.append(issue.description)

            # Find requirement details
            req = next(
                (r for r in validation_matrix.legal_requirements.required_documents
                 if r.document_type == doc_val.document_type),
                None
            )

            if req and req.expires:
                recommendations.append(f"Verificar que documento tenga menos de {req.expiry_days} días")

            report = DocumentGapReport(
                document_type=doc_val.document_type,
                is_present=doc_val.present,
                is_required=doc_val.required,
                is_valid=doc_val.is_valid(),
                gaps=doc_gaps,
                warnings=warnings,
                recommendations=recommendations
            )

            reports.append(report)

        return reports

    @staticmethod
    def analyze(validation_matrix: ValidationMatrix) -> GapAnalysisReport:
        """
        Main analysis method.
        Analyzes validation matrix and generates complete gap report.

        Args:
            validation_matrix: ValidationMatrix from Phase 5

        Returns:
            GapAnalysisReport with detailed gap analysis
        """
        report = GapAnalysisReport(validation_matrix=validation_matrix)

        # Detect all types of gaps
        report.gaps.extend(GapDetector.detect_missing_documents(validation_matrix))
        report.gaps.extend(GapDetector.detect_expired_documents(validation_matrix))
        report.gaps.extend(GapDetector.detect_missing_data(validation_matrix))
        report.gaps.extend(GapDetector.detect_semantic_errors(validation_matrix))  # Detect logical impossibilities first
        report.gaps.extend(GapDetector.detect_inconsistencies(validation_matrix))
        report.gaps.extend(GapDetector.detect_format_issues(validation_matrix))
        report.gaps.extend(GapDetector.detect_catalog_issues(validation_matrix))

        # Create document-specific reports
        report.document_reports = GapDetector.create_document_reports(validation_matrix, report.gaps)

        # Calculate summary
        report.calculate_summary()

        return report

    @staticmethod
    def save_gap_report(report: GapAnalysisReport, output_path: str) -> None:
        """Save gap analysis report to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report.to_json())
        print(f"\n✅ Reporte de brechas guardado en: {output_path}")


def example_usage():
    """Example usage of Phase 6"""

    print("\n" + "="*70)
    print("  EJEMPLOS DE USO - FASE 6: DETECCIÓN DE BRECHAS")
    print("="*70)

    print("\n📌 Ejemplo 1: Crear un gap (brecha)")
    print("-" * 70)

    gap = Gap(
        gap_type=GapType.MISSING_DOCUMENT,
        priority=ActionPriority.URGENT,
        title="Falta Estatuto Social",
        description="Documento obligatorio no encontrado: Estatuto social de la empresa",
        affected_document=DocumentType.ESTATUTO,
        legal_basis="Art. 248",
        current_state="Documento no cargado",
        required_state="Documento presente y válido",
        action_required="Cargar estatuto de la empresa al sistema"
    )

    print(gap.get_display())

    print("\n\n📌 Ejemplo 2: Gap de documento vencido")
    print("-" * 70)

    gap2 = Gap(
        gap_type=GapType.EXPIRED_DOCUMENT,
        priority=ActionPriority.URGENT,
        title="Certificado BPS vencido",
        description="Certificado BPS tiene más de 30 días de antigüedad",
        affected_document=DocumentType.CERTIFICADO_BPS,
        legal_basis="Requisito BPS",
        current_state="Certificado con fecha 01/10/2024 (más de 30 días)",
        required_state="Certificado con menos de 30 días de antigüedad",
        action_required="Obtener certificado BPS actualizado",
        deadline=datetime.now() + timedelta(days=7)
    )

    print(gap2.get_display())

    print("\n\n📌 Ejemplo 3: Flujo completo (requiere Fases 1-5)")
    print("-" * 70)
    print("Para ejecutar análisis de brechas completo:")
    print("""
    # Fases 1-5: Obtener matriz de validación
    validation_matrix = LegalValidator.validate(requirements, extraction_result)

    # Fase 6: Análisis de brechas
    gap_report = GapDetector.analyze(validation_matrix)

    # Ver resumen
    print(gap_report.get_summary())

    # Ver plan de acción
    print(gap_report.get_action_plan())

    # Verificar si puede emitir certificado
    if gap_report.ready_for_certificate:
        print("✅ Listo para certificado!")
    else:
        print(f"❌ Corregir {gap_report.urgent_gaps} problemas urgentes")
    """)


if __name__ == "__main__":
    example_usage()
