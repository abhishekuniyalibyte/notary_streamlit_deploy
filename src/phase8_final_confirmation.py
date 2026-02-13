"""
Phase 8: Final Legal Confirmation

This module handles:
- Re-running all legal validations after Phase 7 updates
- Ensuring 100% compliance with Articles 248-255
- Verifying all institution-specific requirements
- Generating final compliance report
- Making go/no-go decision for certificate generation

This is a CRITICAL phase - certificates can only be generated if this phase passes.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from datetime import datetime
from enum import Enum
import json

from src.phase2_legal_requirements import LegalRequirements
from src.phase4_text_extraction import CollectionExtractionResult
from src.phase5_legal_validation import (
    LegalValidator,
    ValidationMatrix,
    ValidationIssue,
    ValidationSeverity,
    ValidationStatus
)
from src.phase6_gap_detection import GapDetector, GapAnalysisReport, Gap, ActionPriority
from src.phase7_data_update import UpdateAttemptResult


class ComplianceLevel(Enum):
    """Overall compliance level"""
    FULLY_COMPLIANT = "fully_compliant"  # 100% ready for certificate
    SUBSTANTIALLY_COMPLIANT = "substantially_compliant"  # Minor issues only
    PARTIALLY_COMPLIANT = "partially_compliant"  # Major issues remain
    NON_COMPLIANT = "non_compliant"  # Critical issues blocking


class CertificateDecision(Enum):
    """Final decision on certificate generation"""
    APPROVED = "approved"  # Proceed to Phase 9
    APPROVED_WITH_WARNINGS = "approved_with_warnings"  # Proceed but notify
    REJECTED = "rejected"  # Cannot proceed
    REQUIRES_REVIEW = "requires_review"  # Manual notary review needed


@dataclass
class ComplianceCheck:
    """
    Represents a single compliance check result.
    """
    check_name: str
    check_category: str  # "document", "legal", "institution"
    is_compliant: bool
    severity: ValidationSeverity
    details: str
    legal_basis: Optional[str] = None
    blocking: bool = False  # Does this block certificate generation?

    def to_dict(self) -> dict:
        return {
            "check_name": self.check_name,
            "check_category": self.check_category,
            "is_compliant": self.is_compliant,
            "severity": self.severity.value,
            "details": self.details,
            "legal_basis": self.legal_basis,
            "blocking": self.blocking
        }

    def get_display(self) -> str:
        """Get formatted display string"""
        status_icon = "✅" if self.is_compliant else "❌"
        severity_icons = {
            ValidationSeverity.CRITICAL: "🔴",
            ValidationSeverity.ERROR: "🟠",
            ValidationSeverity.WARNING: "🟡",
            ValidationSeverity.INFO: "🔵"
        }
        severity_icon = severity_icons.get(self.severity, "⚪")

        display = f"{status_icon} {severity_icon} {self.check_name}"
        if not self.is_compliant:
            display += f"\n   ⚠️  {self.details}"
        if self.legal_basis:
            display += f"\n   📖 Base legal: {self.legal_basis}"
        if self.blocking:
            display += "\n   🚫 BLOQUEANTE"

        return display


@dataclass
class FinalConfirmationReport:
    """
    Final legal confirmation report after Phase 7 updates.
    Contains all validation results and final decision.
    """
    # Input data
    legal_requirements: LegalRequirements
    update_result: UpdateAttemptResult

    # Validation results
    validation_matrix: Optional[ValidationMatrix] = None
    gap_report: Optional[GapAnalysisReport] = None

    # Compliance checks
    compliance_checks: List[ComplianceCheck] = field(default_factory=list)

    # Summary metrics
    total_checks: int = 0
    passed_checks: int = 0
    failed_checks: int = 0
    blocking_issues: int = 0
    critical_issues: int = 0
    warnings: int = 0

    # Final determination
    compliance_level: ComplianceLevel = ComplianceLevel.NON_COMPLIANT
    certificate_decision: CertificateDecision = CertificateDecision.REJECTED
    decision_rationale: str = ""

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    validated_by: str = "Sistema Automatizado"

    # Remaining issues (if any)
    remaining_issues: List[str] = field(default_factory=list)

    def calculate_summary(self):
        """Calculate summary statistics"""
        self.total_checks = len(self.compliance_checks)
        self.passed_checks = sum(1 for c in self.compliance_checks if c.is_compliant)
        self.failed_checks = self.total_checks - self.passed_checks

        self.blocking_issues = sum(1 for c in self.compliance_checks
                                   if not c.is_compliant and c.blocking)
        self.critical_issues = sum(1 for c in self.compliance_checks
                                   if not c.is_compliant and c.severity == ValidationSeverity.CRITICAL)
        self.warnings = sum(1 for c in self.compliance_checks
                           if not c.is_compliant and c.severity == ValidationSeverity.WARNING)

    def to_dict(self) -> dict:
        return {
            "legal_requirements": self.legal_requirements.to_dict(),
            "update_result": self.update_result.to_dict(),
            "validation_matrix": self.validation_matrix.to_dict() if self.validation_matrix else None,
            "gap_report": self.gap_report.to_dict() if self.gap_report else None,
            "compliance_checks": [c.to_dict() for c in self.compliance_checks],
            "total_checks": self.total_checks,
            "passed_checks": self.passed_checks,
            "failed_checks": self.failed_checks,
            "blocking_issues": self.blocking_issues,
            "critical_issues": self.critical_issues,
            "warnings": self.warnings,
            "compliance_level": self.compliance_level.value,
            "certificate_decision": self.certificate_decision.value,
            "decision_rationale": self.decision_rationale,
            "timestamp": self.timestamp.isoformat(),
            "validated_by": self.validated_by,
            "remaining_issues": self.remaining_issues
        }

    def to_json(self) -> str:
        """Serialize to JSON"""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def get_summary(self) -> str:
        """Get formatted summary"""
        self.calculate_summary()

        border = "=" * 70

        # Decision icon
        decision_icons = {
            CertificateDecision.APPROVED: "✅",
            CertificateDecision.APPROVED_WITH_WARNINGS: "⚠️",
            CertificateDecision.REJECTED: "❌",
            CertificateDecision.REQUIRES_REVIEW: "🔍"
        }
        decision_icon = decision_icons.get(self.certificate_decision, "❓")

        # Compliance icon
        compliance_icons = {
            ComplianceLevel.FULLY_COMPLIANT: "✅",
            ComplianceLevel.SUBSTANTIALLY_COMPLIANT: "🟢",
            ComplianceLevel.PARTIALLY_COMPLIANT: "🟡",
            ComplianceLevel.NON_COMPLIANT: "🔴"
        }
        compliance_icon = compliance_icons.get(self.compliance_level, "⚪")

        summary = f"""
{border}
           FASE 8: CONFIRMACIÓN LEGAL FINAL
{border}

{decision_icon} DECISIÓN: {self.certificate_decision.value.upper().replace('_', ' ')}
{compliance_icon} NIVEL DE CUMPLIMIENTO: {self.compliance_level.value.upper().replace('_', ' ')}

📊 RESUMEN DE VERIFICACIONES:
   Total de verificaciones: {self.total_checks}
   Verificaciones exitosas: {self.passed_checks} ✅
   Verificaciones fallidas: {self.failed_checks} ❌

   Problemas bloqueantes: {self.blocking_issues} 🚫
   Problemas críticos: {self.critical_issues} 🔴
   Advertencias: {self.warnings} 🟡

📋 RATIONALE:
{self.decision_rationale}

⏰ Timestamp: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
👤 Validado por: {self.validated_by}

"""

        if self.certificate_decision == CertificateDecision.APPROVED:
            summary += f"""
{'='*70}
✅ CERTIFICADO APROBADO PARA GENERACIÓN
{'='*70}

Puede proceder a Fase 9: Generación de Certificado

"""
        elif self.certificate_decision == CertificateDecision.APPROVED_WITH_WARNINGS:
            summary += f"""
{'='*70}
⚠️  CERTIFICADO APROBADO CON ADVERTENCIAS
{'='*70}

Advertencias no bloqueantes:
"""
            for issue in self.remaining_issues:
                summary += f"  • {issue}\n"

            summary += "\nPuede proceder a Fase 9, pero revisar advertencias.\n\n"

        elif self.certificate_decision == CertificateDecision.REJECTED:
            summary += f"""
{'='*70}
❌ CERTIFICADO RECHAZADO
{'='*70}

Problemas que deben ser resueltos:
"""
            for issue in self.remaining_issues:
                summary += f"  • {issue}\n"

            summary += "\nNo puede proceder a Fase 9 hasta resolver estos problemas.\n\n"

        elif self.certificate_decision == CertificateDecision.REQUIRES_REVIEW:
            summary += f"""
{'='*70}
🔍 REQUIERE REVISIÓN MANUAL DEL NOTARIO
{'='*70}

Aspectos que requieren revisión:
"""
            for issue in self.remaining_issues:
                summary += f"  • {issue}\n"

            summary += "\nRevisión del notario necesaria antes de proceder.\n\n"

        summary += border + "\n"

        return summary

    def get_detailed_report(self) -> str:
        """Get detailed compliance report"""
        report = "\n" + "=" * 70 + "\n"
        report += "    REPORTE DETALLADO DE CUMPLIMIENTO LEGAL - FASE 8\n"
        report += "=" * 70 + "\n\n"

        # Group checks by category
        categories = {}
        for check in self.compliance_checks:
            if check.check_category not in categories:
                categories[check.check_category] = []
            categories[check.check_category].append(check)

        for category, checks in categories.items():
            report += f"\n📁 {category.upper()}\n"
            report += "-" * 70 + "\n"

            passed = [c for c in checks if c.is_compliant]
            failed = [c for c in checks if not c.is_compliant]

            report += f"✅ Aprobadas: {len(passed)} / {len(checks)}\n\n"

            if failed:
                report += "❌ Verificaciones fallidas:\n\n"
                for check in failed:
                    report += check.get_display() + "\n\n"

        return report

    def can_proceed_to_phase9(self) -> bool:
        """Check if can proceed to Phase 9 (Certificate Generation)"""
        return self.certificate_decision in [
            CertificateDecision.APPROVED,
            CertificateDecision.APPROVED_WITH_WARNINGS
        ]


class FinalConfirmationEngine:
    """
    Main class for Phase 8: Final Legal Confirmation
    """

    @staticmethod
    def confirm(
        legal_requirements: LegalRequirements,
        update_result: UpdateAttemptResult
    ) -> FinalConfirmationReport:
        """
        Perform final legal confirmation.

        This is the main method that:
        1. Re-runs Phase 5 validation on updated data
        2. Re-runs Phase 6 gap detection
        3. Performs additional compliance checks
        4. Makes final go/no-go decision

        Args:
            legal_requirements: LegalRequirements from Phase 2
            update_result: UpdateAttemptResult from Phase 7

        Returns:
            FinalConfirmationReport with final decision
        """
        report = FinalConfirmationReport(
            legal_requirements=legal_requirements,
            update_result=update_result
        )

        print("\n" + "="*70)
        print("   FASE 8: CONFIRMACIÓN LEGAL FINAL")
        print("="*70 + "\n")

        # Step 1: Verify we have updated extraction result
        if not update_result.updated_extraction_result:
            report.certificate_decision = CertificateDecision.REJECTED
            report.compliance_level = ComplianceLevel.NON_COMPLIANT
            report.decision_rationale = "No se pudo extraer datos de documentos actualizados"
            report.remaining_issues.append("Falta resultado de extracción de datos")
            return report

        print("🔄 Paso 1: Re-validando documentos actualizados...")

        # Step 2: Re-run Phase 5 validation
        validation_matrix = LegalValidator.validate(
            legal_requirements,
            update_result.updated_extraction_result
        )
        report.validation_matrix = validation_matrix

        print(f"   ✓ Validación completada: {len(validation_matrix.document_validations)} documentos")

        # Step 3: Re-run Phase 6 gap detection
        print("\n🔍 Paso 2: Analizando brechas restantes...")

        gap_report = GapDetector.analyze(validation_matrix)
        report.gap_report = gap_report

        print(f"   ✓ Análisis completado: {len(gap_report.gaps)} brechas detectadas")

        # Step 4: Create compliance checks
        print("\n✅ Paso 3: Verificando cumplimiento legal...")

        report.compliance_checks = FinalConfirmationEngine._create_compliance_checks(
            validation_matrix,
            gap_report,
            legal_requirements,
            update_result
        )

        print(f"   ✓ {len(report.compliance_checks)} verificaciones realizadas")

        # Step 5: Make final decision
        print("\n⚖️  Paso 4: Determinando decisión final...")

        FinalConfirmationEngine._make_decision(report)

        print(f"   ✓ Decisión: {report.certificate_decision.value}")

        return report

    @staticmethod
    def _create_compliance_checks(
        validation_matrix: ValidationMatrix,
        gap_report: GapAnalysisReport,
        legal_requirements: LegalRequirements,
        update_result: Optional[UpdateAttemptResult] = None
    ) -> List[ComplianceCheck]:
        """Create detailed compliance checks"""
        checks = []

        # Check 1: All required documents present
        missing_docs = [dv for dv in validation_matrix.document_validations
                       if dv.required and not dv.present]

        checks.append(ComplianceCheck(
            check_name="Documentos requeridos presentes",
            check_category="document",
            is_compliant=len(missing_docs) == 0,
            severity=ValidationSeverity.CRITICAL,
            details=f"{len(missing_docs)} documento(s) faltante(s)" if missing_docs else "Todos los documentos presentes",
            legal_basis="Art. 248, 249",
            blocking=True
        ))

        review_items = update_result.review_required if update_result else []
        if review_items:
            checks.append(ComplianceCheck(
                check_name="Documentos con revisión pendiente",
                check_category="review",
                is_compliant=False,
                severity=ValidationSeverity.WARNING,
                details=f"{len(review_items)} documento(s) requieren verificación manual",
                legal_basis="Revisión manual",
                blocking=False
            ))

        # Check 2: No expired documents (derive from validation issues)
        expired_docs = [
            dv for dv in validation_matrix.document_validations
            if dv.present and any(
                issue.issue_type == "expired_document" for issue in dv.issues
            )
        ]
        expiry_review_docs = [
            dv for dv in validation_matrix.document_validations
            if dv.present and any(
                issue.issue_type == "expiry_check_needed" for issue in dv.issues
            )
        ]

        checks.append(ComplianceCheck(
            check_name="Documentos vigentes",
            check_category="document",
            is_compliant=len(expired_docs) == 0,
            severity=ValidationSeverity.CRITICAL,
            details=f"{len(expired_docs)} documento(s) vencido(s)" if expired_docs else "Todos los documentos vigentes",
            legal_basis="Requisitos institucionales",
            blocking=True
        ))
        if expiry_review_docs:
            checks.append(ComplianceCheck(
                check_name="Revisión de vigencia pendiente",
                check_category="document",
                is_compliant=False,
                severity=ValidationSeverity.WARNING,
                details=f"{len(expiry_review_docs)} documento(s) requieren verificación de vigencia",
                legal_basis="Requisitos institucionales",
                blocking=False
            ))

        # Check 3: Required elements present
        missing_elements = [ev for ev in validation_matrix.element_validations
                          if ev.status != ValidationStatus.VALID]

        checks.append(ComplianceCheck(
            check_name="Elementos requeridos presentes",
            check_category="legal",
            is_compliant=len(missing_elements) == 0,
            severity=ValidationSeverity.CRITICAL,
            details=f"{len(missing_elements)} elemento(s) faltante(s)" if missing_elements else "Todos los elementos presentes",
            legal_basis="Art. 255",
            blocking=True
        ))

        # Check 4: Data consistency (cross-document issues)
        consistency_issues = validation_matrix.cross_document_issues

        checks.append(ComplianceCheck(
            check_name="Consistencia de datos",
            check_category="legal",
            is_compliant=len(consistency_issues) == 0,
            severity=ValidationSeverity.ERROR,
            details=f"{len(consistency_issues)} inconsistencia(s) detectada(s)" if consistency_issues else "Datos consistentes",
            blocking=len(consistency_issues) > 0
        ))

        # Check 5: Critical validation issues
        critical_issues = [issue for issue in validation_matrix.get_all_issues()
                         if issue.severity == ValidationSeverity.CRITICAL]

        checks.append(ComplianceCheck(
            check_name="Sin problemas críticos",
            check_category="legal",
            is_compliant=len(critical_issues) == 0,
            severity=ValidationSeverity.CRITICAL,
            details=f"{len(critical_issues)} problema(s) crítico(s)" if critical_issues else "Sin problemas críticos",
            blocking=True
        ))

        # Check 6: Urgent gaps resolved
        urgent_gaps = [g for g in gap_report.gaps if g.priority == ActionPriority.URGENT]

        checks.append(ComplianceCheck(
            check_name="Brechas urgentes resueltas",
            check_category="document",
            is_compliant=len(urgent_gaps) == 0,
            severity=ValidationSeverity.CRITICAL,
            details=f"{len(urgent_gaps)} brecha(s) urgente(s) sin resolver" if urgent_gaps else "Todas las brechas urgentes resueltas",
            blocking=True
        ))

        # Check 7: Institution-specific requirements
        if legal_requirements.institution_rules:
            # Check if there are institution-related issues
            inst_issues = [issue for issue in validation_matrix.get_all_issues()
                          if legal_requirements.institution_rules.institution.lower() in issue.description.lower()]
            inst_compliant = len(inst_issues) == 0

            checks.append(ComplianceCheck(
                check_name=f"Requisitos {legal_requirements.institution_rules.institution}",
                check_category="institution",
                is_compliant=inst_compliant,
                severity=ValidationSeverity.CRITICAL,
                details="Requisitos institucionales cumplidos" if inst_compliant else f"{len(inst_issues)} requisito(s) institucional(es) no cumplido(s)",
                legal_basis=f"Requisitos {legal_requirements.institution_rules.institution}",
                blocking=True
            ))

        # Check 8: Article compliance
        articles_compliant = validation_matrix.can_issue_certificate
        checks.append(ComplianceCheck(
            check_name="Cumplimiento de Artículos 248-255",
            check_category="legal",
            is_compliant=articles_compliant,
            severity=ValidationSeverity.CRITICAL,
            details="Todos los artículos cumplidos" if articles_compliant else "Artículos no cumplidos",
            legal_basis="Arts. 248-255 Reglamento Notarial",
            blocking=True
        ))

        return checks

    @staticmethod
    def _make_decision(report: FinalConfirmationReport):
        """Make final certificate generation decision"""
        report.calculate_summary()
        review_failed = any(
            not check.is_compliant and check.check_category == "review"
            for check in report.compliance_checks
        )

        # Decision logic
        if report.blocking_issues == 0 and report.critical_issues == 0:
            if review_failed:
                report.compliance_level = ComplianceLevel.PARTIALLY_COMPLIANT
                report.certificate_decision = CertificateDecision.REQUIRES_REVIEW
                report.decision_rationale = (
                    "Documentos requieren verificación manual antes de emitir certificado."
                )
                for check in report.compliance_checks:
                    if not check.is_compliant and check.check_category == "review":
                        report.remaining_issues.append(f"{check.check_name}: {check.details}")
                return
            if report.warnings == 0:
                # Perfect compliance
                report.compliance_level = ComplianceLevel.FULLY_COMPLIANT
                report.certificate_decision = CertificateDecision.APPROVED
                report.decision_rationale = (
                    "Todos los requisitos legales cumplidos. "
                    "Documentación completa y válida. "
                    "Sin problemas bloqueantes ni advertencias. "
                    "Aprobado para generación de certificado."
                )
            else:
                # Minor warnings only
                report.compliance_level = ComplianceLevel.SUBSTANTIALLY_COMPLIANT
                report.certificate_decision = CertificateDecision.APPROVED_WITH_WARNINGS
                report.decision_rationale = (
                    f"Requisitos legales cumplidos con {report.warnings} advertencia(s) menor(es). "
                    "Documentación completa y válida. "
                    "Sin problemas bloqueantes. "
                    "Aprobado para generación de certificado con advertencias."
                )
                # List warnings
                for check in report.compliance_checks:
                    if not check.is_compliant and check.severity == ValidationSeverity.WARNING:
                        report.remaining_issues.append(f"{check.check_name}: {check.details}")

        elif report.blocking_issues > 0 or report.critical_issues > 0:
            # Critical issues present
            report.compliance_level = ComplianceLevel.NON_COMPLIANT
            report.certificate_decision = CertificateDecision.REJECTED
            report.decision_rationale = (
                f"Certificado rechazado: {report.blocking_issues} problema(s) bloqueante(s), "
                f"{report.critical_issues} problema(s) crítico(s). "
                "Debe resolver estos problemas antes de generar certificado."
            )
            # List blocking issues
            for check in report.compliance_checks:
                if not check.is_compliant and (check.blocking or check.severity == ValidationSeverity.CRITICAL):
                    report.remaining_issues.append(f"{check.check_name}: {check.details}")

        else:
            # Edge case: needs review
            report.compliance_level = ComplianceLevel.PARTIALLY_COMPLIANT
            report.certificate_decision = CertificateDecision.REQUIRES_REVIEW
            report.decision_rationale = (
                "Situación ambigua detectada. "
                "Se requiere revisión manual del notario antes de proceder."
            )
            for check in report.compliance_checks:
                if not check.is_compliant:
                    report.remaining_issues.append(f"{check.check_name}: {check.details}")

    @staticmethod
    def save_confirmation_report(report: FinalConfirmationReport, output_path: str) -> None:
        """Save final confirmation report to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report.to_json())
        print(f"\n✅ Reporte de confirmación guardado en: {output_path}")

    @staticmethod
    def load_confirmation_report(input_path: str) -> Dict:
        """Load confirmation report from JSON file (simplified)"""
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ Reporte de confirmación cargado desde: {input_path}")
        return data


def example_usage():
    """Example usage of Phase 8"""

    print("\n" + "="*70)
    print("  EJEMPLOS DE USO - FASE 8: CONFIRMACIÓN LEGAL FINAL")
    print("="*70)

    print("\n📌 Ejemplo 1: Confirmación legal completa")
    print("-" * 70)
    print("""
from src.phase8_final_confirmation import FinalConfirmationEngine

# Asumiendo que tienes legal_requirements (Fase 2) y update_result (Fase 7):

# Realizar confirmación final
confirmation_report = FinalConfirmationEngine.confirm(
    legal_requirements=legal_requirements,
    update_result=update_result
)

# Ver resultado
print(confirmation_report.get_summary())

# Verificar si puede proceder
if confirmation_report.can_proceed_to_phase9():
    print("✅ Puede proceder a Fase 9: Generación de Certificado")
else:
    print("❌ No puede proceder. Resolver problemas primero.")
    print(confirmation_report.get_detailed_report())
    """)

    print("\n📌 Ejemplo 2: Flujo completo (Fases 1-8)")
    print("-" * 70)
    print("""
from src.phase1_certificate_intent import CertificateIntentCapture
from src.phase2_legal_requirements import LegalRequirementsEngine
from src.phase3_document_intake import DocumentIntake
from src.phase4_text_extraction import TextExtractor
from src.phase5_legal_validation import LegalValidator
from src.phase6_gap_detection import GapDetector
from src.phase7_data_update import DataUpdater
from src.phase8_final_confirmation import FinalConfirmationEngine

# Fases 1-3: Preparación
intent = CertificateIntentCapture.capture_intent_from_params(...)
requirements = LegalRequirementsEngine.resolve_requirements(intent)
collection = DocumentIntake.create_collection(intent, requirements)
collection = DocumentIntake.scan_directory_for_client(...)

# Fase 4: Extracción inicial
extraction = TextExtractor.process_collection(collection)

# Fase 5: Validación inicial
validation = LegalValidator.validate(requirements, extraction)

# Fase 6: Detección de brechas
gap_report = GapDetector.analyze(validation)

# Fase 7: Actualizar documentos (si hay brechas)
if not gap_report.ready_for_certificate:
    update_result = DataUpdater.create_update_session(gap_report, collection)

    # Cargar documentos faltantes
    for gap in gap_report.gaps:
        if gap.priority == ActionPriority.URGENT:
            # Notario carga documentos...
            pass

    # Re-extraer
    update_result = DataUpdater.re_extract_data(update_result)
else:
    # Si no hay brechas, crear update_result vacío
    update_result = DataUpdater.create_update_session(gap_report, collection)
    update_result.updated_extraction_result = extraction

# Fase 8: Confirmación final
confirmation = FinalConfirmationEngine.confirm(requirements, update_result)

print(confirmation.get_summary())

if confirmation.certificate_decision == CertificateDecision.APPROVED:
    print("\\n✅ TODO LISTO PARA FASE 9: GENERACIÓN DE CERTIFICADO")
    # Proceder a Fase 9...
    """)

    print("\n📌 Ejemplo 3: Guardar y cargar reporte")
    print("-" * 70)
    print("""
# Guardar reporte
FinalConfirmationEngine.save_confirmation_report(
    confirmation_report,
    "confirmation_report.json"
)

# Cargar reporte (para referencia)
data = FinalConfirmationEngine.load_confirmation_report("confirmation_report.json")
    """)

    print("\n📌 Ejemplo 4: Análisis detallado de cumplimiento")
    print("-" * 70)
    print("""
# Ver reporte detallado
print(confirmation_report.get_detailed_report())

# Verificar checks específicos
for check in confirmation_report.compliance_checks:
    if not check.is_compliant:
        print(f"❌ {check.check_name}")
        print(f"   {check.details}")
        if check.blocking:
            print("   🚫 BLOQUEANTE")

# Estadísticas
print(f"Total de verificaciones: {confirmation_report.total_checks}")
print(f"Exitosas: {confirmation_report.passed_checks}")
print(f"Fallidas: {confirmation_report.failed_checks}")
print(f"Bloqueantes: {confirmation_report.blocking_issues}")
    """)


if __name__ == "__main__":
    example_usage()
