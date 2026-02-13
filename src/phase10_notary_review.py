"""
Phase 10: Notary Review & Learning

This module handles:
- Presenting draft certificate to notary for review
- Capturing notary edits and corrections
- Recording feedback for system improvement
- Tracking approval/rejection decisions
- Learning from corrections to improve future generations
- Managing review workflow and version control

This is the human-in-the-loop phase that ensures quality and enables continuous improvement.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
from datetime import datetime
from enum import Enum
import json
import difflib

from src.phase1_certificate_intent import CertificateIntent
from src.phase9_certificate_generation import GeneratedCertificate, CertificateSection


class ReviewStatus(Enum):
    """Status of notary review"""
    PENDING = "pending"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    APPROVED_WITH_CHANGES = "approved_with_changes"
    REJECTED = "rejected"
    REQUIRES_REVISION = "requires_revision"


class ChangeType(Enum):
    """Type of change made by notary"""
    WORDING = "wording"  # Better phrasing
    LEGAL_ACCURACY = "legal_accuracy"  # Legal correction
    DATA_CORRECTION = "data_correction"  # Factual error
    FORMATTING = "formatting"  # Format/style change
    ADDITION = "addition"  # Added content
    DELETION = "deletion"  # Removed content
    OTHER = "other"


class FeedbackCategory(Enum):
    """Category of feedback"""
    TEMPLATE_IMPROVEMENT = "template_improvement"
    DATA_EXTRACTION = "data_extraction"
    LEGAL_INTERPRETATION = "legal_interpretation"
    INSTITUTION_RULES = "institution_rules"
    FORMATTING = "formatting"
    GENERAL = "general"


@dataclass
class NotaryEdit:
    """
    Represents a single edit made by the notary.
    """
    section_type: Optional[str] = None
    original_text: str = ""
    edited_text: str = ""
    change_type: ChangeType = ChangeType.OTHER
    reason: str = ""
    line_number: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "section_type": self.section_type,
            "original_text": self.original_text,
            "edited_text": self.edited_text,
            "change_type": self.change_type.value,
            "reason": self.reason,
            "line_number": self.line_number,
            "timestamp": self.timestamp.isoformat()
        }

    def get_diff(self) -> str:
        """Get formatted diff of the change"""
        diff = difflib.unified_diff(
            self.original_text.splitlines(keepends=True),
            self.edited_text.splitlines(keepends=True),
            fromfile='original',
            tofile='edited',
            lineterm=''
        )
        return ''.join(diff)


@dataclass
class NotaryFeedback:
    """
    Feedback from notary for system improvement.
    """
    category: FeedbackCategory
    feedback_text: str
    severity: str = "low"  # low, medium, high
    actionable: bool = True
    related_certificate_type: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "category": self.category.value,
            "feedback_text": self.feedback_text,
            "severity": self.severity,
            "actionable": self.actionable,
            "related_certificate_type": self.related_certificate_type,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class ReviewSession:
    """
    Complete review session for a certificate.
    """
    certificate: GeneratedCertificate
    reviewer_name: str

    # Review data
    status: ReviewStatus = ReviewStatus.PENDING
    edits: List[NotaryEdit] = field(default_factory=list)
    feedback: List[NotaryFeedback] = field(default_factory=list)

    # Versions
    original_text: str = ""
    reviewed_text: str = ""

    # Review metadata
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    review_duration_minutes: Optional[int] = None

    # Decision
    approval_decision: Optional[str] = None
    rejection_reason: Optional[str] = None
    notary_notes: str = ""

    # Learning
    key_corrections: List[str] = field(default_factory=list)
    template_suggestions: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "certificate": self.certificate.to_dict(),
            "reviewer_name": self.reviewer_name,
            "status": self.status.value,
            "edits": [e.to_dict() for e in self.edits],
            "feedback": [f.to_dict() for f in self.feedback],
            "original_text": self.original_text,
            "reviewed_text": self.reviewed_text,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "review_duration_minutes": self.review_duration_minutes,
            "approval_decision": self.approval_decision,
            "rejection_reason": self.rejection_reason,
            "notary_notes": self.notary_notes,
            "key_corrections": self.key_corrections,
            "template_suggestions": self.template_suggestions
        }

    def to_json(self) -> str:
        """Serialize to JSON"""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def get_summary(self) -> str:
        """Get formatted summary"""
        border = "=" * 70

        status_icons = {
            ReviewStatus.APPROVED: "✅",
            ReviewStatus.APPROVED_WITH_CHANGES: "✏️",
            ReviewStatus.REJECTED: "❌",
            ReviewStatus.REQUIRES_REVISION: "🔄",
            ReviewStatus.IN_REVIEW: "👁️",
            ReviewStatus.PENDING: "⏳"
        }
        status_icon = status_icons.get(self.status, "❓")

        summary = f"""
{border}
           FASE 10: REVISIÓN DEL NOTARIO
{border}

{status_icon} ESTADO: {self.status.value.upper().replace('_', ' ')}
👤 Revisor: {self.reviewer_name}

📋 CERTIFICADO:
   Tipo: {self.certificate.certificate_intent.certificate_type.value}
   Sujeto: {self.certificate.certificate_intent.subject_name}

📝 REVISIÓN:
   Ediciones realizadas: {len(self.edits)}
   Comentarios: {len(self.feedback)}
   Duración: {self.review_duration_minutes or '---'} minutos

"""

        if self.status == ReviewStatus.APPROVED:
            summary += "✅ APROBADO SIN CAMBIOS\n"
            summary += "   El certificado está listo para firma.\n\n"

        elif self.status == ReviewStatus.APPROVED_WITH_CHANGES:
            summary += f"✏️  APROBADO CON {len(self.edits)} CAMBIO(S)\n"
            summary += "   Cambios incorporados. Listo para firma.\n\n"

        elif self.status == ReviewStatus.REJECTED:
            summary += f"❌ RECHAZADO\n"
            summary += f"   Razón: {self.rejection_reason}\n\n"

        if self.edits:
            summary += "📝 EDICIONES PRINCIPALES:\n"
            for edit in self.edits[:5]:  # Show first 5
                summary += f"   • {edit.change_type.value}: {edit.reason[:50]}...\n"
            if len(self.edits) > 5:
                summary += f"   ... y {len(self.edits) - 5} más\n"
            summary += "\n"

        if self.key_corrections:
            summary += "🔑 CORRECCIONES CLAVE:\n"
            for correction in self.key_corrections:
                summary += f"   • {correction}\n"
            summary += "\n"

        summary += border + "\n"

        return summary


class NotaryReviewSystem:
    """
    Main class for Phase 10: Notary Review & Learning
    """

    @staticmethod
    def start_review(
        certificate: GeneratedCertificate,
        reviewer_name: str
    ) -> ReviewSession:
        """
        Start a new review session.

        Args:
            certificate: GeneratedCertificate from Phase 9
            reviewer_name: Name of the reviewing notary

        Returns:
            ReviewSession initialized and ready for review
        """
        print("\n" + "="*70)
        print("   FASE 10: REVISIÓN DEL NOTARIO")
        print("="*70 + "\n")

        session = ReviewSession(
            certificate=certificate,
            reviewer_name=reviewer_name,
            original_text=certificate.get_formatted_text(),
            reviewed_text=certificate.get_formatted_text(),
            status=ReviewStatus.IN_REVIEW
        )

        print(f"✅ Sesión de revisión iniciada")
        print(f"   Revisor: {reviewer_name}")
        print(f"   Certificado: {certificate.certificate_intent.certificate_type.value}")
        print(f"   Hora inicio: {session.start_time.strftime('%H:%M:%S')}\n")

        return session

    @staticmethod
    def add_edit(
        session: ReviewSession,
        original_text: str,
        edited_text: str,
        change_type: ChangeType,
        reason: str,
        section_type: Optional[str] = None
    ) -> ReviewSession:
        """
        Add an edit to the review session.

        Args:
            session: Current ReviewSession
            original_text: Original text
            edited_text: Edited text
            change_type: Type of change
            reason: Reason for the change
            section_type: Optional section identifier

        Returns:
            Updated ReviewSession
        """
        edit = NotaryEdit(
            section_type=section_type,
            original_text=original_text,
            edited_text=edited_text,
            change_type=change_type,
            reason=reason
        )

        session.edits.append(edit)

        # Update reviewed text
        session.reviewed_text = session.reviewed_text.replace(original_text, edited_text)

        print(f"✏️  Edición agregada: {change_type.value}")
        print(f"   Razón: {reason}")

        return session

    @staticmethod
    def add_feedback(
        session: ReviewSession,
        category: FeedbackCategory,
        feedback_text: str,
        severity: str = "low",
        actionable: bool = True
    ) -> ReviewSession:
        """
        Add feedback for system improvement.

        Args:
            session: Current ReviewSession
            category: Feedback category
            feedback_text: The feedback text
            severity: Importance level
            actionable: Whether this can be acted upon

        Returns:
            Updated ReviewSession
        """
        feedback = NotaryFeedback(
            category=category,
            feedback_text=feedback_text,
            severity=severity,
            actionable=actionable,
            related_certificate_type=session.certificate.certificate_intent.certificate_type.value
        )

        session.feedback.append(feedback)

        print(f"💬 Feedback agregado: {category.value}")
        print(f"   Severidad: {severity}")

        return session

    @staticmethod
    def approve_certificate(
        session: ReviewSession,
        notes: str = ""
    ) -> ReviewSession:
        """
        Approve the certificate.

        Args:
            session: Current ReviewSession
            notes: Optional approval notes

        Returns:
            Updated ReviewSession with approved status
        """
        session.end_time = datetime.now()
        session.review_duration_minutes = int(
            (session.end_time - session.start_time).total_seconds() / 60
        )

        if len(session.edits) == 0:
            session.status = ReviewStatus.APPROVED
            session.approval_decision = "Aprobado sin cambios"
        else:
            session.status = ReviewStatus.APPROVED_WITH_CHANGES
            session.approval_decision = f"Aprobado con {len(session.edits)} cambio(s)"

        session.notary_notes = notes

        # Extract key corrections for learning
        session.key_corrections = NotaryReviewSystem._extract_key_corrections(session)

        print(f"\n✅ Certificado aprobado")
        print(f"   {session.approval_decision}")
        print(f"   Duración de revisión: {session.review_duration_minutes} minutos\n")

        return session

    @staticmethod
    def reject_certificate(
        session: ReviewSession,
        reason: str,
        notes: str = ""
    ) -> ReviewSession:
        """
        Reject the certificate.

        Args:
            session: Current ReviewSession
            reason: Reason for rejection
            notes: Additional notes

        Returns:
            Updated ReviewSession with rejected status
        """
        session.end_time = datetime.now()
        session.review_duration_minutes = int(
            (session.end_time - session.start_time).total_seconds() / 60
        )

        session.status = ReviewStatus.REJECTED
        session.rejection_reason = reason
        session.notary_notes = notes

        print(f"\n❌ Certificado rechazado")
        print(f"   Razón: {reason}\n")

        return session

    @staticmethod
    def _extract_key_corrections(session: ReviewSession) -> List[str]:
        """Extract key corrections for learning"""
        key_corrections = []

        # Group edits by type
        legal_edits = [e for e in session.edits if e.change_type == ChangeType.LEGAL_ACCURACY]
        data_edits = [e for e in session.edits if e.change_type == ChangeType.DATA_CORRECTION]

        if legal_edits:
            key_corrections.append(
                f"{len(legal_edits)} corrección(es) de precisión legal"
            )

        if data_edits:
            key_corrections.append(
                f"{len(data_edits)} corrección(es) de datos factuales"
            )

        # Extract common patterns
        if len(session.edits) > 5:
            key_corrections.append(
                "Múltiples ediciones - revisar plantilla"
            )

        return key_corrections

    @staticmethod
    def get_change_report(session: ReviewSession) -> str:
        """
        Get detailed change report.

        Args:
            session: ReviewSession

        Returns:
            Formatted change report
        """
        report = "\n" + "=" * 70 + "\n"
        report += "         REPORTE DETALLADO DE CAMBIOS - FASE 10\n"
        report += "=" * 70 + "\n\n"

        if not session.edits:
            report += "ℹ️  No se realizaron cambios.\n"
            return report

        # Group by change type
        by_type: Dict[ChangeType, List[NotaryEdit]] = {}
        for edit in session.edits:
            if edit.change_type not in by_type:
                by_type[edit.change_type] = []
            by_type[edit.change_type].append(edit)

        for change_type, edits in by_type.items():
            report += f"\n📝 {change_type.value.upper()} ({len(edits)} cambios)\n"
            report += "-" * 70 + "\n"

            for i, edit in enumerate(edits, 1):
                report += f"\n{i}. {edit.reason}\n"
                if edit.section_type:
                    report += f"   Sección: {edit.section_type}\n"
                report += f"\n   Original:\n   {edit.original_text[:100]}...\n"
                report += f"\n   Editado:\n   {edit.edited_text[:100]}...\n"

        return report

    @staticmethod
    def get_learning_insights(session: ReviewSession) -> Dict[str, any]:
        """
        Extract learning insights from review session.

        Args:
            session: Completed ReviewSession

        Returns:
            Dictionary of learning insights
        """
        insights = {
            "certificate_type": session.certificate.certificate_intent.certificate_type.value,
            "total_edits": len(session.edits),
            "edit_types": {},
            "feedback_categories": {},
            "common_issues": [],
            "template_improvements": []
        }

        # Count edit types
        for edit in session.edits:
            edit_type = edit.change_type.value
            insights["edit_types"][edit_type] = insights["edit_types"].get(edit_type, 0) + 1

        # Count feedback categories
        for fb in session.feedback:
            cat = fb.category.value
            insights["feedback_categories"][cat] = insights["feedback_categories"].get(cat, 0) + 1

        # Identify common issues
        if insights["edit_types"].get("legal_accuracy", 0) > 2:
            insights["common_issues"].append("Precisión legal requiere atención")

        if insights["edit_types"].get("data_correction", 0) > 2:
            insights["common_issues"].append("Extracción de datos necesita mejora")

        if len(session.edits) > 10:
            insights["common_issues"].append("Plantilla requiere revisión significativa")

        # Template improvements
        for fb in session.feedback:
            if fb.category == FeedbackCategory.TEMPLATE_IMPROVEMENT and fb.actionable:
                insights["template_improvements"].append(fb.feedback_text)

        return insights

    @staticmethod
    def compare_versions(original: str, reviewed: str) -> List[Tuple[str, str]]:
        """
        Compare original and reviewed versions.

        Args:
            original: Original certificate text
            reviewed: Reviewed certificate text

        Returns:
            List of (line_type, content) tuples for diff display
        """
        diff = difflib.unified_diff(
            original.splitlines(keepends=True),
            reviewed.splitlines(keepends=True),
            fromfile='Original',
            tofile='Revisado',
            lineterm=''
        )

        diff_output = []
        for line in diff:
            if line.startswith('+'):
                diff_output.append(('added', line))
            elif line.startswith('-'):
                diff_output.append(('removed', line))
            elif line.startswith('@@'):
                diff_output.append(('context', line))
            else:
                diff_output.append(('unchanged', line))

        return diff_output

    @staticmethod
    def save_review_session(session: ReviewSession, output_path: str) -> None:
        """Save review session to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(session.to_json())
        print(f"\n✅ Sesión de revisión guardada en: {output_path}")

    @staticmethod
    def load_review_session(input_path: str) -> Dict:
        """Load review session from JSON file (simplified)"""
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ Sesión de revisión cargada desde: {input_path}")
        return data


def example_usage():
    """Example usage of Phase 10"""

    print("\n" + "="*70)
    print("  EJEMPLOS DE USO - FASE 10: REVISIÓN DEL NOTARIO")
    print("="*70)

    print("\n📌 Ejemplo 1: Iniciar revisión")
    print("-" * 70)
    print("""
from src.phase10_notary_review import NotaryReviewSystem, ChangeType, FeedbackCategory

# Asumiendo que tienes certificate (Fase 9)

# Iniciar sesión de revisión
session = NotaryReviewSystem.start_review(
    certificate=certificate,
    reviewer_name="Dr. Juan Pérez"
)

print(session.get_summary())
    """)

    print("\n📌 Ejemplo 2: Agregar ediciones")
    print("-" * 70)
    print("""
# El notario encuentra un error de redacción
session = NotaryReviewSystem.add_edit(
    session=session,
    original_text="Que la sociedad se encuentra vigente y activa",
    edited_text="Que la sociedad se encuentra vigente y en pleno funcionamiento",
    change_type=ChangeType.WORDING,
    reason="Mejor redacción conforme a práctica notarial",
    section_type="certifications"
)

# Corrección legal
session = NotaryReviewSystem.add_edit(
    session=session,
    original_text="conforme a lo dispuesto en el artículo 248",
    edited_text="conforme a lo dispuesto en los artículos 248 y 249",
    change_type=ChangeType.LEGAL_ACCURACY,
    reason="Faltó citar artículo 249 (fuente de documentos)",
    section_type="legal_basis"
)
    """)

    print("\n📌 Ejemplo 3: Agregar feedback")
    print("-" * 70)
    print("""
# Feedback para mejorar el sistema
session = NotaryReviewSystem.add_feedback(
    session=session,
    category=FeedbackCategory.TEMPLATE_IMPROVEMENT,
    feedback_text="La plantilla para BPS debería incluir mención explícita de aportes al día",
    severity="medium",
    actionable=True
)

session = NotaryReviewSystem.add_feedback(
    session=session,
    category=FeedbackCategory.DATA_EXTRACTION,
    feedback_text="El sistema no extrajo correctamente el número de acta",
    severity="high",
    actionable=True
)
    """)

    print("\n📌 Ejemplo 4: Aprobar o rechazar")
    print("-" * 70)
    print("""
# Aprobar con cambios
session = NotaryReviewSystem.approve_certificate(
    session=session,
    notes="Cambios menores de redacción. Certificado listo para firma."
)

# O rechazar si hay problemas graves
# session = NotaryReviewSystem.reject_certificate(
#     session=session,
#     reason="Falta documentación requerida por BPS",
#     notes="Solicitar certificado BPS actualizado y volver a Fase 7"
# )

print(session.get_summary())
    """)

    print("\n📌 Ejemplo 5: Obtener insights de aprendizaje")
    print("-" * 70)
    print("""
# Extraer aprendizaje de la sesión
insights = NotaryReviewSystem.get_learning_insights(session)

print("Insights de aprendizaje:")
print(f"  Tipo de certificado: {insights['certificate_type']}")
print(f"  Total ediciones: {insights['total_edits']}")
print(f"  Tipos de edición: {insights['edit_types']}")
print(f"  Problemas comunes: {insights['common_issues']}")
print(f"  Mejoras de plantilla: {insights['template_improvements']}")
    """)

    print("\n📌 Ejemplo 6: Flujo completo (Fases 9-10)")
    print("-" * 70)
    print("""
from src.phase9_certificate_generation import CertificateGenerator
from src.phase10_notary_review import NotaryReviewSystem, ChangeType, ReviewStatus

# Fase 9: Generar certificado
certificate = CertificateGenerator.generate(
    intent, requirements, extraction_result, confirmation_report,
    notary_name="Dr. Juan Pérez"
)

# Fase 10: Revisión del notario
session = NotaryReviewSystem.start_review(certificate, "Dr. Juan Pérez")

# Notario revisa y hace cambios...
session = NotaryReviewSystem.add_edit(
    session, original_text, edited_text,
    ChangeType.WORDING, "Mejor redacción"
)

# Aprobar
session = NotaryReviewSystem.approve_certificate(session)

# Ver reporte final
print(session.get_summary())
print(NotaryReviewSystem.get_change_report(session))

# Guardar para Phase 11
NotaryReviewSystem.save_review_session(session, "review_session.json")

# Si aprobado, continuar a Fase 11
if session.status in [ReviewStatus.APPROVED, ReviewStatus.APPROVED_WITH_CHANGES]:
    print("\\n✅ Proceder a Fase 11: Salida Final")
    final_text = session.reviewed_text  # Usar texto revisado
else:
    print("\\n❌ Volver a fase anterior para correcciones")
    """)


if __name__ == "__main__":
    example_usage()
