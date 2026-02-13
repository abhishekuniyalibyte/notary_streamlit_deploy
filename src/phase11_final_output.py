"""
Phase 11: Final Output

This module handles:
- Generating final certificate in production format
- Digital signature preparation
- Multiple output formats (PDF, DOCX, etc.)
- Archiving with full audit trail
- Metadata and tracking
- Final delivery preparation

This is the final phase that produces the official notarial certificate ready for use.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from datetime import datetime
from enum import Enum
import json
import hashlib
import os

from src.phase1_certificate_intent import CertificateIntent
from src.phase9_certificate_generation import GeneratedCertificate
from src.phase10_notary_review import ReviewSession, ReviewStatus


class OutputFormat(Enum):
    """Available output formats"""
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    HTML = "html"
    JSON = "json"


class SignatureStatus(Enum):
    """Digital signature status"""
    NOT_SIGNED = "not_signed"
    PENDING_SIGNATURE = "pending_signature"
    SIGNED = "signed"
    VERIFIED = "verified"


class DeliveryMethod(Enum):
    """How the certificate will be delivered"""
    PHYSICAL = "physical"
    EMAIL = "email"
    DOWNLOAD = "download"
    API = "api"
    GOVERNMENT_PORTAL = "government_portal"


@dataclass
class CertificateMetadata:
    """
    Metadata for the final certificate.
    """
    certificate_id: str
    certificate_number: str  # Official notarial number
    issue_date: datetime
    issuing_notary: str
    notary_office: str

    # Subject info
    subject_name: str
    subject_type: str  # person or company
    certificate_type: str
    purpose: str
    destination: str

    # Processing info
    generation_date: datetime
    review_date: Optional[datetime] = None
    finalization_date: datetime = field(default_factory=datetime.now)

    # Audit trail
    phases_completed: List[str] = field(default_factory=list)
    total_processing_time_minutes: Optional[int] = None

    # Signature
    signature_status: SignatureStatus = SignatureStatus.NOT_SIGNED
    signature_date: Optional[datetime] = None
    signature_hash: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "certificate_id": self.certificate_id,
            "certificate_number": self.certificate_number,
            "issue_date": self.issue_date.isoformat(),
            "issuing_notary": self.issuing_notary,
            "notary_office": self.notary_office,
            "subject_name": self.subject_name,
            "subject_type": self.subject_type,
            "certificate_type": self.certificate_type,
            "purpose": self.purpose,
            "destination": self.destination,
            "generation_date": self.generation_date.isoformat(),
            "review_date": self.review_date.isoformat() if self.review_date else None,
            "finalization_date": self.finalization_date.isoformat(),
            "phases_completed": self.phases_completed,
            "total_processing_time_minutes": self.total_processing_time_minutes,
            "signature_status": self.signature_status.value,
            "signature_date": self.signature_date.isoformat() if self.signature_date else None,
            "signature_hash": self.signature_hash
        }


@dataclass
class FinalCertificate:
    """
    The final certificate package ready for delivery.
    """
    metadata: CertificateMetadata
    certificate_text: str

    # Output files
    output_files: Dict[str, str] = field(default_factory=dict)  # format -> file_path

    # Audit trail
    original_draft: Optional[str] = None
    review_changes_count: int = 0

    # Delivery
    delivery_method: Optional[DeliveryMethod] = None
    delivered: bool = False
    delivery_date: Optional[datetime] = None
    delivery_confirmation: Optional[str] = None

    # Archival
    archive_path: Optional[str] = None
    archived: bool = False

    def to_dict(self) -> dict:
        return {
            "metadata": self.metadata.to_dict(),
            "certificate_text": self.certificate_text,
            "output_files": self.output_files,
            "original_draft": self.original_draft,
            "review_changes_count": self.review_changes_count,
            "delivery_method": self.delivery_method.value if self.delivery_method else None,
            "delivered": self.delivered,
            "delivery_date": self.delivery_date.isoformat() if self.delivery_date else None,
            "delivery_confirmation": self.delivery_confirmation,
            "archive_path": self.archive_path,
            "archived": self.archived
        }

    def to_json(self) -> str:
        """Serialize to JSON"""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def get_summary(self) -> str:
        """Get formatted summary"""
        border = "=" * 70

        sig_icons = {
            SignatureStatus.NOT_SIGNED: "⚪",
            SignatureStatus.PENDING_SIGNATURE: "🔶",
            SignatureStatus.SIGNED: "✅",
            SignatureStatus.VERIFIED: "✅🔒"
        }
        sig_icon = sig_icons.get(self.metadata.signature_status, "❓")

        summary = f"""
{border}
           FASE 11: CERTIFICADO FINAL
{border}

📋 CERTIFICADO: {self.metadata.certificate_number}
📅 Fecha de emisión: {self.metadata.issue_date.strftime('%Y-%m-%d')}
👤 Notario: {self.metadata.issuing_notary}

📝 DETALLES:
   Sujeto: {self.metadata.subject_name}
   Tipo: {self.metadata.certificate_type}
   Propósito: {self.metadata.purpose}
   Destino: {self.metadata.destination}

{sig_icon} FIRMA: {self.metadata.signature_status.value.upper().replace('_', ' ')}

📦 ARCHIVOS GENERADOS:
"""

        if self.output_files:
            for format_type, file_path in self.output_files.items():
                file_name = os.path.basename(file_path)
                summary += f"   ✓ {format_type.upper()}: {file_name}\n"
        else:
            summary += "   (ninguno)\n"

        if self.review_changes_count > 0:
            summary += f"\n✏️  Cambios de revisión: {self.review_changes_count}\n"

        if self.delivered:
            summary += f"\n📤 ENTREGADO: {self.delivery_date.strftime('%Y-%m-%d %H:%M')}\n"
            summary += f"   Método: {self.delivery_method.value if self.delivery_method else 'N/A'}\n"

        if self.archived:
            summary += f"\n📁 ARCHIVADO: {self.archive_path}\n"

        summary += f"\n{border}\n"

        return summary


class FinalOutputGenerator:
    """
    Main class for Phase 11: Final Output
    """

    @staticmethod
    def generate_final_certificate(
        certificate: GeneratedCertificate,
        review_session: ReviewSession,
        certificate_number: str,
        issuing_notary: str,
        notary_office: str
    ) -> FinalCertificate:
        """
        Generate the final certificate package.

        Args:
            certificate: GeneratedCertificate from Phase 9
            review_session: ReviewSession from Phase 10
            certificate_number: Official certificate number
            issuing_notary: Notary name
            notary_office: Notary office details

        Returns:
            FinalCertificate ready for output
        """
        print("\n" + "="*70)
        print("   FASE 11: GENERACIÓN DE SALIDA FINAL")
        print("="*70 + "\n")

        # Verify review was approved
        if review_session.status not in [ReviewStatus.APPROVED, ReviewStatus.APPROVED_WITH_CHANGES]:
            raise ValueError(
                f"No se puede generar certificado final: "
                f"Revisión no aprobada (estado: {review_session.status.value})"
            )

        print(f"✅ Revisión aprobada: {review_session.status.value}")

        # Generate certificate ID
        certificate_id = FinalOutputGenerator._generate_certificate_id(
            certificate.certificate_intent,
            certificate_number
        )

        print(f"📋 ID generado: {certificate_id}")

        # Create metadata
        metadata = CertificateMetadata(
            certificate_id=certificate_id,
            certificate_number=certificate_number,
            issue_date=datetime.now(),
            issuing_notary=issuing_notary,
            notary_office=notary_office,
            subject_name=certificate.certificate_intent.subject_name,
            subject_type=certificate.certificate_intent.subject_type,
            certificate_type=certificate.certificate_intent.certificate_type.value,
            purpose=certificate.certificate_intent.purpose.value,
            destination=FinalOutputGenerator._format_destination(certificate.certificate_intent.purpose.value),
            generation_date=certificate.generation_timestamp,
            review_date=review_session.end_time,
            phases_completed=[
                "Phase 1: Intent Definition",
                "Phase 2: Legal Requirements",
                "Phase 3: Document Intake",
                "Phase 4: Text Extraction",
                "Phase 5: Legal Validation",
                "Phase 6: Gap Detection",
                "Phase 7: Data Update",
                "Phase 8: Final Confirmation",
                "Phase 9: Certificate Generation",
                "Phase 10: Notary Review",
                "Phase 11: Final Output"
            ]
        )

        # Calculate total processing time
        if review_session.end_time:
            total_time = (datetime.now() - certificate.generation_timestamp).total_seconds() / 60
            metadata.total_processing_time_minutes = int(total_time)

        # Use reviewed text if changes were made, otherwise use original
        final_text = review_session.reviewed_text if review_session.edits else certificate.get_formatted_text()

        final_cert = FinalCertificate(
            metadata=metadata,
            certificate_text=final_text,
            original_draft=certificate.get_formatted_text(),
            review_changes_count=len(review_session.edits)
        )

        print(f"✅ Certificado final generado")
        print(f"   Número: {certificate_number}")
        print(f"   Cambios de revisión: {len(review_session.edits)}")
        print(f"   Tiempo total: {metadata.total_processing_time_minutes} minutos\n")

        return final_cert

    @staticmethod
    def _generate_certificate_id(intent: CertificateIntent, cert_number: str) -> str:
        """Generate unique certificate ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        subject_hash = hashlib.md5(intent.subject_name.encode()).hexdigest()[:8]
        return f"CERT-{cert_number}-{timestamp}-{subject_hash}"

    @staticmethod
    def _format_destination(purpose: str) -> str:
        """Format destination for metadata"""
        return purpose.replace("_", " ").title()

    @staticmethod
    def export_to_format(
        final_cert: FinalCertificate,
        output_format: OutputFormat,
        output_path: str
    ) -> FinalCertificate:
        """
        Export certificate to specific format.

        Args:
            final_cert: FinalCertificate to export
            output_format: Desired format
            output_path: Output file path

        Returns:
            Updated FinalCertificate with file path recorded
        """
        if output_format == OutputFormat.TXT:
            FinalOutputGenerator._export_txt(final_cert, output_path)
        elif output_format == OutputFormat.HTML:
            FinalOutputGenerator._export_html(final_cert, output_path)
        elif output_format == OutputFormat.JSON:
            FinalOutputGenerator._export_json(final_cert, output_path)
        elif output_format == OutputFormat.PDF:
            FinalOutputGenerator._export_pdf_placeholder(final_cert, output_path)
        elif output_format == OutputFormat.DOCX:
            FinalOutputGenerator._export_docx_placeholder(final_cert, output_path)
        else:
            raise ValueError(f"Formato no soportado: {output_format}")

        final_cert.output_files[output_format.value] = output_path
        print(f"✅ Exportado a {output_format.value.upper()}: {output_path}")

        return final_cert

    @staticmethod
    def _export_txt(final_cert: FinalCertificate, output_path: str):
        """Export as plain text"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_cert.certificate_text)

    @staticmethod
    def _export_html(final_cert: FinalCertificate, output_path: str):
        """Export as HTML"""
        html = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Certificado Notarial - {final_cert.metadata.certificate_number}</title>
    <style>
        body {{
            font-family: 'Times New Roman', serif;
            max-width: 800px;
            margin: 40px auto;
            padding: 40px;
            line-height: 1.8;
            background-color: #f5f5f5;
        }}
        .certificate {{
            background: white;
            padding: 60px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
            border: 2px solid #333;
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            border-bottom: 2px solid #333;
            padding-bottom: 20px;
        }}
        .cert-number {{
            font-weight: bold;
            font-size: 14px;
            color: #666;
        }}
        .content {{
            white-space: pre-wrap;
            font-size: 14px;
        }}
        .footer {{
            margin-top: 40px;
            text-align: right;
            font-size: 12px;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="certificate">
        <div class="header">
            <div class="cert-number">Certificado N° {final_cert.metadata.certificate_number}</div>
            <div style="margin-top: 10px;">{final_cert.metadata.issue_date.strftime('%d de %B de %Y')}</div>
        </div>
        <div class="content">{final_cert.certificate_text}</div>
        <div class="footer">
            <div>Certificado ID: {final_cert.metadata.certificate_id}</div>
            <div>Generado: {final_cert.metadata.finalization_date.strftime('%Y-%m-%d %H:%M:%S')}</div>
        </div>
    </div>
</body>
</html>"""

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)

    @staticmethod
    def _export_json(final_cert: FinalCertificate, output_path: str):
        """Export as JSON (full metadata + content)"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_cert.to_json())

    @staticmethod
    def _export_pdf_placeholder(final_cert: FinalCertificate, output_path: str):
        """Placeholder for PDF export (would use reportlab or similar)"""
        # In production, would use reportlab, weasyprint, or similar
        placeholder = f"""PDF Export Placeholder
Certificate Number: {final_cert.metadata.certificate_number}
Subject: {final_cert.metadata.subject_name}

To implement PDF generation, install reportlab:
    pip install reportlab

Then use reportlab.pdfgen.canvas to create professional PDFs.
"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(placeholder)

        print("⚠️  PDF export is a placeholder - implement with reportlab for production")

    @staticmethod
    def _export_docx_placeholder(final_cert: FinalCertificate, output_path: str):
        """Placeholder for DOCX export (would use python-docx)"""
        # In production, would use python-docx
        placeholder = f"""DOCX Export Placeholder
Certificate Number: {final_cert.metadata.certificate_number}
Subject: {final_cert.metadata.subject_name}

To implement DOCX generation, install python-docx:
    pip install python-docx

Then use docx.Document to create Word documents.
"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(placeholder)

        print("⚠️  DOCX export is a placeholder - implement with python-docx for production")

    @staticmethod
    def prepare_for_signature(final_cert: FinalCertificate) -> FinalCertificate:
        """
        Prepare certificate for digital signature.

        Args:
            final_cert: FinalCertificate

        Returns:
            Updated certificate with signature preparation
        """
        # Generate hash of certificate content
        content_hash = hashlib.sha256(final_cert.certificate_text.encode()).hexdigest()

        final_cert.metadata.signature_status = SignatureStatus.PENDING_SIGNATURE
        final_cert.metadata.signature_hash = content_hash

        print(f"\n🔐 Certificado preparado para firma digital")
        print(f"   Hash: {content_hash[:16]}...")

        return final_cert

    @staticmethod
    def mark_as_signed(
        final_cert: FinalCertificate,
        signature_data: Optional[str] = None
    ) -> FinalCertificate:
        """
        Mark certificate as digitally signed.

        Args:
            final_cert: FinalCertificate
            signature_data: Optional signature data/hash

        Returns:
            Updated certificate marked as signed
        """
        final_cert.metadata.signature_status = SignatureStatus.SIGNED
        final_cert.metadata.signature_date = datetime.now()

        print(f"\n✅ Certificado firmado digitalmente")
        print(f"   Fecha: {final_cert.metadata.signature_date.strftime('%Y-%m-%d %H:%M:%S')}")

        return final_cert

    @staticmethod
    def archive_certificate(
        final_cert: FinalCertificate,
        archive_directory: str
    ) -> FinalCertificate:
        """
        Archive certificate with full audit trail.

        Args:
            final_cert: FinalCertificate
            archive_directory: Directory to store archive

        Returns:
            Updated certificate with archive info
        """
        # Create archive subdirectory based on date
        date_folder = datetime.now().strftime("%Y/%m")
        archive_path = os.path.join(archive_directory, date_folder, final_cert.metadata.certificate_id)

        os.makedirs(archive_path, exist_ok=True)

        # Save metadata
        metadata_path = os.path.join(archive_path, "metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(final_cert.metadata.to_dict(), f, indent=2, ensure_ascii=False)

        # Save certificate text
        cert_path = os.path.join(archive_path, "certificate.txt")
        with open(cert_path, 'w', encoding='utf-8') as f:
            f.write(final_cert.certificate_text)

        # Save full package
        package_path = os.path.join(archive_path, "full_package.json")
        with open(package_path, 'w', encoding='utf-8') as f:
            f.write(final_cert.to_json())

        final_cert.archive_path = archive_path
        final_cert.archived = True

        print(f"\n📁 Certificado archivado")
        print(f"   Ruta: {archive_path}")

        return final_cert

    @staticmethod
    def mark_as_delivered(
        final_cert: FinalCertificate,
        delivery_method: DeliveryMethod,
        confirmation: Optional[str] = None
    ) -> FinalCertificate:
        """
        Mark certificate as delivered.

        Args:
            final_cert: FinalCertificate
            delivery_method: How it was delivered
            confirmation: Optional delivery confirmation

        Returns:
            Updated certificate marked as delivered
        """
        final_cert.delivered = True
        final_cert.delivery_date = datetime.now()
        final_cert.delivery_method = delivery_method
        final_cert.delivery_confirmation = confirmation

        print(f"\n📤 Certificado entregado")
        print(f"   Método: {delivery_method.value}")
        print(f"   Fecha: {final_cert.delivery_date.strftime('%Y-%m-%d %H:%M:%S')}")

        return final_cert


def example_usage():
    """Example usage of Phase 11"""

    print("\n" + "="*70)
    print("  EJEMPLOS DE USO - FASE 11: SALIDA FINAL")
    print("="*70)

    print("\n📌 Ejemplo 1: Generar certificado final")
    print("-" * 70)
    print("""
from src.phase11_final_output import FinalOutputGenerator, OutputFormat, DeliveryMethod

# Asumiendo que tienes:
# - certificate (Fase 9)
# - review_session (Fase 10 - aprobada)

# Generar certificado final
final_cert = FinalOutputGenerator.generate_final_certificate(
    certificate=certificate,
    review_session=review_session,
    certificate_number="2026-001-ABC",
    issuing_notary="Dr. Juan Pérez",
    notary_office="Escribanía Juan Pérez - Montevideo"
)

print(final_cert.get_summary())
    """)

    print("\n📌 Ejemplo 2: Exportar a múltiples formatos")
    print("-" * 70)
    print("""
# Exportar a texto plano
final_cert = FinalOutputGenerator.export_to_format(
    final_cert,
    OutputFormat.TXT,
    "certificado_final.txt"
)

# Exportar a HTML
final_cert = FinalOutputGenerator.export_to_format(
    final_cert,
    OutputFormat.HTML,
    "certificado_final.html"
)

# Exportar a JSON (con metadata completa)
final_cert = FinalOutputGenerator.export_to_format(
    final_cert,
    OutputFormat.JSON,
    "certificado_final.json"
)

# PDF y DOCX (placeholders - implementar en producción)
final_cert = FinalOutputGenerator.export_to_format(
    final_cert,
    OutputFormat.PDF,
    "certificado_final.pdf"
)
    """)

    print("\n📌 Ejemplo 3: Firma digital")
    print("-" * 70)
    print("""
# Preparar para firma
final_cert = FinalOutputGenerator.prepare_for_signature(final_cert)

# (Aquí integrar con sistema de firma digital)
# signature = external_signature_service.sign(final_cert.metadata.signature_hash)

# Marcar como firmado
final_cert = FinalOutputGenerator.mark_as_signed(final_cert)
    """)

    print("\n📌 Ejemplo 4: Archivar")
    print("-" * 70)
    print("""
# Archivar con audit trail completo
final_cert = FinalOutputGenerator.archive_certificate(
    final_cert,
    archive_directory="/archive/certificates"
)

# Estructura creada:
# /archive/certificates/2026/01/CERT-001-ABC-timestamp/
#   ├── metadata.json
#   ├── certificate.txt
#   └── full_package.json
    """)

    print("\n📌 Ejemplo 5: Marcar como entregado")
    print("-" * 70)
    print("""
# Marcar como entregado por email
final_cert = FinalOutputGenerator.mark_as_delivered(
    final_cert,
    delivery_method=DeliveryMethod.EMAIL,
    confirmation="Enviado a cliente@example.com - ID: MSG-12345"
)
    """)

    print("\n📌 Ejemplo 6: Flujo completo (Fases 9-11)")
    print("-" * 70)
    print("""
from src.phase9_certificate_generation import CertificateGenerator
from src.phase10_notary_review import NotaryReviewSystem, ReviewStatus
from src.phase11_final_output import FinalOutputGenerator, OutputFormat, DeliveryMethod

# Fase 9: Generar certificado
certificate = CertificateGenerator.generate(...)

# Fase 10: Revisión del notario
session = NotaryReviewSystem.start_review(certificate, "Dr. Juan Pérez")
# ... notario revisa y edita ...
session = NotaryReviewSystem.approve_certificate(session)

# Fase 11: Salida final
if session.status in [ReviewStatus.APPROVED, ReviewStatus.APPROVED_WITH_CHANGES]:
    # Generar certificado final
    final_cert = FinalOutputGenerator.generate_final_certificate(
        certificate, session,
        certificate_number="2026-001-ABC",
        issuing_notary="Dr. Juan Pérez",
        notary_office="Escribanía Juan Pérez"
    )

    # Exportar a formatos
    final_cert = FinalOutputGenerator.export_to_format(
        final_cert, OutputFormat.TXT, "output/cert.txt"
    )
    final_cert = FinalOutputGenerator.export_to_format(
        final_cert, OutputFormat.HTML, "output/cert.html"
    )

    # Preparar firma
    final_cert = FinalOutputGenerator.prepare_for_signature(final_cert)

    # Firmar (integrar con sistema de firma)
    final_cert = FinalOutputGenerator.mark_as_signed(final_cert)

    # Archivar
    final_cert = FinalOutputGenerator.archive_certificate(
        final_cert, "/archive/certificates"
    )

    # Entregar
    final_cert = FinalOutputGenerator.mark_as_delivered(
        final_cert,
        DeliveryMethod.EMAIL,
        "Enviado exitosamente"
    )

    print(final_cert.get_summary())

    print("\\n🎉 ¡PROCESO COMPLETO! Certificado notarial generado y entregado.")
    """)


if __name__ == "__main__":
    example_usage()
