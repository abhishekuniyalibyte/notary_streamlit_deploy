"""
Phase 9: Certificate Generation

This module handles:
- Generating notarial certificates in correct legal format
- Applying institution-specific templates
- Inserting verified data into certificate text
- Formatting according to Uruguayan notarial standards
- Creating draft certificates for notary review
- Supporting multiple output formats (text, structured)

This phase creates the actual legal instrument text based on all validated data.
Only executes if Phase 8 approved the certificate for generation.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from datetime import datetime
from enum import Enum
import json
import re

from src.phase1_certificate_intent import CertificateIntent, CertificateType, Purpose
from src.phase2_legal_requirements import LegalRequirements
from src.phase4_text_extraction import CollectionExtractionResult, ExtractedData
from src.phase8_final_confirmation import FinalConfirmationReport


class CertificateFormat(Enum):
    """Output format for certificate"""
    PLAIN_TEXT = "plain_text"
    FORMATTED_TEXT = "formatted_text"
    STRUCTURED_JSON = "structured_json"
    HTML = "html"


class TemplateSection(Enum):
    """Standard sections of a notarial certificate"""
    HEADER = "header"  # Notary identification
    INTRODUCTION = "introduction"  # "CERTIFICO:"
    LEGAL_BASIS = "legal_basis"  # Articles referenced
    SUBJECT_IDENTIFICATION = "subject_identification"  # Who/what is certified
    DOCUMENT_SOURCES = "document_sources"  # Documents reviewed
    CERTIFICATIONS = "certifications"  # What is being certified
    SPECIAL_MENTIONS = "special_mentions"  # Institution-specific requirements
    DESTINATION = "destination"  # Purpose/destination
    CLOSING = "closing"  # Date, place, signature block


@dataclass
class CertificateSection:
    """
    Represents a single section of the certificate.
    """
    section_type: TemplateSection
    content: str
    legal_basis: Optional[str] = None
    required: bool = True
    order: int = 0

    def to_dict(self) -> dict:
        return {
            "section_type": self.section_type.value,
            "content": self.content,
            "legal_basis": self.legal_basis,
            "required": self.required,
            "order": self.order
        }


@dataclass
class GeneratedCertificate:
    """
    Complete generated certificate with all sections.
    """
    certificate_intent: CertificateIntent
    confirmation_report: FinalConfirmationReport

    # Certificate content
    sections: List[CertificateSection] = field(default_factory=list)
    full_text: str = ""

    # Metadata
    generation_timestamp: datetime = field(default_factory=datetime.now)
    template_version: str = "1.0"
    generated_by: str = "Sistema Automatizado"

    # Variables used in generation
    substitutions: Dict[str, str] = field(default_factory=dict)

    # Status
    is_draft: bool = True
    requires_notary_review: bool = True

    def to_dict(self) -> dict:
        return {
            "certificate_intent": self.certificate_intent.to_dict(),
            "sections": [s.to_dict() for s in self.sections],
            "full_text": self.full_text,
            "generation_timestamp": self.generation_timestamp.isoformat(),
            "template_version": self.template_version,
            "generated_by": self.generated_by,
            "substitutions": self.substitutions,
            "is_draft": self.is_draft,
            "requires_notary_review": self.requires_notary_review
        }

    def to_json(self) -> str:
        """Serialize to JSON"""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def get_formatted_text(self) -> str:
        """Get formatted certificate text"""
        if self.full_text:
            return self.full_text

        # Build from sections if full_text not set
        lines = []
        for section in sorted(self.sections, key=lambda s: s.order):
            lines.append(section.content)
            lines.append("")  # Blank line between sections

        return "\n".join(lines)

    def get_summary(self) -> str:
        """Get summary of generated certificate"""
        border = "=" * 70

        summary = f"""
{border}
           FASE 9: CERTIFICADO GENERADO (BORRADOR)
{border}

📋 TIPO: {self.certificate_intent.certificate_type.value}
🎯 PROPÓSITO: {self.certificate_intent.purpose.value}
👤 SUJETO: {self.certificate_intent.subject_name}

📊 CONTENIDO:
   Secciones: {len(self.sections)}
   Palabras: {len(self.full_text.split())}
   Caracteres: {len(self.full_text)}

⏰ Generado: {self.generation_timestamp.strftime('%Y-%m-%d %H:%M:%S')}
🤖 Por: {self.generated_by}
📦 Versión Template: {self.template_version}

⚠️  ESTADO: {'BORRADOR - Requiere revisión del notario' if self.is_draft else 'FINAL'}

{border}

"""
        return summary


class CertificateGenerator:
    """
    Main class for Phase 9: Certificate Generation
    """

    # Standard legal phrases
    LEGAL_PHRASES = {
        "opening": "CERTIFICO:",
        "legal_basis_intro": "Que conforme a lo dispuesto en",
        "document_review": "Que he tenido a la vista y revisado",
        "verification": "Que de acuerdo con la documentación examinada",
        "closing_standard": "Expido el presente certificado a solicitud de parte interesada",
    }

    @staticmethod
    def generate(
        certificate_intent: CertificateIntent,
        legal_requirements: LegalRequirements,
        extraction_result: CollectionExtractionResult,
        confirmation_report: FinalConfirmationReport,
        notary_name: Optional[str] = None,
        notary_office: Optional[str] = None
    ) -> GeneratedCertificate:
        """
        Generate certificate from validated data.

        Args:
            certificate_intent: From Phase 1
            legal_requirements: From Phase 2
            extraction_result: From Phase 4 (updated)
            confirmation_report: From Phase 8
            notary_name: Optional notary name
            notary_office: Optional notary office details

        Returns:
            GeneratedCertificate with all sections
        """
        print("\n" + "="*70)
        print("   FASE 9: GENERACIÓN DE CERTIFICADO")
        print("="*70 + "\n")

        # Verify Phase 8 approval
        if not confirmation_report.can_proceed_to_phase9():
            raise ValueError(
                f"No se puede generar certificado: "
                f"Fase 8 reportó decisión '{confirmation_report.certificate_decision.value}'. "
                f"Debe ser 'approved' o 'approved_with_warnings'."
            )

        certificate = GeneratedCertificate(
            certificate_intent=certificate_intent,
            confirmation_report=confirmation_report
        )

        print("✅ Fase 8 aprobó generación de certificado")
        print(f"   Tipo: {certificate_intent.certificate_type.value}")
        print(f"   Propósito: {certificate_intent.purpose.value}\n")

        # Prepare substitution variables
        print("🔄 Paso 1: Preparando variables...")
        certificate.substitutions = CertificateGenerator._prepare_substitutions(
            certificate_intent,
            legal_requirements,
            extraction_result,
            notary_name,
            notary_office
        )
        print(f"   ✓ {len(certificate.substitutions)} variables preparadas")

        # Generate sections
        print("\n📝 Paso 2: Generando secciones del certificado...")
        certificate.sections = CertificateGenerator._generate_sections(
            certificate_intent,
            legal_requirements,
            extraction_result,
            certificate.substitutions
        )
        print(f"   ✓ {len(certificate.sections)} secciones generadas")

        # Assemble full text
        print("\n🔧 Paso 3: Ensamblando texto completo...")
        certificate.full_text = CertificateGenerator._assemble_certificate_text(
            certificate.sections,
            certificate_intent.certificate_type
        )
        print(f"   ✓ Certificado ensamblado ({len(certificate.full_text)} caracteres)")

        print("\n✅ Certificado generado exitosamente (BORRADOR)")
        print("   Requiere revisión del notario antes de firma\n")

        return certificate

    @staticmethod
    def _prepare_substitutions(
        intent: CertificateIntent,
        requirements: LegalRequirements,
        extraction: CollectionExtractionResult,
        notary_name: Optional[str],
        notary_office: Optional[str]
    ) -> Dict[str, str]:
        """Prepare all variable substitutions for certificate"""
        subs = {}

        # Notary information
        subs["{{NOTARY_NAME}}"] = notary_name or "[NOMBRE DEL NOTARIO]"
        subs["{{NOTARY_OFFICE}}"] = notary_office or "[ESCRIBANÍA]"

        # Date and place
        today = datetime.now()
        subs["{{DATE}}"] = today.strftime("%d de %B de %Y")
        subs["{{PLACE}}"] = "Montevideo"  # Default, should be configurable

        # Subject information
        subs["{{SUBJECT_NAME}}"] = intent.subject_name
        subs["{{SUBJECT_TYPE}}"] = "persona física" if intent.subject_type == "person" else "sociedad"

        # Purpose/destination
        subs["{{PURPOSE}}"] = intent.purpose.value.replace("_", " ").title()
        subs["{{DESTINATION}}"] = CertificateGenerator._format_destination(intent.purpose)

        # Extract data from extraction result
        if extraction.extracted_data:
            data = extraction.extracted_data
            subs["{{COMPANY_NAME}}"] = data.company_name or intent.subject_name
            subs["{{RUT}}"] = data.rut or "[RUT]"
            subs["{{REGISTRY_NUMBER}}"] = data.registro_comercio or "[N° REGISTRO]"
            subs["{{REPRESENTATIVE}}"] = "[REPRESENTANTE]"  # Would need to be extracted from acta
            subs["{{CI}}"] = data.ci or "[CI]"
        else:
            subs["{{COMPANY_NAME}}"] = intent.subject_name
            subs["{{RUT}}"] = "[RUT]"
            subs["{{REGISTRY_NUMBER}}"] = "[N° REGISTRO]"
            subs["{{REPRESENTATIVE}}"] = "[REPRESENTANTE]"
            subs["{{CI}}"] = "[CI]"

        # Legal articles
        articles = [art.value for art in requirements.mandatory_articles]
        subs["{{ARTICLES}}"] = CertificateGenerator._format_articles(articles)

        # Institution-specific
        if requirements.institution_rules:
            subs["{{INSTITUTION}}"] = requirements.institution_rules.institution
        else:
            subs["{{INSTITUTION}}"] = ""

        return subs

    @staticmethod
    def _format_destination(purpose: Purpose) -> str:
        """Format destination text based on purpose"""
        destination_map = {
            Purpose.BPS: "el Banco de Previsión Social (BPS)",
            Purpose.ABITAB: "ABITAB",
            Purpose.DGI: "la Dirección General Impositiva (DGI)",
            Purpose.BANCO: "entidad bancaria",
            Purpose.ZONA_FRANCA: "Zona Franca",
            Purpose.MTOP: "el Ministerio de Transporte y Obras Públicas (MTOP)",
            Purpose.RUPE: "el Registro Único de Proveedores del Estado (RUPE)",
            Purpose.MSP: "el Ministerio de Salud Pública (MSP)",
            Purpose.ANTEL: "ANTEL",
            Purpose.UTE: "UTE",
            Purpose.IMM: "la Intendencia Municipal de Montevideo",
            Purpose.MEF: "el Ministerio de Economía y Finanzas",
            Purpose.BASE_DATOS: "base de datos",
            Purpose.MIGRACIONES: "Dirección Nacional de Migración",
            Purpose.COMPRAVENTA: "compraventa",
            Purpose.OTROS: "los fines que corresponda"
        }
        return destination_map.get(purpose, "los fines solicitados")

    @staticmethod
    def _format_articles(articles: List[str]) -> str:
        """Format article list for legal basis"""
        if not articles:
            return "la normativa vigente"

        if len(articles) == 1:
            return f"el artículo {articles[0]} del Reglamento Notarial"

        articles_str = ", ".join(articles[:-1]) + f" y {articles[-1]}"
        return f"los artículos {articles_str} del Reglamento Notarial"

    @staticmethod
    def _generate_sections(
        intent: CertificateIntent,
        requirements: LegalRequirements,
        extraction: CollectionExtractionResult,
        substitutions: Dict[str, str]
    ) -> List[CertificateSection]:
        """Generate all certificate sections"""
        sections = []

        # Section 1: Header
        sections.append(CertificateSection(
            section_type=TemplateSection.HEADER,
            content=CertificateGenerator._generate_header(substitutions),
            order=1
        ))

        # Section 2: Introduction
        sections.append(CertificateSection(
            section_type=TemplateSection.INTRODUCTION,
            content=CertificateGenerator.LEGAL_PHRASES["opening"],
            legal_basis="Art. 248",
            order=2
        ))

        # Section 3: Legal Basis
        sections.append(CertificateSection(
            section_type=TemplateSection.LEGAL_BASIS,
            content=CertificateGenerator._generate_legal_basis(requirements, substitutions),
            legal_basis="Art. 248, 255",
            order=3
        ))

        # Section 4: Subject Identification
        sections.append(CertificateSection(
            section_type=TemplateSection.SUBJECT_IDENTIFICATION,
            content=CertificateGenerator._generate_subject_identification(intent, substitutions),
            legal_basis="Art. 130, 248",
            order=4
        ))

        # Section 5: Document Sources
        sections.append(CertificateSection(
            section_type=TemplateSection.DOCUMENT_SOURCES,
            content=CertificateGenerator._generate_document_sources(extraction, substitutions),
            legal_basis="Art. 249",
            order=5
        ))

        # Section 6: Certifications (main content)
        sections.append(CertificateSection(
            section_type=TemplateSection.CERTIFICATIONS,
            content=CertificateGenerator._generate_certifications(
                intent.certificate_type,
                requirements,
                extraction,
                substitutions
            ),
            legal_basis="Art. 250-254",
            order=6
        ))

        # Section 7: Special Mentions (if institution-specific)
        if requirements.institution_rules:
            sections.append(CertificateSection(
                section_type=TemplateSection.SPECIAL_MENTIONS,
                content=CertificateGenerator._generate_special_mentions(requirements, substitutions),
                legal_basis="Art. 254",
                order=7,
                required=True
            ))

        # Section 8: Destination
        sections.append(CertificateSection(
            section_type=TemplateSection.DESTINATION,
            content=CertificateGenerator._generate_destination(intent, substitutions),
            legal_basis="Art. 255",
            order=8
        ))

        # Section 9: Closing
        sections.append(CertificateSection(
            section_type=TemplateSection.CLOSING,
            content=CertificateGenerator._generate_closing(substitutions),
            legal_basis="Art. 253, 255",
            order=9
        ))

        return sections

    @staticmethod
    def _generate_header(subs: Dict[str, str]) -> str:
        """Generate certificate header"""
        return f"""{subs["{{NOTARY_NAME}}"]}
{subs["{{NOTARY_OFFICE}}"]}"""

    @staticmethod
    def _generate_legal_basis(requirements: LegalRequirements, subs: Dict[str, str]) -> str:
        """Generate legal basis section"""
        return f"{CertificateGenerator.LEGAL_PHRASES['legal_basis_intro']} {subs['{{ARTICLES}}']}:"

    @staticmethod
    def _generate_subject_identification(intent: CertificateIntent, subs: Dict[str, str]) -> str:
        """Generate subject identification section"""
        if intent.subject_type == "company":
            company_name = subs['{{COMPANY_NAME}}']
            registry_num = subs['{{REGISTRY_NUMBER}}']
            rut = subs['{{RUT}}']
            return (
                f"Que {company_name} es una sociedad comercial inscripta "
                f"en el Registro de Comercio bajo el número {registry_num}, "
                f"con RUT número {rut}."
            )
        else:
            subject_name = subs['{{SUBJECT_NAME}}']
            ci = subs['{{CI}}']
            return (
                f"Que {subject_name}, titular de la cédula de identidad "
                f"número {ci}."
            )

    @staticmethod
    def _generate_document_sources(extraction: CollectionExtractionResult, subs: Dict[str, str]) -> str:
        """Generate document sources section"""
        docs = extraction.extraction_results

        if not docs:
            return f"{CertificateGenerator.LEGAL_PHRASES['document_review']} la documentación presentada."

        doc_list = []
        for doc in docs[:5]:  # List up to 5 documents
            doc_type = doc.extracted_data.document_type.value.replace("_", " ").title()
            doc_list.append(f"- {doc_type}")

        docs_text = "\n".join(doc_list)

        if len(docs) > 5:
            docs_text += f"\n- Y {len(docs) - 5} documento(s) adicional(es)"

        return f"{CertificateGenerator.LEGAL_PHRASES['document_review']}:\n\n{docs_text}"

    @staticmethod
    def _generate_certifications(
        cert_type: CertificateType,
        requirements: LegalRequirements,
        extraction: CollectionExtractionResult,
        subs: Dict[str, str]
    ) -> str:
        """Generate main certification content based on certificate type"""

        if cert_type == CertificateType.CERTIFICADO_PERSONERIA:
            return CertificateGenerator._generate_personeria_certification(subs, extraction)
        elif cert_type == CertificateType.CERTIFICADO_REPRESENTACION:
            return CertificateGenerator._generate_representacion_certification(subs, extraction)
        elif cert_type == CertificateType.CERTIFICACION_FIRMAS:
            return CertificateGenerator._generate_firmas_certification(subs, extraction)
        elif cert_type == CertificateType.CERTIFICADO_VIGENCIA:
            return CertificateGenerator._generate_vigencia_certification(subs, extraction)
        else:
            return CertificateGenerator._generate_generic_certification(cert_type, subs)

    @staticmethod
    def _generate_personeria_certification(subs: Dict[str, str], extraction: CollectionExtractionResult) -> str:
        """Generate certification for personería (legal personality)"""
        company = subs['{{COMPANY_NAME}}']
        representative = subs['{{REPRESENTATIVE}}']
        ci = subs['{{CI}}']

        return f"""Que {company} es una sociedad legalmente constituida e inscripta en el Registro Nacional de Comercio, con domicilio en la República Oriental del Uruguay.

Que de acuerdo con la documentación examinada, la sociedad se encuentra debidamente constituida y sus autoridades designadas conforme a derecho.

Que {representative}, titular de la cédula de identidad número {ci}, se encuentra facultado para actuar en representación de {company} de acuerdo con las facultades conferidas por la Asamblea de Accionistas y el Directorio de la sociedad."""

    @staticmethod
    def _generate_representacion_certification(subs: Dict[str, str], extraction: CollectionExtractionResult) -> str:
        """Generate certification for representation"""
        representative = subs['{{REPRESENTATIVE}}']
        ci = subs['{{CI}}']
        company = subs['{{COMPANY_NAME}}']
        rut = subs['{{RUT}}']

        return f"""Que {representative}, titular de la cédula de identidad número {ci}, actúa en representación de {company} con RUT número {rut}.

Que dicha representación surge de la documentación examinada y se encuentra vigente y con plenos poderes para los actos que requieran de su intervención."""

    @staticmethod
    def _generate_firmas_certification(subs: Dict[str, str], extraction: CollectionExtractionResult) -> str:
        """Generate certification for signature authentication"""
        representative = subs['{{REPRESENTATIVE}}']
        ci = subs['{{CI}}']
        company = subs['{{COMPANY_NAME}}']

        return f"""Que la firma inserta en el documento presentado pertenece a {representative}, titular de la cédula de identidad número {ci}, quien la estampó en mi presencia, dándola por reconocida.

Que el firmante se identifica con el documento de identidad mencionado y actúa en representación de {company}."""

    @staticmethod
    def _generate_vigencia_certification(subs: Dict[str, str], extraction: CollectionExtractionResult) -> str:
        """Generate certification for validity/current status"""
        company = subs['{{COMPANY_NAME}}']

        return f"""Que {company} se encuentra debidamente constituida e inscripta en el Registro Nacional de Comercio.

Que de acuerdo con la documentación examinada, la sociedad se encuentra vigente y en pleno funcionamiento, sin que conste ninguna causal de disolución o liquidación."""

    @staticmethod
    def _generate_generic_certification(cert_type: CertificateType, subs: Dict[str, str]) -> str:
        """Generate generic certification for other types"""
        subject = subs['{{SUBJECT_NAME}}']

        return f"""Que de acuerdo con la documentación examinada y que ha sido tenida a la vista, se verifica la información solicitada respecto de {subject}.

Que todos los documentos presentados se encuentran en debida forma y conforme a las disposiciones legales vigentes."""

    @staticmethod
    def _generate_special_mentions(requirements: LegalRequirements, subs: Dict[str, str]) -> str:
        """Generate special mentions for institution-specific requirements"""
        if not requirements.institution_rules:
            return ""

        institution = requirements.institution_rules.institution
        special_reqs = requirements.institution_rules.special_requirements

        mentions = [f"Para {subs['{{DESTINATION}}']} se deja constancia de lo siguiente:"]

        for req in special_reqs:
            mentions.append(f"- {req}")

        return "\n".join(mentions)

    @staticmethod
    def _generate_destination(intent: CertificateIntent, subs: Dict[str, str]) -> str:
        """Generate destination section"""
        return f"""Expido el presente certificado a solicitud de parte interesada, para ser presentado ante {subs['{{DESTINATION}}']}."""

    @staticmethod
    def _generate_closing(subs: Dict[str, str]) -> str:
        """Generate closing section"""
        return f"""Lugar y fecha: {subs['{{PLACE}}']}, {subs['{{DATE}}']}.


_______________________________
{subs['{{NOTARY_NAME}}']}
Escribano Público"""

    @staticmethod
    def _assemble_certificate_text(sections: List[CertificateSection], cert_type: CertificateType) -> str:
        """Assemble all sections into final certificate text"""
        lines = []

        for section in sorted(sections, key=lambda s: s.order):
            lines.append(section.content)

            # Add appropriate spacing between sections
            if section.section_type in [TemplateSection.HEADER, TemplateSection.INTRODUCTION]:
                lines.append("")  # Single blank line
            elif section.section_type == TemplateSection.CLOSING:
                lines.append("")  # Single blank line before closing
            else:
                lines.append("")  # Single blank line between sections

        return "\n".join(lines)

    @staticmethod
    def _apply_substitutions(text: str, substitutions: Dict[str, str]) -> str:
        """Apply variable substitutions to text"""
        result = text
        for key, value in substitutions.items():
            result = result.replace(key, value)
        return result

    @staticmethod
    def export_certificate(
        certificate: GeneratedCertificate,
        output_path: str,
        format: CertificateFormat = CertificateFormat.PLAIN_TEXT
    ) -> None:
        """Export certificate to file"""
        if format == CertificateFormat.PLAIN_TEXT:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(certificate.get_formatted_text())
            print(f"\n✅ Certificado exportado (texto plano): {output_path}")

        elif format == CertificateFormat.STRUCTURED_JSON:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(certificate.to_json())
            print(f"\n✅ Certificado exportado (JSON): {output_path}")

        elif format == CertificateFormat.HTML:
            html_content = CertificateGenerator._generate_html(certificate)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            print(f"\n✅ Certificado exportado (HTML): {output_path}")

        else:
            raise ValueError(f"Formato no soportado: {format}")

    @staticmethod
    def _generate_html(certificate: GeneratedCertificate) -> str:
        """Generate HTML version of certificate"""
        text = certificate.get_formatted_text()

        html = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Certificado Notarial - {certificate.certificate_intent.certificate_type.value}</title>
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
        }}
        .draft-watermark {{
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%) rotate(-45deg);
            font-size: 120px;
            color: rgba(255, 0, 0, 0.1);
            font-weight: bold;
            z-index: -1;
        }}
        pre {{
            white-space: pre-wrap;
            font-family: 'Times New Roman', serif;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="draft-watermark">BORRADOR</div>
    <div class="certificate">
        <pre>{text}</pre>
    </div>
</body>
</html>"""
        return html


def example_usage():
    """Example usage of Phase 9"""

    print("\n" + "="*70)
    print("  EJEMPLOS DE USO - FASE 9: GENERACIÓN DE CERTIFICADO")
    print("="*70)

    print("\n📌 Ejemplo 1: Generar certificado completo")
    print("-" * 70)
    print("""
from src.phase9_certificate_generation import CertificateGenerator, CertificateFormat

# Asumiendo que tienes todos los inputs de fases anteriores:
# - certificate_intent (Fase 1)
# - legal_requirements (Fase 2)
# - extraction_result (Fase 4, actualizado en Fase 7)
# - confirmation_report (Fase 8)

# Generar certificado
certificate = CertificateGenerator.generate(
    certificate_intent=certificate_intent,
    legal_requirements=legal_requirements,
    extraction_result=extraction_result,
    confirmation_report=confirmation_report,
    notary_name="Dr. Juan Pérez",
    notary_office="Escribanía Juan Pérez - Montevideo"
)

# Ver resumen
print(certificate.get_summary())

# Ver certificado completo
print(certificate.get_formatted_text())
    """)

    print("\n📌 Ejemplo 2: Exportar certificado")
    print("-" * 70)
    print("""
# Exportar como texto plano
CertificateGenerator.export_certificate(
    certificate,
    "certificado_borrador.txt",
    format=CertificateFormat.PLAIN_TEXT
)

# Exportar como JSON estructurado
CertificateGenerator.export_certificate(
    certificate,
    "certificado_borrador.json",
    format=CertificateFormat.STRUCTURED_JSON
)

# Exportar como HTML
CertificateGenerator.export_certificate(
    certificate,
    "certificado_borrador.html",
    format=CertificateFormat.HTML
)
    """)

    print("\n📌 Ejemplo 3: Flujo completo (Fases 1-9)")
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
from src.phase9_certificate_generation import CertificateGenerator

# Fases 1-8 (preparación y validación)
intent = CertificateIntentCapture.capture_intent_from_params(...)
requirements = LegalRequirementsEngine.resolve_requirements(intent)
collection = DocumentIntake.scan_directory_for_client(...)
extraction = TextExtractor.process_collection(collection)
validation = LegalValidator.validate(requirements, extraction)
gap_report = GapDetector.analyze(validation)

# Fase 7: Si hay brechas, actualizar
if not gap_report.ready_for_certificate:
    update_result = DataUpdater.create_update_session(gap_report, collection)
    # ... cargar documentos ...
    update_result = DataUpdater.re_extract_data(update_result)
else:
    update_result = DataUpdater.create_update_session(gap_report, collection)
    update_result.updated_extraction_result = extraction

# Fase 8: Confirmación final
confirmation = FinalConfirmationEngine.confirm(requirements, update_result)

# Fase 9: Generar certificado (solo si aprobado)
if confirmation.can_proceed_to_phase9():
    certificate = CertificateGenerator.generate(
        intent,
        requirements,
        update_result.updated_extraction_result,
        confirmation,
        notary_name="Dr. Juan Pérez",
        notary_office="Escribanía Juan Pérez"
    )

    print(certificate.get_summary())
    print("\\n" + "="*70)
    print(certificate.get_formatted_text())

    # Exportar
    CertificateGenerator.export_certificate(
        certificate,
        "certificado_final.txt"
    )

    print("\\n✅ CERTIFICADO GENERADO - Listo para revisión del notario")
else:
    print("❌ No se puede generar certificado")
    print(confirmation.get_summary())
    """)


if __name__ == "__main__":
    example_usage()
