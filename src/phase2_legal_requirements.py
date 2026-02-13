"""
Phase 2: Legal Requirement Resolution (Rules Engine)

This module handles the legal validation rules by:
- Mapping certificate types to applicable Uruguayan laws (Articles 248-255)
- Defining required documents per certificate type
- Handling institution-specific requirements
- Creating structured validation checklists

This is a RULE-DRIVEN system, not guessing - it follows explicit legal requirements.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from enum import Enum
import json

from src.phase1_certificate_intent import CertificateType, Purpose, CertificateIntent


class ArticleReference(Enum):
    """Legal articles from Uruguayan Notarial Regulations"""
    ART_130 = "130"  # Identification rules
    ART_248 = "248"  # General certificate requirements
    ART_249 = "249"  # Document source requirements
    ART_250 = "250"  # Signature certification
    ART_251 = "251"  # Signature presence
    ART_252 = "252"  # Certification content
    ART_253 = "253"  # Certificate format
    ART_254 = "254"  # Special mentions
    ART_255 = "255"  # Required elements (destination, date, etc.)


class RequiredElement(Enum):
    """Required elements for certificate validation"""
    IDENTITY_VERIFICATION = "identity_verification"
    DOCUMENT_SOURCE = "document_source"
    SIGNATURE_PRESENCE = "signature_presence"
    DESTINATION_ENTITY = "destination_entity"
    VALIDITY_DATES = "validity_dates"
    REGISTRY_INSCRIPTION = "registry_inscription"
    COMPANY_NAME = "company_name"
    RUT_NUMBER = "rut_number"
    LEGAL_REPRESENTATIVE = "legal_representative"
    POWER_OF_ATTORNEY = "power_of_attorney"
    BOARD_MINUTES = "board_minutes"
    COMPANY_STATUTE = "company_statute"
    CERTIFICATE_FRESHNESS = "certificate_freshness"
    TAX_COMPLIANCE = "tax_compliance"
    SOCIAL_SECURITY_STATUS = "social_security_status"


class DocumentType(Enum):
    """Types of documents required for validation"""
    CEDULA_IDENTIDAD = "cedula_identidad"
    ESTATUTO = "estatuto"
    ACTA_DIRECTORIO = "acta_directorio"
    CERTIFICADO_BPS = "certificado_bps"
    CERTIFICADO_DGI = "certificado_dgi"
    PODER = "poder"
    REGISTRO_COMERCIO = "registro_comercio"
    PADRON_BPS = "padron_bps"
    CERTIFICADO_VIGENCIA = "certificado_vigencia"
    CONTRATO_SOCIAL = "contrato_social"
    BALANCE = "balance"
    DECLARACION_JURADA = "declaracion_jurada"


@dataclass
class DocumentRequirement:
    """Represents a required document for a certificate"""
    document_type: DocumentType
    description: str
    mandatory: bool = True
    expires: bool = False
    expiry_days: Optional[int] = None
    legal_basis: Optional[str] = None  # Which article requires this
    institution_specific: Optional[str] = None  # e.g., "BPS", "Abitab"

    def to_dict(self) -> dict:
        return {
            "document_type": self.document_type.value,
            "description": self.description,
            "mandatory": self.mandatory,
            "expires": self.expires,
            "expiry_days": self.expiry_days,
            "legal_basis": self.legal_basis,
            "institution_specific": self.institution_specific
        }


@dataclass
class InstitutionRule:
    """Institution-specific rules and requirements"""
    institution: str  # "BPS", "Abitab", "MSP", etc.
    validity_days: Optional[int] = None  # e.g., 30 days for Abitab
    additional_documents: List[DocumentRequirement] = field(default_factory=list)
    special_requirements: List[str] = field(default_factory=list)
    format_rules: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "institution": self.institution,
            "validity_days": self.validity_days,
            "additional_documents": [doc.to_dict() for doc in self.additional_documents],
            "special_requirements": self.special_requirements,
            "format_rules": self.format_rules
        }


@dataclass
class LegalRequirements:
    """
    Complete legal requirements for a specific certificate type.
    This is the output of Phase 2 - a structured checklist.
    """
    certificate_type: CertificateType
    purpose: Purpose
    mandatory_articles: List[ArticleReference]
    cross_references: List[ArticleReference]
    required_elements: List[RequiredElement]
    required_documents: List[DocumentRequirement]
    institution_rules: Optional[InstitutionRule] = None
    validation_rules: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "certificate_type": self.certificate_type.value,
            "purpose": self.purpose.value,
            "mandatory_articles": [art.value for art in self.mandatory_articles],
            "cross_references": [art.value for art in self.cross_references],
            "required_elements": [elem.value for elem in self.required_elements],
            "required_documents": [doc.to_dict() for doc in self.required_documents],
            "institution_rules": self.institution_rules.to_dict() if self.institution_rules else None,
            "validation_rules": self.validation_rules
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def get_summary(self) -> str:
        """Get human-readable summary in Spanish"""
        summary = f"""
╔══════════════════════════════════════════════════════════════╗
║              REQUISITOS LEGALES - FASE 2                     ║
╚══════════════════════════════════════════════════════════════╝

📋 Tipo de Certificado: {self.certificate_type.value.replace('_', ' ').title()}
🎯 Propósito: {self.purpose.value.replace('para_', 'Para ').replace('_', ' ').title()}

📚 ARTÍCULOS APLICABLES:
   Obligatorios: {', '.join([art.value for art in self.mandatory_articles])}
   Referencias cruzadas: {', '.join([art.value for art in self.cross_references])}

📄 DOCUMENTOS REQUERIDOS ({len(self.required_documents)} total):
"""
        for doc in self.required_documents:
            status = "✓ OBLIGATORIO" if doc.mandatory else "○ OPCIONAL"
            expiry = f" [Vence en {doc.expiry_days} días]" if doc.expires else ""
            institution = f" ({doc.institution_specific})" if doc.institution_specific else ""
            summary += f"   {status}: {doc.description}{expiry}{institution}\n"

        if self.institution_rules:
            summary += f"\n🏛️ REGLAS INSTITUCIONALES ({self.institution_rules.institution}):\n"
            if self.institution_rules.validity_days:
                summary += f"   - Validez del certificado: {self.institution_rules.validity_days} días\n"
            for req in self.institution_rules.special_requirements:
                summary += f"   - {req}\n"

        return summary


class LegalRequirementsEngine:
    """
    Rules engine that maps certificate types and purposes to legal requirements.
    This is the core of Phase 2.
    """

    # Base articles required for ALL certificates
    BASE_ARTICLES = [
        ArticleReference.ART_248,
        ArticleReference.ART_249,
        ArticleReference.ART_255
    ]

    @staticmethod
    def _get_base_requirements() -> List[RequiredElement]:
        """Elements required for all certificates"""
        return [
            RequiredElement.IDENTITY_VERIFICATION,
            RequiredElement.DOCUMENT_SOURCE,
            RequiredElement.DESTINATION_ENTITY,
            RequiredElement.VALIDITY_DATES
        ]

    @staticmethod
    def _get_personeria_documents() -> List[DocumentRequirement]:
        """Documents required for personería (legal personality) certificates"""
        return [
            DocumentRequirement(
                document_type=DocumentType.CEDULA_IDENTIDAD,
                description="Cédula de identidad del representante legal",
                mandatory=True,
                expires=False,
                legal_basis="Art. 130"
            ),
            DocumentRequirement(
                document_type=DocumentType.ESTATUTO,
                description="Estatuto social de la empresa",
                mandatory=True,
                expires=False,
                legal_basis="Art. 248"
            ),
            DocumentRequirement(
                document_type=DocumentType.REGISTRO_COMERCIO,
                description="Inscripción en Registro de Comercio",
                mandatory=True,
                expires=False,
                legal_basis="Art. 249"
            ),
            DocumentRequirement(
                document_type=DocumentType.ACTA_DIRECTORIO,
                description="Acta de Directorio designando representantes",
                mandatory=True,
                expires=False,
                legal_basis="Art. 248"
            ),
            DocumentRequirement(
                document_type=DocumentType.CERTIFICADO_DGI,
                description="Certificado de situación tributaria (DGI)",
                mandatory=True,
                expires=True,
                expiry_days=90,
                legal_basis="Ley 17904"
            )
        ]

    @staticmethod
    def _get_firma_documents() -> List[DocumentRequirement]:
        """Documents required for signature certification"""
        return [
            DocumentRequirement(
                document_type=DocumentType.CEDULA_IDENTIDAD,
                description="Cédula de identidad del firmante",
                mandatory=True,
                expires=False,
                legal_basis="Art. 130"
            ),
            DocumentRequirement(
                document_type=DocumentType.PODER,
                description="Poder si actúa en representación",
                mandatory=False,
                expires=False,
                legal_basis="Art. 250"
            )
        ]

    @staticmethod
    def _get_poder_documents() -> List[DocumentRequirement]:
        """Documents required for power of attorney"""
        return [
            DocumentRequirement(
                document_type=DocumentType.CEDULA_IDENTIDAD,
                description="Cédula de identidad del otorgante",
                mandatory=True,
                expires=False,
                legal_basis="Art. 130"
            ),
            DocumentRequirement(
                document_type=DocumentType.ESTATUTO,
                description="Estatuto social (si es empresa)",
                mandatory=True,
                expires=False,
                legal_basis="Art. 248"
            ),
            DocumentRequirement(
                document_type=DocumentType.ACTA_DIRECTORIO,
                description="Acta que autoriza otorgar poder",
                mandatory=True,
                expires=False,
                legal_basis="Art. 248"
            )
        ]

    @staticmethod
    def _get_institution_rules(purpose: Purpose) -> Optional[InstitutionRule]:
        """Get institution-specific rules based on purpose"""
        institution_rules_map = {
            Purpose.BPS: InstitutionRule(
                institution="BPS",
                validity_days=30,
                additional_documents=[
                    DocumentRequirement(
                        document_type=DocumentType.CERTIFICADO_BPS,
                        description="Certificado de situación de BPS",
                        mandatory=True,
                        expires=True,
                        expiry_days=30,
                        institution_specific="BPS"
                    ),
                    DocumentRequirement(
                        document_type=DocumentType.PADRON_BPS,
                        description="Padrón de funcionarios BPS",
                        mandatory=True,
                        expires=True,
                        expiry_days=30,
                        institution_specific="BPS"
                    )
                ],
                special_requirements=[
                    "Debe incluir situación de aportes al día",
                    "Debe mencionar número de patrón BPS"
                ]
            ),
            Purpose.ABITAB: InstitutionRule(
                institution="Abitab",
                validity_days=30,
                additional_documents=[],
                special_requirements=[
                    "Certificado debe tener vigencia de 30 días",
                    "Debe incluir representación legal completa"
                ]
            ),
            Purpose.ZONA_FRANCA: InstitutionRule(
                institution="Zona Franca",
                additional_documents=[
                    DocumentRequirement(
                        document_type=DocumentType.CERTIFICADO_VIGENCIA,
                        description="Certificado de vigencia de Zona Franca",
                        mandatory=True,
                        expires=True,
                        expiry_days=90,
                        institution_specific="Zona Franca"
                    )
                ],
                special_requirements=[
                    "Debe mencionar domicilio en Zona Franca",
                    "Incluir autorización de Zona Franca"
                ]
            ),
            Purpose.MTOP: InstitutionRule(
                institution="MTOP",
                additional_documents=[],
                special_requirements=[
                    "Formato específico MTOP",
                    "Incluir objeto social relacionado con transporte"
                ]
            ),
            Purpose.DGI: InstitutionRule(
                institution="DGI",
                additional_documents=[
                    DocumentRequirement(
                        document_type=DocumentType.CERTIFICADO_DGI,
                        description="Certificado único DGI",
                        mandatory=True,
                        expires=True,
                        expiry_days=90,
                        institution_specific="DGI"
                    )
                ],
                special_requirements=[
                    "Debe incluir RUT",
                    "Situación tributaria al día"
                ]
            ),
            Purpose.RUPE: InstitutionRule(
                institution="RUPE",
                validity_days=180,
                additional_documents=[],
                special_requirements=[
                    "Certificado válido por 180 días",
                    "Incluir Ley 18930 (protección datos personales)",
                    "Incluir Ley 17904 (prevención lavado de activos)"
                ],
                format_rules={
                    "include_law_18930": "true",
                    "include_law_17904": "true"
                }
            ),
            Purpose.MIGRACIONES: InstitutionRule(
                institution="Migraciones",
                additional_documents=[],
                special_requirements=[
                    "Incluir domicilio fiscal",
                    "Incluir domicilio constituido"
                ]
            ),
            Purpose.BASE_DATOS: InstitutionRule(
                institution="Base de Datos",
                additional_documents=[],
                special_requirements=[
                    "Debe incluir Ley 18930 (protección de datos personales)",
                    "Mencionar responsable de base de datos"
                ],
                format_rules={
                    "include_law_18930": "true"
                }
            )
        }

        return institution_rules_map.get(purpose)

    @staticmethod
    def resolve_requirements(intent: CertificateIntent) -> LegalRequirements:
        """
        Main method: Resolve all legal requirements for a given certificate intent.

        This is where the rules engine decides what laws apply and what documents are needed.
        """
        # Start with base articles
        mandatory_articles = list(LegalRequirementsEngine.BASE_ARTICLES)
        cross_references = []
        required_elements = list(LegalRequirementsEngine._get_base_requirements())
        required_documents = []

        # Add certificate-type specific requirements
        if intent.certificate_type == CertificateType.CERTIFICADO_PERSONERIA:
            mandatory_articles.append(ArticleReference.ART_252)
            required_elements.extend([
                RequiredElement.COMPANY_NAME,
                RequiredElement.REGISTRY_INSCRIPTION,
                RequiredElement.LEGAL_REPRESENTATIVE,
                RequiredElement.RUT_NUMBER
            ])
            required_documents.extend(LegalRequirementsEngine._get_personeria_documents())

        elif intent.certificate_type == CertificateType.CERTIFICACION_FIRMAS:
            mandatory_articles.append(ArticleReference.ART_250)
            mandatory_articles.append(ArticleReference.ART_251)
            cross_references.append(ArticleReference.ART_130)
            required_elements.append(RequiredElement.SIGNATURE_PRESENCE)
            required_documents.extend(LegalRequirementsEngine._get_firma_documents())

        elif intent.certificate_type in [CertificateType.PODER_GENERAL, CertificateType.CARTA_PODER]:
            mandatory_articles.append(ArticleReference.ART_252)
            cross_references.append(ArticleReference.ART_130)
            required_elements.extend([
                RequiredElement.POWER_OF_ATTORNEY,
                RequiredElement.LEGAL_REPRESENTATIVE
            ])
            required_documents.extend(LegalRequirementsEngine._get_poder_documents())

        elif intent.certificate_type == CertificateType.CERTIFICADO_REPRESENTACION:
            mandatory_articles.append(ArticleReference.ART_252)
            required_elements.extend([
                RequiredElement.LEGAL_REPRESENTATIVE,
                RequiredElement.BOARD_MINUTES
            ])
            required_documents.extend(LegalRequirementsEngine._get_personeria_documents())

        # Get institution-specific rules
        institution_rules = LegalRequirementsEngine._get_institution_rules(intent.purpose)

        # Add institution-specific documents
        if institution_rules and institution_rules.additional_documents:
            required_documents.extend(institution_rules.additional_documents)

        # Add certificate freshness requirement
        if institution_rules and institution_rules.validity_days:
            required_elements.append(RequiredElement.CERTIFICATE_FRESHNESS)

        # Validation rules
        validation_rules = {
            "check_identity": "Verificar identidad según Art. 130",
            "check_source": "Verificar fuente documental según Art. 249",
            "check_destination": "Verificar destinatario según Art. 255",
            "check_dates": "Verificar vigencia de documentos"
        }

        if institution_rules:
            validation_rules["check_institution"] = f"Verificar requisitos específicos de {institution_rules.institution}"

        return LegalRequirements(
            certificate_type=intent.certificate_type,
            purpose=intent.purpose,
            mandatory_articles=mandatory_articles,
            cross_references=cross_references,
            required_elements=required_elements,
            required_documents=required_documents,
            institution_rules=institution_rules,
            validation_rules=validation_rules
        )

    @staticmethod
    def get_all_applicable_articles(requirements: LegalRequirements) -> Set[str]:
        """Get all applicable article numbers (mandatory + cross-references)"""
        all_articles = set()
        all_articles.update([art.value for art in requirements.mandatory_articles])
        all_articles.update([art.value for art in requirements.cross_references])
        return all_articles


def example_usage():
    """Example usage of Phase 2"""

    print("\n" + "="*70)
    print("  EJEMPLOS DE USO - FASE 2: REQUISITOS LEGALES")
    print("="*70)

    from src.phase1_certificate_intent import CertificateIntentCapture

    # Example 1: GIRTEC BPS Certificate
    print("\n📌 Ejemplo 1: Certificado de Personería para BPS (GIRTEC S.A.)")
    print("-" * 70)

    intent1 = CertificateIntentCapture.capture_intent_from_params(
        certificate_type="certificado_de_personeria",
        purpose="BPS",
        subject_name="GIRTEC S.A.",
        subject_type="company"
    )

    requirements1 = LegalRequirementsEngine.resolve_requirements(intent1)
    print(requirements1.get_summary())

    # Example 2: NETKLA Zona Franca
    print("\n\n📌 Ejemplo 2: Certificado de Personería para Zona Franca (NETKLA)")
    print("-" * 70)

    intent2 = CertificateIntentCapture.capture_intent_from_params(
        certificate_type="certificado_de_personeria",
        purpose="zona franca",
        subject_name="NETKLA TRADING S.A.",
        subject_type="company"
    )

    requirements2 = LegalRequirementsEngine.resolve_requirements(intent2)
    print(requirements2.get_summary())

    # Example 3: Signature certification for Abitab
    print("\n\n📌 Ejemplo 3: Certificación de Firmas para Abitab")
    print("-" * 70)

    intent3 = CertificateIntentCapture.capture_intent_from_params(
        certificate_type="certificacion_de_firmas",
        purpose="Abitab",
        subject_name="INVERSORA RINLEN S.A.",
        subject_type="company"
    )

    requirements3 = LegalRequirementsEngine.resolve_requirements(intent3)
    print(requirements3.get_summary())

    # Example 4: JSON output
    print("\n\n📌 Ejemplo 4: Salida JSON para integración")
    print("-" * 70)
    print(requirements3.to_json())


if __name__ == "__main__":
    example_usage()
