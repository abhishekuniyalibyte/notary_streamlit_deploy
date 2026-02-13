"""
Phase 1: Certificate Intent Definition

This module handles the initial step where the notary defines:
- Certificate type
- Purpose/destination
- Subject (person or company)

This triggers the entire legal validation pipeline.
"""

from dataclasses import dataclass
from typing import Optional, List
from enum import Enum
import json


class CertificateType(Enum):
    """Enumeration of supported certificate types"""
    CERTIFICACION_FIRMAS = "certificacion_de_firmas"
    CERTIFICADO_PERSONERIA = "certificado_de_personeria"
    CERTIFICADO_REPRESENTACION = "certificado_de_representacion"
    CERTIFICADO_SITUACION_JURIDICA = "certificado_de_situacion_juridica"
    CERTIFICADO_VIGENCIA = "certificado_de_vigencia"
    CARTA_PODER = "carta_poder"
    PODER_GENERAL = "poder_general"
    PODER_PLEITOS = "poder_para_pleitos"
    DECLARATORIA = "declaratoria"
    OTROS = "otros"

    @classmethod
    def from_string(cls, cert_type: str) -> 'CertificateType':
        """Convert string to CertificateType enum"""
        cert_type_normalized = cert_type.lower().replace(" ", "_")
        for cert in cls:
            if cert.value == cert_type_normalized:
                return cert
        return cls.OTROS


class Purpose(Enum):
    """Common purposes/destinations for certificates"""
    BPS = "para_bps"
    MSP = "para_msp"
    ABITAB = "para_abitab"
    UTE = "para_ute"
    ANTEL = "para_antel"
    DGI = "para_dgi"
    BANCO = "para_banco"
    COMPRAVENTA = "para_compraventa"
    ZONA_FRANCA = "para_zona_franca"
    MTOP = "para_mtop"
    IMM = "para_imm"
    MEF = "para_mef"
    RUPE = "para_rupe"
    BASE_DATOS = "para_base_datos"
    MIGRACIONES = "para_migraciones"
    OTROS = "otros"

    @classmethod
    def from_string(cls, purpose: str) -> 'Purpose':
        """Convert string to Purpose enum"""
        purpose_normalized = purpose.lower().replace(" ", "_")
        if not purpose_normalized.startswith("para_"):
            purpose_normalized = f"para_{purpose_normalized}"

        for purp in cls:
            if purp.value == purpose_normalized:
                return purp
        return cls.OTROS


@dataclass
class CertificateIntent:
    """
    Represents the notary's intent to create a specific certificate.

    This is the trigger for the entire legal validation pipeline.
    """
    certificate_type: CertificateType
    purpose: Purpose
    subject_name: str
    subject_type: str  # "person" or "company"
    additional_notes: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "certificate_type": self.certificate_type.value,
            "purpose": self.purpose.value,
            "subject_name": self.subject_name,
            "subject_type": self.subject_type,
            "additional_notes": self.additional_notes
        }

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> 'CertificateIntent':
        """Create CertificateIntent from dictionary"""
        return cls(
            certificate_type=CertificateType(data["certificate_type"]),
            purpose=Purpose(data["purpose"]),
            subject_name=data["subject_name"],
            subject_type=data["subject_type"],
            additional_notes=data.get("additional_notes")
        )

    def get_display_summary(self) -> str:
        """Get human-readable summary in Spanish"""
        cert_type_display = self.certificate_type.value.replace("_", " ").title()
        purpose_display = self.purpose.value.replace("para_", "Para ").replace("_", " ").title()

        summary = f"""
╔══════════════════════════════════════════════════════════════╗
║              DEFINICIÓN DE INTENCIÓN DE CERTIFICADO          ║
╚══════════════════════════════════════════════════════════════╝

📋 Tipo de Certificado: {cert_type_display}
🎯 Propósito/Destino:   {purpose_display}
👤 Sujeto:              {self.subject_name}
📂 Tipo de Sujeto:      {self.subject_type.capitalize()}
"""
        if self.additional_notes:
            summary += f"📝 Notas Adicionales:   {self.additional_notes}\n"

        return summary


class CertificateIntentCapture:
    """
    Service class to capture certificate intent from the notary.

    This can be used via CLI, API, or GUI interface.
    """

    @staticmethod
    def get_available_certificate_types() -> List[dict]:
        """Get list of all available certificate types"""
        return [
            {
                "value": cert.value,
                "label": cert.value.replace("_", " ").title()
            }
            for cert in CertificateType
        ]

    @staticmethod
    def get_available_purposes() -> List[dict]:
        """Get list of all available purposes"""
        return [
            {
                "value": purp.value,
                "label": purp.value.replace("para_", "Para ").replace("_", " ").title()
            }
            for purp in Purpose
        ]

    @staticmethod
    def capture_intent_interactive() -> CertificateIntent:
        """
        Capture certificate intent interactively via CLI.
        This is a simple implementation - can be replaced with GUI/API.
        """
        print("\n" + "="*60)
        print("  FASE 1: DEFINICIÓN DE INTENCIÓN DE CERTIFICADO")
        print("="*60 + "\n")

        # Certificate Type
        print("Tipos de certificado disponibles:")
        cert_types = list(CertificateType)
        for idx, cert in enumerate(cert_types, 1):
            print(f"  {idx}. {cert.value.replace('_', ' ').title()}")

        cert_choice = int(input("\nSeleccione tipo de certificado (número): ")) - 1
        certificate_type = cert_types[cert_choice]

        # Purpose
        print("\nPropósitos/Destinos disponibles:")
        purposes = list(Purpose)
        for idx, purp in enumerate(purposes, 1):
            print(f"  {idx}. {purp.value.replace('para_', 'Para ').replace('_', ' ').title()}")

        purpose_choice = int(input("\nSeleccione propósito/destino (número): ")) - 1
        purpose = purposes[purpose_choice]

        # Subject
        subject_name = input("\nIngrese nombre del sujeto (persona o empresa): ").strip()

        print("\nTipo de sujeto:")
        print("  1. Persona")
        print("  2. Empresa")
        subject_type_choice = int(input("\nSeleccione tipo de sujeto (número): "))
        subject_type = "person" if subject_type_choice == 1 else "company"

        # Additional notes
        additional_notes = input("\nNotas adicionales (opcional, presione Enter para omitir): ").strip()
        additional_notes = additional_notes if additional_notes else None

        # Create intent
        intent = CertificateIntent(
            certificate_type=certificate_type,
            purpose=purpose,
            subject_name=subject_name,
            subject_type=subject_type,
            additional_notes=additional_notes
        )

        return intent

    @staticmethod
    def capture_intent_from_params(
        certificate_type: str,
        purpose: str,
        subject_name: str,
        subject_type: str = "company",
        additional_notes: Optional[str] = None
    ) -> CertificateIntent:
        """
        Capture certificate intent from parameters.
        Useful for API or programmatic usage.

        Args:
            certificate_type: Type of certificate (e.g., "certificado_de_personeria")
            purpose: Purpose/destination (e.g., "para_abitab", "BPS", "Abitab")
            subject_name: Name of person or company
            subject_type: "person" or "company"
            additional_notes: Optional additional notes

        Returns:
            CertificateIntent object
        """
        # Normalize inputs
        cert_type = CertificateType.from_string(certificate_type)
        purp = Purpose.from_string(purpose)

        return CertificateIntent(
            certificate_type=cert_type,
            purpose=purp,
            subject_name=subject_name,
            subject_type=subject_type,
            additional_notes=additional_notes
        )

    @staticmethod
    def save_intent(intent: CertificateIntent, filepath: str) -> None:
        """Save certificate intent to JSON file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(intent.to_json())
        print(f"\n✅ Intención guardada en: {filepath}")

    @staticmethod
    def load_intent(filepath: str) -> CertificateIntent:
        """Load certificate intent from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return CertificateIntent.from_dict(data)


def example_usage():
    """Example usage of Phase 1"""

    print("\n" + "="*60)
    print("  EJEMPLOS DE USO - FASE 1")
    print("="*60)

    # Example 1: Programmatic creation
    print("\n📌 Ejemplo 1: Creación programática")
    print("-" * 60)

    intent1 = CertificateIntentCapture.capture_intent_from_params(
        certificate_type="certificado_de_personeria",
        purpose="Abitab",  # Will be normalized to "para_abitab"
        subject_name="INVERSORA RINLEN S.A.",
        subject_type="company"
    )

    print(intent1.get_display_summary())
    print("\nJSON generado:")
    print(intent1.to_json())

    # Example 2: Another certificate type
    print("\n\n📌 Ejemplo 2: Certificación de firmas para BPS")
    print("-" * 60)

    intent2 = CertificateIntentCapture.capture_intent_from_params(
        certificate_type="certificacion_de_firmas",
        purpose="BPS",
        subject_name="GIRTEC S.A.",
        subject_type="company",
        additional_notes="Requiere verificación de representantes actuales"
    )

    print(intent2.get_display_summary())

    # Example 3: Save and load
    print("\n\n📌 Ejemplo 3: Guardar y cargar desde archivo")
    print("-" * 60)

    import tempfile
    import os

    temp_file = os.path.join(tempfile.gettempdir(), "certificate_intent.json")
    CertificateIntentCapture.save_intent(intent2, temp_file)

    loaded_intent = CertificateIntentCapture.load_intent(temp_file)
    print(f"\n✅ Intención cargada desde: {temp_file}")
    print(loaded_intent.get_display_summary())

    # Clean up
    os.remove(temp_file)


if __name__ == "__main__":
    # Run examples
    example_usage()

    # Uncomment below to run interactive mode
    # print("\n\n" + "="*60)
    # print("  MODO INTERACTIVO")
    # print("="*60)
    # intent = CertificateIntentCapture.capture_intent_interactive()
    # print("\n\n" + "="*60)
    # print("  RESUMEN DE INTENCIÓN CAPTURADA")
    # print("="*60)
    # print(intent.get_display_summary())
    # print("\nJSON:")
    # print(intent.to_json())
