"""
Unit tests for Phase 1: Certificate Intent Definition
"""

import unittest
import json
import os
import tempfile
from src.phase1_certificate_intent import (
    CertificateType,
    Purpose,
    CertificateIntent,
    CertificateIntentCapture
)


class TestCertificateType(unittest.TestCase):
    """Test CertificateType enum"""

    def test_from_string_valid(self):
        """Test converting valid strings to CertificateType"""
        cert = CertificateType.from_string("certificado de personeria")
        self.assertEqual(cert, CertificateType.CERTIFICADO_PERSONERIA)

        cert = CertificateType.from_string("certificacion_de_firmas")
        self.assertEqual(cert, CertificateType.CERTIFICACION_FIRMAS)

    def test_from_string_invalid(self):
        """Test converting invalid strings returns OTROS"""
        cert = CertificateType.from_string("certificado_invalido")
        self.assertEqual(cert, CertificateType.OTROS)


class TestPurpose(unittest.TestCase):
    """Test Purpose enum"""

    def test_from_string_valid(self):
        """Test converting valid strings to Purpose"""
        purp = Purpose.from_string("BPS")
        self.assertEqual(purp, Purpose.BPS)

        purp = Purpose.from_string("para_abitab")
        self.assertEqual(purp, Purpose.ABITAB)

        purp = Purpose.from_string("Zona Franca")
        self.assertEqual(purp, Purpose.ZONA_FRANCA)

    def test_from_string_invalid(self):
        """Test converting invalid strings returns OTROS"""
        purp = Purpose.from_string("institucion_desconocida")
        self.assertEqual(purp, Purpose.OTROS)


class TestCertificateIntent(unittest.TestCase):
    """Test CertificateIntent dataclass"""

    def setUp(self):
        """Set up test data"""
        self.intent = CertificateIntent(
            certificate_type=CertificateType.CERTIFICADO_PERSONERIA,
            purpose=Purpose.ABITAB,
            subject_name="INVERSORA RINLEN S.A.",
            subject_type="company",
            additional_notes="Test note"
        )

    def test_to_dict(self):
        """Test conversion to dictionary"""
        result = self.intent.to_dict()

        self.assertEqual(result["certificate_type"], "certificado_de_personeria")
        self.assertEqual(result["purpose"], "para_abitab")
        self.assertEqual(result["subject_name"], "INVERSORA RINLEN S.A.")
        self.assertEqual(result["subject_type"], "company")
        self.assertEqual(result["additional_notes"], "Test note")

    def test_to_json(self):
        """Test conversion to JSON"""
        json_str = self.intent.to_json()
        data = json.loads(json_str)

        self.assertEqual(data["certificate_type"], "certificado_de_personeria")
        self.assertEqual(data["subject_name"], "INVERSORA RINLEN S.A.")

    def test_from_dict(self):
        """Test creating CertificateIntent from dictionary"""
        data = {
            "certificate_type": "certificacion_de_firmas",
            "purpose": "para_bps",
            "subject_name": "GIRTEC S.A.",
            "subject_type": "company",
            "additional_notes": None
        }

        intent = CertificateIntent.from_dict(data)

        self.assertEqual(intent.certificate_type, CertificateType.CERTIFICACION_FIRMAS)
        self.assertEqual(intent.purpose, Purpose.BPS)
        self.assertEqual(intent.subject_name, "GIRTEC S.A.")
        self.assertEqual(intent.subject_type, "company")
        self.assertIsNone(intent.additional_notes)

    def test_get_display_summary(self):
        """Test display summary generation"""
        summary = self.intent.get_display_summary()

        self.assertIn("INVERSORA RINLEN S.A.", summary)
        self.assertIn("company", summary.lower())
        self.assertIn("Test note", summary)


class TestCertificateIntentCapture(unittest.TestCase):
    """Test CertificateIntentCapture service class"""

    def test_get_available_certificate_types(self):
        """Test getting available certificate types"""
        types = CertificateIntentCapture.get_available_certificate_types()

        self.assertIsInstance(types, list)
        self.assertGreater(len(types), 0)
        self.assertIn("value", types[0])
        self.assertIn("label", types[0])

    def test_get_available_purposes(self):
        """Test getting available purposes"""
        purposes = CertificateIntentCapture.get_available_purposes()

        self.assertIsInstance(purposes, list)
        self.assertGreater(len(purposes), 0)
        self.assertIn("value", purposes[0])
        self.assertIn("label", purposes[0])

    def test_capture_intent_from_params(self):
        """Test capturing intent from parameters"""
        intent = CertificateIntentCapture.capture_intent_from_params(
            certificate_type="certificado de personeria",
            purpose="Abitab",
            subject_name="TEST COMPANY S.A.",
            subject_type="company",
            additional_notes="Test"
        )

        self.assertEqual(intent.certificate_type, CertificateType.CERTIFICADO_PERSONERIA)
        self.assertEqual(intent.purpose, Purpose.ABITAB)
        self.assertEqual(intent.subject_name, "TEST COMPANY S.A.")
        self.assertEqual(intent.subject_type, "company")
        self.assertEqual(intent.additional_notes, "Test")

    def test_save_and_load_intent(self):
        """Test saving and loading intent from file"""
        intent = CertificateIntent(
            certificate_type=CertificateType.CERTIFICACION_FIRMAS,
            purpose=Purpose.BPS,
            subject_name="TEST S.A.",
            subject_type="company"
        )

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name

        try:
            # Save
            CertificateIntentCapture.save_intent(intent, temp_path)

            # Load
            loaded_intent = CertificateIntentCapture.load_intent(temp_path)

            # Verify
            self.assertEqual(loaded_intent.certificate_type, intent.certificate_type)
            self.assertEqual(loaded_intent.purpose, intent.purpose)
            self.assertEqual(loaded_intent.subject_name, intent.subject_name)
            self.assertEqual(loaded_intent.subject_type, intent.subject_type)

        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)


class TestRealWorldScenarios(unittest.TestCase):
    """Test real-world scenarios based on client data"""

    def test_girtec_bps_certificate(self):
        """Test creating intent for GIRTEC BPS certificate"""
        intent = CertificateIntentCapture.capture_intent_from_params(
            certificate_type="certificado_de_personeria",
            purpose="BPS",
            subject_name="GIRTEC S.A.",
            subject_type="company"
        )

        self.assertEqual(intent.certificate_type, CertificateType.CERTIFICADO_PERSONERIA)
        self.assertEqual(intent.purpose, Purpose.BPS)
        self.assertEqual(intent.subject_name, "GIRTEC S.A.")

    def test_netkla_zona_franca(self):
        """Test creating intent for NETKLA Zona Franca certificate"""
        intent = CertificateIntentCapture.capture_intent_from_params(
            certificate_type="certificado_de_personeria",
            purpose="zona franca",
            subject_name="NETKLA TRADING S.A.",
            subject_type="company"
        )

        self.assertEqual(intent.purpose, Purpose.ZONA_FRANCA)
        self.assertEqual(intent.subject_name, "NETKLA TRADING S.A.")

    def test_saterix_ute(self):
        """Test creating intent for SATERIX UTE certificate"""
        intent = CertificateIntentCapture.capture_intent_from_params(
            certificate_type="certificado_de_personeria",
            purpose="UTE",
            subject_name="SATERIX S.A.",
            subject_type="company"
        )

        self.assertEqual(intent.purpose, Purpose.UTE)

    def test_poder_general(self):
        """Test creating intent for poder general (general power of attorney)"""
        intent = CertificateIntentCapture.capture_intent_from_params(
            certificate_type="poder general",
            purpose="banco",
            subject_name="GIRTEC S.A.",
            subject_type="company",
            additional_notes="Poder a favor de Carolina Bomio"
        )

        self.assertEqual(intent.certificate_type, CertificateType.PODER_GENERAL)
        self.assertEqual(intent.purpose, Purpose.BANCO)


if __name__ == '__main__':
    unittest.main()
