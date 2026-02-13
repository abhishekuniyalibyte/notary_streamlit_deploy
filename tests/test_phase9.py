"""
Unit tests for Phase 9: Certificate Generation
"""

import unittest
import tempfile
import os
from datetime import datetime

from src.phase1_certificate_intent import CertificateIntentCapture, CertificateType, Purpose
from src.phase2_legal_requirements import LegalRequirementsEngine, DocumentType
from src.phase3_document_intake import DocumentIntake
from src.phase4_text_extraction import CollectionExtractionResult, ExtractedData
from src.phase5_legal_validation import ValidationMatrix, ValidationStatus
from src.phase6_gap_detection import GapAnalysisReport, GapDetector
from src.phase7_data_update import DataUpdater, UpdateAttemptResult
from src.phase8_final_confirmation import (
    FinalConfirmationEngine,
    FinalConfirmationReport,
    CertificateDecision,
    ComplianceLevel
)
from src.phase9_certificate_generation import (
    CertificateGenerator,
    GeneratedCertificate,
    CertificateSection,
    TemplateSection,
    CertificateFormat
)


class TestPhase9CertificateGeneration(unittest.TestCase):
    """Test Phase 9: Certificate Generation functionality"""

    def setUp(self):
        """Set up test fixtures"""
        # Create basic intent and requirements
        self.intent = CertificateIntentCapture.capture_intent_from_params(
            certificate_type="certificado_de_personeria",
            purpose="BPS",
            subject_name="TEST COMPANY S.A.",
            subject_type="company"
        )

        self.requirements = LegalRequirementsEngine.resolve_requirements(self.intent)

        # Create a simple document collection
        self.collection = DocumentIntake.create_collection(self.intent, self.requirements)

        # Create extraction result with data
        self.extraction_result = CollectionExtractionResult(collection=self.collection)
        self.extraction_result.extracted_data = ExtractedData(
            document_type=DocumentType.ESTATUTO,
            raw_text="TEST COMPANY S.A. RUT: 21234567890 Registro: 12345 Juan Pérez CI: 1234567-8",
            normalized_text="TEST COMPANY S.A. RUT: 21234567890 Registro: 12345 Juan Pérez CI: 1234567-8",
            company_name="TEST COMPANY S.A.",
            rut="212345678901",
            registro_comercio="12345",
            ci="1234567-8"
        )

        # Create validation matrix
        self.validation_matrix = ValidationMatrix(
            legal_requirements=self.requirements,
            extraction_result=self.extraction_result
        )
        self.validation_matrix.overall_status = ValidationStatus.VALID
        self.validation_matrix.can_issue_certificate = True

        # Create gap report
        self.gap_report = GapAnalysisReport(validation_matrix=self.validation_matrix)

        # Create update result
        self.update_result = UpdateAttemptResult(
            original_gap_report=self.gap_report,
            updated_collection=self.collection,
            updated_extraction_result=self.extraction_result
        )

        # Create approved confirmation report
        self.confirmation_report = FinalConfirmationReport(
            legal_requirements=self.requirements,
            update_result=self.update_result,
            compliance_level=ComplianceLevel.FULLY_COMPLIANT,
            certificate_decision=CertificateDecision.APPROVED,
            decision_rationale="Test approval"
        )

    def test_certificate_section_creation(self):
        """Test creating a CertificateSection"""
        section = CertificateSection(
            section_type=TemplateSection.HEADER,
            content="Test Header",
            legal_basis="Art. 248",
            required=True,
            order=1
        )

        self.assertEqual(section.section_type, TemplateSection.HEADER)
        self.assertEqual(section.content, "Test Header")
        self.assertEqual(section.legal_basis, "Art. 248")
        self.assertTrue(section.required)

    def test_certificate_section_serialization(self):
        """Test CertificateSection to_dict"""
        section = CertificateSection(
            section_type=TemplateSection.INTRODUCTION,
            content="CERTIFICO:",
            order=2
        )

        data = section.to_dict()

        self.assertIn('section_type', data)
        self.assertIn('content', data)
        self.assertEqual(data['section_type'], 'introduction')
        self.assertEqual(data['content'], 'CERTIFICO:')

    def test_generated_certificate_creation(self):
        """Test creating a GeneratedCertificate"""
        certificate = GeneratedCertificate(
            certificate_intent=self.intent,
            confirmation_report=self.confirmation_report
        )

        self.assertEqual(certificate.certificate_intent, self.intent)
        self.assertEqual(certificate.confirmation_report, self.confirmation_report)
        self.assertTrue(certificate.is_draft)
        self.assertTrue(certificate.requires_notary_review)

    def test_generated_certificate_serialization(self):
        """Test GeneratedCertificate to_dict and to_json"""
        certificate = GeneratedCertificate(
            certificate_intent=self.intent,
            confirmation_report=self.confirmation_report,
            full_text="Test certificate text"
        )

        data = certificate.to_dict()

        self.assertIn('certificate_intent', data)
        self.assertIn('sections', data)
        self.assertIn('full_text', data)
        self.assertEqual(data['full_text'], "Test certificate text")

        json_str = certificate.to_json()
        self.assertIsInstance(json_str, str)
        self.assertIn('"full_text"', json_str)

    def test_prepare_substitutions(self):
        """Test preparation of substitution variables"""
        subs = CertificateGenerator._prepare_substitutions(
            self.intent,
            self.requirements,
            self.extraction_result,
            notary_name="Dr. Test",
            notary_office="Test Office"
        )

        self.assertIn("{{NOTARY_NAME}}", subs)
        self.assertIn("{{COMPANY_NAME}}", subs)
        self.assertIn("{{RUT}}", subs)
        self.assertIn("{{REPRESENTATIVE}}", subs)

        self.assertEqual(subs["{{NOTARY_NAME}}"], "Dr. Test")
        self.assertEqual(subs["{{COMPANY_NAME}}"], "TEST COMPANY S.A.")
        self.assertEqual(subs["{{RUT}}"], "212345678901")

    def test_format_destination(self):
        """Test destination formatting"""
        dest_bps = CertificateGenerator._format_destination(Purpose.BPS)
        self.assertIn("BPS", dest_bps)
        self.assertIn("Banco de Previsión Social", dest_bps)

        dest_abitab = CertificateGenerator._format_destination(Purpose.ABITAB)
        self.assertIn("ABITAB", dest_abitab)

    def test_format_articles(self):
        """Test article formatting"""
        # Single article
        result = CertificateGenerator._format_articles(["248"])
        self.assertIn("artículo 248", result)

        # Multiple articles
        result = CertificateGenerator._format_articles(["248", "249", "250"])
        self.assertIn("248, 249 y 250", result)

        # No articles
        result = CertificateGenerator._format_articles([])
        self.assertIn("normativa vigente", result)

    def test_generate_sections(self):
        """Test section generation"""
        subs = CertificateGenerator._prepare_substitutions(
            self.intent,
            self.requirements,
            self.extraction_result,
            notary_name="Dr. Test",
            notary_office="Test Office"
        )

        sections = CertificateGenerator._generate_sections(
            self.intent,
            self.requirements,
            self.extraction_result,
            subs
        )

        # Should have all standard sections
        self.assertGreater(len(sections), 0)

        # Check that required sections exist
        section_types = [s.section_type for s in sections]
        self.assertIn(TemplateSection.HEADER, section_types)
        self.assertIn(TemplateSection.INTRODUCTION, section_types)
        self.assertIn(TemplateSection.CERTIFICATIONS, section_types)
        self.assertIn(TemplateSection.CLOSING, section_types)

    def test_generate_header(self):
        """Test header generation"""
        subs = {
            "{{NOTARY_NAME}}": "Dr. Juan Pérez",
            "{{NOTARY_OFFICE}}": "Test Office"
        }

        header = CertificateGenerator._generate_header(subs)

        self.assertIn("Dr. Juan Pérez", header)
        self.assertIn("Test Office", header)

    def test_generate_legal_basis(self):
        """Test legal basis generation"""
        subs = {"{{ARTICLES}}": "los artículos 248 y 249 del Reglamento Notarial"}

        legal_basis = CertificateGenerator._generate_legal_basis(self.requirements, subs)

        self.assertIn("conforme a lo dispuesto", legal_basis)
        self.assertIn("artículos", legal_basis)

    def test_generate_subject_identification_company(self):
        """Test subject identification for company"""
        subs = {
            "{{COMPANY_NAME}}": "TEST COMPANY S.A.",
            "{{REGISTRY_NUMBER}}": "12345",
            "{{RUT}}": "212345678901"
        }

        subject_id = CertificateGenerator._generate_subject_identification(self.intent, subs)

        self.assertIn("TEST COMPANY S.A.", subject_id)
        self.assertIn("sociedad comercial", subject_id)
        self.assertIn("12345", subject_id)
        self.assertIn("212345678901", subject_id)

    def test_generate_subject_identification_person(self):
        """Test subject identification for person"""
        intent_person = CertificateIntentCapture.capture_intent_from_params(
            certificate_type="certificado_de_personeria",
            purpose="BPS",
            subject_name="Juan Pérez",
            subject_type="person"
        )

        subs = {
            "{{SUBJECT_NAME}}": "Juan Pérez",
            "{{CI}}": "1234567-8"
        }

        subject_id = CertificateGenerator._generate_subject_identification(intent_person, subs)

        self.assertIn("Juan Pérez", subject_id)
        self.assertIn("1234567-8", subject_id)

    def test_generate_personeria_certification(self):
        """Test personería certification content"""
        subs = {
            "{{COMPANY_NAME}}": "TEST COMPANY S.A.",
            "{{REPRESENTATIVE}}": "Juan Pérez",
            "{{CI}}": "1234567-8"
        }

        cert = CertificateGenerator._generate_personeria_certification(subs, self.extraction_result)

        self.assertIn("TEST COMPANY S.A.", cert)
        self.assertIn("legalmente constituida", cert)
        self.assertIn("Juan Pérez", cert)

    def test_generate_certificate_rejected(self):
        """Test that generation fails if Phase 8 rejected"""
        # Create rejected confirmation report
        rejected_report = FinalConfirmationReport(
            legal_requirements=self.requirements,
            update_result=self.update_result,
            compliance_level=ComplianceLevel.NON_COMPLIANT,
            certificate_decision=CertificateDecision.REJECTED,
            decision_rationale="Test rejection"
        )

        with self.assertRaises(ValueError) as context:
            CertificateGenerator.generate(
                self.intent,
                self.requirements,
                self.extraction_result,
                rejected_report
            )

        self.assertIn("No se puede generar certificado", str(context.exception))

    def test_generate_certificate_success(self):
        """Test successful certificate generation"""
        certificate = CertificateGenerator.generate(
            self.intent,
            self.requirements,
            self.extraction_result,
            self.confirmation_report,
            notary_name="Dr. Test",
            notary_office="Test Office"
        )

        self.assertIsInstance(certificate, GeneratedCertificate)
        self.assertGreater(len(certificate.sections), 0)
        self.assertGreater(len(certificate.full_text), 0)
        self.assertTrue(certificate.is_draft)

    def test_generated_certificate_content(self):
        """Test that generated certificate contains expected content"""
        certificate = CertificateGenerator.generate(
            self.intent,
            self.requirements,
            self.extraction_result,
            self.confirmation_report,
            notary_name="Dr. Test",
            notary_office="Test Office"
        )

        text = certificate.get_formatted_text()

        # Should contain key elements
        self.assertIn("CERTIFICO:", text)
        self.assertIn("TEST COMPANY S.A.", text)
        self.assertIn("Dr. Test", text)
        self.assertIn("Escribano Público", text)

    def test_export_certificate_plain_text(self):
        """Test exporting certificate as plain text"""
        certificate = CertificateGenerator.generate(
            self.intent,
            self.requirements,
            self.extraction_result,
            self.confirmation_report
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            temp_file = f.name

        try:
            CertificateGenerator.export_certificate(
                certificate,
                temp_file,
                format=CertificateFormat.PLAIN_TEXT
            )

            self.assertTrue(os.path.exists(temp_file))

            with open(temp_file, 'r', encoding='utf-8') as f:
                content = f.read()

            self.assertIn("CERTIFICO:", content)
            self.assertGreater(len(content), 0)

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_export_certificate_json(self):
        """Test exporting certificate as JSON"""
        certificate = CertificateGenerator.generate(
            self.intent,
            self.requirements,
            self.extraction_result,
            self.confirmation_report
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name

        try:
            CertificateGenerator.export_certificate(
                certificate,
                temp_file,
                format=CertificateFormat.STRUCTURED_JSON
            )

            self.assertTrue(os.path.exists(temp_file))

            with open(temp_file, 'r', encoding='utf-8') as f:
                import json
                data = json.load(f)

            self.assertIn('certificate_intent', data)
            self.assertIn('full_text', data)

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_export_certificate_html(self):
        """Test exporting certificate as HTML"""
        certificate = CertificateGenerator.generate(
            self.intent,
            self.requirements,
            self.extraction_result,
            self.confirmation_report
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            temp_file = f.name

        try:
            CertificateGenerator.export_certificate(
                certificate,
                temp_file,
                format=CertificateFormat.HTML
            )

            self.assertTrue(os.path.exists(temp_file))

            with open(temp_file, 'r', encoding='utf-8') as f:
                content = f.read()

            self.assertIn("<!DOCTYPE html>", content)
            self.assertIn("CERTIFICO:", content)
            self.assertIn("BORRADOR", content)

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_certificate_summary(self):
        """Test certificate summary display"""
        certificate = CertificateGenerator.generate(
            self.intent,
            self.requirements,
            self.extraction_result,
            self.confirmation_report
        )

        summary = certificate.get_summary()

        self.assertIn("FASE 9", summary)
        self.assertIn("CERTIFICADO GENERADO", summary)
        self.assertIn("TEST COMPANY S.A.", summary)
        self.assertIn("BORRADOR", summary)

    def test_different_certificate_types(self):
        """Test generation of different certificate types"""
        # Test certificación de firmas
        intent_firmas = CertificateIntentCapture.capture_intent_from_params(
            certificate_type="certificacion_de_firmas",
            purpose="banco",
            subject_name="TEST COMPANY S.A.",
            subject_type="company"
        )

        requirements_firmas = LegalRequirementsEngine.resolve_requirements(intent_firmas)

        # Create approved confirmation for firmas
        confirmation_firmas = FinalConfirmationReport(
            legal_requirements=requirements_firmas,
            update_result=self.update_result,
            compliance_level=ComplianceLevel.FULLY_COMPLIANT,
            certificate_decision=CertificateDecision.APPROVED
        )

        certificate_firmas = CertificateGenerator.generate(
            intent_firmas,
            requirements_firmas,
            self.extraction_result,
            confirmation_firmas
        )

        text = certificate_firmas.get_formatted_text()
        self.assertIn("firma", text.lower())


def run_tests():
    """Run all Phase 9 tests"""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == '__main__':
    run_tests()
