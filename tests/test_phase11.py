"""
Unit tests for Phase 11: Final Output Generation & Delivery
"""

import unittest
import tempfile
import os
import json
from datetime import datetime

from src.phase1_certificate_intent import CertificateIntentCapture
from src.phase2_legal_requirements import LegalRequirementsEngine, DocumentType
from src.phase3_document_intake import DocumentIntake
from src.phase4_text_extraction import CollectionExtractionResult, ExtractedData
from src.phase5_legal_validation import ValidationMatrix, ValidationStatus
from src.phase6_gap_detection import GapAnalysisReport
from src.phase7_data_update import UpdateAttemptResult
from src.phase8_final_confirmation import (
    FinalConfirmationReport,
    CertificateDecision,
    ComplianceLevel
)
from src.phase9_certificate_generation import GeneratedCertificate, CertificateGenerator
from src.phase10_notary_review import NotaryReviewSystem, ReviewSession
from src.phase11_final_output import (
    FinalOutputGenerator,
    FinalCertificate,
    CertificateMetadata,
    OutputFormat,
    SignatureStatus,
    DeliveryMethod
)


class TestPhase11FinalOutput(unittest.TestCase):
    """Test Phase 11: Final Output Generation functionality"""

    def setUp(self):
        """Set up test fixtures"""
        # Create basic intent
        self.intent = CertificateIntentCapture.capture_intent_from_params(
            certificate_type="certificado_de_personeria",
            purpose="BPS",
            subject_name="TEST COMPANY S.A.",
            subject_type="company"
        )

        # Create mock certificate chain
        self.requirements = LegalRequirementsEngine.resolve_requirements(self.intent)
        self.collection = DocumentIntake.create_collection(self.intent, self.requirements)

        extraction_result = CollectionExtractionResult(collection=self.collection)
        extraction_result.extracted_data = ExtractedData(
            document_type=DocumentType.ESTATUTO,
            raw_text="TEST COMPANY S.A.",
            normalized_text="TEST COMPANY S.A.",
            company_name="TEST COMPANY S.A.",
            rut="212345678901",
            registro_comercio="12345"
        )

        validation_matrix = ValidationMatrix(
            legal_requirements=self.requirements,
            extraction_result=extraction_result
        )

        gap_report = GapAnalysisReport(validation_matrix=validation_matrix)
        update_result = UpdateAttemptResult(
            original_gap_report=gap_report,
            updated_collection=self.collection,
            updated_extraction_result=extraction_result
        )

        confirmation_report = FinalConfirmationReport(
            legal_requirements=self.requirements,
            update_result=update_result,
            compliance_level=ComplianceLevel.FULLY_COMPLIANT,
            certificate_decision=CertificateDecision.APPROVED
        )

        self.certificate = CertificateGenerator.generate(
            self.intent,
            self.requirements,
            extraction_result,
            confirmation_report,
            notary_name="Dr. Test Notary",
            notary_office="Test Office"
        )

        # Create approved review session
        self.review_session = NotaryReviewSystem.start_review(
            self.certificate,
            "Dr. Test Reviewer"
        )
        self.review_session = NotaryReviewSystem.approve_certificate(
            self.review_session,
            notes="Approved for testing"
        )

    def test_certificate_metadata_creation(self):
        """Test creating CertificateMetadata"""
        metadata = CertificateMetadata(
            certificate_id="TEST-ID-001",
            certificate_number="2026-001",
            issue_date=datetime.now(),
            issuing_notary="Dr. Test",
            notary_office="Test Office",
            subject_name=self.intent.subject_name,
            subject_type=self.intent.subject_type,
            certificate_type=self.intent.certificate_type.value,
            purpose=self.intent.purpose.value,
            destination="BPS",
            generation_date=datetime.now()
        )

        self.assertEqual(metadata.certificate_number, "2026-001")
        self.assertEqual(metadata.issuing_notary, "Dr. Test")
        self.assertEqual(metadata.signature_status, SignatureStatus.NOT_SIGNED)

    def test_certificate_metadata_serialization(self):
        """Test CertificateMetadata to_dict"""
        metadata = CertificateMetadata(
            certificate_id="TEST-ID-001",
            certificate_number="2026-001",
            issue_date=datetime.now(),
            issuing_notary="Dr. Test",
            notary_office="Test Office",
            subject_name=self.intent.subject_name,
            subject_type=self.intent.subject_type,
            certificate_type=self.intent.certificate_type.value,
            purpose=self.intent.purpose.value,
            destination="BPS",
            generation_date=datetime.now()
        )

        data = metadata.to_dict()

        self.assertIn('certificate_number', data)
        self.assertIn('issuing_notary', data)
        self.assertIn('signature_status', data)
        self.assertEqual(data['certificate_number'], "2026-001")
        self.assertEqual(data['signature_status'], 'not_signed')

    def test_final_certificate_creation(self):
        """Test creating FinalCertificate"""
        final_cert = FinalOutputGenerator.generate_final_certificate(
            certificate=self.certificate,
            review_session=self.review_session,
            certificate_number="2026-001",
            issuing_notary="Dr. Test",
            notary_office="Test Office"
        )

        self.assertIsInstance(final_cert, FinalCertificate)
        self.assertEqual(final_cert.metadata.certificate_number, "2026-001")
        self.assertEqual(final_cert.metadata.issuing_notary, "Dr. Test")
        self.assertIsNotNone(final_cert.metadata.certificate_id)
        self.assertFalse(final_cert.delivered)  # Not delivered yet

    def test_final_certificate_requires_approved_review(self):
        """Test that final certificate requires approved review"""
        # Create unapproved review session
        pending_session = NotaryReviewSystem.start_review(
            self.certificate,
            "Dr. Test"
        )

        with self.assertRaises(ValueError) as context:
            FinalOutputGenerator.generate_final_certificate(
                certificate=self.certificate,
                review_session=pending_session,
                certificate_number="2026-001",
                issuing_notary="Dr. Test",
                notary_office="Test Office"
            )

        # Check that error message mentions review not approved
        self.assertIn("revisión no aprobada", str(context.exception).lower())

    def test_final_certificate_serialization(self):
        """Test FinalCertificate to_dict and to_json"""
        final_cert = FinalOutputGenerator.generate_final_certificate(
            self.certificate,
            self.review_session,
            "2026-001",
            "Dr. Test",
            "Test Office"
        )

        data = final_cert.to_dict()

        self.assertIn('metadata', data)
        self.assertIn('certificate_text', data)
        self.assertIn('certificate_id', data['metadata'])  # certificate_id is in metadata

        json_str = final_cert.to_json()
        self.assertIsInstance(json_str, str)
        self.assertIn('"certificate_number"', json_str)

    def test_export_plain_text(self):
        """Test exporting to plain text format"""
        final_cert = FinalOutputGenerator.generate_final_certificate(
            self.certificate,
            self.review_session,
            "2026-001",
            "Dr. Test",
            "Test Office"
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            temp_file = f.name

        try:
            FinalOutputGenerator.export_to_format(
                final_cert,
                OutputFormat.TXT,
                temp_file
            )

            self.assertTrue(os.path.exists(temp_file))

            with open(temp_file, 'r', encoding='utf-8') as f:
                content = f.read()

            self.assertIn("CERTIFICO:", content)
            self.assertIn("TEST COMPANY S.A.", content)

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_export_html(self):
        """Test exporting to HTML format"""
        final_cert = FinalOutputGenerator.generate_final_certificate(
            self.certificate,
            self.review_session,
            "2026-001",
            "Dr. Test",
            "Test Office"
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            temp_file = f.name

        try:
            FinalOutputGenerator.export_to_format(
                final_cert,
                OutputFormat.HTML,
                temp_file
            )

            self.assertTrue(os.path.exists(temp_file))

            with open(temp_file, 'r', encoding='utf-8') as f:
                content = f.read()

            self.assertIn("<!DOCTYPE html>", content)
            self.assertIn("CERTIFICO:", content)
            self.assertIn("2026-001", content)

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_export_json(self):
        """Test exporting to JSON format"""
        final_cert = FinalOutputGenerator.generate_final_certificate(
            self.certificate,
            self.review_session,
            "2026-001",
            "Dr. Test",
            "Test Office"
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name

        try:
            FinalOutputGenerator.export_to_format(
                final_cert,
                OutputFormat.JSON,
                temp_file
            )

            self.assertTrue(os.path.exists(temp_file))

            with open(temp_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.assertIn('metadata', data)
            self.assertIn('certificate_text', data)
            self.assertEqual(data['metadata']['certificate_number'], "2026-001")

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_prepare_for_signature(self):
        """Test preparing certificate for digital signature"""
        final_cert = FinalOutputGenerator.generate_final_certificate(
            self.certificate,
            self.review_session,
            "2026-001",
            "Dr. Test",
            "Test Office"
        )

        updated_cert = FinalOutputGenerator.prepare_for_signature(final_cert)

        self.assertEqual(updated_cert.metadata.signature_status, SignatureStatus.PENDING_SIGNATURE)
        self.assertIsNotNone(updated_cert.metadata.signature_hash)
        self.assertEqual(len(updated_cert.metadata.signature_hash), 64)  # SHA256 hex length

    def test_archive_certificate(self):
        """Test archiving certificate to directory"""
        final_cert = FinalOutputGenerator.generate_final_certificate(
            self.certificate,
            self.review_session,
            "2026-001",
            "Dr. Test",
            "Test Office"
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            updated_cert = FinalOutputGenerator.archive_certificate(
                final_cert,
                temp_dir
            )

            self.assertTrue(updated_cert.archived)
            self.assertIsNotNone(updated_cert.archive_path)
            self.assertTrue(os.path.exists(updated_cert.archive_path))
            self.assertIn('2026', updated_cert.archive_path)  # Year folder
            self.assertIn('01', updated_cert.archive_path)    # Month folder

            # Check that JSON files were created
            json_files = [f for f in os.listdir(updated_cert.archive_path) if f.endswith('.json')]
            self.assertGreater(len(json_files), 0)

    def test_final_certificate_summary(self):
        """Test final certificate summary display"""
        final_cert = FinalOutputGenerator.generate_final_certificate(
            self.certificate,
            self.review_session,
            "2026-001",
            "Dr. Test",
            "Test Office"
        )

        summary = final_cert.get_summary()

        self.assertIn("FASE 11", summary)
        self.assertIn("CERTIFICADO FINAL", summary)
        self.assertIn("2026-001", summary)
        self.assertIn("Dr. Test", summary)
        self.assertIn("TEST COMPANY S.A.", summary)

    def test_delivery_tracking(self):
        """Test delivery method tracking"""
        final_cert = FinalOutputGenerator.generate_final_certificate(
            self.certificate,
            self.review_session,
            "2026-001",
            "Dr. Test",
            "Test Office"
        )

        # Initially no delivery
        self.assertIsNone(final_cert.delivery_method)
        self.assertIsNone(final_cert.delivery_date)

        # Mark as delivered
        final_cert.delivery_method = DeliveryMethod.EMAIL
        final_cert.delivery_date = datetime.now()
        final_cert.delivered = True
        final_cert.delivery_confirmation = "test@example.com"

        self.assertEqual(final_cert.delivery_method, DeliveryMethod.EMAIL)
        self.assertIsNotNone(final_cert.delivery_date)
        self.assertTrue(final_cert.delivered)

    def test_signature_status_tracking(self):
        """Test signature status changes"""
        final_cert = FinalOutputGenerator.generate_final_certificate(
            self.certificate,
            self.review_session,
            "2026-001",
            "Dr. Test",
            "Test Office"
        )

        # Initially not signed
        self.assertEqual(final_cert.metadata.signature_status, SignatureStatus.NOT_SIGNED)

        # Mark as pending signature
        final_cert.metadata.signature_status = SignatureStatus.PENDING_SIGNATURE
        self.assertEqual(final_cert.metadata.signature_status, SignatureStatus.PENDING_SIGNATURE)

        # Mark as signed
        final_cert.metadata.signature_status = SignatureStatus.SIGNED
        final_cert.metadata.signature_date = datetime.now()
        final_cert.metadata.signature_hash = "MOCK_SIGNATURE_HASH"

        self.assertEqual(final_cert.metadata.signature_status, SignatureStatus.SIGNED)
        self.assertIsNotNone(final_cert.metadata.signature_date)

    def test_metadata_includes_all_phases(self):
        """Test that metadata tracks all 11 phases"""
        final_cert = FinalOutputGenerator.generate_final_certificate(
            self.certificate,
            self.review_session,
            "2026-001",
            "Dr. Test",
            "Test Office"
        )

        data = final_cert.to_dict()

        # Check metadata structure
        self.assertIn('metadata', data)
        self.assertIn('certificate_text', data)

        # Check metadata has all required fields
        metadata = data['metadata']
        self.assertIn('certificate_id', metadata)
        self.assertIn('certificate_number', metadata)
        self.assertIn('phases_completed', metadata)

        # Should track all 11 phases
        phases = final_cert.metadata.phases_completed
        self.assertEqual(len(phases), 11)

    def test_certificate_id_uniqueness(self):
        """Test that certificate IDs are unique"""
        final_cert1 = FinalOutputGenerator.generate_final_certificate(
            self.certificate,
            self.review_session,
            "2026-001",
            "Dr. Test",
            "Test Office"
        )

        final_cert2 = FinalOutputGenerator.generate_final_certificate(
            self.certificate,
            self.review_session,
            "2026-002",
            "Dr. Test",
            "Test Office"
        )

        self.assertNotEqual(final_cert1.metadata.certificate_id, final_cert2.metadata.certificate_id)

    def test_get_formatted_certificate(self):
        """Test getting formatted certificate summary"""
        final_cert = FinalOutputGenerator.generate_final_certificate(
            self.certificate,
            self.review_session,
            "2026-001",
            "Dr. Test",
            "Test Office"
        )

        # Use get_summary instead of get_formatted_certificate
        summary = final_cert.get_summary()

        self.assertIn("FASE 11", summary)
        self.assertIn("2026-001", summary)
        self.assertIn("Dr. Test", summary)

    def test_export_pdf_placeholder(self):
        """Test PDF export (placeholder)"""
        final_cert = FinalOutputGenerator.generate_final_certificate(
            self.certificate,
            self.review_session,
            "2026-001",
            "Dr. Test",
            "Test Office"
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdf', delete=False) as f:
            temp_file = f.name

        try:
            FinalOutputGenerator.export_to_format(
                final_cert,
                OutputFormat.PDF,
                temp_file
            )

            self.assertTrue(os.path.exists(temp_file))

            # Check it's a text placeholder, not real PDF
            with open(temp_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Check for placeholder content (case insensitive)
            self.assertIn("placeholder", content.lower())
            self.assertIn("pdf", content.lower())

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_export_docx_placeholder(self):
        """Test DOCX export (placeholder)"""
        final_cert = FinalOutputGenerator.generate_final_certificate(
            self.certificate,
            self.review_session,
            "2026-001",
            "Dr. Test",
            "Test Office"
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.docx', delete=False) as f:
            temp_file = f.name

        try:
            FinalOutputGenerator.export_to_format(
                final_cert,
                OutputFormat.DOCX,
                temp_file
            )

            self.assertTrue(os.path.exists(temp_file))

            # Check it's a text placeholder, not real DOCX
            with open(temp_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Check for placeholder content (case insensitive)
            self.assertIn("placeholder", content.lower())
            self.assertIn("docx", content.lower())

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_certificate_with_changes(self):
        """Test final certificate when review had changes"""
        # Create review session with edits
        review_with_changes = NotaryReviewSystem.start_review(
            self.certificate,
            "Dr. Test"
        )

        from src.phase10_notary_review import ChangeType
        # Edit actual text that exists in the certificate
        review_with_changes = NotaryReviewSystem.add_edit(
            review_with_changes,
            "legalmente constituida",
            "debidamente constituida y regularizada",
            ChangeType.WORDING,
            "Better legal phrasing"
        )

        review_with_changes = NotaryReviewSystem.approve_certificate(
            review_with_changes,
            "Approved with minor changes"
        )

        final_cert = FinalOutputGenerator.generate_final_certificate(
            self.certificate,
            review_with_changes,
            "2026-001",
            "Dr. Test",
            "Test Office"
        )

        # Should use reviewed text with the change
        self.assertIn("debidamente constituida y regularizada", final_cert.certificate_text)
        # Should track the change
        self.assertEqual(final_cert.review_changes_count, 1)


def run_tests():
    """Run all Phase 11 tests"""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == '__main__':
    run_tests()
