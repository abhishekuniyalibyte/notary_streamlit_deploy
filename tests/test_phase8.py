"""
Unit tests for Phase 8: Final Legal Confirmation
"""

import unittest
from datetime import datetime, timedelta

from src.phase1_certificate_intent import CertificateIntentCapture
from src.phase2_legal_requirements import LegalRequirementsEngine, DocumentType
from src.phase3_document_intake import DocumentIntake, DocumentCollection
from src.phase4_text_extraction import CollectionExtractionResult, ExtractedData
from src.phase5_legal_validation import (
    LegalValidator,
    ValidationMatrix,
    ValidationIssue,
    ValidationSeverity,
    ValidationStatus,
    DocumentValidation
)
from src.phase6_gap_detection import GapDetector, GapAnalysisReport, Gap, GapType, ActionPriority
from src.phase7_data_update import DataUpdater, UpdateAttemptResult
from src.phase8_final_confirmation import (
    FinalConfirmationEngine,
    FinalConfirmationReport,
    ComplianceCheck,
    ComplianceLevel,
    CertificateDecision
)


class TestPhase8FinalConfirmation(unittest.TestCase):
    """Test Phase 8: Final Legal Confirmation functionality"""

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

    def test_compliance_check_creation(self):
        """Test creating a ComplianceCheck"""
        check = ComplianceCheck(
            check_name="Documentos presentes",
            check_category="document",
            is_compliant=True,
            severity=ValidationSeverity.CRITICAL,
            details="Todos los documentos presentes",
            legal_basis="Art. 248",
            blocking=True
        )

        self.assertEqual(check.check_name, "Documentos presentes")
        self.assertTrue(check.is_compliant)
        self.assertEqual(check.severity, ValidationSeverity.CRITICAL)
        self.assertTrue(check.blocking)

    def test_compliance_check_serialization(self):
        """Test ComplianceCheck to_dict"""
        check = ComplianceCheck(
            check_name="Test Check",
            check_category="legal",
            is_compliant=False,
            severity=ValidationSeverity.ERROR,
            details="Test details",
            legal_basis="Art. 250"
        )

        data = check.to_dict()

        self.assertIn('check_name', data)
        self.assertIn('check_category', data)
        self.assertIn('is_compliant', data)
        self.assertIn('severity', data)
        self.assertEqual(data['check_name'], "Test Check")
        self.assertFalse(data['is_compliant'])

    def test_compliance_check_display(self):
        """Test ComplianceCheck display format"""
        check = ComplianceCheck(
            check_name="Test Check",
            check_category="document",
            is_compliant=False,
            severity=ValidationSeverity.CRITICAL,
            details="Missing document",
            legal_basis="Art. 248",
            blocking=True
        )

        display = check.get_display()

        self.assertIn("Test Check", display)
        self.assertIn("Missing document", display)
        self.assertIn("Art. 248", display)
        self.assertIn("BLOQUEANTE", display)

    def test_final_confirmation_report_creation(self):
        """Test creating a FinalConfirmationReport"""
        extraction_result = CollectionExtractionResult(collection=self.collection)
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

        report = FinalConfirmationReport(
            legal_requirements=self.requirements,
            update_result=update_result
        )

        self.assertEqual(report.legal_requirements, self.requirements)
        self.assertEqual(report.update_result, update_result)
        self.assertEqual(report.compliance_level, ComplianceLevel.NON_COMPLIANT)
        self.assertEqual(report.certificate_decision, CertificateDecision.REJECTED)

    def test_final_confirmation_report_serialization(self):
        """Test FinalConfirmationReport to_dict and to_json"""
        extraction_result = CollectionExtractionResult(collection=self.collection)
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

        report = FinalConfirmationReport(
            legal_requirements=self.requirements,
            update_result=update_result,
            compliance_level=ComplianceLevel.FULLY_COMPLIANT,
            certificate_decision=CertificateDecision.APPROVED
        )

        data = report.to_dict()

        self.assertIn('legal_requirements', data)
        self.assertIn('update_result', data)
        self.assertIn('compliance_level', data)
        self.assertIn('certificate_decision', data)
        self.assertEqual(data['compliance_level'], 'fully_compliant')
        self.assertEqual(data['certificate_decision'], 'approved')

        json_str = report.to_json()
        self.assertIsInstance(json_str, str)
        self.assertIn('"compliance_level"', json_str)

    def test_final_confirmation_report_summary_calculation(self):
        """Test summary statistics calculation"""
        extraction_result = CollectionExtractionResult(collection=self.collection)
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

        report = FinalConfirmationReport(
            legal_requirements=self.requirements,
            update_result=update_result
        )

        # Add some checks
        report.compliance_checks = [
            ComplianceCheck("Check 1", "document", True, ValidationSeverity.CRITICAL, "OK"),
            ComplianceCheck("Check 2", "legal", False, ValidationSeverity.CRITICAL, "Failed", blocking=True),
            ComplianceCheck("Check 3", "legal", False, ValidationSeverity.WARNING, "Warning"),
        ]

        report.calculate_summary()

        self.assertEqual(report.total_checks, 3)
        self.assertEqual(report.passed_checks, 1)
        self.assertEqual(report.failed_checks, 2)
        self.assertEqual(report.blocking_issues, 1)
        self.assertEqual(report.critical_issues, 1)
        self.assertEqual(report.warnings, 1)

    def test_can_proceed_to_phase9_approved(self):
        """Test can_proceed_to_phase9 when approved"""
        extraction_result = CollectionExtractionResult(collection=self.collection)
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

        report = FinalConfirmationReport(
            legal_requirements=self.requirements,
            update_result=update_result,
            certificate_decision=CertificateDecision.APPROVED
        )

        self.assertTrue(report.can_proceed_to_phase9())

    def test_can_proceed_to_phase9_rejected(self):
        """Test can_proceed_to_phase9 when rejected"""
        extraction_result = CollectionExtractionResult(collection=self.collection)
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

        report = FinalConfirmationReport(
            legal_requirements=self.requirements,
            update_result=update_result,
            certificate_decision=CertificateDecision.REJECTED
        )

        self.assertFalse(report.can_proceed_to_phase9())

    def test_confirm_no_extraction_result(self):
        """Test confirmation when no extraction result available"""
        extraction_result = CollectionExtractionResult(collection=self.collection)
        validation_matrix = ValidationMatrix(
            legal_requirements=self.requirements,
            extraction_result=extraction_result
        )
        gap_report = GapAnalysisReport(validation_matrix=validation_matrix)

        # Create update result WITHOUT extraction result
        update_result = UpdateAttemptResult(
            original_gap_report=gap_report,
            updated_collection=self.collection
        )

        report = FinalConfirmationEngine.confirm(
            self.requirements,
            update_result
        )

        self.assertEqual(report.certificate_decision, CertificateDecision.REJECTED)
        self.assertEqual(report.compliance_level, ComplianceLevel.NON_COMPLIANT)
        self.assertIn("No se pudo extraer datos", report.decision_rationale)

    def test_confirm_with_extraction_result(self):
        """Test confirmation with valid extraction result"""
        # Create extraction result with some data
        extraction_result = CollectionExtractionResult(collection=self.collection)

        # Add some extracted data
        extraction_result.extracted_data = ExtractedData(
            document_type=DocumentType.ESTATUTO,
            raw_text="TEST COMPANY S.A. RUT: 21234567890",
            normalized_text="TEST COMPANY S.A. RUT: 21234567890",
            company_name="TEST COMPANY S.A.",
            rut="21234567890"
        )

        # Create validation matrix
        validation_matrix = ValidationMatrix(
            legal_requirements=self.requirements,
            extraction_result=extraction_result
        )

        gap_report = GapAnalysisReport(validation_matrix=validation_matrix)

        # Create update result WITH extraction result
        update_result = UpdateAttemptResult(
            original_gap_report=gap_report,
            updated_collection=self.collection,
            updated_extraction_result=extraction_result
        )

        report = FinalConfirmationEngine.confirm(
            self.requirements,
            update_result
        )

        # Should complete without error
        self.assertIsNotNone(report.validation_matrix)
        self.assertIsNotNone(report.gap_report)
        self.assertGreater(len(report.compliance_checks), 0)

    def test_create_compliance_checks(self):
        """Test compliance checks creation"""
        extraction_result = CollectionExtractionResult(collection=self.collection)
        validation_matrix = ValidationMatrix(
            legal_requirements=self.requirements,
            extraction_result=extraction_result
        )

        gap_report = GapAnalysisReport(validation_matrix=validation_matrix)

        checks = FinalConfirmationEngine._create_compliance_checks(
            validation_matrix,
            gap_report,
            self.requirements
        )

        # Should create multiple checks
        self.assertGreater(len(checks), 0)

        # Check categories
        categories = set(c.check_category for c in checks)
        self.assertIn('document', categories)
        self.assertIn('legal', categories)

    def test_make_decision_fully_compliant(self):
        """Test decision making when fully compliant"""
        extraction_result = CollectionExtractionResult(collection=self.collection)
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

        report = FinalConfirmationReport(
            legal_requirements=self.requirements,
            update_result=update_result
        )

        # All checks pass
        report.compliance_checks = [
            ComplianceCheck("Check 1", "document", True, ValidationSeverity.CRITICAL, "OK"),
            ComplianceCheck("Check 2", "legal", True, ValidationSeverity.CRITICAL, "OK"),
        ]

        FinalConfirmationEngine._make_decision(report)

        self.assertEqual(report.compliance_level, ComplianceLevel.FULLY_COMPLIANT)
        self.assertEqual(report.certificate_decision, CertificateDecision.APPROVED)
        self.assertIn("Todos los requisitos legales cumplidos", report.decision_rationale)

    def test_make_decision_with_warnings(self):
        """Test decision making with warnings"""
        extraction_result = CollectionExtractionResult(collection=self.collection)
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

        report = FinalConfirmationReport(
            legal_requirements=self.requirements,
            update_result=update_result
        )

        # All critical checks pass, but warnings present
        report.compliance_checks = [
            ComplianceCheck("Check 1", "document", True, ValidationSeverity.CRITICAL, "OK"),
            ComplianceCheck("Check 2", "legal", False, ValidationSeverity.WARNING, "Minor issue"),
        ]

        FinalConfirmationEngine._make_decision(report)

        self.assertEqual(report.compliance_level, ComplianceLevel.SUBSTANTIALLY_COMPLIANT)
        self.assertEqual(report.certificate_decision, CertificateDecision.APPROVED_WITH_WARNINGS)
        self.assertIn("advertencia", report.decision_rationale.lower())

    def test_make_decision_with_blocking_issues(self):
        """Test decision making with blocking issues"""
        extraction_result = CollectionExtractionResult(collection=self.collection)
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

        report = FinalConfirmationReport(
            legal_requirements=self.requirements,
            update_result=update_result
        )

        # Blocking issue present
        report.compliance_checks = [
            ComplianceCheck("Check 1", "document", False, ValidationSeverity.CRITICAL, "Failed", blocking=True),
            ComplianceCheck("Check 2", "legal", True, ValidationSeverity.CRITICAL, "OK"),
        ]

        FinalConfirmationEngine._make_decision(report)

        self.assertEqual(report.compliance_level, ComplianceLevel.NON_COMPLIANT)
        self.assertEqual(report.certificate_decision, CertificateDecision.REJECTED)
        self.assertIn("rechazado", report.decision_rationale.lower())

    def test_make_decision_with_critical_issues(self):
        """Test decision making with critical issues"""
        extraction_result = CollectionExtractionResult(collection=self.collection)
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

        report = FinalConfirmationReport(
            legal_requirements=self.requirements,
            update_result=update_result
        )

        # Critical issue present (not necessarily blocking, but critical severity)
        report.compliance_checks = [
            ComplianceCheck("Check 1", "document", False, ValidationSeverity.CRITICAL, "Critical issue"),
            ComplianceCheck("Check 2", "legal", True, ValidationSeverity.INFO, "OK"),
        ]

        FinalConfirmationEngine._make_decision(report)

        self.assertEqual(report.compliance_level, ComplianceLevel.NON_COMPLIANT)
        self.assertEqual(report.certificate_decision, CertificateDecision.REJECTED)

    def test_final_confirmation_report_summary_display(self):
        """Test summary display format"""
        extraction_result = CollectionExtractionResult(collection=self.collection)
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

        report = FinalConfirmationReport(
            legal_requirements=self.requirements,
            update_result=update_result,
            compliance_level=ComplianceLevel.FULLY_COMPLIANT,
            certificate_decision=CertificateDecision.APPROVED
        )

        report.compliance_checks = [
            ComplianceCheck("Test", "document", True, ValidationSeverity.CRITICAL, "OK")
        ]

        summary = report.get_summary()

        self.assertIn("FASE 8", summary)
        self.assertIn("CONFIRMACIÓN LEGAL FINAL", summary)
        self.assertIn("APPROVED", summary)
        self.assertIn("FULLY COMPLIANT", summary)  # Note: displayed with space, not underscore

    def test_final_confirmation_report_detailed_report(self):
        """Test detailed report display"""
        extraction_result = CollectionExtractionResult(collection=self.collection)
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

        report = FinalConfirmationReport(
            legal_requirements=self.requirements,
            update_result=update_result
        )

        report.compliance_checks = [
            ComplianceCheck("Check 1", "document", True, ValidationSeverity.CRITICAL, "OK"),
            ComplianceCheck("Check 2", "legal", False, ValidationSeverity.ERROR, "Failed"),
        ]

        detailed = report.get_detailed_report()

        self.assertIn("REPORTE DETALLADO", detailed)
        self.assertIn("DOCUMENT", detailed)
        self.assertIn("LEGAL", detailed)


def run_tests():
    """Run all Phase 8 tests"""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == '__main__':
    run_tests()
