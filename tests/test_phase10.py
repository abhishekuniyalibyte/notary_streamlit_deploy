"""
Unit tests for Phase 10: Notary Review & Learning
"""

import unittest
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
from src.phase9_certificate_generation import GeneratedCertificate
from src.phase10_notary_review import (
    NotaryReviewSystem,
    ReviewSession,
    NotaryEdit,
    NotaryFeedback,
    ReviewStatus,
    ChangeType,
    FeedbackCategory
)


class TestPhase10NotaryReview(unittest.TestCase):
    """Test Phase 10: Notary Review functionality"""

    def setUp(self):
        """Set up test fixtures"""
        # Create basic intent
        self.intent = CertificateIntentCapture.capture_intent_from_params(
            certificate_type="certificado_de_personeria",
            purpose="BPS",
            subject_name="TEST COMPANY S.A.",
            subject_type="company"
        )

        # Create mock certificate
        self.requirements = LegalRequirementsEngine.resolve_requirements(self.intent)
        self.collection = DocumentIntake.create_collection(self.intent, self.requirements)

        extraction_result = CollectionExtractionResult(collection=self.collection)
        extraction_result.extracted_data = ExtractedData(
            document_type=DocumentType.ESTATUTO,
            raw_text="TEST COMPANY S.A.",
            normalized_text="TEST COMPANY S.A.",
            company_name="TEST COMPANY S.A."
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

        self.certificate = GeneratedCertificate(
            certificate_intent=self.intent,
            confirmation_report=confirmation_report,
            full_text="CERTIFICO: Que TEST COMPANY S.A. es una sociedad legalmente constituida."
        )

    def test_notary_edit_creation(self):
        """Test creating a NotaryEdit"""
        edit = NotaryEdit(
            section_type="certifications",
            original_text="sociedad constituida",
            edited_text="sociedad legalmente constituida",
            change_type=ChangeType.WORDING,
            reason="Mejor redacción"
        )

        self.assertEqual(edit.section_type, "certifications")
        self.assertEqual(edit.change_type, ChangeType.WORDING)
        self.assertIsNotNone(edit.timestamp)

    def test_notary_edit_serialization(self):
        """Test NotaryEdit to_dict"""
        edit = NotaryEdit(
            original_text="original",
            edited_text="edited",
            change_type=ChangeType.LEGAL_ACCURACY,
            reason="Legal correction"
        )

        data = edit.to_dict()

        self.assertIn('original_text', data)
        self.assertIn('edited_text', data)
        self.assertIn('change_type', data)
        self.assertEqual(data['change_type'], 'legal_accuracy')

    def test_notary_feedback_creation(self):
        """Test creating NotaryFeedback"""
        feedback = NotaryFeedback(
            category=FeedbackCategory.TEMPLATE_IMPROVEMENT,
            feedback_text="Template needs improvement",
            severity="high",
            actionable=True
        )

        self.assertEqual(feedback.category, FeedbackCategory.TEMPLATE_IMPROVEMENT)
        self.assertEqual(feedback.severity, "high")
        self.assertTrue(feedback.actionable)

    def test_notary_feedback_serialization(self):
        """Test NotaryFeedback to_dict"""
        feedback = NotaryFeedback(
            category=FeedbackCategory.DATA_EXTRACTION,
            feedback_text="Data extraction issue",
            severity="medium"
        )

        data = feedback.to_dict()

        self.assertIn('category', data)
        self.assertIn('feedback_text', data)
        self.assertEqual(data['category'], 'data_extraction')

    def test_start_review(self):
        """Test starting a review session"""
        session = NotaryReviewSystem.start_review(
            certificate=self.certificate,
            reviewer_name="Dr. Test"
        )

        self.assertIsInstance(session, ReviewSession)
        self.assertEqual(session.reviewer_name, "Dr. Test")
        self.assertEqual(session.status, ReviewStatus.IN_REVIEW)
        self.assertIsNotNone(session.start_time)
        self.assertEqual(session.original_text, self.certificate.get_formatted_text())

    def test_add_edit(self):
        """Test adding an edit to review session"""
        session = NotaryReviewSystem.start_review(self.certificate, "Dr. Test")

        session = NotaryReviewSystem.add_edit(
            session=session,
            original_text="sociedad legalmente constituida",
            edited_text="sociedad debidamente constituida",
            change_type=ChangeType.WORDING,
            reason="Better wording"
        )

        self.assertEqual(len(session.edits), 1)
        self.assertEqual(session.edits[0].change_type, ChangeType.WORDING)
        self.assertIn("debidamente", session.reviewed_text)

    def test_add_feedback(self):
        """Test adding feedback to review session"""
        session = NotaryReviewSystem.start_review(self.certificate, "Dr. Test")

        session = NotaryReviewSystem.add_feedback(
            session=session,
            category=FeedbackCategory.TEMPLATE_IMPROVEMENT,
            feedback_text="Template should include more details",
            severity="medium",
            actionable=True
        )

        self.assertEqual(len(session.feedback), 1)
        self.assertEqual(session.feedback[0].category, FeedbackCategory.TEMPLATE_IMPROVEMENT)
        self.assertEqual(session.feedback[0].severity, "medium")

    def test_approve_certificate_no_changes(self):
        """Test approving certificate without changes"""
        session = NotaryReviewSystem.start_review(self.certificate, "Dr. Test")

        session = NotaryReviewSystem.approve_certificate(
            session=session,
            notes="Perfect as is"
        )

        self.assertEqual(session.status, ReviewStatus.APPROVED)
        self.assertIsNotNone(session.end_time)
        self.assertIsNotNone(session.review_duration_minutes)
        self.assertIn("sin cambios", session.approval_decision.lower())

    def test_approve_certificate_with_changes(self):
        """Test approving certificate with changes"""
        session = NotaryReviewSystem.start_review(self.certificate, "Dr. Test")

        # Add an edit
        session = NotaryReviewSystem.add_edit(
            session=session,
            original_text="sociedad legalmente constituida",
            edited_text="sociedad debidamente constituida",
            change_type=ChangeType.WORDING,
            reason="Better wording"
        )

        session = NotaryReviewSystem.approve_certificate(
            session=session,
            notes="Minor changes applied"
        )

        self.assertEqual(session.status, ReviewStatus.APPROVED_WITH_CHANGES)
        self.assertIn("1 cambio", session.approval_decision.lower())

    def test_reject_certificate(self):
        """Test rejecting certificate"""
        session = NotaryReviewSystem.start_review(self.certificate, "Dr. Test")

        session = NotaryReviewSystem.reject_certificate(
            session=session,
            reason="Missing required information",
            notes="Need to go back to Phase 7"
        )

        self.assertEqual(session.status, ReviewStatus.REJECTED)
        self.assertEqual(session.rejection_reason, "Missing required information")
        self.assertIsNotNone(session.end_time)

    def test_review_session_serialization(self):
        """Test ReviewSession to_dict and to_json"""
        session = NotaryReviewSystem.start_review(self.certificate, "Dr. Test")
        session = NotaryReviewSystem.approve_certificate(session)

        data = session.to_dict()

        self.assertIn('certificate', data)
        self.assertIn('reviewer_name', data)
        self.assertIn('status', data)
        self.assertEqual(data['status'], 'approved')

        json_str = session.to_json()
        self.assertIsInstance(json_str, str)
        self.assertIn('"status"', json_str)

    def test_review_session_summary(self):
        """Test review session summary display"""
        session = NotaryReviewSystem.start_review(self.certificate, "Dr. Test")

        # Add some edits and feedback
        session = NotaryReviewSystem.add_edit(
            session, "original", "edited",
            ChangeType.WORDING, "Test reason"
        )

        session = NotaryReviewSystem.approve_certificate(session)

        summary = session.get_summary()

        self.assertIn("FASE 10", summary)
        self.assertIn("REVISIÓN DEL NOTARIO", summary)
        self.assertIn("Dr. Test", summary)
        self.assertIn("APPROVED", summary.upper())

    def test_get_change_report(self):
        """Test change report generation"""
        session = NotaryReviewSystem.start_review(self.certificate, "Dr. Test")

        # Add edits
        session = NotaryReviewSystem.add_edit(
            session, "text1", "text2",
            ChangeType.WORDING, "Reason 1"
        )

        session = NotaryReviewSystem.add_edit(
            session, "text3", "text4",
            ChangeType.LEGAL_ACCURACY, "Reason 2"
        )

        report = NotaryReviewSystem.get_change_report(session)

        self.assertIn("REPORTE DETALLADO DE CAMBIOS", report)
        self.assertIn("WORDING", report)
        self.assertIn("LEGAL_ACCURACY", report)

    def test_get_learning_insights(self):
        """Test learning insights extraction"""
        session = NotaryReviewSystem.start_review(self.certificate, "Dr. Test")

        # Add multiple edits
        for i in range(3):
            session = NotaryReviewSystem.add_edit(
                session, f"text{i}", f"edited{i}",
                ChangeType.LEGAL_ACCURACY, f"Reason {i}"
            )

        # Add feedback
        session = NotaryReviewSystem.add_feedback(
            session,
            FeedbackCategory.TEMPLATE_IMPROVEMENT,
            "Improve template",
            actionable=True
        )

        session = NotaryReviewSystem.approve_certificate(session)

        insights = NotaryReviewSystem.get_learning_insights(session)

        self.assertIn('certificate_type', insights)
        self.assertIn('total_edits', insights)
        self.assertEqual(insights['total_edits'], 3)
        self.assertIn('edit_types', insights)
        self.assertIn('common_issues', insights)

    def test_compare_versions(self):
        """Test version comparison"""
        original = "Line 1\nLine 2\nLine 3"
        reviewed = "Line 1\nLine 2 modified\nLine 3"

        diff = NotaryReviewSystem.compare_versions(original, reviewed)

        self.assertIsInstance(diff, list)
        self.assertGreater(len(diff), 0)

    def test_multiple_edits_different_types(self):
        """Test handling multiple edits of different types"""
        session = NotaryReviewSystem.start_review(self.certificate, "Dr. Test")

        # Add different types of edits
        session = NotaryReviewSystem.add_edit(
            session, "text1", "edited1",
            ChangeType.WORDING, "Wording improvement"
        )

        session = NotaryReviewSystem.add_edit(
            session, "text2", "edited2",
            ChangeType.LEGAL_ACCURACY, "Legal correction"
        )

        session = NotaryReviewSystem.add_edit(
            session, "text3", "edited3",
            ChangeType.DATA_CORRECTION, "Data fix"
        )

        session = NotaryReviewSystem.approve_certificate(session)

        insights = NotaryReviewSystem.get_learning_insights(session)

        self.assertEqual(insights['total_edits'], 3)
        self.assertIn('wording', insights['edit_types'])
        self.assertIn('legal_accuracy', insights['edit_types'])
        self.assertIn('data_correction', insights['edit_types'])


def run_tests():
    """Run all Phase 10 tests"""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == '__main__':
    run_tests()
