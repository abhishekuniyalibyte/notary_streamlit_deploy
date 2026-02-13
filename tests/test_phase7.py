"""
Unit tests for Phase 7: Data Update Attempt
"""

import unittest
import tempfile
import os
from datetime import datetime, timedelta

from src.phase1_certificate_intent import CertificateIntentCapture
from src.phase2_legal_requirements import LegalRequirementsEngine, DocumentType
from src.phase3_document_intake import DocumentIntake, DocumentCollection, UploadedDocument, FileFormat
from src.phase6_gap_detection import Gap, GapType, GapAnalysisReport, ActionPriority
from src.phase7_data_update import (
    DataUpdater,
    UpdateAttemptResult,
    DocumentUpdate,
    UpdateSource,
    UpdateStatus
)


class TestPhase7DataUpdate(unittest.TestCase):
    """Test Phase 7: Data Update functionality"""

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

    def test_create_update_session(self):
        """Test creating an update session"""
        # Create a mock gap report
        gap = Gap(
            gap_type=GapType.MISSING_DOCUMENT,
            priority=ActionPriority.URGENT,
            title="Falta Estatuto",
            description="Estatuto no encontrado",
            affected_document=DocumentType.ESTATUTO,
            legal_basis="Art. 248"
        )

        # Create a minimal gap report (we need validation_matrix)
        # For testing, create a mock
        from src.phase5_legal_validation import ValidationMatrix
        from src.phase4_text_extraction import CollectionExtractionResult

        extraction_result = CollectionExtractionResult(collection=self.collection)
        validation_matrix = ValidationMatrix(
            legal_requirements=self.requirements,
            extraction_result=extraction_result
        )

        gap_report = GapAnalysisReport(validation_matrix=validation_matrix)
        gap_report.gaps = [gap]

        # Create update session
        update_result = DataUpdater.create_update_session(gap_report, self.collection)

        self.assertIsInstance(update_result, UpdateAttemptResult)
        self.assertEqual(update_result.original_gap_report, gap_report)
        self.assertEqual(update_result.updated_collection, self.collection)
        self.assertEqual(len(update_result.updates), 0)

    def test_document_update_creation(self):
        """Test creating a DocumentUpdate"""
        gap = Gap(
            gap_type=GapType.MISSING_DOCUMENT,
            priority=ActionPriority.URGENT,
            title="Falta Estatuto",
            description="Estatuto no encontrado",
            affected_document=DocumentType.ESTATUTO
        )

        update = DocumentUpdate(
            document_type=DocumentType.ESTATUTO,
            gap_addressed=gap,
            update_source=UpdateSource.MANUAL_UPLOAD,
            update_status=UpdateStatus.SUCCESS,
            previous_state="Sin documento",
            new_state="Documento cargado"
        )

        self.assertEqual(update.document_type, DocumentType.ESTATUTO)
        self.assertEqual(update.update_source, UpdateSource.MANUAL_UPLOAD)
        self.assertEqual(update.update_status, UpdateStatus.SUCCESS)
        self.assertIsNotNone(update.timestamp)

    def test_document_update_serialization(self):
        """Test DocumentUpdate to_dict"""
        gap = Gap(
            gap_type=GapType.EXPIRED_DOCUMENT,
            priority=ActionPriority.HIGH,
            title="BPS Vencido",
            description="Certificado BPS vencido",
            affected_document=DocumentType.CERTIFICADO_BPS
        )

        update = DocumentUpdate(
            document_type=DocumentType.CERTIFICADO_BPS,
            gap_addressed=gap,
            update_source=UpdateSource.MANUAL_UPLOAD,
            update_status=UpdateStatus.SUCCESS,
            notes="Certificado actualizado"
        )

        data = update.to_dict()

        self.assertIn('document_type', data)
        self.assertIn('gap_addressed', data)
        self.assertIn('update_source', data)
        self.assertIn('update_status', data)
        self.assertIn('timestamp', data)
        self.assertEqual(data['notes'], "Certificado actualizado")

    def test_document_update_display(self):
        """Test DocumentUpdate display format"""
        gap = Gap(
            gap_type=GapType.MISSING_DOCUMENT,
            priority=ActionPriority.URGENT,
            title="Falta Acta",
            description="Acta de directorio no encontrada",
            affected_document=DocumentType.ACTA_DIRECTORIO
        )

        update = DocumentUpdate(
            document_type=DocumentType.ACTA_DIRECTORIO,
            gap_addressed=gap,
            update_source=UpdateSource.MANUAL_UPLOAD,
            update_status=UpdateStatus.SUCCESS,
            previous_state="Sin documento",
            new_state="Acta cargada"
        )

        display = update.get_display()

        self.assertIn("ACTA_DIRECTORIO", display)
        self.assertIn("manual_upload", display)
        self.assertIn("success", display)
        self.assertIn("✅", display)

    def test_upload_updated_document_file_not_found(self):
        """Test uploading non-existent file"""
        gap = Gap(
            gap_type=GapType.MISSING_DOCUMENT,
            priority=ActionPriority.URGENT,
            title="Falta Estatuto",
            description="Estatuto no encontrado",
            affected_document=DocumentType.ESTATUTO
        )

        from src.phase5_legal_validation import ValidationMatrix
        from src.phase4_text_extraction import CollectionExtractionResult

        extraction_result = CollectionExtractionResult(collection=self.collection)
        validation_matrix = ValidationMatrix(
            legal_requirements=self.requirements,
            extraction_result=extraction_result
        )

        gap_report = GapAnalysisReport(validation_matrix=validation_matrix)
        gap_report.gaps = [gap]

        update_result = DataUpdater.create_update_session(gap_report, self.collection)

        # Try to upload non-existent file
        update_result = DataUpdater.upload_updated_document(
            update_result,
            gap,
            "/nonexistent/file.pdf"
        )

        # Should have recorded a failed update
        self.assertEqual(len(update_result.updates), 1)
        self.assertEqual(update_result.updates[0].update_status, UpdateStatus.FAILED)
        self.assertIsNotNone(update_result.updates[0].error_message)

    def test_upload_updated_document_success(self):
        """Test successfully uploading a document"""
        gap = Gap(
            gap_type=GapType.MISSING_DOCUMENT,
            priority=ActionPriority.URGENT,
            title="Falta Estatuto",
            description="Estatuto no encontrado",
            affected_document=DocumentType.ESTATUTO
        )

        from src.phase5_legal_validation import ValidationMatrix
        from src.phase4_text_extraction import CollectionExtractionResult

        extraction_result = CollectionExtractionResult(collection=self.collection)
        validation_matrix = ValidationMatrix(
            legal_requirements=self.requirements,
            extraction_result=extraction_result
        )

        gap_report = GapAnalysisReport(validation_matrix=validation_matrix)
        gap_report.gaps = [gap]

        update_result = DataUpdater.create_update_session(gap_report, self.collection)

        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test estatuto content")
            temp_file = f.name

        try:
            initial_count = len(update_result.updated_collection.documents)

            update_result = DataUpdater.upload_updated_document(
                update_result,
                gap,
                temp_file,
                notes="Test upload"
            )

            # Check update was recorded
            self.assertEqual(len(update_result.updates), 1)
            self.assertEqual(update_result.updates[0].update_status, UpdateStatus.SUCCESS)
            self.assertEqual(update_result.updates[0].notes, "Test upload")

            # Check document was added to collection
            self.assertEqual(len(update_result.updated_collection.documents), initial_count + 1)

        finally:
            os.unlink(temp_file)

    def test_mark_gap_not_addressed(self):
        """Test marking a gap as not addressed"""
        gap = Gap(
            gap_type=GapType.MISSING_DOCUMENT,
            priority=ActionPriority.LOW,
            title="Documento opcional faltante",
            description="Documento opcional no encontrado",
            affected_document=DocumentType.DECLARACION_JURADA
        )

        from src.phase5_legal_validation import ValidationMatrix
        from src.phase4_text_extraction import CollectionExtractionResult

        extraction_result = CollectionExtractionResult(collection=self.collection)
        validation_matrix = ValidationMatrix(
            legal_requirements=self.requirements,
            extraction_result=extraction_result
        )

        gap_report = GapAnalysisReport(validation_matrix=validation_matrix)
        gap_report.gaps = [gap]

        update_result = DataUpdater.create_update_session(gap_report, self.collection)

        update_result = DataUpdater.mark_gap_not_addressed(
            update_result,
            gap,
            reason="Documento opcional, no requerido para certificado"
        )

        self.assertEqual(len(update_result.updates), 1)
        self.assertEqual(update_result.updates[0].update_status, UpdateStatus.NOT_ATTEMPTED)
        self.assertEqual(update_result.updates[0].update_source, UpdateSource.NOT_UPDATED)

    def test_attempt_public_registry_fetch(self):
        """Test public registry fetch (placeholder)"""
        gap = Gap(
            gap_type=GapType.MISSING_DATA,
            priority=ActionPriority.HIGH,
            title="Falta RUT",
            description="RUT no encontrado",
            affected_document=DocumentType.ESTATUTO
        )

        from src.phase5_legal_validation import ValidationMatrix
        from src.phase4_text_extraction import CollectionExtractionResult

        extraction_result = CollectionExtractionResult(collection=self.collection)
        validation_matrix = ValidationMatrix(
            legal_requirements=self.requirements,
            extraction_result=extraction_result
        )

        gap_report = GapAnalysisReport(validation_matrix=validation_matrix)
        gap_report.gaps = [gap]

        update_result = DataUpdater.create_update_session(gap_report, self.collection)

        update_result = DataUpdater.attempt_public_registry_fetch(
            update_result,
            gap,
            company_name="TEST COMPANY S.A.",
            rut="21234567890"
        )

        # Should record as not attempted (placeholder)
        self.assertEqual(len(update_result.updates), 1)
        self.assertEqual(update_result.updates[0].update_status, UpdateStatus.NOT_ATTEMPTED)
        self.assertEqual(update_result.updates[0].update_source, UpdateSource.PUBLIC_REGISTRY)

    def test_update_result_summary_calculation(self):
        """Test summary statistics calculation"""
        from src.phase5_legal_validation import ValidationMatrix
        from src.phase4_text_extraction import CollectionExtractionResult

        extraction_result = CollectionExtractionResult(collection=self.collection)
        validation_matrix = ValidationMatrix(
            legal_requirements=self.requirements,
            extraction_result=extraction_result
        )

        gap_report = GapAnalysisReport(validation_matrix=validation_matrix)

        update_result = UpdateAttemptResult(original_gap_report=gap_report)

        # Add some updates
        gap1 = Gap(
            gap_type=GapType.MISSING_DOCUMENT,
            priority=ActionPriority.URGENT,
            title="Gap 1",
            description="Test gap 1",
            affected_document=DocumentType.ESTATUTO
        )

        gap2 = Gap(
            gap_type=GapType.EXPIRED_DOCUMENT,
            priority=ActionPriority.HIGH,
            title="Gap 2",
            description="Test gap 2",
            affected_document=DocumentType.CERTIFICADO_BPS
        )

        update_result.updates.append(DocumentUpdate(
            document_type=DocumentType.ESTATUTO,
            gap_addressed=gap1,
            update_source=UpdateSource.MANUAL_UPLOAD,
            update_status=UpdateStatus.SUCCESS
        ))

        update_result.updates.append(DocumentUpdate(
            document_type=DocumentType.CERTIFICADO_BPS,
            gap_addressed=gap2,
            update_source=UpdateSource.MANUAL_UPLOAD,
            update_status=UpdateStatus.FAILED,
            error_message="Test error"
        ))

        update_result.calculate_summary()

        self.assertEqual(update_result.successful_updates, 1)
        self.assertEqual(update_result.failed_updates, 1)
        self.assertEqual(update_result.gaps_addressed, 1)

    def test_update_result_serialization(self):
        """Test UpdateAttemptResult to_dict and to_json"""
        from src.phase5_legal_validation import ValidationMatrix
        from src.phase4_text_extraction import CollectionExtractionResult

        extraction_result = CollectionExtractionResult(collection=self.collection)
        validation_matrix = ValidationMatrix(
            legal_requirements=self.requirements,
            extraction_result=extraction_result
        )

        gap_report = GapAnalysisReport(validation_matrix=validation_matrix)

        update_result = UpdateAttemptResult(
            original_gap_report=gap_report,
            updated_collection=self.collection
        )

        data = update_result.to_dict()

        self.assertIn('original_gap_report', data)
        self.assertIn('updates', data)
        self.assertIn('total_gaps', data)
        self.assertIn('timestamp', data)

        json_str = update_result.to_json()
        self.assertIsInstance(json_str, str)
        self.assertIn('"original_gap_report"', json_str)

    def test_get_remaining_gaps(self):
        """Test getting remaining unaddressed gaps"""
        gap1 = Gap(
            gap_type=GapType.MISSING_DOCUMENT,
            priority=ActionPriority.URGENT,
            title="Gap 1",
            description="Test gap 1",
            affected_document=DocumentType.ESTATUTO
        )

        gap2 = Gap(
            gap_type=GapType.EXPIRED_DOCUMENT,
            priority=ActionPriority.HIGH,
            title="Gap 2",
            description="Test gap 2",
            affected_document=DocumentType.CERTIFICADO_BPS
        )

        gap3 = Gap(
            gap_type=GapType.MISSING_DATA,
            priority=ActionPriority.MEDIUM,
            title="Gap 3",
            description="Test gap 3",
            affected_document=DocumentType.ACTA_DIRECTORIO
        )

        from src.phase5_legal_validation import ValidationMatrix
        from src.phase4_text_extraction import CollectionExtractionResult

        extraction_result = CollectionExtractionResult(collection=self.collection)
        validation_matrix = ValidationMatrix(
            legal_requirements=self.requirements,
            extraction_result=extraction_result
        )

        gap_report = GapAnalysisReport(validation_matrix=validation_matrix)
        gap_report.gaps = [gap1, gap2, gap3]

        update_result = UpdateAttemptResult(original_gap_report=gap_report)

        # Address gap1 successfully
        update_result.updates.append(DocumentUpdate(
            document_type=DocumentType.ESTATUTO,
            gap_addressed=gap1,
            update_source=UpdateSource.MANUAL_UPLOAD,
            update_status=UpdateStatus.SUCCESS
        ))

        # Fail to address gap2
        update_result.updates.append(DocumentUpdate(
            document_type=DocumentType.CERTIFICADO_BPS,
            gap_addressed=gap2,
            update_source=UpdateSource.MANUAL_UPLOAD,
            update_status=UpdateStatus.FAILED
        ))

        # Don't address gap3 at all

        remaining = DataUpdater.get_remaining_gaps(update_result)

        # gap1 was successfully addressed, so shouldn't be in remaining
        # gap2 failed, so should be in remaining
        # gap3 not attempted, so should be in remaining
        self.assertEqual(len(remaining), 2)

    def test_update_result_summary_display(self):
        """Test summary display format"""
        from src.phase5_legal_validation import ValidationMatrix
        from src.phase4_text_extraction import CollectionExtractionResult

        extraction_result = CollectionExtractionResult(collection=self.collection)
        validation_matrix = ValidationMatrix(
            legal_requirements=self.requirements,
            extraction_result=extraction_result
        )

        gap_report = GapAnalysisReport(validation_matrix=validation_matrix)

        update_result = UpdateAttemptResult(
            original_gap_report=gap_report,
            updated_collection=self.collection
        )

        summary = update_result.get_summary()

        self.assertIn("FASE 7", summary)
        self.assertIn("RESULTADO DE ACTUALIZACIÓN", summary)
        self.assertIn("Total de brechas detectadas", summary)
        self.assertIn("Timestamp", summary)

    def test_update_result_changes_report(self):
        """Test changes report display"""
        from src.phase5_legal_validation import ValidationMatrix
        from src.phase4_text_extraction import CollectionExtractionResult

        extraction_result = CollectionExtractionResult(collection=self.collection)
        validation_matrix = ValidationMatrix(
            legal_requirements=self.requirements,
            extraction_result=extraction_result
        )

        gap_report = GapAnalysisReport(validation_matrix=validation_matrix)

        update_result = UpdateAttemptResult(
            original_gap_report=gap_report,
            updated_collection=self.collection
        )

        gap = Gap(
            gap_type=GapType.MISSING_DOCUMENT,
            priority=ActionPriority.URGENT,
            title="Test Gap",
            description="Test gap",
            affected_document=DocumentType.ESTATUTO
        )

        update_result.updates.append(DocumentUpdate(
            document_type=DocumentType.ESTATUTO,
            gap_addressed=gap,
            update_source=UpdateSource.MANUAL_UPLOAD,
            update_status=UpdateStatus.SUCCESS
        ))

        report = update_result.get_changes_report()

        self.assertIn("REPORTE DETALLADO DE CAMBIOS", report)
        self.assertIn("ESTATUTO", report)


def run_tests():
    """Run all Phase 7 tests"""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == '__main__':
    run_tests()
