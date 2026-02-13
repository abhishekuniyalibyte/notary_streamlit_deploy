"""
Unit tests for Phase 6: Gap & Error Detection
"""

import unittest
from datetime import datetime, timedelta
from pathlib import Path

from src.phase1_certificate_intent import CertificateIntent, CertificateType, Purpose
from src.phase2_legal_requirements import (
    LegalRequirementsEngine,
    DocumentType,
    DocumentRequirement,
    RequiredElement
)
from src.phase3_document_intake import DocumentCollection, UploadedDocument, FileFormat
from src.phase4_text_extraction import (
    ExtractedData,
    DocumentExtractionResult,
    CollectionExtractionResult
)
from src.phase5_legal_validation import (
    ValidationMatrix,
    ValidationIssue,
    ValidationSeverity,
    ValidationStatus,
    DocumentValidation,
    ElementValidation,
    LegalValidator
)
from src.phase6_gap_detection import (
    GapType,
    ActionPriority,
    Gap,
    DocumentGapReport,
    GapAnalysisReport,
    GapDetector
)


class TestGap(unittest.TestCase):
    """Test Gap dataclass"""

    def test_create_gap(self):
        """Test creating a gap"""
        gap = Gap(
            gap_type=GapType.MISSING_DOCUMENT,
            priority=ActionPriority.URGENT,
            title="Falta estatuto",
            description="Documento obligatorio no encontrado",
            affected_document=DocumentType.ESTATUTO,
            legal_basis="Art. 248"
        )

        self.assertEqual(gap.gap_type, GapType.MISSING_DOCUMENT)
        self.assertEqual(gap.priority, ActionPriority.URGENT)
        self.assertEqual(gap.affected_document, DocumentType.ESTATUTO)

    def test_to_dict(self):
        """Test conversion to dictionary"""
        gap = Gap(
            gap_type=GapType.EXPIRED_DOCUMENT,
            priority=ActionPriority.HIGH,
            title="Documento vencido",
            description="Certificado vencido",
            affected_document=DocumentType.CERTIFICADO_BPS,
            deadline=datetime(2025, 2, 1)
        )

        result = gap.to_dict()

        self.assertEqual(result["gap_type"], "expired_document")
        self.assertEqual(result["priority"], "high")
        self.assertEqual(result["affected_document"], "certificado_bps")
        self.assertIn("2025-02-01", result["deadline"])

    def test_get_priority_icon(self):
        """Test priority icon generation"""
        gap_urgent = Gap(
            gap_type=GapType.MISSING_DOCUMENT,
            priority=ActionPriority.URGENT,
            title="Test",
            description="Test"
        )

        gap_low = Gap(
            gap_type=GapType.MISSING_DOCUMENT,
            priority=ActionPriority.LOW,
            title="Test",
            description="Test"
        )

        self.assertEqual(gap_urgent.get_priority_icon(), "🔴")
        self.assertEqual(gap_low.get_priority_icon(), "🟢")

    def test_get_display(self):
        """Test display string generation"""
        gap = Gap(
            gap_type=GapType.MISSING_DOCUMENT,
            priority=ActionPriority.URGENT,
            title="Falta estatuto",
            description="Documento obligatorio",
            affected_document=DocumentType.ESTATUTO,
            legal_basis="Art. 248",
            action_required="Cargar estatuto"
        )

        display = gap.get_display()

        self.assertIn("Falta estatuto", display)
        self.assertIn("URGENT", display)
        self.assertIn("Art. 248", display)
        self.assertIn("Cargar estatuto", display)


class TestDocumentGapReport(unittest.TestCase):
    """Test DocumentGapReport"""

    def test_create_report(self):
        """Test creating document gap report"""
        report = DocumentGapReport(
            document_type=DocumentType.ESTATUTO,
            is_present=False,
            is_required=True,
            is_valid=False
        )

        self.assertEqual(report.document_type, DocumentType.ESTATUTO)
        self.assertFalse(report.is_present)
        self.assertTrue(report.is_required)
        self.assertFalse(report.is_valid)

    def test_has_critical_gaps(self):
        """Test checking for critical gaps"""
        report = DocumentGapReport(
            document_type=DocumentType.ESTATUTO,
            is_present=True,
            is_required=True,
            is_valid=False
        )

        # Add critical gap
        report.gaps.append(Gap(
            gap_type=GapType.MISSING_DATA,
            priority=ActionPriority.URGENT,
            title="Critical",
            description="Critical issue"
        ))

        self.assertTrue(report.has_critical_gaps())

    def test_no_critical_gaps(self):
        """Test when there are no critical gaps"""
        report = DocumentGapReport(
            document_type=DocumentType.ESTATUTO,
            is_present=True,
            is_required=True,
            is_valid=True
        )

        # Add non-critical gap
        report.gaps.append(Gap(
            gap_type=GapType.MISSING_DATA,
            priority=ActionPriority.LOW,
            title="Minor",
            description="Minor issue"
        ))

        self.assertFalse(report.has_critical_gaps())

    def test_to_dict(self):
        """Test conversion to dictionary"""
        report = DocumentGapReport(
            document_type=DocumentType.ESTATUTO,
            is_present=True,
            is_required=True,
            is_valid=True,
            warnings=["Warning 1"],
            recommendations=["Recommendation 1"]
        )

        result = report.to_dict()

        self.assertEqual(result["document_type"], "estatuto")
        self.assertTrue(result["is_present"])
        self.assertEqual(len(result["warnings"]), 1)
        self.assertEqual(len(result["recommendations"]), 1)


class TestGapAnalysisReport(unittest.TestCase):
    """Test GapAnalysisReport"""

    def setUp(self):
        """Set up test data"""
        # Create minimal validation matrix
        intent = CertificateIntent(
            certificate_type=CertificateType.CERTIFICADO_PERSONERIA,
            purpose=Purpose.BPS,
            subject_name="GIRTEC S.A.",
            subject_type="company"
        )

        requirements = LegalRequirementsEngine.resolve_requirements(intent)
        collection = DocumentCollection(
            certificate_intent=intent,
            legal_requirements=requirements
        )

        extraction_result = CollectionExtractionResult(collection=collection)

        self.validation_matrix = ValidationMatrix(
            legal_requirements=requirements,
            extraction_result=extraction_result
        )

        self.report = GapAnalysisReport(validation_matrix=self.validation_matrix)

    def test_calculate_summary(self):
        """Test summary calculation"""
        # Add gaps of different priorities
        self.report.gaps = [
            Gap(GapType.MISSING_DOCUMENT, ActionPriority.URGENT, "U1", "Urgent 1"),
            Gap(GapType.MISSING_DOCUMENT, ActionPriority.URGENT, "U2", "Urgent 2"),
            Gap(GapType.EXPIRED_DOCUMENT, ActionPriority.HIGH, "H1", "High 1"),
            Gap(GapType.MISSING_DATA, ActionPriority.MEDIUM, "M1", "Medium 1"),
            Gap(GapType.INCONSISTENT_DATA, ActionPriority.LOW, "L1", "Low 1"),
        ]

        self.report.calculate_summary()

        self.assertEqual(self.report.total_gaps, 5)
        self.assertEqual(self.report.urgent_gaps, 2)
        self.assertEqual(self.report.high_priority_gaps, 1)
        self.assertEqual(self.report.medium_priority_gaps, 1)
        self.assertEqual(self.report.low_priority_gaps, 1)
        self.assertEqual(self.report.blocking_issues_count, 3)  # urgent + high
        self.assertFalse(self.report.ready_for_certificate)  # Has urgent gaps

    def test_ready_for_certificate(self):
        """Test when ready for certificate"""
        # Only low priority gaps
        self.report.gaps = [
            Gap(GapType.INCONSISTENT_DATA, ActionPriority.LOW, "L1", "Low 1"),
        ]

        self.report.calculate_summary()

        self.assertTrue(self.report.ready_for_certificate)

    def test_get_gaps_by_priority(self):
        """Test filtering gaps by priority"""
        self.report.gaps = [
            Gap(GapType.MISSING_DOCUMENT, ActionPriority.URGENT, "U1", "Urgent"),
            Gap(GapType.EXPIRED_DOCUMENT, ActionPriority.HIGH, "H1", "High"),
            Gap(GapType.MISSING_DATA, ActionPriority.URGENT, "U2", "Urgent 2"),
        ]

        urgent = self.report.get_gaps_by_priority(ActionPriority.URGENT)
        high = self.report.get_gaps_by_priority(ActionPriority.HIGH)

        self.assertEqual(len(urgent), 2)
        self.assertEqual(len(high), 1)

    def test_get_gaps_by_type(self):
        """Test filtering gaps by type"""
        self.report.gaps = [
            Gap(GapType.MISSING_DOCUMENT, ActionPriority.URGENT, "M1", "Missing 1"),
            Gap(GapType.MISSING_DOCUMENT, ActionPriority.HIGH, "M2", "Missing 2"),
            Gap(GapType.EXPIRED_DOCUMENT, ActionPriority.HIGH, "E1", "Expired 1"),
        ]

        missing = self.report.get_gaps_by_type(GapType.MISSING_DOCUMENT)
        expired = self.report.get_gaps_by_type(GapType.EXPIRED_DOCUMENT)

        self.assertEqual(len(missing), 2)
        self.assertEqual(len(expired), 1)

    def test_to_dict(self):
        """Test conversion to dictionary"""
        self.report.gaps = [
            Gap(GapType.MISSING_DOCUMENT, ActionPriority.URGENT, "U1", "Urgent")
        ]
        self.report.calculate_summary()

        result = self.report.to_dict()

        self.assertIn("analysis_timestamp", result)
        self.assertIn("summary", result)
        self.assertEqual(result["summary"]["total_gaps"], 1)
        self.assertEqual(result["summary"]["urgent"], 1)

    def test_to_json(self):
        """Test JSON conversion"""
        self.report.calculate_summary()
        json_str = self.report.to_json()

        self.assertIn("analysis_timestamp", json_str)
        self.assertIn("summary", json_str)

    def test_get_summary(self):
        """Test summary generation"""
        self.report.gaps = [
            Gap(GapType.MISSING_DOCUMENT, ActionPriority.URGENT, "Falta estatuto", "Documento faltante")
        ]
        self.report.calculate_summary()

        summary = self.report.get_summary()

        self.assertIn("ANÁLISIS DE BRECHAS", summary)
        self.assertIn("URGENTE", summary)

    def test_get_action_plan(self):
        """Test action plan generation"""
        self.report.gaps = [
            Gap(GapType.MISSING_DOCUMENT, ActionPriority.URGENT, "U1", "Urgent",
                action_required="Cargar documento"),
            Gap(GapType.EXPIRED_DOCUMENT, ActionPriority.HIGH, "H1", "High",
                action_required="Actualizar documento"),
        ]
        self.report.calculate_summary()

        action_plan = self.report.get_action_plan()

        self.assertIn("PLAN DE ACCIÓN", action_plan)
        self.assertIn("PASO 1", action_plan)
        self.assertIn("PASO 2", action_plan)


class TestGapDetector(unittest.TestCase):
    """Test GapDetector"""

    def setUp(self):
        """Set up test validation matrix"""
        intent = CertificateIntent(
            certificate_type=CertificateType.CERTIFICADO_PERSONERIA,
            purpose=Purpose.BPS,
            subject_name="GIRTEC S.A.",
            subject_type="company"
        )

        self.requirements = LegalRequirementsEngine.resolve_requirements(intent)
        collection = DocumentCollection(
            certificate_intent=intent,
            legal_requirements=self.requirements
        )

        extraction_result = CollectionExtractionResult(collection=collection)

        self.validation_matrix = ValidationMatrix(
            legal_requirements=self.requirements,
            extraction_result=extraction_result
        )

    def test_detect_missing_documents(self):
        """Test detecting missing documents"""
        # Add a missing document validation
        self.validation_matrix.document_validations = [
            DocumentValidation(
                document_type=DocumentType.ESTATUTO,
                required=True,
                present=False,
                status=ValidationStatus.MISSING
            )
        ]

        gaps = GapDetector.detect_missing_documents(self.validation_matrix)

        self.assertGreater(len(gaps), 0)
        self.assertEqual(gaps[0].gap_type, GapType.MISSING_DOCUMENT)
        self.assertEqual(gaps[0].priority, ActionPriority.URGENT)

    def test_detect_expired_documents(self):
        """Test detecting expired documents"""
        # Create document validation with expiry issue
        doc_val = DocumentValidation(
            document_type=DocumentType.CERTIFICADO_BPS,
            required=True,
            present=True,
            status=ValidationStatus.EXPIRED
        )

        doc_val.issues.append(ValidationIssue(
            field="certificado_bps",
            issue_type="expired",
            severity=ValidationSeverity.CRITICAL,
            description="Certificado BPS vencido",
            legal_basis="Requisito BPS"
        ))

        self.validation_matrix.document_validations = [doc_val]

        gaps = GapDetector.detect_expired_documents(self.validation_matrix)

        self.assertGreater(len(gaps), 0)
        self.assertEqual(gaps[0].gap_type, GapType.EXPIRED_DOCUMENT)

    def test_detect_missing_data(self):
        """Test detecting missing data elements"""
        # Add missing element validation
        self.validation_matrix.element_validations = [
            ElementValidation(
                element=RequiredElement.COMPANY_NAME,
                status=ValidationStatus.MISSING,
                issues=[
                    ValidationIssue(
                        field="company_name",
                        issue_type="missing",
                        severity=ValidationSeverity.CRITICAL,
                        description="No se encontró nombre de empresa",
                        legal_basis="Art. 248"
                    )
                ]
            )
        ]

        gaps = GapDetector.detect_missing_data(self.validation_matrix)

        self.assertGreater(len(gaps), 0)
        self.assertEqual(gaps[0].gap_type, GapType.MISSING_DATA)
        self.assertEqual(gaps[0].priority, ActionPriority.URGENT)

    def test_detect_inconsistencies(self):
        """Test detecting inconsistencies"""
        # Add cross-document issue
        self.validation_matrix.cross_document_issues = [
            ValidationIssue(
                field="company_name",
                issue_type="inconsistent",
                severity=ValidationSeverity.ERROR,
                description="Nombre de empresa inconsistente",
                recommendation="Verificar documentos"
            )
        ]

        gaps = GapDetector.detect_inconsistencies(self.validation_matrix)

        self.assertGreater(len(gaps), 0)
        self.assertEqual(gaps[0].gap_type, GapType.INCONSISTENT_DATA)

    def test_analyze_complete(self):
        """Test complete gap analysis"""
        # Add various validation issues
        self.validation_matrix.document_validations = [
            DocumentValidation(
                document_type=DocumentType.ESTATUTO,
                required=True,
                present=False,
                status=ValidationStatus.MISSING
            )
        ]

        self.validation_matrix.element_validations = [
            ElementValidation(
                element=RequiredElement.RUT_NUMBER,
                status=ValidationStatus.MISSING
            )
        ]

        report = GapDetector.analyze(self.validation_matrix)

        # Should have gaps
        self.assertGreater(len(report.gaps), 0)

        # Should have calculated summary
        self.assertEqual(report.total_gaps, len(report.gaps))

        # Should have document reports
        self.assertGreater(len(report.document_reports), 0)

    def test_create_document_reports(self):
        """Test creating document reports"""
        self.validation_matrix.document_validations = [
            DocumentValidation(
                document_type=DocumentType.ESTATUTO,
                required=True,
                present=False,
                status=ValidationStatus.MISSING
            )
        ]

        gaps = [
            Gap(
                gap_type=GapType.MISSING_DOCUMENT,
                priority=ActionPriority.URGENT,
                title="Falta estatuto",
                description="Documento faltante",
                affected_document=DocumentType.ESTATUTO
            )
        ]

        reports = GapDetector.create_document_reports(self.validation_matrix, gaps)

        self.assertGreater(len(reports), 0)
        self.assertGreater(len(reports[0].warnings), 0)
        self.assertGreater(len(reports[0].recommendations), 0)


class TestRealWorldScenarios(unittest.TestCase):
    """Test real-world gap detection scenarios"""

    def test_girtec_bps_missing_documents(self):
        """Test gap detection for GIRTEC BPS with missing documents"""
        intent = CertificateIntent(
            certificate_type=CertificateType.CERTIFICADO_PERSONERIA,
            purpose=Purpose.BPS,
            subject_name="GIRTEC S.A.",
            subject_type="company"
        )

        requirements = LegalRequirementsEngine.resolve_requirements(intent)
        collection = DocumentCollection(
            certificate_intent=intent,
            legal_requirements=requirements
        )

        extraction_result = CollectionExtractionResult(collection=collection)
        validation_matrix = LegalValidator.validate(requirements, extraction_result)

        # Analyze gaps
        gap_report = GapDetector.analyze(validation_matrix)

        # Should detect missing documents
        self.assertGreater(gap_report.total_gaps, 0)
        self.assertGreater(gap_report.urgent_gaps, 0)
        self.assertFalse(gap_report.ready_for_certificate)


if __name__ == '__main__':
    unittest.main()
