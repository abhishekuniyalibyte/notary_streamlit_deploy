"""
Unit tests for Phase 5: Legal Validation Engine
"""

import unittest
from pathlib import Path
from datetime import datetime

from src.phase1_certificate_intent import CertificateIntent, CertificateType, Purpose
from src.phase2_legal_requirements import (
    LegalRequirementsEngine,
    DocumentType,
    DocumentRequirement,
    RequiredElement
)
from src.phase3_document_intake import (
    DocumentCollection,
    UploadedDocument,
    FileFormat
)
from src.phase4_text_extraction import (
    ExtractedData,
    DocumentExtractionResult,
    CollectionExtractionResult
)
from src.phase5_legal_validation import (
    ValidationStatus,
    ValidationSeverity,
    ValidationIssue,
    DocumentValidation,
    ElementValidation,
    ValidationMatrix,
    LegalValidator
)


class TestValidationIssue(unittest.TestCase):
    """Test ValidationIssue"""

    def test_create_issue(self):
        """Test creating a validation issue"""
        issue = ValidationIssue(
            field="estatuto",
            issue_type="missing_document",
            severity=ValidationSeverity.CRITICAL,
            description="Falta estatuto social",
            legal_basis="Art. 248"
        )

        self.assertEqual(issue.field, "estatuto")
        self.assertEqual(issue.severity, ValidationSeverity.CRITICAL)

    def test_to_dict(self):
        """Test conversion to dictionary"""
        issue = ValidationIssue(
            field="rut",
            issue_type="missing_element",
            severity=ValidationSeverity.ERROR,
            description="No se encontró RUT"
        )

        result = issue.to_dict()

        self.assertEqual(result["field"], "rut")
        self.assertEqual(result["severity"], "error")

    def test_get_display(self):
        """Test display string generation"""
        issue = ValidationIssue(
            field="certificado_bps",
            issue_type="expired",
            severity=ValidationSeverity.CRITICAL,
            description="Certificado BPS vencido",
            legal_basis="Requisito BPS"
        )

        display = issue.get_display()
        self.assertIn("certificado_bps", display)
        self.assertIn("Certificado BPS vencido", display)
        self.assertIn("Requisito BPS", display)


class TestDocumentValidation(unittest.TestCase):
    """Test DocumentValidation"""

    def test_valid_document(self):
        """Test validation of valid document"""
        validation = DocumentValidation(
            document_type=DocumentType.ESTATUTO,
            required=True,
            present=True,
            status=ValidationStatus.VALID
        )

        self.assertTrue(validation.is_valid())

    def test_missing_required_document(self):
        """Test validation of missing required document"""
        validation = DocumentValidation(
            document_type=DocumentType.ESTATUTO,
            required=True,
            present=False,
            status=ValidationStatus.MISSING
        )

        self.assertFalse(validation.is_valid())

    def test_document_with_critical_issue(self):
        """Test document with critical issue"""
        validation = DocumentValidation(
            document_type=DocumentType.CERTIFICADO_BPS,
            required=True,
            present=True,
            status=ValidationStatus.EXPIRED
        )

        validation.issues.append(ValidationIssue(
            field="certificado_bps",
            issue_type="expired",
            severity=ValidationSeverity.CRITICAL,
            description="Certificado vencido"
        ))

        self.assertFalse(validation.is_valid())

    def test_to_dict(self):
        """Test conversion to dictionary"""
        validation = DocumentValidation(
            document_type=DocumentType.ESTATUTO,
            required=True,
            present=True,
            status=ValidationStatus.VALID
        )

        result = validation.to_dict()

        self.assertEqual(result["document_type"], "estatuto")
        self.assertTrue(result["required"])
        self.assertTrue(result["present"])


class TestElementValidation(unittest.TestCase):
    """Test ElementValidation"""

    def test_valid_element(self):
        """Test validation of valid element"""
        validation = ElementValidation(
            element=RequiredElement.COMPANY_NAME,
            status=ValidationStatus.VALID,
            value_found="GIRTEC S.A."
        )

        self.assertTrue(validation.is_valid())
        self.assertEqual(validation.value_found, "GIRTEC S.A.")

    def test_missing_element(self):
        """Test validation of missing element"""
        validation = ElementValidation(
            element=RequiredElement.RUT_NUMBER,
            status=ValidationStatus.MISSING
        )

        self.assertFalse(validation.is_valid())

    def test_to_dict(self):
        """Test conversion to dictionary"""
        validation = ElementValidation(
            element=RequiredElement.COMPANY_NAME,
            status=ValidationStatus.VALID,
            value_found="TEST S.A."
        )

        result = validation.to_dict()

        self.assertEqual(result["element"], "company_name")
        self.assertEqual(result["status"], "valid")
        self.assertEqual(result["value_found"], "TEST S.A.")


class TestValidationMatrix(unittest.TestCase):
    """Test ValidationMatrix"""

    def setUp(self):
        """Set up test data"""
        # Create mock intent and requirements
        intent = CertificateIntent(
            certificate_type=CertificateType.CERTIFICADO_PERSONERIA,
            purpose=Purpose.BPS,
            subject_name="GIRTEC S.A.",
            subject_type="company"
        )

        self.requirements = LegalRequirementsEngine.resolve_requirements(intent)

        # Create mock collection
        collection = DocumentCollection(
            certificate_intent=intent,
            legal_requirements=self.requirements
        )

        # Create mock extraction result
        self.extraction_result = CollectionExtractionResult(collection=collection)

        # Create validation matrix
        self.matrix = ValidationMatrix(
            legal_requirements=self.requirements,
            extraction_result=self.extraction_result
        )

    def test_get_all_issues(self):
        """Test getting all issues"""
        # Add some issues
        doc_val = DocumentValidation(
            document_type=DocumentType.ESTATUTO,
            required=True,
            present=False,
            status=ValidationStatus.MISSING
        )
        doc_val.issues.append(ValidationIssue(
            field="estatuto",
            issue_type="missing",
            severity=ValidationSeverity.CRITICAL,
            description="Falta estatuto"
        ))

        self.matrix.document_validations.append(doc_val)

        all_issues = self.matrix.get_all_issues()
        self.assertEqual(len(all_issues), 1)

    def test_get_critical_issues(self):
        """Test getting only critical issues"""
        # Add critical issue
        issue1 = ValidationIssue(
            field="estatuto",
            issue_type="missing",
            severity=ValidationSeverity.CRITICAL,
            description="Falta estatuto"
        )

        # Add warning issue
        issue2 = ValidationIssue(
            field="dates",
            issue_type="verification",
            severity=ValidationSeverity.WARNING,
            description="Verificar fechas"
        )

        self.matrix.cross_document_issues = [issue1, issue2]

        critical = self.matrix.get_critical_issues()
        self.assertEqual(len(critical), 1)
        self.assertEqual(critical[0].severity, ValidationSeverity.CRITICAL)

    def test_get_issue_count_by_severity(self):
        """Test counting issues by severity"""
        self.matrix.cross_document_issues = [
            ValidationIssue("f1", "t1", ValidationSeverity.CRITICAL, "d1"),
            ValidationIssue("f2", "t2", ValidationSeverity.CRITICAL, "d2"),
            ValidationIssue("f3", "t3", ValidationSeverity.ERROR, "d3"),
            ValidationIssue("f4", "t4", ValidationSeverity.WARNING, "d4"),
        ]

        counts = self.matrix.get_issue_count_by_severity()

        self.assertEqual(counts[ValidationSeverity.CRITICAL], 2)
        self.assertEqual(counts[ValidationSeverity.ERROR], 1)
        self.assertEqual(counts[ValidationSeverity.WARNING], 1)
        self.assertEqual(counts[ValidationSeverity.INFO], 0)

    def test_to_dict(self):
        """Test conversion to dictionary"""
        result = self.matrix.to_dict()

        self.assertIn("validation_timestamp", result)
        self.assertIn("overall_status", result)
        self.assertIn("can_issue_certificate", result)
        self.assertIn("issue_summary", result)

    def test_to_json(self):
        """Test JSON conversion"""
        json_str = self.matrix.to_json()
        self.assertIn("validation_timestamp", json_str)


class TestLegalValidator(unittest.TestCase):
    """Test LegalValidator"""

    def setUp(self):
        """Set up test data"""
        # Create intent and requirements
        intent = CertificateIntent(
            certificate_type=CertificateType.CERTIFICADO_PERSONERIA,
            purpose=Purpose.BPS,
            subject_name="GIRTEC S.A.",
            subject_type="company"
        )

        self.requirements = LegalRequirementsEngine.resolve_requirements(intent)

        # Create collection with some documents
        collection = DocumentCollection(
            certificate_intent=intent,
            legal_requirements=self.requirements
        )

        # Create extraction result
        self.extraction_result = CollectionExtractionResult(collection=collection)

        # Add some mock extraction results
        doc1 = UploadedDocument(
            file_path=Path("/test/estatuto.pdf"),
            file_name="estatuto.pdf",
            file_format=FileFormat.PDF,
            file_size_bytes=1024,
            upload_timestamp=datetime.now(),
            detected_type=DocumentType.ESTATUTO
        )

        extracted1 = ExtractedData(
            document_type=DocumentType.ESTATUTO,
            raw_text="Raw",
            normalized_text="GIRTEC S.A. RUT: 212345678901",
            company_name="GIRTEC S.A.",
            rut="212345678901"
        )

        self.extraction_result.extraction_results.append(
            DocumentExtractionResult(
                document=doc1,
                extracted_data=extracted1,
                success=True
            )
        )

    def test_validate_document_presence_all_present(self):
        """Test validation when all documents are present"""
        # Add all required documents
        for req_doc in self.requirements.required_documents:
            doc = UploadedDocument(
                file_path=Path(f"/test/{req_doc.document_type.value}.pdf"),
                file_name=f"{req_doc.document_type.value}.pdf",
                file_format=FileFormat.PDF,
                file_size_bytes=1024,
                upload_timestamp=datetime.now(),
                detected_type=req_doc.document_type
            )

            extracted = ExtractedData(
                document_type=req_doc.document_type,
                raw_text="Raw",
                normalized_text="Normalized"
            )

            self.extraction_result.extraction_results.append(
                DocumentExtractionResult(
                    document=doc,
                    extracted_data=extracted,
                    success=True
                )
            )

        validations = LegalValidator.validate_document_presence(
            self.requirements,
            self.extraction_result
        )

        # All required documents should be marked as present
        for validation in validations:
            if validation.required:
                self.assertTrue(validation.present)

    def test_validate_document_presence_missing(self):
        """Test validation when documents are missing"""
        validations = LegalValidator.validate_document_presence(
            self.requirements,
            self.extraction_result
        )

        # Should have missing documents
        missing = [v for v in validations if not v.present and v.required]
        self.assertGreater(len(missing), 0)

        # Missing required documents should have critical issues
        for validation in missing:
            has_critical = any(
                issue.severity == ValidationSeverity.CRITICAL
                for issue in validation.issues
            )
            self.assertTrue(has_critical)

    def test_validate_required_elements(self):
        """Test validation of required elements"""
        validations = LegalValidator.validate_required_elements(
            self.requirements,
            self.extraction_result
        )

        # Should have validations for all required elements
        self.assertGreater(len(validations), 0)

        # Check that company name was found
        company_val = next(
            (v for v in validations if v.element == RequiredElement.COMPANY_NAME),
            None
        )
        if company_val:
            self.assertEqual(company_val.status, ValidationStatus.VALID)
            self.assertEqual(company_val.value_found, "GIRTEC S.A.")

    def test_validate_cross_document_consistency(self):
        """Test cross-document consistency validation"""
        # Add another document with different company name
        doc2 = UploadedDocument(
            file_path=Path("/test/acta.pdf"),
            file_name="acta.pdf",
            file_format=FileFormat.PDF,
            file_size_bytes=1024,
            upload_timestamp=datetime.now(),
            detected_type=DocumentType.ACTA_DIRECTORIO
        )

        extracted2 = ExtractedData(
            document_type=DocumentType.ACTA_DIRECTORIO,
            raw_text="Raw",
            normalized_text="DIFFERENT S.A.",
            company_name="DIFFERENT S.A."  # Different company name!
        )

        self.extraction_result.extraction_results.append(
            DocumentExtractionResult(
                document=doc2,
                extracted_data=extracted2,
                success=True
            )
        )

        issues = LegalValidator.validate_cross_document_consistency(
            self.extraction_result
        )

        # Should detect inconsistency
        self.assertGreater(len(issues), 0)

        # Should be an error about company name
        has_company_issue = any("company_name" in issue.field for issue in issues)
        self.assertTrue(has_company_issue)

    def test_validate_complete(self):
        """Test complete validation"""
        matrix = LegalValidator.validate(
            self.requirements,
            self.extraction_result
        )

        # Should have validation results
        self.assertGreater(len(matrix.document_validations), 0)
        self.assertGreater(len(matrix.element_validations), 0)

        # Should have overall status
        self.assertIsNotNone(matrix.overall_status)
        self.assertIsInstance(matrix.can_issue_certificate, bool)


class TestRealWorldScenarios(unittest.TestCase):
    """Test real-world validation scenarios"""

    def test_girtec_bps_complete_validation(self):
        """Test complete validation flow for GIRTEC BPS"""
        # Create intent
        intent = CertificateIntent(
            certificate_type=CertificateType.CERTIFICADO_PERSONERIA,
            purpose=Purpose.BPS,
            subject_name="GIRTEC S.A.",
            subject_type="company"
        )

        # Get requirements
        requirements = LegalRequirementsEngine.resolve_requirements(intent)

        # Create collection
        collection = DocumentCollection(
            certificate_intent=intent,
            legal_requirements=requirements
        )

        # Create extraction result
        extraction_result = CollectionExtractionResult(collection=collection)

        # Validate
        matrix = LegalValidator.validate(requirements, extraction_result)

        # Should have validation results
        self.assertIsNotNone(matrix)
        self.assertIsNotNone(matrix.overall_status)


if __name__ == '__main__':
    unittest.main()
