"""
Unit tests for Phase 3: Document Intake
"""

import unittest
import tempfile
import os
from pathlib import Path
from datetime import datetime

from src.phase1_certificate_intent import CertificateIntent, CertificateType, Purpose
from src.phase2_legal_requirements import (
    LegalRequirementsEngine,
    DocumentType,
    DocumentRequirement
)
from src.phase3_document_intake import (
    FileFormat,
    ProcessingStatus,
    UploadedDocument,
    DocumentCollection,
    DocumentTypeDetector,
    DocumentIntake
)


class TestFileFormat(unittest.TestCase):
    """Test FileFormat enum"""

    def test_from_extension_valid(self):
        """Test converting valid extensions"""
        self.assertEqual(FileFormat.from_extension(".pdf"), FileFormat.PDF)
        self.assertEqual(FileFormat.from_extension("pdf"), FileFormat.PDF)
        self.assertEqual(FileFormat.from_extension(".DOCX"), FileFormat.DOCX)
        self.assertEqual(FileFormat.from_extension("jpg"), FileFormat.JPG)

    def test_from_extension_invalid(self):
        """Test converting invalid extensions"""
        self.assertEqual(FileFormat.from_extension(".xyz"), FileFormat.UNKNOWN)
        self.assertEqual(FileFormat.from_extension(""), FileFormat.UNKNOWN)


class TestUploadedDocument(unittest.TestCase):
    """Test UploadedDocument dataclass"""

    def setUp(self):
        """Set up test data"""
        self.doc = UploadedDocument(
            file_path=Path("/test/estatuto.pdf"),
            file_name="estatuto.pdf",
            file_format=FileFormat.PDF,
            file_size_bytes=102400,
            upload_timestamp=datetime(2025, 1, 1, 12, 0, 0),
            detected_type=DocumentType.ESTATUTO
        )

    def test_to_dict(self):
        """Test conversion to dictionary"""
        result = self.doc.to_dict()

        self.assertEqual(result["file_name"], "estatuto.pdf")
        self.assertEqual(result["file_format"], "pdf")
        self.assertEqual(result["detected_type"], "estatuto")

    def test_get_display_info(self):
        """Test display info generation"""
        info = self.doc.get_display_info()

        self.assertIn("estatuto.pdf", info)
        self.assertIn("PDF", info)
        self.assertIn("estatuto", info)


class TestDocumentTypeDetector(unittest.TestCase):
    """Test DocumentTypeDetector"""

    def test_detect_estatuto(self):
        """Test detecting estatuto from filename"""
        detected = DocumentTypeDetector.detect_from_filename("estatuto_girtec.pdf")
        self.assertEqual(detected, DocumentType.ESTATUTO)

        detected = DocumentTypeDetector.detect_from_filename("ESTATUTOS.docx")
        self.assertEqual(detected, DocumentType.ESTATUTO)

    def test_detect_acta(self):
        """Test detecting acta from filename"""
        detected = DocumentTypeDetector.detect_from_filename("acta_directorio_2023.pdf")
        self.assertEqual(detected, DocumentType.ACTA_DIRECTORIO)

        detected = DocumentTypeDetector.detect_from_filename("acta asamblea.doc")
        self.assertEqual(detected, DocumentType.ACTA_DIRECTORIO)

    def test_detect_bps(self):
        """Test detecting BPS certificate"""
        detected = DocumentTypeDetector.detect_from_filename("certificado_bps.pdf")
        self.assertEqual(detected, DocumentType.CERTIFICADO_BPS)

        detected = DocumentTypeDetector.detect_from_filename("BPS_2025.pdf")
        self.assertEqual(detected, DocumentType.CERTIFICADO_BPS)

    def test_detect_dgi(self):
        """Test detecting DGI certificate"""
        detected = DocumentTypeDetector.detect_from_filename("certificado_dgi.pdf")
        self.assertEqual(detected, DocumentType.CERTIFICADO_DGI)

        detected = DocumentTypeDetector.detect_from_filename("situacion_tributaria.pdf")
        self.assertEqual(detected, DocumentType.CERTIFICADO_DGI)

    def test_detect_cedula(self):
        """Test detecting cédula"""
        detected = DocumentTypeDetector.detect_from_filename("cedula_identidad.jpg")
        self.assertEqual(detected, DocumentType.CEDULA_IDENTIDAD)

        detected = DocumentTypeDetector.detect_from_filename("ci_scan.pdf")
        self.assertEqual(detected, DocumentType.CEDULA_IDENTIDAD)

    def test_detect_poder(self):
        """Test detecting poder"""
        detected = DocumentTypeDetector.detect_from_filename("poder_general.pdf")
        self.assertEqual(detected, DocumentType.PODER)

        detected = DocumentTypeDetector.detect_from_filename("apoderado.docx")
        self.assertEqual(detected, DocumentType.PODER)

    def test_detect_none(self):
        """Test when no type can be detected"""
        detected = DocumentTypeDetector.detect_from_filename("random_file.pdf")
        self.assertIsNone(detected)

    def test_detect_filename_avoids_substring_false_positives(self):
        """Short keywords like 'ci' should not match inside longer words (e.g. 'certificacion')."""
        detected = DocumentTypeDetector.detect_from_filename("certificacion_control_leyes.doc")
        self.assertIsNone(detected)

    def test_detect_from_text(self):
        """Test detecting document types from extracted content text."""
        detected = DocumentTypeDetector.detect_from_text("Este es el Estatuto Social de la empresa GIRTEC S.A.")
        self.assertEqual(detected, DocumentType.ESTATUTO)

        detected = DocumentTypeDetector.detect_from_text("Se presenta la Declaración Jurada (DDJJ) ante BCU.")
        self.assertEqual(detected, DocumentType.DECLARACION_JURADA)

        detected = DocumentTypeDetector.detect_from_text("Cédula de Identidad del compareciente.")
        self.assertEqual(detected, DocumentType.CEDULA_IDENTIDAD)

        detected = DocumentTypeDetector.detect_from_text("Texto sin indicadores claros.")
        self.assertIsNone(detected)

    def test_is_likely_scanned(self):
        """Test scanned detection"""
        self.assertTrue(DocumentTypeDetector.is_likely_scanned(FileFormat.JPG))
        self.assertTrue(DocumentTypeDetector.is_likely_scanned(FileFormat.PNG))
        self.assertFalse(DocumentTypeDetector.is_likely_scanned(FileFormat.PDF))
        self.assertFalse(DocumentTypeDetector.is_likely_scanned(FileFormat.DOCX))


class TestDocumentCollection(unittest.TestCase):
    """Test DocumentCollection"""

    def setUp(self):
        """Set up test data"""
        self.intent = CertificateIntent(
            certificate_type=CertificateType.CERTIFICADO_PERSONERIA,
            purpose=Purpose.BPS,
            subject_name="GIRTEC S.A.",
            subject_type="company"
        )

        self.requirements = LegalRequirementsEngine.resolve_requirements(self.intent)
        self.collection = DocumentCollection(
            certificate_intent=self.intent,
            legal_requirements=self.requirements
        )

    def test_add_document(self):
        """Test adding documents"""
        doc = UploadedDocument(
            file_path=Path("/test/estatuto.pdf"),
            file_name="estatuto.pdf",
            file_format=FileFormat.PDF,
            file_size_bytes=100000,
            upload_timestamp=datetime.now(),
            detected_type=DocumentType.ESTATUTO
        )

        self.collection.add_document(doc)
        self.assertEqual(len(self.collection.documents), 1)

    def test_get_documents_by_type(self):
        """Test filtering documents by type"""
        doc1 = UploadedDocument(
            file_path=Path("/test/estatuto.pdf"),
            file_name="estatuto.pdf",
            file_format=FileFormat.PDF,
            file_size_bytes=100000,
            upload_timestamp=datetime.now(),
            detected_type=DocumentType.ESTATUTO
        )

        doc2 = UploadedDocument(
            file_path=Path("/test/acta.pdf"),
            file_name="acta.pdf",
            file_format=FileFormat.PDF,
            file_size_bytes=50000,
            upload_timestamp=datetime.now(),
            detected_type=DocumentType.ACTA_DIRECTORIO
        )

        doc3 = UploadedDocument(
            file_path=Path("/test/estatuto2.pdf"),
            file_name="estatuto2.pdf",
            file_format=FileFormat.PDF,
            file_size_bytes=100000,
            upload_timestamp=datetime.now(),
            detected_type=DocumentType.ESTATUTO
        )

        self.collection.add_document(doc1)
        self.collection.add_document(doc2)
        self.collection.add_document(doc3)

        estatutos = self.collection.get_documents_by_type(DocumentType.ESTATUTO)
        self.assertEqual(len(estatutos), 2)

        actas = self.collection.get_documents_by_type(DocumentType.ACTA_DIRECTORIO)
        self.assertEqual(len(actas), 1)

    def test_get_missing_documents(self):
        """Test getting missing documents"""
        # Initially all documents are missing
        missing = self.collection.get_missing_documents()
        self.assertGreater(len(missing), 0)

        # Add estatuto
        doc = UploadedDocument(
            file_path=Path("/test/estatuto.pdf"),
            file_name="estatuto.pdf",
            file_format=FileFormat.PDF,
            file_size_bytes=100000,
            upload_timestamp=datetime.now(),
            detected_type=DocumentType.ESTATUTO
        )
        self.collection.add_document(doc)

        # Check if estatuto is no longer missing
        missing_after = self.collection.get_missing_documents()
        self.assertNotIn(DocumentType.ESTATUTO, missing_after)

    def test_get_coverage_summary(self):
        """Test coverage summary calculation"""
        # Initially 0% coverage
        summary = self.collection.get_coverage_summary()
        self.assertEqual(summary['coverage_percentage'], 0.0)

        # Add required documents
        required_types = [req.document_type for req in self.requirements.required_documents if req.mandatory]

        for doc_type in required_types[:2]:  # Add first 2 documents
            doc = UploadedDocument(
                file_path=Path(f"/test/{doc_type.value}.pdf"),
                file_name=f"{doc_type.value}.pdf",
                file_format=FileFormat.PDF,
                file_size_bytes=100000,
                upload_timestamp=datetime.now(),
                detected_type=doc_type
            )
            self.collection.add_document(doc)

        summary = self.collection.get_coverage_summary()
        self.assertGreater(summary['coverage_percentage'], 0)
        self.assertEqual(summary['present'], 2)

    def test_to_dict(self):
        """Test conversion to dictionary"""
        result = self.collection.to_dict()

        self.assertIn('certificate_intent', result)
        self.assertIn('legal_requirements', result)
        self.assertIn('documents', result)
        self.assertIn('coverage_summary', result)

    def test_to_json(self):
        """Test JSON conversion"""
        json_str = self.collection.to_json()
        self.assertIn('GIRTEC S.A.', json_str)

    def test_get_summary(self):
        """Test summary generation"""
        summary = self.collection.get_summary()
        self.assertIn('GIRTEC S.A.', summary)
        self.assertIn('COLECCIÓN DE DOCUMENTOS', summary)


class TestDocumentIntake(unittest.TestCase):
    """Test DocumentIntake service"""

    def setUp(self):
        """Set up test data"""
        self.intent = CertificateIntent(
            certificate_type=CertificateType.CERTIFICADO_PERSONERIA,
            purpose=Purpose.BPS,
            subject_name="GIRTEC S.A.",
            subject_type="company"
        )
        self.requirements = LegalRequirementsEngine.resolve_requirements(self.intent)

    def test_create_collection(self):
        """Test creating a collection"""
        collection = DocumentIntake.create_collection(self.intent, self.requirements)

        self.assertIsInstance(collection, DocumentCollection)
        self.assertEqual(collection.certificate_intent, self.intent)
        self.assertEqual(len(collection.documents), 0)

    def test_process_file_real(self):
        """Test processing a real temporary file"""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.pdf', prefix='estatuto_') as f:
            temp_path = f.name
            f.write("test content")

        try:
            doc = DocumentIntake.process_file(temp_path)

            self.assertEqual(doc.file_format, FileFormat.PDF)
            self.assertGreater(doc.file_size_bytes, 0)
            self.assertEqual(doc.detected_type, DocumentType.ESTATUTO)
            self.assertFalse(doc.is_scanned)

        finally:
            os.remove(temp_path)

    def test_process_file_not_found(self):
        """Test processing non-existent file"""
        with self.assertRaises(FileNotFoundError):
            DocumentIntake.process_file("/nonexistent/file.pdf")

    def test_add_files_to_collection(self):
        """Test adding multiple files to collection"""
        collection = DocumentIntake.create_collection(self.intent, self.requirements)

        # Create temporary files
        temp_files = []
        try:
            for name in ['estatuto.pdf', 'acta.pdf', 'bps.pdf']:
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.pdf', prefix=name.replace('.pdf', '_')) as f:
                    temp_files.append(f.name)
                    f.write("test")

            collection = DocumentIntake.add_files_to_collection(collection, temp_files)

            self.assertEqual(len(collection.documents), 3)

        finally:
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)

    def test_save_and_load_collection(self):
        """Test saving and loading collection"""
        collection = DocumentIntake.create_collection(self.intent, self.requirements)

        # Add a document
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.pdf', prefix='estatuto_') as f:
            temp_doc_path = f.name
            f.write("test")

        try:
            doc = DocumentIntake.process_file(temp_doc_path)
            collection.add_document(doc)

            # Save collection
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
                temp_json_path = f.name

            try:
                DocumentIntake.save_collection(collection, temp_json_path)

                # Verify file exists
                self.assertTrue(os.path.exists(temp_json_path))

                # Load collection
                loaded = DocumentIntake.load_collection(temp_json_path)

                self.assertEqual(len(loaded.documents), 1)
                self.assertEqual(loaded.certificate_intent.subject_name, "GIRTEC S.A.")

            finally:
                if os.path.exists(temp_json_path):
                    os.remove(temp_json_path)

        finally:
            if os.path.exists(temp_doc_path):
                os.remove(temp_doc_path)


class TestRealWorldScenarios(unittest.TestCase):
    """Test real-world scenarios"""

    def test_girtec_bps_workflow(self):
        """Test complete workflow for GIRTEC BPS"""
        # Phase 1: Intent
        intent = CertificateIntent(
            certificate_type=CertificateType.CERTIFICADO_PERSONERIA,
            purpose=Purpose.BPS,
            subject_name="GIRTEC S.A.",
            subject_type="company"
        )

        # Phase 2: Requirements
        requirements = LegalRequirementsEngine.resolve_requirements(intent)

        # Phase 3: Collection
        collection = DocumentIntake.create_collection(intent, requirements)

        # Verify workflow
        self.assertEqual(collection.certificate_intent.subject_name, "GIRTEC S.A.")
        self.assertGreater(len(requirements.required_documents), 0)

    def test_document_type_detection_real_patterns(self):
        """Test document type detection with real filename patterns"""
        test_cases = [
            ("GIRTEC S.A., CERTIFICACION PERSONERIA, REPRESENTACION, PODERES.doc", None),  # Complex, hard to detect
            ("Certificado - SISA -BASE DE DATOS.doc", None),  # Certificate, not source doc
            ("ESTATUTO GIRTEC SA.pdf", DocumentType.ESTATUTO),
            ("Acta Directorio 2023-05-15.pdf", DocumentType.ACTA_DIRECTORIO),
            ("Certificado BPS vigente.pdf", DocumentType.CERTIFICADO_BPS),
            ("CI_Director.jpg", DocumentType.CEDULA_IDENTIDAD),
            ("Poder General Carolina Bomio.doc", DocumentType.PODER),
        ]

        for filename, expected_type in test_cases:
            detected = DocumentTypeDetector.detect_from_filename(filename)
            if expected_type is not None:
                self.assertEqual(detected, expected_type, f"Failed for: {filename}")


if __name__ == '__main__':
    unittest.main()
