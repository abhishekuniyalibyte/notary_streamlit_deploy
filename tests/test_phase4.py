"""
Unit tests for Phase 4: Text Extraction & Structuring
"""

import unittest
from pathlib import Path
from datetime import datetime

from src.phase4_text_extraction import (
    TextNormalizer,
    DataExtractor,
    ExtractedData,
    DocumentExtractionResult,
    CollectionExtractionResult
)
from src.phase3_document_intake import DocumentType, FileFormat, UploadedDocument


class TestTextNormalizer(unittest.TestCase):
    """Test TextNormalizer"""

    def test_fix_encoding_tildes(self):
        """Test fixing Spanish tildes"""
        text = "ResoluciÃ³n del Directorio"
        fixed = TextNormalizer.fix_encoding(text)
        self.assertEqual(fixed, "Resolución del Directorio")

    def test_fix_encoding_multiple(self):
        """Test fixing multiple encoding errors"""
        text = "SituaciÃ³n jurÃ­dica de la compaÃ±Ã­a"
        fixed = TextNormalizer.fix_encoding(text)
        self.assertEqual(fixed, "Situación jurídica de la compañía")

    def test_normalize_whitespace(self):
        """Test whitespace normalization"""
        text = "  Text   with    extra    spaces  "
        normalized = TextNormalizer.normalize_whitespace(text)
        self.assertEqual(normalized, "Text with extra spaces")

    def test_normalize_text_complete(self):
        """Test complete normalization"""
        text = "  ResoluciÃ³n   del   Directorio  "
        normalized = TextNormalizer.normalize_text(text)
        self.assertEqual(normalized, "Resolución del Directorio")


class TestDataExtractor(unittest.TestCase):
    """Test DataExtractor"""

    def test_extract_rut_standard(self):
        """Test extracting standard RUT format"""
        text = "RUT: 212345678901"
        rut = DataExtractor.extract_rut(text)
        self.assertEqual(rut, "212345678901")

    def test_extract_rut_with_spaces(self):
        """Test extracting RUT with spaces"""
        text = "RUT: 21 234 567 8901"
        rut = DataExtractor.extract_rut(text)
        self.assertEqual(rut, "212345678901")

    def test_extract_ci(self):
        """Test extracting CI"""
        text = "CI: 1.234.567-8"
        ci = DataExtractor.extract_ci(text)
        self.assertIsNotNone(ci)
        self.assertIn("1.234.567", ci)

    def test_extract_dates(self):
        """Test extracting dates"""
        text = "Fecha: 15/06/2023 y también 01-12-2024"
        dates = DataExtractor.extract_dates(text)
        self.assertGreater(len(dates), 0)
        self.assertIn("15/06/2023", dates)

    def test_extract_emails(self):
        """Test extracting emails"""
        text = "Contacto: info@girtec.com.uy y ventas@empresa.com"
        emails = DataExtractor.extract_emails(text)
        self.assertEqual(len(emails), 2)
        self.assertIn("info@girtec.com.uy", emails)

    def test_extract_registro_comercio(self):
        """Test extracting Registro de Comercio"""
        text = "Registro de Comercio N° 12345"
        registro = DataExtractor.extract_registro_comercio(text)
        self.assertEqual(registro, "12345")

    def test_extract_acta_number(self):
        """Test extracting Acta number"""
        text = "Acta N° 45 del Directorio"
        acta = DataExtractor.extract_acta_number(text)
        self.assertEqual(acta, "45")

    def test_extract_padron_bps(self):
        """Test extracting Padrón BPS"""
        text = "Padrón BPS Número 98765"
        padron = DataExtractor.extract_padron_bps(text)
        self.assertEqual(padron, "98765")

    def test_extract_company_name_sa(self):
        """Test extracting company name (S.A.)"""
        text = "GIRTEC SOCIEDAD ANÓNIMA inscrita en el Registro"
        company = DataExtractor.extract_company_name(text)
        self.assertIsNotNone(company)
        self.assertIn("GIRTEC", company)


class TestExtractedData(unittest.TestCase):
    """Test ExtractedData dataclass"""

    def test_create_extracted_data(self):
        """Test creating ExtractedData"""
        data = ExtractedData(
            document_type=DocumentType.ESTATUTO,
            raw_text="Raw text here",
            normalized_text="Normalized text here",
            company_name="GIRTEC S.A.",
            rut="212345678901"
        )

        self.assertEqual(data.document_type, DocumentType.ESTATUTO)
        self.assertEqual(data.company_name, "GIRTEC S.A.")
        self.assertEqual(data.rut, "212345678901")

    def test_to_dict(self):
        """Test conversion to dictionary"""
        data = ExtractedData(
            document_type=DocumentType.ESTATUTO,
            raw_text="Raw",
            normalized_text="Normalized",
            company_name="TEST S.A.",
            rut="123456789012"
        )

        result = data.to_dict()

        self.assertEqual(result["document_type"], "estatuto")
        self.assertEqual(result["company_name"], "TEST S.A.")
        self.assertEqual(result["rut"], "123456789012")

    def test_to_json(self):
        """Test JSON conversion"""
        data = ExtractedData(
            document_type=DocumentType.ESTATUTO,
            raw_text="Raw",
            normalized_text="Normalized"
        )

        json_str = data.to_json()
        self.assertIn("estatuto", json_str)

    def test_get_summary(self):
        """Test summary generation"""
        data = ExtractedData(
            document_type=DocumentType.ESTATUTO,
            raw_text="Raw",
            normalized_text="Normalized",
            company_name="GIRTEC S.A.",
            rut="212345678901"
        )

        summary = data.get_summary()
        self.assertIn("ESTATUTO", summary)
        self.assertIn("GIRTEC S.A.", summary)
        self.assertIn("212345678901", summary)


class TestDocumentExtractionResult(unittest.TestCase):
    """Test DocumentExtractionResult"""

    def test_successful_extraction(self):
        """Test successful extraction result"""
        doc = UploadedDocument(
            file_path=Path("/test/estatuto.pdf"),
            file_name="estatuto.pdf",
            file_format=FileFormat.PDF,
            file_size_bytes=1024,
            upload_timestamp=datetime.now(),
            detected_type=DocumentType.ESTATUTO
        )

        extracted_data = ExtractedData(
            document_type=DocumentType.ESTATUTO,
            raw_text="Raw",
            normalized_text="Normalized"
        )

        result = DocumentExtractionResult(
            document=doc,
            extracted_data=extracted_data,
            success=True
        )

        self.assertTrue(result.success)
        self.assertIsNotNone(result.extracted_data)
        self.assertIsNone(result.error)

    def test_failed_extraction(self):
        """Test failed extraction result"""
        doc = UploadedDocument(
            file_path=Path("/test/broken.pdf"),
            file_name="broken.pdf",
            file_format=FileFormat.PDF,
            file_size_bytes=1024,
            upload_timestamp=datetime.now()
        )

        result = DocumentExtractionResult(
            document=doc,
            error="File not found",
            success=False
        )

        self.assertFalse(result.success)
        self.assertIsNone(result.extracted_data)
        self.assertEqual(result.error, "File not found")

    def test_to_dict(self):
        """Test conversion to dictionary"""
        doc = UploadedDocument(
            file_path=Path("/test/estatuto.pdf"),
            file_name="estatuto.pdf",
            file_format=FileFormat.PDF,
            file_size_bytes=1024,
            upload_timestamp=datetime.now(),
            detected_type=DocumentType.ESTATUTO
        )

        result = DocumentExtractionResult(
            document=doc,
            success=True
        )

        result_dict = result.to_dict()

        self.assertEqual(result_dict["file_name"], "estatuto.pdf")
        self.assertEqual(result_dict["document_type"], "estatuto")
        self.assertTrue(result_dict["success"])


class TestRealWorldScenarios(unittest.TestCase):
    """Test real-world extraction scenarios"""

    def test_girtec_sample_text(self):
        """Test extraction from GIRTEC-like text"""
        sample_text = """
        GIRTEC SOCIEDAD ANÓNIMA
        RUT: 21 234 567 8901
        Registro de Comercio Nro. 12345
        Acta N° 45 del 15/06/2023
        Padrón BPS: 98765
        Email: contacto@girtec.com.uy
        """

        # Extract data
        company = DataExtractor.extract_company_name(sample_text)
        rut = DataExtractor.extract_rut(sample_text)
        registro = DataExtractor.extract_registro_comercio(sample_text)
        acta = DataExtractor.extract_acta_number(sample_text)
        padron = DataExtractor.extract_padron_bps(sample_text)
        emails = DataExtractor.extract_emails(sample_text)

        # Verify
        self.assertIsNotNone(company)
        self.assertIn("GIRTEC", company)
        self.assertEqual(rut, "212345678901")
        self.assertEqual(registro, "12345")
        self.assertEqual(acta, "45")
        self.assertEqual(padron, "98765")
        self.assertIn("contacto@girtec.com.uy", emails)

    def test_encoding_fix_real_example(self):
        """Test fixing real OCR encoding errors"""
        ocr_text = "La resoluciÃ³n del directorio de la compaÃ±Ã­a fue aprobada"
        fixed = TextNormalizer.fix_encoding(ocr_text)
        self.assertEqual(fixed, "La resolución del directorio de la compañía fue aprobada")


if __name__ == '__main__':
    unittest.main()
