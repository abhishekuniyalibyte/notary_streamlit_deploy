"""
Unit tests for Phase 2: Legal Requirement Resolution
"""

import unittest
from src.phase1_certificate_intent import CertificateType, Purpose, CertificateIntent
from src.phase2_legal_requirements import (
    ArticleReference,
    RequiredElement,
    DocumentType,
    DocumentRequirement,
    InstitutionRule,
    LegalRequirements,
    LegalRequirementsEngine
)


class TestArticleReference(unittest.TestCase):
    """Test ArticleReference enum"""

    def test_article_values(self):
        """Test that article references have correct values"""
        self.assertEqual(ArticleReference.ART_130.value, "130")
        self.assertEqual(ArticleReference.ART_248.value, "248")
        self.assertEqual(ArticleReference.ART_255.value, "255")


class TestDocumentRequirement(unittest.TestCase):
    """Test DocumentRequirement dataclass"""

    def test_document_requirement_creation(self):
        """Test creating a document requirement"""
        req = DocumentRequirement(
            document_type=DocumentType.ESTATUTO,
            description="Estatuto social",
            mandatory=True,
            expires=False,
            legal_basis="Art. 248"
        )

        self.assertEqual(req.document_type, DocumentType.ESTATUTO)
        self.assertEqual(req.description, "Estatuto social")
        self.assertTrue(req.mandatory)
        self.assertFalse(req.expires)
        self.assertEqual(req.legal_basis, "Art. 248")

    def test_to_dict(self):
        """Test conversion to dictionary"""
        req = DocumentRequirement(
            document_type=DocumentType.CERTIFICADO_BPS,
            description="Certificado BPS",
            mandatory=True,
            expires=True,
            expiry_days=30,
            institution_specific="BPS"
        )

        result = req.to_dict()

        self.assertEqual(result["document_type"], "certificado_bps")
        self.assertEqual(result["expiry_days"], 30)
        self.assertEqual(result["institution_specific"], "BPS")


class TestInstitutionRule(unittest.TestCase):
    """Test InstitutionRule dataclass"""

    def test_institution_rule_creation(self):
        """Test creating institution rules"""
        rule = InstitutionRule(
            institution="BPS",
            validity_days=30,
            special_requirements=["Aportes al día", "Padrón actualizado"]
        )

        self.assertEqual(rule.institution, "BPS")
        self.assertEqual(rule.validity_days, 30)
        self.assertEqual(len(rule.special_requirements), 2)

    def test_to_dict(self):
        """Test conversion to dictionary"""
        rule = InstitutionRule(
            institution="Abitab",
            validity_days=30,
            special_requirements=["Test requirement"]
        )

        result = rule.to_dict()

        self.assertEqual(result["institution"], "Abitab")
        self.assertEqual(result["validity_days"], 30)


class TestLegalRequirements(unittest.TestCase):
    """Test LegalRequirements dataclass"""

    def setUp(self):
        """Set up test data"""
        self.requirements = LegalRequirements(
            certificate_type=CertificateType.CERTIFICADO_PERSONERIA,
            purpose=Purpose.BPS,
            mandatory_articles=[ArticleReference.ART_248, ArticleReference.ART_255],
            cross_references=[ArticleReference.ART_130],
            required_elements=[RequiredElement.IDENTITY_VERIFICATION],
            required_documents=[
                DocumentRequirement(
                    document_type=DocumentType.ESTATUTO,
                    description="Estatuto",
                    mandatory=True
                )
            ]
        )

    def test_to_dict(self):
        """Test conversion to dictionary"""
        result = self.requirements.to_dict()

        self.assertEqual(result["certificate_type"], "certificado_de_personeria")
        self.assertEqual(result["purpose"], "para_bps")
        self.assertIn("248", result["mandatory_articles"])
        self.assertIn("130", result["cross_references"])

    def test_to_json(self):
        """Test JSON conversion"""
        json_str = self.requirements.to_json()
        self.assertIn("certificado_de_personeria", json_str)
        self.assertIn("para_bps", json_str)

    def test_get_summary(self):
        """Test summary generation"""
        summary = self.requirements.get_summary()
        self.assertIn("REQUISITOS LEGALES", summary)
        self.assertIn("BPS", summary)


class TestLegalRequirementsEngine(unittest.TestCase):
    """Test LegalRequirementsEngine"""

    def test_base_articles(self):
        """Test that base articles are defined"""
        self.assertGreater(len(LegalRequirementsEngine.BASE_ARTICLES), 0)
        self.assertIn(ArticleReference.ART_248, LegalRequirementsEngine.BASE_ARTICLES)
        self.assertIn(ArticleReference.ART_255, LegalRequirementsEngine.BASE_ARTICLES)

    def test_get_base_requirements(self):
        """Test getting base requirements"""
        base_req = LegalRequirementsEngine._get_base_requirements()
        self.assertGreater(len(base_req), 0)
        self.assertIn(RequiredElement.IDENTITY_VERIFICATION, base_req)

    def test_get_personeria_documents(self):
        """Test getting personería documents"""
        docs = LegalRequirementsEngine._get_personeria_documents()
        self.assertGreater(len(docs), 0)

        # Check that estatuto is required
        estatuto_found = any(doc.document_type == DocumentType.ESTATUTO for doc in docs)
        self.assertTrue(estatuto_found)

    def test_get_firma_documents(self):
        """Test getting signature certification documents"""
        docs = LegalRequirementsEngine._get_firma_documents()
        self.assertGreater(len(docs), 0)

        # Check that cedula is required
        cedula_found = any(doc.document_type == DocumentType.CEDULA_IDENTIDAD for doc in docs)
        self.assertTrue(cedula_found)

    def test_get_institution_rules_bps(self):
        """Test getting BPS institution rules"""
        rules = LegalRequirementsEngine._get_institution_rules(Purpose.BPS)

        self.assertIsNotNone(rules)
        self.assertEqual(rules.institution, "BPS")
        self.assertEqual(rules.validity_days, 30)
        self.assertGreater(len(rules.additional_documents), 0)

    def test_get_institution_rules_abitab(self):
        """Test getting Abitab institution rules"""
        rules = LegalRequirementsEngine._get_institution_rules(Purpose.ABITAB)

        self.assertIsNotNone(rules)
        self.assertEqual(rules.institution, "Abitab")
        self.assertEqual(rules.validity_days, 30)

    def test_get_institution_rules_zona_franca(self):
        """Test getting Zona Franca institution rules"""
        rules = LegalRequirementsEngine._get_institution_rules(Purpose.ZONA_FRANCA)

        self.assertIsNotNone(rules)
        self.assertEqual(rules.institution, "Zona Franca")

    def test_resolve_requirements_personeria_bps(self):
        """Test resolving requirements for personería BPS certificate"""
        intent = CertificateIntent(
            certificate_type=CertificateType.CERTIFICADO_PERSONERIA,
            purpose=Purpose.BPS,
            subject_name="GIRTEC S.A.",
            subject_type="company"
        )

        requirements = LegalRequirementsEngine.resolve_requirements(intent)

        # Check basic properties
        self.assertEqual(requirements.certificate_type, CertificateType.CERTIFICADO_PERSONERIA)
        self.assertEqual(requirements.purpose, Purpose.BPS)

        # Check articles
        self.assertGreater(len(requirements.mandatory_articles), 0)
        self.assertIn(ArticleReference.ART_248, requirements.mandatory_articles)
        self.assertIn(ArticleReference.ART_255, requirements.mandatory_articles)

        # Check required documents
        self.assertGreater(len(requirements.required_documents), 0)

        # Check BPS-specific documents
        bps_docs = [doc for doc in requirements.required_documents
                    if doc.institution_specific == "BPS"]
        self.assertGreater(len(bps_docs), 0)

        # Check institution rules
        self.assertIsNotNone(requirements.institution_rules)
        self.assertEqual(requirements.institution_rules.institution, "BPS")

    def test_resolve_requirements_firma_abitab(self):
        """Test resolving requirements for signature certification for Abitab"""
        intent = CertificateIntent(
            certificate_type=CertificateType.CERTIFICACION_FIRMAS,
            purpose=Purpose.ABITAB,
            subject_name="INVERSORA RINLEN S.A.",
            subject_type="company"
        )

        requirements = LegalRequirementsEngine.resolve_requirements(intent)

        # Check certificate type
        self.assertEqual(requirements.certificate_type, CertificateType.CERTIFICACION_FIRMAS)

        # Check that Art. 250 (signature) is included
        self.assertIn(ArticleReference.ART_250, requirements.mandatory_articles)

        # Check cross-reference to Art. 130 (identification)
        self.assertIn(ArticleReference.ART_130, requirements.cross_references)

        # Check signature presence requirement
        self.assertIn(RequiredElement.SIGNATURE_PRESENCE, requirements.required_elements)

    def test_resolve_requirements_poder_banco(self):
        """Test resolving requirements for poder general for bank"""
        intent = CertificateIntent(
            certificate_type=CertificateType.PODER_GENERAL,
            purpose=Purpose.BANCO,
            subject_name="GIRTEC S.A.",
            subject_type="company"
        )

        requirements = LegalRequirementsEngine.resolve_requirements(intent)

        # Check certificate type
        self.assertEqual(requirements.certificate_type, CertificateType.PODER_GENERAL)

        # Check power of attorney requirement
        self.assertIn(RequiredElement.POWER_OF_ATTORNEY, requirements.required_elements)

    def test_get_all_applicable_articles(self):
        """Test getting all applicable articles"""
        intent = CertificateIntent(
            certificate_type=CertificateType.CERTIFICACION_FIRMAS,
            purpose=Purpose.BPS,
            subject_name="Test",
            subject_type="company"
        )

        requirements = LegalRequirementsEngine.resolve_requirements(intent)
        all_articles = LegalRequirementsEngine.get_all_applicable_articles(requirements)

        # Should include both mandatory and cross-referenced articles
        self.assertGreater(len(all_articles), 0)
        self.assertIsInstance(all_articles, set)


class TestRealWorldScenarios(unittest.TestCase):
    """Test real-world scenarios from client data"""

    def test_girtec_bps_complete(self):
        """Test complete GIRTEC BPS certificate requirements"""
        intent = CertificateIntent(
            certificate_type=CertificateType.CERTIFICADO_PERSONERIA,
            purpose=Purpose.BPS,
            subject_name="GIRTEC S.A.",
            subject_type="company"
        )

        requirements = LegalRequirementsEngine.resolve_requirements(intent)

        # Should have BPS-specific requirements
        self.assertIsNotNone(requirements.institution_rules)
        self.assertEqual(requirements.institution_rules.validity_days, 30)

        # Should require BPS certificate
        has_bps_cert = any(
            doc.document_type == DocumentType.CERTIFICADO_BPS
            for doc in requirements.required_documents
        )
        self.assertTrue(has_bps_cert)

    def test_netkla_zona_franca(self):
        """Test NETKLA Zona Franca requirements"""
        intent = CertificateIntent(
            certificate_type=CertificateType.CERTIFICADO_PERSONERIA,
            purpose=Purpose.ZONA_FRANCA,
            subject_name="NETKLA TRADING S.A.",
            subject_type="company"
        )

        requirements = LegalRequirementsEngine.resolve_requirements(intent)

        # Should have Zona Franca specific rules
        self.assertIsNotNone(requirements.institution_rules)
        self.assertEqual(requirements.institution_rules.institution, "Zona Franca")

    def test_saterix_base_datos(self):
        """Test SATERIX Base de Datos requirements"""
        intent = CertificateIntent(
            certificate_type=CertificateType.CERTIFICADO_PERSONERIA,
            purpose=Purpose.BASE_DATOS,
            subject_name="SATERIX S.A.",
            subject_type="company"
        )

        requirements = LegalRequirementsEngine.resolve_requirements(intent)

        # Should mention Law 18930 (data protection)
        self.assertIsNotNone(requirements.institution_rules)
        has_law_18930 = any(
            "18930" in req
            for req in requirements.institution_rules.special_requirements
        )
        self.assertTrue(has_law_18930)

    def test_girtec_rupe(self):
        """Test GIRTEC RUPE requirements"""
        intent = CertificateIntent(
            certificate_type=CertificateType.CERTIFICADO_PERSONERIA,
            purpose=Purpose.RUPE,
            subject_name="GIRTEC S.A.",
            subject_type="company"
        )

        requirements = LegalRequirementsEngine.resolve_requirements(intent)

        # RUPE should have 180-day validity
        self.assertIsNotNone(requirements.institution_rules)
        self.assertEqual(requirements.institution_rules.validity_days, 180)

        # Should require both Law 18930 and 17904
        special_reqs = requirements.institution_rules.special_requirements
        has_18930 = any("18930" in req for req in special_reqs)
        has_17904 = any("17904" in req for req in special_reqs)
        self.assertTrue(has_18930)
        self.assertTrue(has_17904)


if __name__ == '__main__':
    unittest.main()
