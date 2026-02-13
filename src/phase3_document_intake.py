"""
Phase 3: Document Intake

This module handles document collection and indexing:
- Direct file uploads (PDF, DOCX, JPG)
- Indexing documents by client, type, date
- Detecting document types
- Organizing evidence for validation

This prepares documents for Phase 4 (text extraction).
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
from pathlib import Path
from datetime import datetime
import json
import mimetypes
import os
import re
import unicodedata
from enum import Enum

from src.phase1_certificate_intent import CertificateIntent
from src.phase2_legal_requirements import LegalRequirements, DocumentType


class FileFormat(Enum):
    """Supported file formats"""
    PDF = "pdf"
    DOCX = "docx"
    DOC = "doc"
    JPG = "jpg"
    JPEG = "jpeg"
    PNG = "png"
    TXT = "txt"
    UNKNOWN = "unknown"

    @classmethod
    def from_extension(cls, extension: str) -> 'FileFormat':
        """Get FileFormat from file extension"""
        ext = extension.lower().lstrip('.')
        for fmt in cls:
            if fmt.value == ext:
                return fmt
        return cls.UNKNOWN


class ProcessingStatus(Enum):
    """Status of document processing"""
    PENDING = "pending"
    INDEXED = "indexed"
    SCANNED = "scanned"
    DIGITAL = "digital"
    ERROR = "error"


@dataclass
class UploadedDocument:
    """Represents a single uploaded document"""
    file_path: Path
    file_name: str
    file_format: FileFormat
    file_size_bytes: int
    upload_timestamp: datetime
    processing_status: ProcessingStatus = ProcessingStatus.PENDING
    detected_type: Optional[DocumentType] = None
    is_scanned: bool = False
    metadata: Dict[str, any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "file_path": str(self.file_path),
            "file_name": self.file_name,
            "file_format": self.file_format.value,
            "file_size_bytes": self.file_size_bytes,
            "upload_timestamp": self.upload_timestamp.isoformat(),
            "processing_status": self.processing_status.value,
            "detected_type": self.detected_type.value if self.detected_type else None,
            "is_scanned": self.is_scanned,
            "metadata": self.metadata
        }

    def get_display_info(self) -> str:
        """Get display information about the document"""
        size_kb = self.file_size_bytes / 1024
        doc_type = self.detected_type.value if self.detected_type else "no detectado"
        scan_status = "Escaneado" if self.is_scanned else "Digital"

        return f"📄 {self.file_name} [{self.file_format.value.upper()}] ({size_kb:.1f} KB) - Tipo: {doc_type} - {scan_status}"


@dataclass
class DocumentCollection:
    """Collection of documents for a certificate request"""
    certificate_intent: CertificateIntent
    legal_requirements: LegalRequirements
    documents: List[UploadedDocument] = field(default_factory=list)
    collection_timestamp: datetime = field(default_factory=datetime.now)
    catalog_info: Dict[str, any] = field(default_factory=dict)
    catalog_entries: List[Dict[str, any]] = field(default_factory=list)

    def add_document(self, document: UploadedDocument) -> None:
        """Add a document to the collection"""
        self.documents.append(document)

    def get_documents_by_type(self, doc_type: DocumentType) -> List[UploadedDocument]:
        """Get all documents of a specific type"""
        return [doc for doc in self.documents if doc.detected_type == doc_type]

    def get_missing_documents(self) -> List[DocumentType]:
        """Get list of required documents that are missing"""
        present_types = {doc.detected_type for doc in self.documents if doc.detected_type}
        required_types = {req.document_type for req in self.legal_requirements.required_documents if req.mandatory}

        missing = []
        for req_type in required_types:
            if req_type not in present_types:
                missing.append(req_type)

        return missing

    def get_coverage_summary(self) -> Dict[str, any]:
        """Get summary of document coverage"""
        total_required = len([req for req in self.legal_requirements.required_documents if req.mandatory])
        missing_count = len(self.get_missing_documents())
        present_count = total_required - missing_count
        coverage_pct = (present_count / total_required * 100) if total_required > 0 else 0

        return {
            "total_required": total_required,
            "present": present_count,
            "missing": missing_count,
            "coverage_percentage": coverage_pct
        }

    def to_dict(self) -> dict:
        data = {
            "certificate_intent": self.certificate_intent.to_dict(),
            "legal_requirements": self.legal_requirements.to_dict(),
            "documents": [doc.to_dict() for doc in self.documents],
            "collection_timestamp": self.collection_timestamp.isoformat(),
            "coverage_summary": self.get_coverage_summary()
        }

        if self.catalog_info:
            data["catalog_info"] = self.catalog_info
        if self.catalog_entries:
            data["catalog_entries"] = self.catalog_entries

        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def get_summary(self) -> str:
        """Get human-readable summary in Spanish"""
        coverage = self.get_coverage_summary()

        summary = f"""
╔══════════════════════════════════════════════════════════════╗
║              COLECCIÓN DE DOCUMENTOS - FASE 3                ║
╚══════════════════════════════════════════════════════════════╝

👤 Sujeto: {self.certificate_intent.subject_name}
📋 Tipo: {self.certificate_intent.certificate_type.value.replace('_', ' ').title()}
🎯 Propósito: {self.certificate_intent.purpose.value.replace('para_', 'Para ').replace('_', ' ').title()}

📊 COBERTURA DE DOCUMENTOS:
   Total requeridos: {coverage['total_required']}
   Presentes: {coverage['present']}
   Faltantes: {coverage['missing']}
   Cobertura: {coverage['coverage_percentage']:.1f}%

📁 DOCUMENTOS CARGADOS ({len(self.documents)} total):
"""
        if self.catalog_info:
            catalog_matches = sum(
                1 for doc in self.documents if doc.metadata.get("catalog")
            )
            summary += (
                f"\n📎 CATALOGO CLIENTE: {self.catalog_info.get('source_file', 'catalogo')}\n"
                f"   Entradas: {len(self.catalog_entries)} | Coincidencias: {catalog_matches}\n"
            )
        for doc in self.documents:
            summary += f"   {doc.get_display_info()}\n"

        missing = self.get_missing_documents()
        if missing:
            summary += f"\n⚠️  DOCUMENTOS FALTANTES ({len(missing)}):\n"
            for doc_type in missing:
                # Find the requirement details
                req = next((r for r in self.legal_requirements.required_documents
                           if r.document_type == doc_type), None)
                if req:
                    summary += f"   ❌ {req.description}\n"

        return summary


class DocumentTypeDetector:
    """
    Detects document types using simple keyword heuristics.

    - Phase 3: filename-based detection (fast, best-effort)
    - Phase 4+: content-based detection fallback (works even when users name files "1.pdf")
    """

    # Keyword patterns for document type detection
    PATTERNS = {
        DocumentType.CEDULA_IDENTIDAD: ["cedula", "ci", "identidad", "documento"],
        DocumentType.ESTATUTO: ["estatuto", "estatutos"],
        DocumentType.ACTA_DIRECTORIO: ["acta", "directorio", "asamblea"],
        DocumentType.CERTIFICADO_BPS: ["bps", "prevision"],
        DocumentType.CERTIFICADO_DGI: ["dgi", "tributaria", "impositiva"],
        DocumentType.PODER: ["poder", "apoderado"],
        DocumentType.REGISTRO_COMERCIO: ["registro", "comercio", "rnc"],
        DocumentType.PADRON_BPS: ["padron"],
        DocumentType.CERTIFICADO_VIGENCIA: ["vigencia"],
        DocumentType.CONTRATO_SOCIAL: ["contrato social"],
        DocumentType.BALANCE: ["balance", "estado financiero"],
        DocumentType.DECLARACION_JURADA: ["declaracion jurada", "ddjj"]
    }

    @staticmethod
    def _normalize_for_match(value: str) -> str:
        if not value:
            return ""
        normalized = unicodedata.normalize("NFKD", value)
        normalized = normalized.encode("ascii", "ignore").decode("ascii")
        normalized = normalized.lower()
        normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
        return " ".join(normalized.split())

    @staticmethod
    def _score_keyword_matches(text_norm: str) -> Dict[DocumentType, int]:
        if not text_norm:
            return {}
        tokens = set(text_norm.split())

        scores: Dict[DocumentType, int] = {}
        for doc_type, keywords in DocumentTypeDetector.PATTERNS.items():
            score = 0
            for keyword in keywords:
                keyword_norm = DocumentTypeDetector._normalize_for_match(keyword)
                if not keyword_norm:
                    continue
                if " " in keyword_norm:
                    if keyword_norm in text_norm:
                        score += len(keyword_norm.replace(" ", ""))  # Longer phrase = higher score
                else:
                    if keyword_norm in tokens:
                        score += len(keyword_norm)  # Longer token = higher score
            if score > 0:
                scores[doc_type] = score
        return scores

    @staticmethod
    def detect_from_filename(filename: str) -> Optional[DocumentType]:
        """
        Detect document type from filename using keyword matching.

        This is a simple implementation. Phase 4 will use actual content analysis.
        """
        text_norm = DocumentTypeDetector._normalize_for_match(filename)
        scores = DocumentTypeDetector._score_keyword_matches(text_norm)
        if scores:
            return max(scores.items(), key=lambda item: item[1])[0]

        return None

    @staticmethod
    def detect_from_text(text: str) -> Optional[DocumentType]:
        """
        Detect document type from extracted text using keyword matching.

        Used as a fallback when filename-based detection fails (e.g., "1.pdf").
        """
        text_norm = DocumentTypeDetector._normalize_for_match(text)
        scores = DocumentTypeDetector._score_keyword_matches(text_norm)
        if scores:
            return max(scores.items(), key=lambda item: item[1])[0]
        return None

    @staticmethod
    def is_likely_scanned(file_format: FileFormat) -> bool:
        """Determine if file is likely scanned (will be refined in Phase 4)"""
        # Images are likely scanned
        return file_format in [FileFormat.JPG, FileFormat.JPEG, FileFormat.PNG]


class DocumentIntake:
    """
    Service class for document intake operations.
    Handles uploading, indexing, and organizing documents.
    """

    @staticmethod
    def _normalize_catalog_key(value: str) -> str:
        if not value:
            return ""
        normalized = unicodedata.normalize("NFKD", value)
        normalized = normalized.encode("ascii", "ignore").decode("ascii")
        normalized = normalized.lower()
        normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
        return " ".join(normalized.split())

    @staticmethod
    def _catalog_entry_keys(entry: Dict[str, any]) -> List[str]:
        keys = []
        for key in ("matched_filename", "raw_name", "matched_path"):
            value = entry.get(key)
            if value:
                keys.append(os.path.basename(value))
        candidates = entry.get("candidate_filenames") or []
        keys.extend(candidates)
        return keys

    @staticmethod
    def load_client_catalog(
        catalog_path: str,
        customer_name: str
    ) -> Tuple[Dict[str, any], List[Dict[str, any]]]:
        if not catalog_path or not customer_name:
            return {}, []
        path = Path(catalog_path)
        if not path.exists():
            return {}, []

        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        customer_entry = data.get(customer_name)
        if not customer_entry:
            return {}, []

        entries = customer_entry.get("entries", [])
        info = {
            "customer": customer_name,
            "source_file": customer_entry.get("source_file"),
            "entry_count": customer_entry.get("entry_count", len(entries)),
            "catalog_path": str(path)
        }
        return info, entries

    @staticmethod
    def match_catalog_entry(
        filename: str,
        catalog_entries: List[Dict[str, any]]
    ) -> Optional[Dict[str, any]]:
        if not filename or not catalog_entries:
            return None

        normalized_filename = DocumentIntake._normalize_catalog_key(filename)
        normalized_stem = DocumentIntake._normalize_catalog_key(Path(filename).stem)

        for entry in catalog_entries:
            for key in DocumentIntake._catalog_entry_keys(entry):
                if DocumentIntake._normalize_catalog_key(key) == normalized_filename:
                    return entry

        for entry in catalog_entries:
            for key in DocumentIntake._catalog_entry_keys(entry):
                key_norm = DocumentIntake._normalize_catalog_key(Path(key).stem)
                if key_norm == normalized_stem and key_norm:
                    return entry

        return None

    @staticmethod
    def create_collection(
        intent: CertificateIntent,
        requirements: LegalRequirements,
        catalog_path: Optional[str] = None,
        catalog_customer: Optional[str] = None
    ) -> DocumentCollection:
        """Create a new document collection"""
        collection = DocumentCollection(
            certificate_intent=intent,
            legal_requirements=requirements
        )
        if catalog_path and catalog_customer:
            catalog_info, catalog_entries = DocumentIntake.load_client_catalog(
                catalog_path,
                catalog_customer
            )
            collection.catalog_info = catalog_info
            collection.catalog_entries = catalog_entries
        return collection

    @staticmethod
    def process_file(
        file_path: str,
        catalog_entries: Optional[List[Dict[str, any]]] = None,
        catalog_info: Optional[Dict[str, any]] = None,
        display_name: Optional[str] = None
    ) -> UploadedDocument:
        """
        Process a single file and create an UploadedDocument object.

        Args:
            file_path: Path to the file
            catalog_entries: Optional catalog entries for filename matching
            catalog_info: Optional catalog metadata for the customer
            display_name: Optional original filename for matching and type detection

        Returns:
            UploadedDocument object
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Get file info
        file_name = display_name or path.name
        file_size = path.stat().st_size
        file_extension = path.suffix
        file_format = FileFormat.from_extension(file_extension)

        # Detect document type from filename
        detected_type = DocumentTypeDetector.detect_from_filename(file_name)

        # Check if likely scanned
        is_scanned = DocumentTypeDetector.is_likely_scanned(file_format)

        # Get modification time as proxy for upload time
        upload_time = datetime.fromtimestamp(path.stat().st_mtime)

        # Determine processing status
        if file_format == FileFormat.UNKNOWN:
            status = ProcessingStatus.ERROR
        else:
            status = ProcessingStatus.INDEXED

        metadata = {
            "original_path": str(path),
            "mime_type": mimetypes.guess_type(str(path))[0]
        }
        if detected_type:
            metadata["detected_type_source"] = "filename"
        if display_name:
            metadata["original_filename"] = display_name
        catalog_entry = DocumentIntake.match_catalog_entry(
            file_name,
            catalog_entries or []
        )
        if catalog_entry:
            metadata["catalog"] = {
                "customer": (catalog_info or {}).get("customer"),
                "source_file": (catalog_info or {}).get("source_file"),
                "raw_name": catalog_entry.get("raw_name"),
                "description": catalog_entry.get("description"),
                "type_raw": catalog_entry.get("type_raw"),
                "expected_extensions": catalog_entry.get("expected_extensions", []),
                "match_status": catalog_entry.get("match_status"),
                "match_method": catalog_entry.get("match_method"),
                "match_score": catalog_entry.get("match_score"),
                "matched_filename": catalog_entry.get("matched_filename"),
                "matched_path": catalog_entry.get("matched_path"),
                "type_mismatch": catalog_entry.get("type_mismatch"),
            }

        return UploadedDocument(
            file_path=path,
            file_name=file_name,
            file_format=file_format,
            file_size_bytes=file_size,
            upload_timestamp=upload_time,
            processing_status=status,
            detected_type=detected_type,
            is_scanned=is_scanned,
            metadata=metadata
        )

    @staticmethod
    def add_files_to_collection(
        collection: DocumentCollection,
        file_paths: List[str],
        file_name_overrides: Optional[Dict[str, str]] = None
    ) -> DocumentCollection:
        """
        Add multiple files to a document collection.

        Args:
            collection: DocumentCollection to add files to
            file_paths: List of file paths to add
            file_name_overrides: Optional map of file path to original filename

        Returns:
            Updated DocumentCollection
        """
        for file_path in file_paths:
            try:
                display_name = None
                if file_name_overrides:
                    display_name = file_name_overrides.get(file_path)
                document = DocumentIntake.process_file(
                    file_path,
                    collection.catalog_entries,
                    collection.catalog_info,
                    display_name
                )
                collection.add_document(document)
            except Exception as e:
                print(f"⚠️  Error procesando {file_path}: {str(e)}")

        return collection

    @staticmethod
    def scan_directory_for_client(
        directory_path: str,
        client_name: str,
        collection: DocumentCollection
    ) -> DocumentCollection:
        """
        Scan a directory for documents related to a specific client.

        Args:
            directory_path: Path to directory to scan
            client_name: Name of client to filter for
            collection: DocumentCollection to add files to

        Returns:
            Updated DocumentCollection
        """
        dir_path = Path(directory_path)

        if not dir_path.exists() or not dir_path.is_dir():
            raise ValueError(f"Directory not found: {directory_path}")

        # Supported extensions
        supported_extensions = ['.pdf', '.docx', '.doc', '.jpg', '.jpeg', '.png']

        # Find all files
        found_files = []
        for ext in supported_extensions:
            found_files.extend(dir_path.glob(f"**/*{ext}"))

        print(f"\n📂 Escaneando directorio: {directory_path}")
        print(f"   Cliente: {client_name}")
        print(f"   Archivos encontrados: {len(found_files)}")

        # Process files
        for file_path in found_files:
            try:
                document = DocumentIntake.process_file(
                    str(file_path),
                    collection.catalog_entries,
                    collection.catalog_info
                )
                collection.add_document(document)
            except Exception as e:
                print(f"⚠️  Error procesando {file_path}: {str(e)}")

        return collection

    @staticmethod
    def save_collection(collection: DocumentCollection, output_path: str) -> None:
        """Save document collection to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(collection.to_json())
        print(f"\n✅ Colección guardada en: {output_path}")

    @staticmethod
    def load_collection(input_path: str) -> DocumentCollection:
        """Load document collection from JSON file"""
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Reconstruct objects
        from src.phase1_certificate_intent import CertificateIntent
        from src.phase2_legal_requirements import LegalRequirements

        intent = CertificateIntent.from_dict(data['certificate_intent'])

        # Simplified loading - in production would need full reconstruction
        collection = DocumentCollection(
            certificate_intent=intent,
            legal_requirements=None,  # Would need to reconstruct
            collection_timestamp=datetime.fromisoformat(data['collection_timestamp']),
            catalog_info=data.get("catalog_info", {}),
            catalog_entries=data.get("catalog_entries", [])
        )

        # Add documents
        for doc_data in data['documents']:
            doc = UploadedDocument(
                file_path=Path(doc_data['file_path']),
                file_name=doc_data['file_name'],
                file_format=FileFormat(doc_data['file_format']),
                file_size_bytes=doc_data['file_size_bytes'],
                upload_timestamp=datetime.fromisoformat(doc_data['upload_timestamp']),
                processing_status=ProcessingStatus(doc_data['processing_status']),
                detected_type=DocumentType(doc_data['detected_type']) if doc_data['detected_type'] else None,
                is_scanned=doc_data['is_scanned'],
                metadata=doc_data['metadata']
            )
            collection.add_document(doc)

        return collection


def example_usage():
    """Example usage of Phase 3"""

    print("\n" + "="*70)
    print("  EJEMPLOS DE USO - FASE 3: INGESTA DE DOCUMENTOS")
    print("="*70)

    from src.phase1_certificate_intent import CertificateIntentCapture
    from src.phase2_legal_requirements import LegalRequirementsEngine

    # Example 1: Create collection for GIRTEC BPS certificate
    print("\n📌 Ejemplo 1: Crear colección para GIRTEC BPS")
    print("-" * 70)

    intent = CertificateIntentCapture.capture_intent_from_params(
        certificate_type="certificado_de_personeria",
        purpose="BPS",
        subject_name="GIRTEC S.A.",
        subject_type="company"
    )

    requirements = LegalRequirementsEngine.resolve_requirements(intent)
    collection = DocumentIntake.create_collection(intent, requirements)

    print(f"✅ Colección creada para: {intent.subject_name}")
    print(f"   Documentos requeridos: {len(requirements.required_documents)}")

    # Example 2: Scan client directory
    print("\n\n📌 Ejemplo 2: Escanear directorio de cliente GIRTEC")
    print("-" * 70)

    girtec_path = "/home/abhishek/Documents/NOTARY_5Jan/Notaria_client_data/Girtec"

    try:
        collection = DocumentIntake.scan_directory_for_client(
            directory_path=girtec_path,
            client_name="GIRTEC S.A.",
            collection=collection
        )

        print(collection.get_summary())

    except Exception as e:
        print(f"⚠️  No se pudo escanear directorio: {str(e)}")
        print("   (Esto es normal si el directorio no existe en el ejemplo)")

    # Example 3: Manual file processing
    print("\n\n📌 Ejemplo 3: Procesamiento manual de archivos")
    print("-" * 70)

    # Create a mock collection
    intent2 = CertificateIntentCapture.capture_intent_from_params(
        certificate_type="certificacion_de_firmas",
        purpose="Abitab",
        subject_name="NETKLA TRADING S.A.",
        subject_type="company"
    )

    requirements2 = LegalRequirementsEngine.resolve_requirements(intent2)
    collection2 = DocumentIntake.create_collection(intent2, requirements2)

    print(f"✅ Colección creada para: {intent2.subject_name}")
    print(f"   Cobertura inicial: {collection2.get_coverage_summary()['coverage_percentage']:.1f}%")

    # Example 4: Document type detection
    print("\n\n📌 Ejemplo 4: Detección de tipo de documento")
    print("-" * 70)

    test_filenames = [
        "estatuto_girtec.pdf",
        "acta_directorio_2023.pdf",
        "certificado_BPS.pdf",
        "cedula_identidad.jpg",
        "poder_general.docx"
    ]

    for filename in test_filenames:
        detected = DocumentTypeDetector.detect_from_filename(filename)
        print(f"   📄 {filename}")
        print(f"      → Tipo detectado: {detected.value if detected else 'No detectado'}")


if __name__ == "__main__":
    example_usage()
