"""
Phase 7: Data Update Attempt

This module handles:
- Attempting to fetch missing/outdated information
- Accepting manual document updates from notary
- Validating updated documents
- Tracking what was changed
- Preparing updated data for Phase 8 (final validation)

This is an OPTIONAL phase that reduces manual work by attempting
to auto-fetch public registry information or accepting updated uploads.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from datetime import datetime, timedelta
from enum import Enum
import json
import os
import re
import unicodedata

from src.phase1_certificate_intent import CertificateIntent
from src.phase2_legal_requirements import DocumentType, LegalRequirements
from src.phase3_document_intake import UploadedDocument, DocumentCollection, DocumentIntake
from src.phase4_text_extraction import TextExtractor, CollectionExtractionResult
from src.phase6_gap_detection import Gap, GapType, GapAnalysisReport, ActionPriority


class UpdateSource(Enum):
    """Source of the update"""
    MANUAL_UPLOAD = "manual_upload"  # Notary uploaded new document
    PUBLIC_REGISTRY = "public_registry"  # Fetched from online registry
    SYSTEM_CORRECTION = "system_correction"  # Auto-corrected by system
    NOT_UPDATED = "not_updated"  # No update attempted/successful


class UpdateStatus(Enum):
    """Status of update attempt"""
    SUCCESS = "success"
    FAILED = "failed"
    NOT_ATTEMPTED = "not_attempted"
    PARTIAL = "partial"  # Some data updated, some not


@dataclass
class DocumentUpdate:
    """
    Represents an update to a single document or data field.
    """
    document_type: DocumentType
    gap_addressed: Gap
    update_source: UpdateSource
    update_status: UpdateStatus
    timestamp: datetime = field(default_factory=datetime.now)

    # Before/after tracking
    previous_state: Optional[str] = None
    new_state: Optional[str] = None

    # New document info (if uploaded)
    new_document: Optional[UploadedDocument] = None

    # Fetched data (if from registry)
    fetched_data: Optional[Dict] = None

    # Error info (if failed)
    error_message: Optional[str] = None

    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "document_type": self.document_type.value,
            "gap_addressed": self.gap_addressed.to_dict(),
            "update_source": self.update_source.value,
            "update_status": self.update_status.value,
            "timestamp": self.timestamp.isoformat(),
            "previous_state": self.previous_state,
            "new_state": self.new_state,
            "new_document": self.new_document.to_dict() if self.new_document else None,
            "fetched_data": self.fetched_data,
            "error_message": self.error_message,
            "notes": self.notes
        }

    def get_display(self) -> str:
        """Get formatted display string"""
        status_icons = {
            UpdateStatus.SUCCESS: "✅",
            UpdateStatus.FAILED: "❌",
            UpdateStatus.PARTIAL: "⚠️",
            UpdateStatus.NOT_ATTEMPTED: "⏭️"
        }

        icon = status_icons.get(self.update_status, "❓")

        display = f"""
{icon} Update: {self.document_type.value.upper()}
   Gap: {self.gap_addressed.title}
   Source: {self.update_source.value}
   Status: {self.update_status.value}
   Timestamp: {self.timestamp.strftime('%Y-%m-%d %H:%M')}
"""

        if self.previous_state:
            display += f"   Before: {self.previous_state}\n"

        if self.new_state:
            display += f"   After: {self.new_state}\n"

        if self.new_document:
            display += f"   New Document: {self.new_document.filename}\n"

        if self.error_message:
            display += f"   Error: {self.error_message}\n"

        if self.notes:
            display += f"   Notes: {self.notes}\n"

        return display


@dataclass
class UpdateAttemptResult:
    """
    Results from attempting to update missing/outdated data.
    Contains the original gap report, all update attempts, and updated collection.
    """
    original_gap_report: GapAnalysisReport
    updates: List[DocumentUpdate] = field(default_factory=list)
    updated_collection: Optional[DocumentCollection] = None
    updated_extraction_result: Optional[CollectionExtractionResult] = None
    review_required: List[Dict[str, Any]] = field(default_factory=list)
    system_note: Optional[str] = None

    # Summary stats
    total_gaps: int = 0
    gaps_addressed: int = 0
    successful_updates: int = 0
    failed_updates: int = 0
    not_attempted: int = 0

    timestamp: datetime = field(default_factory=datetime.now)

    def calculate_summary(self):
        """Calculate summary statistics"""
        self.total_gaps = len(self.original_gap_report.gaps)

        for update in self.updates:
            if update.update_status == UpdateStatus.SUCCESS:
                self.successful_updates += 1
                self.gaps_addressed += 1
            elif update.update_status == UpdateStatus.FAILED:
                self.failed_updates += 1
            elif update.update_status == UpdateStatus.PARTIAL:
                self.gaps_addressed += 1
            elif update.update_status == UpdateStatus.NOT_ATTEMPTED:
                self.not_attempted += 1

    def to_dict(self) -> dict:
        return {
            "original_gap_report": self.original_gap_report.to_dict(),
            "updates": [u.to_dict() for u in self.updates],
            "updated_collection": self.updated_collection.to_dict() if self.updated_collection else None,
            "total_gaps": self.total_gaps,
            "gaps_addressed": self.gaps_addressed,
            "successful_updates": self.successful_updates,
            "failed_updates": self.failed_updates,
            "not_attempted": self.not_attempted,
            "review_required": self.review_required,
            "system_note": self.system_note,
            "timestamp": self.timestamp.isoformat()
        }

    def to_json(self) -> str:
        """Serialize to JSON"""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def get_summary(self) -> str:
        """Get formatted summary"""
        self.calculate_summary()

        border = "=" * 70

        summary = f"""
{border}
           FASE 7: RESULTADO DE ACTUALIZACIÓN DE DATOS
{border}

📊 RESUMEN DE ACTUALIZACIONES:
   Total de brechas detectadas: {self.total_gaps}
   Brechas atendidas: {self.gaps_addressed}
   Actualizaciones exitosas: {self.successful_updates} ✅
   Actualizaciones fallidas: {self.failed_updates} ❌
   No intentadas: {self.not_attempted} ⏭️

📁 ESTADO DE COLECCIÓN:
   Documentos antes: {len(self.original_gap_report.validation_matrix.document_validations)}
   Documentos después: {len(self.updated_collection.documents) if self.updated_collection else 0}

⏰ Timestamp: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

"""
        if self.system_note:
            summary += f"\nℹ️  {self.system_note}\n"

        if self.review_required:
            summary += "\n🔍 DOCUMENTOS PARA REVISIÓN MANUAL:\n"
            summary += "-" * 70 + "\n"
            for entry in self.review_required:
                filename = entry.get("filename", "documento")
                reasons = entry.get("reasons", []) or []
                summary += f"   • {filename}\n"
                for reason in reasons:
                    summary += f"     - {reason}\n"

        if self.successful_updates > 0:
            summary += "\n✅ ACTUALIZACIONES EXITOSAS:\n"
            summary += "-" * 70 + "\n"
            for update in self.updates:
                if update.update_status == UpdateStatus.SUCCESS:
                    summary += update.get_display() + "\n"

        if self.failed_updates > 0:
            summary += "\n❌ ACTUALIZACIONES FALLIDAS:\n"
            summary += "-" * 70 + "\n"
            for update in self.updates:
                if update.update_status == UpdateStatus.FAILED:
                    summary += update.get_display() + "\n"

        summary += "\n" + border + "\n"

        return summary

    def get_changes_report(self) -> str:
        """Get detailed report of what changed"""
        report = "\n" + "=" * 70 + "\n"
        report += "         REPORTE DETALLADO DE CAMBIOS - FASE 7\n"
        report += "=" * 70 + "\n\n"

        if not self.updates:
            report += "ℹ️  No se realizaron actualizaciones.\n"
            return report

        # Group by document type
        by_doc_type: Dict[DocumentType, List[DocumentUpdate]] = {}
        for update in self.updates:
            if update.document_type not in by_doc_type:
                by_doc_type[update.document_type] = []
            by_doc_type[update.document_type].append(update)

        for doc_type, updates_list in by_doc_type.items():
            report += f"\n📄 {doc_type.value.upper()}\n"
            report += "-" * 70 + "\n"

            for update in updates_list:
                report += update.get_display()

            report += "\n"

        return report


class DataUpdater:
    """
    Main class for Phase 7: Data Update Attempt
    """

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
    def _pick_most_common(values: List[str]) -> Optional[str]:
        cleaned = [v.strip() for v in values if v and v.strip()]
        if not cleaned:
            return None

        counts: Dict[str, int] = {}
        originals: Dict[str, List[str]] = {}
        for v in cleaned:
            key = DataUpdater._normalize_for_match(v)
            if not key:
                continue
            counts[key] = counts.get(key, 0) + 1
            originals.setdefault(key, []).append(v)

        if not counts:
            return cleaned[0]

        best_key = max(counts.items(), key=lambda kv: (kv[1], len(kv[0])))[0]
        # Prefer the longest original form for readability.
        return max(originals.get(best_key, [cleaned[0]]), key=len)

    @staticmethod
    def _find_gap_for_element(gaps: List[Gap], element_label: str) -> Optional[Gap]:
        if not gaps or not element_label:
            return None
        needle = DataUpdater._normalize_for_match(element_label)
        if not needle:
            return None
        for gap in gaps:
            if gap.gap_type != GapType.MISSING_DATA:
                continue
            title = gap.title or ""
            if needle in DataUpdater._normalize_for_match(title):
                return gap
        return None

    @staticmethod
    def create_update_session(gap_report: GapAnalysisReport, collection: DocumentCollection) -> UpdateAttemptResult:
        """
        Create a new update session from gap analysis report.

        Args:
            gap_report: GapAnalysisReport from Phase 6
            collection: Current DocumentCollection

        Returns:
            UpdateAttemptResult (initially empty, to be filled)
        """
        return UpdateAttemptResult(
            original_gap_report=gap_report,
            updated_collection=collection
        )

    @staticmethod
    def upload_updated_document(
        update_result: UpdateAttemptResult,
        gap: Gap,
        file_path: str,
        notes: str = ""
    ) -> UpdateAttemptResult:
        """
        Upload a new/updated document to address a gap.

        Args:
            update_result: Current UpdateAttemptResult
            gap: The gap being addressed
            file_path: Path to the new document file
            notes: Optional notes about this update

        Returns:
            Updated UpdateAttemptResult
        """
        try:
            # Verify file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            # Process the new document
            new_doc = DocumentIntake.process_file(file_path)

            # Get previous state
            previous_docs = update_result.updated_collection.get_documents_by_type(gap.affected_document)
            previous_state = f"{len(previous_docs)} documento(s) previo(s)" if previous_docs else "Sin documento previo"

            # Add to collection
            update_result.updated_collection.add_document(new_doc)

            # Create update record
            update = DocumentUpdate(
                document_type=gap.affected_document if gap.affected_document else DocumentType.DECLARACION_JURADA,
                gap_addressed=gap,
                update_source=UpdateSource.MANUAL_UPLOAD,
                update_status=UpdateStatus.SUCCESS,
                previous_state=previous_state,
                new_state=f"Documento cargado: {new_doc.file_name}",
                new_document=new_doc,
                notes=notes
            )

            update_result.updates.append(update)

            print(f"✅ Documento cargado exitosamente: {new_doc.file_name}")

        except Exception as e:
            # Record failed update
            update = DocumentUpdate(
                document_type=gap.affected_document if gap.affected_document else DocumentType.DECLARACION_JURADA,
                gap_addressed=gap,
                update_source=UpdateSource.MANUAL_UPLOAD,
                update_status=UpdateStatus.FAILED,
                error_message=str(e),
                notes=notes
            )

            update_result.updates.append(update)

            print(f"❌ Error al cargar documento: {str(e)}")

        return update_result

    @staticmethod
    def upload_multiple_documents(
        update_result: UpdateAttemptResult,
        gap_file_map: Dict[Gap, str],
        notes: str = ""
    ) -> UpdateAttemptResult:
        """
        Upload multiple documents at once.

        Args:
            update_result: Current UpdateAttemptResult
            gap_file_map: Dictionary mapping gaps to file paths
            notes: Optional notes

        Returns:
            Updated UpdateAttemptResult
        """
        for gap, file_path in gap_file_map.items():
            update_result = DataUpdater.upload_updated_document(
                update_result, gap, file_path, notes
            )

        return update_result

    @staticmethod
    def attempt_public_registry_fetch(
        update_result: UpdateAttemptResult,
        gap: Gap,
        company_name: Optional[str] = None,
        rut: Optional[str] = None
    ) -> UpdateAttemptResult:
        """
        Attempt to fetch missing data from public registries.

        NOTE: This is a PLACEHOLDER for future implementation.
        In production, this would:
        - Connect to Registro de Comercio API
        - Connect to DGI API
        - Connect to BPS API
        - Fetch and validate public records

        Args:
            update_result: Current UpdateAttemptResult
            gap: The gap being addressed
            company_name: Company name (for search)
            rut: RUT number (for search)

        Returns:
            Updated UpdateAttemptResult
        """
        # PLACEHOLDER IMPLEMENTATION
        # In real system, would call external APIs here

        update = DocumentUpdate(
            document_type=gap.affected_document if gap.affected_document else DocumentType.DECLARACION_JURADA,
            gap_addressed=gap,
            update_source=UpdateSource.PUBLIC_REGISTRY,
            update_status=UpdateStatus.NOT_ATTEMPTED,
            error_message="Función de consulta de registros públicos no implementada aún",
            notes="Requiere integración con APIs de: Registro de Comercio, DGI, BPS"
        )

        update_result.updates.append(update)

        print(f"⏭️  Consulta de registro público no disponible aún para: {gap.title}")

        return update_result

    @staticmethod
    def attempt_system_corrections(
        update_result: UpdateAttemptResult,
        intent: Optional[CertificateIntent] = None,
    ) -> UpdateAttemptResult:
        """
        Attempt safe, local "auto-fixes" based on already-uploaded documents.

        Current behavior (intentionally conservative):
        - Fill missing `company_name`, `rut`, `ci`, `registro_comercio` in documents if a value was extracted in *any* other document.
        - Does NOT override conflicting values, and does NOT attempt any network fetches.

        This is meant to reduce "missing data" failures when the information exists in the set but wasn't attached to every doc.
        """
        extraction = update_result.updated_extraction_result
        if not extraction or not extraction.extraction_results:
            return update_result

        company_candidates: List[str] = []
        rut_candidates: List[str] = []
        ci_candidates: List[str] = []
        registry_candidates: List[str] = []
        company_values_by_norm: Dict[str, List[str]] = {}
        rut_digits_set: Set[str] = set()

        for res in extraction.extraction_results:
            if not res.success or not res.extracted_data:
                continue
            data = res.extracted_data
            if data.company_name:
                company_candidates.append(data.company_name)
                company_values_by_norm.setdefault(
                    DataUpdater._normalize_for_match(data.company_name), []
                ).append(data.company_name)
            if data.rut:
                rut_candidates.append(data.rut)
                rut_digits_set.add(re.sub(r"[^0-9]", "", data.rut))
            if data.ci:
                ci_candidates.append(data.ci)
            if data.registro_comercio:
                registry_candidates.append(data.registro_comercio)

        if intent and getattr(intent, "subject_name", None):
            company_candidates.append(str(intent.subject_name))

        canonical_company = DataUpdater._pick_most_common(company_candidates)
        canonical_rut = DataUpdater._pick_most_common(rut_candidates)
        canonical_ci = DataUpdater._pick_most_common(ci_candidates)
        canonical_registry = DataUpdater._pick_most_common(registry_candidates)

        if canonical_rut:
            canonical_rut = re.sub(r"[^0-9]", "", canonical_rut)
        if canonical_ci:
            canonical_ci = canonical_ci.strip()
        if canonical_registry:
            canonical_registry = re.sub(r"[^0-9]", "", canonical_registry)

        if not any([canonical_company, canonical_rut, canonical_ci, canonical_registry]):
            return update_result

        gaps = (update_result.original_gap_report.gaps if update_result.original_gap_report else []) or []

        def record_update(doc: UploadedDocument, gap: Gap, field: str, before: str, after: str) -> None:
            update_result.updates.append(
                DocumentUpdate(
                    document_type=doc.detected_type or DocumentType.DECLARACION_JURADA,
                    gap_addressed=gap,
                    update_source=UpdateSource.SYSTEM_CORRECTION,
                    update_status=UpdateStatus.SUCCESS,
                    previous_state=f"{field}={before or '(vacío)'}",
                    new_state=f"{field}={after}",
                    new_document=None,
                    fetched_data=None,
                    error_message=None,
                    notes="Auto-fill basado en otros documentos cargados; requiere revisión del notario.",
                )
            )

        for res in extraction.extraction_results:
            if not res.success or not res.extracted_data:
                continue
            doc = res.document
            data = res.extracted_data

            changes: List[Tuple[str, str, str]] = []

            # Safe unification: if company names differ only in punctuation/case/accents, normalize to canonical.
            if canonical_company and (data.company_name or "").strip():
                current_norm = DataUpdater._normalize_for_match(data.company_name)
                canonical_norm = DataUpdater._normalize_for_match(canonical_company)
                if current_norm == canonical_norm and data.company_name != canonical_company:
                    before = data.company_name
                    data.company_name = canonical_company
                    changes.append(("company_name", before, canonical_company))

            if canonical_company and not (data.company_name or "").strip():
                before = data.company_name or ""
                data.company_name = canonical_company
                changes.append(("company_name", before, canonical_company))

            if canonical_rut and not (data.rut or "").strip():
                before = data.rut or ""
                data.rut = canonical_rut
                changes.append(("rut", before, canonical_rut))
            elif canonical_rut and (data.rut or "").strip():
                # Safe unification: if RUT digits match, normalize formatting to canonical digits.
                current_digits = re.sub(r"[^0-9]", "", data.rut or "")
                canonical_digits = re.sub(r"[^0-9]", "", canonical_rut or "")
                if current_digits and canonical_digits and current_digits == canonical_digits and data.rut != canonical_digits:
                    before = data.rut or ""
                    data.rut = canonical_digits
                    changes.append(("rut", before, canonical_digits))

            if canonical_ci and not (data.ci or "").strip():
                before = data.ci or ""
                data.ci = canonical_ci
                changes.append(("ci", before, canonical_ci))

            if canonical_registry and not (data.registro_comercio or "").strip():
                before = data.registro_comercio or ""
                data.registro_comercio = canonical_registry
                changes.append(("registro_comercio", before, canonical_registry))

            if not changes:
                continue

            data.additional_fields.setdefault("system_corrections", [])
            for field, before, after in changes:
                data.additional_fields["system_corrections"].append(
                    {"field": field, "before": before, "after": after, "source": "cross_document_fill"}
                )

                element_label_map = {
                    "company_name": "Company Name",
                    "rut": "Rut Number",
                    "ci": "Identity Verification",
                    "registro_comercio": "Registry Inscription",
                }
                gap = DataUpdater._find_gap_for_element(gaps, element_label_map.get(field, field))
                if not gap:
                    gap = Gap(
                        gap_type=GapType.MISSING_DATA,
                        priority=ActionPriority.LOW,
                        title=f"Auto-corrección: {field}",
                        description="Campo faltante completado automáticamente a partir de otros documentos.",
                        affected_document=doc.detected_type,
                        required_state="Campo presente y consistente",
                        action_required="Revisar el valor auto-completado.",
                    )

                record_update(doc, gap, field, before, after)

        return update_result

    @staticmethod
    def apply_system_corrections_to_extraction(
        extraction_result: CollectionExtractionResult,
        intent: Optional[CertificateIntent] = None,
    ) -> List[Dict[str, str]]:
        """
        Apply conservative, local system corrections directly to a CollectionExtractionResult.

        Returns a list of applied corrections for display/logging.
        """
        if not extraction_result or not extraction_result.extraction_results:
            return []

        company_candidates: List[str] = []
        rut_candidates: List[str] = []
        ci_candidates: List[str] = []
        registry_candidates: List[str] = []

        for res in extraction_result.extraction_results:
            if not res.success or not res.extracted_data:
                continue
            data = res.extracted_data
            if data.company_name:
                company_candidates.append(data.company_name)
            if data.rut:
                rut_candidates.append(data.rut)
            if data.ci:
                ci_candidates.append(data.ci)
            if data.registro_comercio:
                registry_candidates.append(data.registro_comercio)

        if intent and getattr(intent, "subject_name", None):
            company_candidates.append(str(intent.subject_name))

        canonical_company = DataUpdater._pick_most_common(company_candidates)
        canonical_rut = DataUpdater._pick_most_common(rut_candidates)
        canonical_ci = DataUpdater._pick_most_common(ci_candidates)
        canonical_registry = DataUpdater._pick_most_common(registry_candidates)

        if canonical_rut:
            canonical_rut = re.sub(r"[^0-9]", "", canonical_rut)
        if canonical_registry:
            canonical_registry = re.sub(r"[^0-9]", "", canonical_registry)
        if canonical_ci:
            canonical_ci = canonical_ci.strip()

        applied: List[Dict[str, str]] = []

        for res in extraction_result.extraction_results:
            if not res.success or not res.extracted_data:
                continue
            doc_name = getattr(res.document, "file_name", "document")
            data = res.extracted_data

            changes: List[Tuple[str, str, str]] = []

            if canonical_company and (data.company_name or "").strip():
                current_norm = DataUpdater._normalize_for_match(data.company_name)
                canonical_norm = DataUpdater._normalize_for_match(canonical_company)
                if current_norm == canonical_norm and data.company_name != canonical_company:
                    before = data.company_name
                    data.company_name = canonical_company
                    changes.append(("company_name", before, canonical_company))

            if canonical_company and not (data.company_name or "").strip():
                before = data.company_name or ""
                data.company_name = canonical_company
                changes.append(("company_name", before, canonical_company))

            if canonical_rut and not (data.rut or "").strip():
                before = data.rut or ""
                data.rut = canonical_rut
                changes.append(("rut", before, canonical_rut))
            elif canonical_rut and (data.rut or "").strip():
                current_digits = re.sub(r"[^0-9]", "", data.rut or "")
                canonical_digits = re.sub(r"[^0-9]", "", canonical_rut or "")
                if current_digits and canonical_digits and current_digits == canonical_digits and data.rut != canonical_digits:
                    before = data.rut or ""
                    data.rut = canonical_digits
                    changes.append(("rut", before, canonical_digits))

            if canonical_ci and not (data.ci or "").strip():
                before = data.ci or ""
                data.ci = canonical_ci
                changes.append(("ci", before, canonical_ci))

            if canonical_registry and not (data.registro_comercio or "").strip():
                before = data.registro_comercio or ""
                data.registro_comercio = canonical_registry
                changes.append(("registro_comercio", before, canonical_registry))

            if not changes:
                continue

            data.additional_fields.setdefault("system_corrections", [])
            for field, before, after in changes:
                data.additional_fields["system_corrections"].append(
                    {"field": field, "before": before, "after": after, "source": "cross_document_fill"}
                )
                applied.append(
                    {
                        "document": str(doc_name),
                        "field": str(field),
                        "before": str(before),
                        "after": str(after),
                    }
                )

        return applied

    @staticmethod
    def mark_gap_not_addressed(
        update_result: UpdateAttemptResult,
        gap: Gap,
        reason: str = "No se intentó actualización"
    ) -> UpdateAttemptResult:
        """
        Mark a gap as not being addressed.

        Args:
            update_result: Current UpdateAttemptResult
            gap: The gap not being addressed
            reason: Reason why not addressed

        Returns:
            Updated UpdateAttemptResult
        """
        update = DocumentUpdate(
            document_type=gap.affected_document if gap.affected_document else DocumentType.DECLARACION_JURADA,
            gap_addressed=gap,
            update_source=UpdateSource.NOT_UPDATED,
            update_status=UpdateStatus.NOT_ATTEMPTED,
            notes=reason
        )

        update_result.updates.append(update)

        return update_result

    @staticmethod
    def re_extract_data(update_result: UpdateAttemptResult) -> UpdateAttemptResult:
        """
        Re-run text extraction on the updated collection.

        Args:
            update_result: UpdateAttemptResult with updated documents

        Returns:
            UpdateAttemptResult with updated_extraction_result populated
        """
        if not update_result.updated_collection:
            print("⚠️  No hay colección actualizada para re-extraer")
            return update_result

        print("\n🔄 Re-extrayendo datos de documentos actualizados...")

        try:
            extraction_result = TextExtractor.process_collection(
                update_result.updated_collection
            )

            update_result.updated_extraction_result = extraction_result

            print(f"✅ Extracción completada: {len(extraction_result.document_extractions)} documentos procesados")

        except Exception as e:
            print(f"❌ Error en re-extracción: {str(e)}")

        return update_result

    @staticmethod
    def get_remaining_gaps(update_result: UpdateAttemptResult) -> List[Gap]:
        """
        Get list of gaps that were not successfully addressed.

        Args:
            update_result: UpdateAttemptResult

        Returns:
            List of remaining gaps
        """
        addressed_gap_ids = set()

        for update in update_result.updates:
            if update.update_status == UpdateStatus.SUCCESS:
                # Create unique ID from gap
                gap_id = f"{update.gap_addressed.gap_type.value}_{update.gap_addressed.affected_document.value if update.gap_addressed.affected_document else 'none'}"
                addressed_gap_ids.add(gap_id)

        remaining = []
        for gap in update_result.original_gap_report.gaps:
            gap_id = f"{gap.gap_type.value}_{gap.affected_document.value if gap.affected_document else 'none'}"
            if gap_id not in addressed_gap_ids:
                remaining.append(gap)

        return remaining

    @staticmethod
    def save_update_result(update_result: UpdateAttemptResult, output_path: str) -> None:
        """Save update result to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(update_result.to_json())
        print(f"\n✅ Resultado de actualización guardado en: {output_path}")

    @staticmethod
    def load_update_result(input_path: str) -> UpdateAttemptResult:
        """Load update result from JSON file (simplified)"""
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # NOTE: This is a simplified loader
        # In production, would need full deserialization
        print(f"✅ Resultado de actualización cargado desde: {input_path}")
        return data


def example_usage():
    """Example usage of Phase 7"""

    print("\n" + "="*70)
    print("  EJEMPLOS DE USO - FASE 7: ACTUALIZACIÓN DE DATOS")
    print("="*70)

    print("\n📌 Ejemplo 1: Crear sesión de actualización")
    print("-" * 70)
    print("""
# Asumiendo que tienes gap_report de Fase 6 y collection de Fase 3:
from src.phase7_data_update import DataUpdater

# Crear sesión de actualización
update_result = DataUpdater.create_update_session(gap_report, collection)
    """)

    print("\n📌 Ejemplo 2: Cargar documento actualizado")
    print("-" * 70)
    print("""
# Obtener gap urgente (documento faltante)
urgent_gap = gap_report.gaps[0]  # Por ejemplo, estatuto faltante

# Cargar el documento
update_result = DataUpdater.upload_updated_document(
    update_result=update_result,
    gap=urgent_gap,
    file_path="/path/to/estatuto_actualizado.pdf",
    notes="Estatuto actualizado con nueva acta"
)

print(update_result.get_summary())
    """)

    print("\n📌 Ejemplo 3: Cargar múltiples documentos")
    print("-" * 70)
    print("""
# Mapear gaps a archivos
gap_file_map = {
    gaps[0]: "/path/to/estatuto.pdf",
    gaps[1]: "/path/to/certificado_bps_nuevo.pdf",
    gaps[2]: "/path/to/acta_directorio.pdf"
}

# Cargar todos
update_result = DataUpdater.upload_multiple_documents(
    update_result=update_result,
    gap_file_map=gap_file_map,
    notes="Documentos actualizados del cliente"
)
    """)

    print("\n📌 Ejemplo 4: Re-extraer datos")
    print("-" * 70)
    print("""
# Después de cargar documentos, re-extraer datos
update_result = DataUpdater.re_extract_data(update_result)

# Ver resultado
print(update_result.get_summary())
print(update_result.get_changes_report())
    """)

    print("\n📌 Ejemplo 5: Verificar brechas restantes")
    print("-" * 70)
    print("""
# Ver qué gaps aún no se han resuelto
remaining_gaps = DataUpdater.get_remaining_gaps(update_result)

print(f"Brechas restantes: {len(remaining_gaps)}")
for gap in remaining_gaps:
    print(f"  - {gap.title} ({gap.priority.value})")
    """)

    print("\n📌 Ejemplo 6: Flujo completo (Fase 6 → Fase 7)")
    print("-" * 70)
    print("""
from src.phase6_gap_detection import GapDetector
from src.phase7_data_update import DataUpdater

# Fase 6: Análisis de brechas
gap_report = GapDetector.analyze(validation_matrix)

# Fase 7: Crear sesión de actualización
update_result = DataUpdater.create_update_session(gap_report, collection)

# Cargar documentos faltantes
for gap in gap_report.gaps:
    if gap.priority == ActionPriority.URGENT:
        if gap.gap_type == GapType.MISSING_DOCUMENT:
            # Aquí el notario carga el documento
            file_path = input(f"Cargar documento para {gap.title}: ")
            update_result = DataUpdater.upload_updated_document(
                update_result, gap, file_path
            )

# Re-extraer datos
update_result = DataUpdater.re_extract_data(update_result)

# Verificar resultado
print(update_result.get_summary())

# Guardar para Fase 8
DataUpdater.save_update_result(update_result, "update_result.json")

# Si todo está bien, continuar a Fase 8 (validación final)
if len(DataUpdater.get_remaining_gaps(update_result)) == 0:
    print("✅ Todas las brechas resueltas. Continuar a Fase 8.")
else:
    print("⚠️  Aún hay brechas sin resolver.")
    """)


if __name__ == "__main__":
    example_usage()
