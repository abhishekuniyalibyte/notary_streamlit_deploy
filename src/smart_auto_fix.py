"""
Smart Auto-Fix System
Automatically detects and corrects errors in notarial documents.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Correction:
    """Represents a single correction to be applied."""
    field_name: str
    old_value: str
    new_value: str
    reason: str
    confidence: float  # 0.0 to 1.0
    document_file: str
    correction_type: str  # 'date_logic', 'consistency', 'format', etc.


class SmartAutoFix:
    """
    Smart Auto-Fix System that automatically corrects errors in documents.
    """

    @staticmethod
    def analyze_date_logic_error(
        registration_date: Optional[str],
        constitution_date: Optional[str],
        acta_date: Optional[str],
        document_name: str,
        statute_approval_date: Optional[str] = None,
    ) -> List[Correction]:
        """
        Analyze date logic errors and determine corrections.

        Rules:
        1. Constitution date must come FIRST (company founding)
        2. Statute approval must come AFTER constitution
        3. Registration date must come AFTER constitution
        4. Acta dates must come AFTER constitution
        """
        corrections = []

        # Parse dates
        def parse_date(date_str: Optional[str]) -> Optional[datetime]:
            if not date_str:
                return None
            for fmt in ["%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"]:
                try:
                    return datetime.strptime(date_str, fmt)
                except:
                    continue
            return None

        reg_dt = parse_date(registration_date)
        const_dt = parse_date(constitution_date)
        acta_dt = parse_date(acta_date)
        statute_dt = parse_date(statute_approval_date)

        # Rule 1: Statute BEFORE Constitution (IMPOSSIBLE)
        if statute_dt and const_dt and statute_dt < const_dt:
            corrected_statute = const_dt + timedelta(days=1)
            corrections.append(Correction(
                field_name="statute_approval_date",
                old_value=statute_approval_date,
                new_value=corrected_statute.strftime("%Y-%m-%d"),
                reason=f"Statute approval date ({statute_approval_date}) cannot be before constitution date ({constitution_date}). "
                       f"Corrected to 1 day after constitution.",
                confidence=0.85,
                document_file=document_name,
                correction_type="date_logic"
            ))

        # Rule 2: Registration BEFORE Constitution (IMPOSSIBLE)
        if reg_dt and const_dt and reg_dt < const_dt:
            corrected_reg = const_dt + timedelta(days=7)
            corrections.append(Correction(
                field_name="registration_date",
                old_value=registration_date,
                new_value=corrected_reg.strftime("%Y-%m-%d"),
                reason=f"Registration date ({registration_date}) cannot be before constitution date ({constitution_date}). "
                       f"Corrected to 1 week after constitution.",
                confidence=0.85,
                document_file=document_name,
                correction_type="date_logic"
            ))

        # Rule 3: Acta BEFORE Constitution (IMPOSSIBLE)
        if acta_dt and const_dt and acta_dt < const_dt:
            corrected_acta = const_dt + timedelta(days=14)
            corrections.append(Correction(
                field_name="acta_date",
                old_value=acta_date,
                new_value=corrected_acta.strftime("%Y-%m-%d"),
                reason=f"Acta date ({acta_date}) cannot be before constitution date ({constitution_date}). "
                       f"Corrected to 2 weeks after constitution.",
                confidence=0.85,
                document_file=document_name,
                correction_type="date_logic"
            ))

        return corrections

    @staticmethod
    def fix_cross_document_inconsistency(
        extraction_results: List[Any]
    ) -> List[Correction]:
        """
        Fix inconsistencies across multiple documents.
        Uses majority voting and confidence scores to determine correct value.
        """
        corrections = []

        # Collect all company names with frequency
        company_names = {}
        for result in extraction_results:
            if not result.success or not result.extracted_data:
                continue

            name = result.extracted_data.company_name
            doc_name = result.document.file_name if result.document else "unknown"

            if name:
                name_clean = name.strip()
                if name_clean not in company_names:
                    company_names[name_clean] = []
                company_names[name_clean].append(doc_name)

        # If multiple variants exist, pick the most common one
        if len(company_names) > 1:
            # Sort by frequency (most common first)
            sorted_names = sorted(company_names.items(), key=lambda x: len(x[1]), reverse=True)
            correct_name = sorted_names[0][0]
            correct_count = len(sorted_names[0][1])
            total_count = sum(len(docs) for _, docs in sorted_names)

            # Fix incorrect variants
            for name, docs in sorted_names[1:]:
                for doc in docs:
                    corrections.append(Correction(
                        field_name="company_name",
                        old_value=name,
                        new_value=correct_name,
                        reason=f"Company name '{name}' is inconsistent. Most documents ({correct_count}/{total_count}) use '{correct_name}'.",
                        confidence=correct_count / total_count,
                        document_file=doc,
                        correction_type="consistency"
                    ))

        # Same logic for RUT
        ruts = {}
        for result in extraction_results:
            if not result.success or not result.extracted_data:
                continue

            rut = result.extracted_data.rut
            doc_name = result.document.file_name if result.document else "unknown"

            if rut:
                rut_clean = rut.strip()
                if rut_clean not in ruts:
                    ruts[rut_clean] = []
                ruts[rut_clean].append(doc_name)

        if len(ruts) > 1:
            sorted_ruts = sorted(ruts.items(), key=lambda x: len(x[1]), reverse=True)
            correct_rut = sorted_ruts[0][0]
            correct_count = len(sorted_ruts[0][1])
            total_count = sum(len(docs) for _, docs in sorted_ruts)

            for rut, docs in sorted_ruts[1:]:
                for doc in docs:
                    corrections.append(Correction(
                        field_name="rut",
                        old_value=rut,
                        new_value=correct_rut,
                        reason=f"RUT '{rut}' is inconsistent. Most documents ({correct_count}/{total_count}) use '{correct_rut}'.",
                        confidence=correct_count / total_count,
                        document_file=doc,
                        correction_type="consistency"
                    ))

        return corrections

    @staticmethod
    def apply_corrections(
        extraction_result,
        corrections: List[Correction]
    ):
        """
        Apply corrections to the extraction result.
        Modifies the extracted data in place.
        """
        applied_corrections = []

        for correction in corrections:
            # Find the document to correct
            for result in extraction_result.extraction_results:
                doc_name = result.document.file_name if result.document else "unknown"

                if doc_name != correction.document_file:
                    continue

                if not result.success or not result.extracted_data:
                    continue

                # Apply the correction based on field name
                if correction.field_name == "company_name":
                    old_val = result.extracted_data.company_name
                    if old_val == correction.old_value:
                        result.extracted_data.company_name = correction.new_value
                        applied_corrections.append(correction)

                elif correction.field_name == "rut":
                    old_val = result.extracted_data.rut
                    if old_val == correction.old_value:
                        result.extracted_data.rut = correction.new_value
                        applied_corrections.append(correction)

                elif correction.field_name == "statute_approval_date":
                    old_val = result.extracted_data.statute_approval_date
                    if old_val and correction.old_value and old_val.strip() == correction.old_value.strip():
                        result.extracted_data.statute_approval_date = correction.new_value
                        applied_corrections.append(correction)

                elif correction.field_name == "registration_date":
                    old_val = result.extracted_data.registration_date
                    if old_val and correction.old_value and old_val.strip() == correction.old_value.strip():
                        result.extracted_data.registration_date = correction.new_value
                        applied_corrections.append(correction)

                elif correction.field_name == "acta_date":
                    old_val = result.extracted_data.acta_date
                    if old_val and correction.old_value and old_val.strip() == correction.old_value.strip():
                        result.extracted_data.acta_date = correction.new_value
                        applied_corrections.append(correction)

        return applied_corrections

    @classmethod
    def auto_fix_all(
        cls,
        extraction_result
    ) -> Tuple[List[Correction], int]:
        """
        Run all auto-fix logic and apply corrections.
        Returns: (all_corrections, applied_count)
        """
        all_corrections = []

        # 1. Fix cross-document inconsistencies
        consistency_corrections = cls.fix_cross_document_inconsistency(
            extraction_result.extraction_results
        )
        all_corrections.extend(consistency_corrections)

        # 2. Fix date logic errors
        for result in extraction_result.extraction_results:
            if not result.success or not result.extracted_data:
                continue

            doc_name = result.document.file_name if result.document else "unknown"
            extracted = result.extracted_data

            date_corrections = cls.analyze_date_logic_error(
                registration_date=extracted.registration_date,
                constitution_date=extracted.company_constitution_date,
                acta_date=extracted.acta_date,
                document_name=doc_name
            )
            all_corrections.extend(date_corrections)

        # 3. Apply all corrections
        applied = cls.apply_corrections(extraction_result, all_corrections)

        return all_corrections, len(applied)


def generate_correction_report(corrections: List[Correction]) -> str:
    """
    Generate a human-readable report of corrections applied.
    """
    if not corrections:
        return "No corrections were needed."

    report_lines = [
        "SMART AUTO-FIX CORRECTION REPORT",
        "=" * 60,
        f"Total corrections applied: {len(corrections)}",
        "",
    ]

    # Group by correction type
    by_type = {}
    for corr in corrections:
        if corr.correction_type not in by_type:
            by_type[corr.correction_type] = []
        by_type[corr.correction_type].append(corr)

    for corr_type, corr_list in by_type.items():
        report_lines.append(f"{corr_type.upper()} CORRECTIONS ({len(corr_list)}):")
        report_lines.append("-" * 60)

        for corr in corr_list:
            report_lines.append(f"File: {corr.document_file}")
            report_lines.append(f"Field: {corr.field_name}")
            report_lines.append(f"Changed: '{corr.old_value}' → '{corr.new_value}'")
            report_lines.append(f"Reason: {corr.reason}")
            report_lines.append(f"Confidence: {corr.confidence:.0%}")
            report_lines.append("")

    return "\n".join(report_lines)
