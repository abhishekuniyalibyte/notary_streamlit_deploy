"""
Error Pattern Analyzer

This module learns from ERROR folders containing documents with known mistakes
and uses those patterns to detect similar errors in new documents.

Key Features:
- Pattern extraction from ERROR folders
- Pattern matching against new documents
- Confidence scoring for error detection
- Learning from historical mistakes
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
from datetime import datetime
from pathlib import Path
import json
import re

from src.phase4_text_extraction import ExtractedData, CollectionExtractionResult
from src.phase5_legal_validation import ValidationIssue, ValidationSeverity


class ErrorPatternType:
    """Types of error patterns that can be detected"""
    DATE_INCONSISTENCY = "date_inconsistency"
    WRONG_REGISTRY_NUMBER = "wrong_registry_number"
    MISMATCHED_COMPANY_NAME = "mismatched_company_name"
    INVALID_RUT = "invalid_rut"
    EXPIRED_DOCUMENT = "expired_document"
    MISSING_SIGNATURE = "missing_signature"
    INCORRECT_FORMAT = "incorrect_format"


@dataclass
class ErrorPattern:
    """
    Represents a known error pattern learned from ERROR folders.
    """
    pattern_id: str  # Unique identifier
    pattern_type: str  # Type of error (from ErrorPatternType)
    description: str  # Human-readable description
    detection_rule: str  # Rule for detecting this pattern
    severity: ValidationSeverity

    # Examples from ERROR folders
    example_cases: List[Dict[str, str]] = field(default_factory=list)

    # Confidence and statistics
    occurrence_count: int = 0  # How many times seen in ERROR folders
    false_positive_rate: float = 0.0

    # When this pattern was added
    created_at: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "pattern_id": self.pattern_id,
            "pattern_type": self.pattern_type,
            "description": self.description,
            "detection_rule": self.detection_rule,
            "severity": self.severity.value,
            "example_cases": self.example_cases,
            "occurrence_count": self.occurrence_count,
            "false_positive_rate": self.false_positive_rate,
            "created_at": self.created_at.isoformat(),
            "last_seen": self.last_seen.isoformat()
        }

    @staticmethod
    def from_dict(data: dict) -> 'ErrorPattern':
        return ErrorPattern(
            pattern_id=data["pattern_id"],
            pattern_type=data["pattern_type"],
            description=data["description"],
            detection_rule=data["detection_rule"],
            severity=ValidationSeverity(data["severity"]),
            example_cases=data.get("example_cases", []),
            occurrence_count=data.get("occurrence_count", 0),
            false_positive_rate=data.get("false_positive_rate", 0.0),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
            last_seen=datetime.fromisoformat(data["last_seen"]) if "last_seen" in data else datetime.now()
        )


@dataclass
class PatternMatch:
    """
    Represents a match of an error pattern in a document.
    """
    pattern: ErrorPattern
    confidence: float  # 0.0 to 1.0
    matched_field: str
    matched_value: str
    evidence: str  # What specifically triggered the match
    recommendation: str

    def to_validation_issue(self) -> ValidationIssue:
        """Convert pattern match to validation issue"""
        return ValidationIssue(
            field=self.matched_field,
            issue_type=f"pattern_match_{self.pattern.pattern_type}",
            severity=self.pattern.severity,
            description=(
                f"Patrón de error detectado (confianza {self.confidence:.0%}): "
                f"{self.pattern.description}. {self.evidence}"
            ),
            legal_basis=f"Aprendizaje de errores históricos",
            recommendation=self.recommendation
        )


class ErrorPatternAnalyzer:
    """
    Main service for analyzing error patterns.

    This class:
    1. Learns patterns from ERROR folders
    2. Stores patterns in a knowledge base
    3. Matches new documents against known patterns
    4. Provides warnings when similar errors are detected
    """

    def __init__(self, patterns_file: Optional[str] = None):
        """
        Initialize analyzer.

        Args:
            patterns_file: Path to JSON file containing known patterns.
                          If None, uses default "config/error_patterns.json"
        """
        self.patterns_file = patterns_file or "config/error_patterns.json"
        self.patterns: List[ErrorPattern] = []
        self.load_patterns()

    def load_patterns(self) -> None:
        """Load patterns from file"""
        if Path(self.patterns_file).exists():
            try:
                with open(self.patterns_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.patterns = [ErrorPattern.from_dict(p) for p in data.get("patterns", [])]
                print(f"✅ Loaded {len(self.patterns)} error patterns from {self.patterns_file}")
            except Exception as e:
                print(f"⚠️  Error loading patterns: {e}")
                self.patterns = []
        else:
            # Initialize with default patterns
            self._initialize_default_patterns()
            self.save_patterns()

    def save_patterns(self) -> None:
        """Save patterns to file"""
        data = {
            "patterns": [p.to_dict() for p in self.patterns],
            "last_updated": datetime.now().isoformat(),
            "total_patterns": len(self.patterns)
        }

        with open(self.patterns_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"✅ Saved {len(self.patterns)} patterns to {self.patterns_file}")

    def _initialize_default_patterns(self) -> None:
        """Initialize with default error patterns"""

        # Pattern 1: Statute approved before company constitution
        self.patterns.append(ErrorPattern(
            pattern_id="date_statute_before_constitution",
            pattern_type=ErrorPatternType.DATE_INCONSISTENCY,
            description="Estatuto aprobado antes de la constitución de la empresa",
            detection_rule="statute_approval_date < company_constitution_date",
            severity=ValidationSeverity.CRITICAL,
            example_cases=[
                {
                    "company": "INVERSORA RINLEN S.A.",
                    "constitution_date": "2007-05-24",
                    "statute_date": "2000-12-08",
                    "error": "Estatuto 7 años antes de constitución"
                }
            ],
            occurrence_count=1
        ))

        # Pattern 2: Registration before constitution
        self.patterns.append(ErrorPattern(
            pattern_id="date_registration_before_constitution",
            pattern_type=ErrorPatternType.DATE_INCONSISTENCY,
            description="Registro antes de la constitución de la empresa",
            detection_rule="registration_date < company_constitution_date",
            severity=ValidationSeverity.CRITICAL,
            example_cases=[],
            occurrence_count=0
        ))

        # Pattern 3: Acta before constitution
        self.patterns.append(ErrorPattern(
            pattern_id="date_acta_before_constitution",
            pattern_type=ErrorPatternType.DATE_INCONSISTENCY,
            description="Acta de directorio fechada antes de la constitución",
            detection_rule="acta_date < company_constitution_date",
            severity=ValidationSeverity.ERROR,
            example_cases=[],
            occurrence_count=0
        ))

        # Pattern 4: Very old documents (>5 years) being used
        self.patterns.append(ErrorPattern(
            pattern_id="date_very_old_document",
            pattern_type=ErrorPatternType.EXPIRED_DOCUMENT,
            description="Documento con más de 5 años de antigüedad",
            detection_rule="document_age > 5 years",
            severity=ValidationSeverity.WARNING,
            example_cases=[],
            occurrence_count=0
        ))

        # Pattern 5: Wrong registry number format
        self.patterns.append(ErrorPattern(
            pattern_id="format_invalid_registry_number",
            pattern_type=ErrorPatternType.WRONG_REGISTRY_NUMBER,
            description="Número de registro en formato incorrecto",
            detection_rule="registry_number not matching expected format",
            severity=ValidationSeverity.ERROR,
            example_cases=[],
            occurrence_count=0
        ))

        print(f"✅ Initialized {len(self.patterns)} default error patterns")

    def analyze_document(
        self,
        extracted_data: ExtractedData,
        document_name: Optional[str] = None
    ) -> List[PatternMatch]:
        """
        Analyze a single document for known error patterns.

        Args:
            extracted_data: Extracted data from document
            document_name: Optional document name for context

        Returns:
            List of pattern matches found
        """
        matches = []

        for pattern in self.patterns:
            match = self._check_pattern(pattern, extracted_data, document_name)
            if match:
                matches.append(match)

        return matches

    def analyze_collection(
        self,
        extraction_result: CollectionExtractionResult
    ) -> List[PatternMatch]:
        """
        Analyze a collection of documents for error patterns.

        Args:
            extraction_result: Results from Phase 4 extraction

        Returns:
            List of all pattern matches found across documents
        """
        all_matches = []

        for result in extraction_result.extraction_results:
            if result.success and result.extracted_data:
                matches = self.analyze_document(
                    result.extracted_data,
                    result.document.file_name
                )
                all_matches.extend(matches)

        return all_matches

    def _check_pattern(
        self,
        pattern: ErrorPattern,
        data: ExtractedData,
        document_name: Optional[str]
    ) -> Optional[PatternMatch]:
        """
        Check if a specific pattern matches the data.

        Returns:
            PatternMatch if pattern matches, None otherwise
        """

        # Check date inconsistency patterns
        if pattern.pattern_type == ErrorPatternType.DATE_INCONSISTENCY:
            return self._check_date_pattern(pattern, data, document_name)

        # Check expired document patterns
        elif pattern.pattern_type == ErrorPatternType.EXPIRED_DOCUMENT:
            return self._check_expiry_pattern(pattern, data, document_name)

        # Check registry number patterns
        elif pattern.pattern_type == ErrorPatternType.WRONG_REGISTRY_NUMBER:
            return self._check_registry_pattern(pattern, data, document_name)

        return None

    def _check_date_pattern(
        self,
        pattern: ErrorPattern,
        data: ExtractedData,
        document_name: Optional[str]
    ) -> Optional[PatternMatch]:
        """Check date-related error patterns"""

        # Pattern: Statute before constitution
        if pattern.pattern_id == "date_statute_before_constitution":
            if data.company_constitution_date and data.statute_approval_date:
                try:
                    from src.phase5_legal_validation import LegalValidator
                    const_date = LegalValidator._parse_date_string(data.company_constitution_date)
                    statute_date = LegalValidator._parse_date_string(data.statute_approval_date)

                    if const_date and statute_date and statute_date < const_date:
                        return PatternMatch(
                            pattern=pattern,
                            confidence=0.95,  # High confidence - this is a clear logical error
                            matched_field="statute_approval_date",
                            matched_value=data.statute_approval_date,
                            evidence=(
                                f"Estatuto aprobado {data.statute_approval_date} "
                                f"ANTES de constitución {data.company_constitution_date}"
                            ),
                            recommendation=(
                                "Verificar fechas en documentos originales. "
                                "Posible error de transcripción o documento incorrecto."
                            )
                        )
                except:
                    pass

        # Pattern: Registration before constitution
        elif pattern.pattern_id == "date_registration_before_constitution":
            if data.company_constitution_date and data.registration_date:
                try:
                    from src.phase5_legal_validation import LegalValidator
                    const_date = LegalValidator._parse_date_string(data.company_constitution_date)
                    reg_date = LegalValidator._parse_date_string(data.registration_date)

                    if const_date and reg_date and reg_date < const_date:
                        return PatternMatch(
                            pattern=pattern,
                            confidence=0.95,
                            matched_field="registration_date",
                            matched_value=data.registration_date,
                            evidence=(
                                f"Registro {data.registration_date} "
                                f"ANTES de constitución {data.company_constitution_date}"
                            ),
                            recommendation="Verificar fecha de registro en documento oficial."
                        )
                except:
                    pass

        # Pattern: Acta before constitution
        elif pattern.pattern_id == "date_acta_before_constitution":
            if data.company_constitution_date and data.acta_date:
                try:
                    from src.phase5_legal_validation import LegalValidator
                    const_date = LegalValidator._parse_date_string(data.company_constitution_date)
                    acta_date = LegalValidator._parse_date_string(data.acta_date)

                    if const_date and acta_date and acta_date < const_date:
                        return PatternMatch(
                            pattern=pattern,
                            confidence=0.90,
                            matched_field="acta_date",
                            matched_value=data.acta_date,
                            evidence=(
                                f"Acta {data.acta_date} "
                                f"ANTES de constitución {data.company_constitution_date}"
                            ),
                            recommendation="Verificar fecha del acta de directorio."
                        )
                except:
                    pass

        return None

    def _check_expiry_pattern(
        self,
        pattern: ErrorPattern,
        data: ExtractedData,
        document_name: Optional[str]
    ) -> Optional[PatternMatch]:
        """Check document expiry patterns"""

        # Pattern: Very old documents (>5 years)
        if pattern.pattern_id == "date_very_old_document":
            # Check all date fields
            date_fields = [
                ("company_constitution_date", data.company_constitution_date),
                ("statute_approval_date", data.statute_approval_date),
                ("registration_date", data.registration_date),
                ("acta_date", data.acta_date)
            ]

            for field_name, date_str in date_fields:
                if date_str:
                    try:
                        from src.phase5_legal_validation import LegalValidator
                        parsed_date = LegalValidator._parse_date_string(date_str)

                        if parsed_date:
                            age_years = (datetime.now() - parsed_date).days / 365.25

                            if age_years > 5:
                                return PatternMatch(
                                    pattern=pattern,
                                    confidence=0.70,  # Medium confidence - old documents might be valid
                                    matched_field=field_name,
                                    matched_value=date_str,
                                    evidence=(
                                        f"Documento con fecha {date_str} "
                                        f"({age_years:.1f} años de antigüedad)"
                                    ),
                                    recommendation=(
                                        "Verificar si se requiere versión más reciente "
                                        "según la institución destinataria."
                                    )
                                )
                    except:
                        pass

        return None

    def _check_registry_pattern(
        self,
        pattern: ErrorPattern,
        data: ExtractedData,
        document_name: Optional[str]
    ) -> Optional[PatternMatch]:
        """Check registry number format patterns"""

        if pattern.pattern_id == "format_invalid_registry_number":
            if data.registry_number:
                # Check for common invalid formats
                # Valid format examples: "1234/2020", "FICHA 12345"

                # Invalid if contains multiple slashes or invalid characters
                if data.registry_number.count('/') > 1:
                    return PatternMatch(
                        pattern=pattern,
                        confidence=0.75,
                        matched_field="registry_number",
                        matched_value=data.registry_number,
                        evidence=f"Número de registro con formato inusual: {data.registry_number}",
                        recommendation="Verificar formato del número de registro en documento original."
                    )

        return None

    def learn_from_error_folder(
        self,
        error_folder_path: str,
        extraction_result: CollectionExtractionResult
    ) -> int:
        """
        Learn new patterns from an ERROR folder.

        Args:
            error_folder_path: Path to ERROR folder
            extraction_result: Extraction results from ERROR folder documents

        Returns:
            Number of new patterns learned
        """
        patterns_learned = 0

        # Analyze documents in ERROR folder
        for result in extraction_result.extraction_results:
            if not result.success or not result.extracted_data:
                continue

            data = result.extracted_data

            # Check for date inconsistencies we might not have seen before
            # Update occurrence counts for existing patterns
            matches = self.analyze_document(data, result.document.file_name)

            for match in matches:
                # Update pattern statistics
                for pattern in self.patterns:
                    if pattern.pattern_id == match.pattern.pattern_id:
                        pattern.occurrence_count += 1
                        pattern.last_seen = datetime.now()

                        # Add example case if not already present
                        example = {
                            "company": data.company_name or "Unknown",
                            "document": result.document.file_name,
                            "error": match.evidence
                        }

                        if example not in pattern.example_cases:
                            pattern.example_cases.append(example)
                            patterns_learned += 1

        if patterns_learned > 0:
            self.save_patterns()
            print(f"✅ Learned {patterns_learned} new pattern examples from ERROR folder")

        return patterns_learned


def example_usage():
    """Example usage of error pattern analyzer"""

    print("\n" + "="*70)
    print("  ERROR PATTERN ANALYZER - EXAMPLES")
    print("="*70)

    # Initialize analyzer
    analyzer = ErrorPatternAnalyzer("example_error_patterns.json")

    print(f"\n✅ Loaded {len(analyzer.patterns)} error patterns")

    # Show patterns
    print("\n📋 Available Patterns:")
    for i, pattern in enumerate(analyzer.patterns, 1):
        print(f"\n{i}. {pattern.pattern_id}")
        print(f"   Type: {pattern.pattern_type}")
        print(f"   Description: {pattern.description}")
        print(f"   Severity: {pattern.severity.value}")
        print(f"   Seen {pattern.occurrence_count} times")

        if pattern.example_cases:
            print(f"   Example: {pattern.example_cases[0]}")

    print("\n\n📌 Usage in validation pipeline:")
    print("""
    # After Phase 4 extraction
    analyzer = ErrorPatternAnalyzer()
    matches = analyzer.analyze_collection(extraction_result)

    # Convert matches to validation issues
    for match in matches:
        validation_issue = match.to_validation_issue()
        print(f"⚠️  {validation_issue.description}")
    """)


if __name__ == "__main__":
    example_usage()
