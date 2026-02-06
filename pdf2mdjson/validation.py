"""
Document quality validation for extracted scientific papers.

Scores extraction quality across key fields (title, abstract, DOI,
references, sections) and returns an overall score with issue descriptions.
"""

from typing import Dict, List, Tuple

from .models import DocumentModel
from .extractors.fast import _validate_title


def validate_document(document: DocumentModel) -> Tuple[float, List[str]]:
    """
    Validate an extracted document and return a quality score with issues.

    Args:
        document: The extracted DocumentModel to validate

    Returns:
        Tuple of (overall_score: float 0-1, issues: list of strings)
    """
    issues = []
    scores: Dict[str, float] = {}

    # --- Title validation ---
    title_score = 1.0
    if not document.title or document.title == "Untitled":
        title_score = 0.0
        issues.append("No title extracted")
    elif not _validate_title(document.title):
        title_score = 0.2
        issues.append(f"Title looks suspicious: '{document.title[:60]}'")
    elif len(document.title) < 20:
        title_score = 0.5
        issues.append(f"Title unusually short ({len(document.title)} chars)")
    scores["title"] = title_score

    # --- Abstract validation ---
    if document.abstract:
        abstract_len = len(document.abstract)
        if abstract_len < 100:
            scores["abstract"] = 0.4
            issues.append(f"Abstract very short ({abstract_len} chars)")
        else:
            scores["abstract"] = min(abstract_len / 200, 1.0)
    else:
        scores["abstract"] = 0.0
        issues.append("No abstract extracted")

    # --- DOI validation ---
    if document.doi:
        scores["doi"] = 1.0
    else:
        scores["doi"] = 0.0
        issues.append("No document DOI found")

    # --- References validation ---
    if document.references:
        ref_count = len(document.references)
        doi_count = sum(1 for r in document.references if r.doi)
        doi_ratio = doi_count / ref_count

        if doi_ratio < 0.1:
            scores["references"] = 0.2
            issues.append(f"Very low DOI rate in references: {doi_count}/{ref_count} ({doi_ratio:.0%})")
        elif doi_ratio < 0.3:
            scores["references"] = 0.5
            issues.append(f"Low DOI rate in references: {doi_count}/{ref_count} ({doi_ratio:.0%})")
        else:
            scores["references"] = doi_ratio
    else:
        scores["references"] = 0.0
        issues.append("No references extracted")

    # --- Sections validation ---
    section_count = len(document.sections)
    if section_count == 0:
        scores["sections"] = 0.0
        issues.append("No sections extracted")
    elif section_count < 3:
        scores["sections"] = 0.3
        issues.append(f"Very few sections extracted ({section_count})")
    else:
        scores["sections"] = min(section_count / 5, 1.0)

    # --- Overall weighted score ---
    # DOI and references are weighted highest (critical for metascience)
    weights = {
        "title": 0.15,
        "abstract": 0.15,
        "doi": 0.20,
        "references": 0.30,
        "sections": 0.20,
    }
    overall = sum(scores.get(k, 0) * w for k, w in weights.items())

    return round(overall, 3), issues
