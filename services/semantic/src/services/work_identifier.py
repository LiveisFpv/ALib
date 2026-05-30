from __future__ import annotations

import re


OPENALEX_WORK_RE = re.compile(r"^W\d+$", re.IGNORECASE)
OPENALEX_URL_RE = re.compile(r"^https?://(?:www\.|api\.)?openalex\.org/(?:works/)?", re.IGNORECASE)
DOI_RE = re.compile(r"10\.\d{4,9}/\S+", re.IGNORECASE)


def normalize_openalex_work_id(value: str | None) -> str | None:
    text = _clean(value)
    if not text:
        return None
    text = OPENALEX_URL_RE.sub("", text).strip().strip("/")
    if OPENALEX_WORK_RE.match(text):
        return text.upper()
    return None


def openalex_aliases(value: str | None) -> list[str]:
    normalized = normalize_openalex_work_id(value)
    if not normalized:
        return []
    return [normalized, f"https://openalex.org/{normalized}"]


def normalize_doi(value: str | None) -> str | None:
    text = _clean(value)
    if not text:
        return None
    text = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(?:doi\s*:|doi\s+)", "", text, flags=re.IGNORECASE).strip()
    match = DOI_RE.search(text)
    if not match:
        return None
    return match.group(0).strip().strip(".,;").lower()


def doi_aliases(value: str | None) -> list[str]:
    normalized = normalize_doi(value)
    if not normalized:
        return []
    return [
        normalized,
        f"https://doi.org/{normalized}",
        f"http://doi.org/{normalized}",
        f"https://dx.doi.org/{normalized}",
        f"http://dx.doi.org/{normalized}",
    ]


def normalize_work_identifier(value: str | None) -> str | None:
    return normalize_openalex_work_id(value) or normalize_doi(value) or _clean(value)


def work_identifier_aliases(value: str | None) -> tuple[str, list[str]]:
    openalex = normalize_openalex_work_id(value)
    if openalex:
        return "openalex", openalex_aliases(openalex)
    doi = normalize_doi(value)
    if doi:
        return "doi", doi_aliases(doi)
    cleaned = _clean(value)
    return "submission", [cleaned] if cleaned else []


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


__all__ = [
    "doi_aliases",
    "normalize_doi",
    "normalize_openalex_work_id",
    "normalize_work_identifier",
    "openalex_aliases",
    "work_identifier_aliases",
]
