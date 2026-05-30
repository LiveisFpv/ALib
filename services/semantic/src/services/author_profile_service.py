from __future__ import annotations

import re
from typing import Protocol

from src.domain.models.author_profile import AuthorProfileModel
from src.domain.models.paper import PaperModel
from src.services.author_profile_errors import (
    AuthorProfileConflictError,
    AuthorProfileValidationError,
)


ORCID_RE = re.compile(r"^[0-9]{4}-[0-9]{4}-[0-9]{4}-[0-9]{3}[0-9X]$")


class UserEnsurerProtocol(Protocol):
    def ensure_user(self, user_id: int): ...


class AuthorProfileRepositoryProtocol(Protocol):
    def get_profile(self, user_id: int) -> AuthorProfileModel: ...
    def upsert_profile(self, *, user_id: int, orcid: str) -> AuthorProfileModel: ...
    def delete_profile(self, user_id: int) -> None: ...
    def list_catalog_papers(self, user_id: int) -> list[dict]: ...


class AuthorProfileService:
    def __init__(
        self,
        repository: AuthorProfileRepositoryProtocol | None = None,
        *,
        user_service: UserEnsurerProtocol | None = None,
    ) -> None:
        if repository is None:
            from src.storage.author_profile_repository import AuthorProfileRepository

            repository = AuthorProfileRepository()
        self.repository = repository
        self.user_service = user_service

    def get_my_profile(self, *, user_id: int) -> AuthorProfileModel:
        return self.repository.get_profile(int(user_id))

    def update_my_profile(
        self,
        *,
        user_id: int,
        orcid: str,
        confirm_authorship: bool,
    ) -> AuthorProfileModel:
        if not confirm_authorship:
            raise AuthorProfileValidationError("authorship confirmation is required")
        normalized_orcid = normalize_orcid(orcid)
        self._ensure_user(int(user_id))
        return self.repository.upsert_profile(user_id=int(user_id), orcid=normalized_orcid)

    def delete_my_profile(self, *, user_id: int) -> None:
        self.repository.delete_profile(int(user_id))

    def list_my_catalog_papers(self, *, user_id: int) -> list[PaperModel]:
        rows = self.repository.list_catalog_papers(int(user_id))
        return [_paper_from_row(row) for row in rows]

    def _ensure_user(self, user_id: int) -> None:
        if self.user_service is not None:
            self.user_service.ensure_user(user_id)


def normalize_orcid(value: str | None) -> str:
    text = str(value or "").strip()
    if not text:
        raise AuthorProfileValidationError("ORCID is required")

    lowered = text.lower()
    for prefix in ("https://orcid.org/", "http://orcid.org/", "orcid.org/"):
        if lowered.startswith(prefix):
            text = text[len(prefix):]
            break
    text = text.strip().strip("/").upper()

    if not ORCID_RE.match(text):
        raise AuthorProfileValidationError("ORCID must use format 0000-0000-0000-000X")
    if not _valid_orcid_checksum(text):
        raise AuthorProfileValidationError("ORCID checksum is invalid")
    return text


def _valid_orcid_checksum(orcid: str) -> bool:
    digits = orcid.replace("-", "")
    total = 0
    for char in digits[:15]:
        total = (total + int(char)) * 2
    check = (12 - (total % 11)) % 11
    expected = "X" if check == 10 else str(check)
    return digits[-1] == expected


def _paper_from_row(row: dict) -> PaperModel:
    return PaperModel(
        row.get("paper_id"),
        row.get("title") or "",
        row.get("abstract") or "",
        row.get("year"),
        row.get("best_oa_location") or "",
        Referenced_works=list(row.get("referenced_works") or []),
        Related_works=list(row.get("related_works") or []),
        Cited_by_count=int(row.get("cited_by_count") or 0),
        Authors=list(row.get("authors") or []),
        Institutions=list(row.get("institutions") or []),
        Identifiers=list(row.get("identifiers") or []),
        State=row.get("state") or "",
    )


__all__ = [
    "AuthorProfileConflictError",
    "AuthorProfileService",
    "AuthorProfileValidationError",
    "normalize_orcid",
]
