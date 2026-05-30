from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.domain.models.author_profile import AuthorProfileModel  # noqa: E402
from src.services.author_profile_service import (  # noqa: E402
    AuthorProfileConflictError,
    AuthorProfileService,
    AuthorProfileValidationError,
    normalize_orcid,
)


class FakeAuthorProfileRepository:
    def __init__(self) -> None:
        self.profiles: dict[int, AuthorProfileModel] = {}
        self.papers: dict[int, list[dict]] = {}

    def get_profile(self, user_id: int) -> AuthorProfileModel:
        return self.profiles.get(int(user_id), AuthorProfileModel(user_id=int(user_id)))

    def upsert_profile(self, *, user_id: int, orcid: str) -> AuthorProfileModel:
        for existing_user_id, profile in self.profiles.items():
            if existing_user_id != int(user_id) and profile.orcid == orcid:
                raise AuthorProfileConflictError("ORCID is already linked to another user")
        profile = AuthorProfileModel(
            user_id=int(user_id),
            orcid=orcid,
            confirmed_at=datetime.now(timezone.utc),
            paper_count=len(self.papers.get(int(user_id), [])),
        )
        self.profiles[int(user_id)] = profile
        return profile

    def delete_profile(self, user_id: int) -> None:
        self.profiles.pop(int(user_id), None)

    def list_catalog_papers(self, user_id: int) -> list[dict]:
        return list(self.papers.get(int(user_id), []))


class FakeUserService:
    def __init__(self) -> None:
        self.ensured: list[int] = []

    def ensure_user(self, user_id: int) -> None:
        self.ensured.append(int(user_id))


def test_normalize_orcid_accepts_url_and_uppercases_checksum() -> None:
    assert normalize_orcid("https://orcid.org/0000-0002-1825-0097/") == "0000-0002-1825-0097"
    assert normalize_orcid("0000-0002-1694-233x") == "0000-0002-1694-233X"


def test_normalize_orcid_rejects_invalid_checksum() -> None:
    with pytest.raises(AuthorProfileValidationError):
        normalize_orcid("0000-0002-1825-0098")


def test_update_requires_manual_confirmation() -> None:
    service = AuthorProfileService(FakeAuthorProfileRepository(), user_service=FakeUserService())

    with pytest.raises(AuthorProfileValidationError):
        service.update_my_profile(
            user_id=10,
            orcid="0000-0002-1825-0097",
            confirm_authorship=False,
        )


def test_update_saves_profile_and_ensures_user() -> None:
    repo = FakeAuthorProfileRepository()
    user_service = FakeUserService()
    service = AuthorProfileService(repo, user_service=user_service)

    profile = service.update_my_profile(
        user_id=10,
        orcid="https://orcid.org/0000-0002-1825-0097",
        confirm_authorship=True,
    )

    assert profile.orcid == "0000-0002-1825-0097"
    assert profile.confirmed is True
    assert user_service.ensured == [10]


def test_update_rejects_orcid_linked_to_another_user() -> None:
    repo = FakeAuthorProfileRepository()
    service = AuthorProfileService(repo, user_service=FakeUserService())
    service.update_my_profile(
        user_id=10,
        orcid="0000-0002-1825-0097",
        confirm_authorship=True,
    )

    with pytest.raises(AuthorProfileConflictError):
        service.update_my_profile(
            user_id=11,
            orcid="0000-0002-1825-0097",
            confirm_authorship=True,
        )


def test_delete_removes_profile() -> None:
    repo = FakeAuthorProfileRepository()
    service = AuthorProfileService(repo, user_service=FakeUserService())
    service.update_my_profile(
        user_id=10,
        orcid="0000-0002-1825-0097",
        confirm_authorship=True,
    )

    service.delete_my_profile(user_id=10)

    assert service.get_my_profile(user_id=10).confirmed is False


def test_list_catalog_papers_maps_rows() -> None:
    repo = FakeAuthorProfileRepository()
    repo.papers[10] = [
        {
            "paper_id": 42,
            "title": "Linked paper",
            "abstract": "Abstract",
            "year": 2024,
            "best_oa_location": "https://example.test/paper",
            "referenced_works": ["W1"],
            "related_works": ["W2"],
            "cited_by_count": 3,
            "authors": ["Ada Lovelace"],
            "institutions": ["Example University"],
            "identifiers": [{"type": "openalex", "value": "W42"}],
            "state": "article",
        }
    ]
    service = AuthorProfileService(repo, user_service=FakeUserService())

    papers = service.list_my_catalog_papers(user_id=10)

    assert len(papers) == 1
    assert papers[0].ID == 42
    assert papers[0].Title == "Linked paper"
    assert papers[0].Authors == ["Ada Lovelace"]
