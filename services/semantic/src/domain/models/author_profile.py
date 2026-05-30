from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(slots=True)
class AuthorProfileModel:
    user_id: int
    orcid: str | None = None
    confirmed_at: datetime | None = None
    paper_count: int = 0
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @property
    def confirmed(self) -> bool:
        return bool(self.orcid and self.confirmed_at)
