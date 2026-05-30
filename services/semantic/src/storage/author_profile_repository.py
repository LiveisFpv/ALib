from __future__ import annotations

import psycopg
from psycopg import errors
from psycopg.rows import dict_row

from src.config.config import DATABASE_SETTINGS
from src.domain.models.author_profile import AuthorProfileModel
from src.services.author_profile_errors import AuthorProfileConflictError
from src.storage.paper_repository import PaperRepository


class AuthorProfileRepository:
    def __init__(
        self,
        *,
        dsn: str | None = None,
        paper_repository: PaperRepository | None = None,
    ) -> None:
        self.dsn = dsn or DATABASE_SETTINGS.psycopg_dsn()
        self.paper_repository = paper_repository or PaperRepository(dsn=self.dsn)

    def get_profile(self, user_id: int) -> AuthorProfileModel:
        with psycopg.connect(self.dsn, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT user_id, orcid, confirmed_at, created_at, updated_at
                    FROM user_orcid_links
                    WHERE user_id = %s
                    """,
                    (int(user_id),),
                )
                row = cur.fetchone()

                if not row:
                    return AuthorProfileModel(user_id=int(user_id))

                paper_count = self._count_papers_by_orcid(cur, row["orcid"])

        return AuthorProfileModel(
            user_id=int(row["user_id"]),
            orcid=row["orcid"],
            confirmed_at=row["confirmed_at"],
            paper_count=paper_count,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def upsert_profile(self, *, user_id: int, orcid: str) -> AuthorProfileModel:
        with psycopg.connect(self.dsn, row_factory=dict_row) as conn:
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO user_orcid_links (user_id, orcid, confirmed_at)
                        VALUES (%s, %s, NOW())
                        ON CONFLICT (user_id) DO UPDATE
                        SET orcid = EXCLUDED.orcid,
                            confirmed_at = NOW(),
                            updated_at = NOW()
                        RETURNING user_id
                        """,
                        (int(user_id), orcid),
                    )
                    if cur.fetchone() is None:
                        raise RuntimeError("failed to save ORCID profile")
                conn.commit()
            except errors.UniqueViolation as exc:
                conn.rollback()
                raise AuthorProfileConflictError("ORCID is already linked to another user") from exc
        return self.get_profile(int(user_id))

    def delete_profile(self, user_id: int) -> None:
        with psycopg.connect(self.dsn, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM user_orcid_links WHERE user_id = %s",
                    (int(user_id),),
                )
            conn.commit()

    def list_catalog_papers(self, user_id: int) -> list[dict]:
        paper_ids = self._list_catalog_paper_ids(user_id)
        return [paper for paper in self.paper_repository.fetch_ordered(paper_ids) if paper is not None]

    def _list_catalog_paper_ids(self, user_id: int) -> list[int]:
        with psycopg.connect(self.dsn, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT DISTINCT p.paper_id
                    FROM user_orcid_links uol
                    JOIN authors a ON lower(a.orcid) = lower(uol.orcid)
                    JOIN paper_authors pa ON pa.author_id = a.author_id
                    JOIN papers p ON p.paper_id = pa.paper_id
                    WHERE uol.user_id = %s
                      AND uol.confirmed_at IS NOT NULL
                    ORDER BY p.paper_id DESC
                    """,
                    (int(user_id),),
                )
                rows = cur.fetchall()
        return [int(row["paper_id"]) for row in rows]

    @staticmethod
    def _count_papers_by_orcid(cur: psycopg.Cursor, orcid: str | None) -> int:
        if not orcid:
            return 0
        cur.execute(
            """
            SELECT COUNT(DISTINCT pa.paper_id) AS total
            FROM authors a
            JOIN paper_authors pa ON pa.author_id = a.author_id
            WHERE lower(a.orcid) = lower(%s)
            """,
            (orcid,),
        )
        row = cur.fetchone()
        return int(row["total"] or 0) if row else 0


__all__ = [
    "AuthorProfileConflictError",
    "AuthorProfileRepository",
]
