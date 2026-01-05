from __future__ import annotations

from typing import Optional
from src.domain.models.user import UserModel

import psycopg
from psycopg.rows import dict_row

from src.config.config import DATABASE_SETTINGS

class UserRepository:
    def __init__(self, *, dsn: str | None = None) -> None:
        self.dsn = dsn or DATABASE_SETTINGS.psycopg_dsn()
    def create_user(self, user: UserModel) -> str:
        with psycopg.connect(self.dsn,row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                if self._has_email_column(cur):
                    query = """
                        INSERT INTO users (id, email)
                        VALUES (%s, %s)
                        ON CONFLICT (id) DO NOTHING
                        RETURNING id
                    """
                    cur.execute(query, (user.id, self._build_email(user.id)))
                else:
                    query = """
                        INSERT INTO users (id)
                        VALUES (%s)
                        ON CONFLICT (id) DO NOTHING
                        RETURNING id
                    """
                    cur.execute(query, (user.id,))
                cur.fetchone()
        return str(user.id)
            
    def get_user(self, user_id:int)->Optional[UserModel]:
        with psycopg.connect(self.dsn,row_factory=dict_row) as conn:
            query = """
                SELECT id
                FROM users
                WHERE id = %s
            """
            with conn.cursor() as cur:
                cur.execute(query, (user_id,))
                row = cur.fetchone()
        if not row:
            return None
        return UserModel(row["id"])

    @staticmethod
    def _build_email(user_id: int) -> str:
        return f"user_{user_id}@local"

    @staticmethod
    def _has_email_column(cur: psycopg.Cursor) -> bool:
        cur.execute(
            """
            SELECT 1
            FROM information_schema.columns
            WHERE table_name = 'users' AND column_name = 'email'
            """
        )
        return cur.fetchone() is not None
