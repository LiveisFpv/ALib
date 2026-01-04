from __future__ import annotations

from typing import Any, Iterable, List, Optional
from domain.models.user import UserModel

import psycopg
from psycopg.rows import dict_row

from src.config.config import DATABASE_SETTINGS

# ! TODO all methods
class UserRepository:
    def __init__(self, *, dsn: str | None = None) -> None:
        self.dsn = dsn or DATABASE_SETTINGS.psycopg_dsn()
    def create_user(self, user: UserModel) -> str:
        with psycopg.connect(self.dsn,row_factory=dict_row) as conn:
            query = """
            
            """
            with conn.cursor() as cur:
                cur.execute(
                    query,
                    (user.id),
                )
                rows=cur.fetchall()
            
    def get_user(self, user_id:int)->UserModel|None:
        with psycopg.connect(self.dsn,row_factory=dict_row) as conn:
            query = """
            
            """
            with conn.cursor() as cur:
                cur.execute(
                    query,
                    (user_id),
                )
                rows=cur.fetchall()