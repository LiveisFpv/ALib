from __future__ import annotations

from typing import Any, Iterable, List, Optional

import psycopg
from psycopg.rows import dict_row

from src.config.config import DATABASE_SETTINGS

# ! TODO all methods
class AuthoryRepository:
    def __init__(self, *, dsn: str | None = None) -> None:
        self.dsn = dsn or DATABASE_SETTINGS.psycopg_dsn()