"""Helpers for obtaining database connections."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import psycopg
from psycopg import Connection
from psycopg.rows import dict_row

from src.config.config import DATABASE_SETTINGS


def get_connection(*, autocommit: bool = False) -> Connection:
    conn = psycopg.connect(DATABASE_SETTINGS.psycopg_dsn(driver="postgresql"), row_factory=dict_row)
    conn.autocommit = autocommit
    return conn


@contextmanager
def get_cursor(*, autocommit: bool = False):
    conn = get_connection(autocommit=autocommit)
    try:
        with conn.cursor() as cur:
            yield cur
            if not autocommit:
                conn.commit()
    except Exception:
        if not autocommit:
            conn.rollback()
        raise
    finally:
        conn.close()


@contextmanager
def transaction() -> Iterator[Connection]:
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
