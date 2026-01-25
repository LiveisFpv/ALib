from __future__ import annotations

from typing import List, Optional
from src.domain.models.chat import ChatMessage, ChatModel
from src.domain.models.paper import PaperModel

import psycopg
from psycopg.rows import dict_row

from src.config.config import DATABASE_SETTINGS

class ChatRepository:
    def __init__(self, *, dsn: str | None = None) -> None:
        self.dsn = dsn or DATABASE_SETTINGS.psycopg_dsn()

    def create_chat(self, user_id: int, title: Optional[str] = None) -> ChatModel:
        title_value = title.strip() if isinstance(title, str) and title.strip() else "New chat"
        with psycopg.connect(self.dsn, row_factory=dict_row) as conn:
            query = """
                INSERT INTO chat (user_id, title)
                VALUES (%s, %s)
                RETURNING chat_id, user_id, updated_at, title
            """
            with conn.cursor() as cur:
                cur.execute(query, (user_id, title_value))
                row = cur.fetchone()
        if not row:
            raise RuntimeError("Failed to create chat")
        return ChatModel(row["chat_id"], row["user_id"], row["updated_at"], row["title"])

    def update_chat(self, chat: ChatModel) -> ChatModel:
        title_value = chat.title.strip() if isinstance(chat.title, str) and chat.title.strip() else None
        with psycopg.connect(self.dsn, row_factory=dict_row) as conn:
            query = """
                UPDATE chat
                SET title = COALESCE(%s, title),
                    updated_at = NOW()
                WHERE chat_id = %s and user_id = %s
                RETURNING chat_id, user_id, updated_at, title
            """
            with conn.cursor() as cur:
                cur.execute(query, (title_value, chat.id, chat.user_id))
                row = cur.fetchone()
        if not row:
            raise RuntimeError("Chat not found")
        return ChatModel(row["chat_id"], row["user_id"], row["updated_at"], row["title"])

    def delete_chat(self, chat_id:int ,user_id: int)->str|None:
        with psycopg.connect(self.dsn,row_factory=dict_row) as conn:
            query = """
                DELETE chat WHERE chat_id = %s and user_id = %s
            """
            with conn.cursor() as cur:
                cur.execute(query,chat_id,user_id)
                row = cur.fetchone()
            if not row:
                raise RuntimeError("Chat doesn't delete")
        return None

    def create_chat_message(self, chat_id: int, search_query: str, papers: List[PaperModel]) -> ChatMessage:
        with psycopg.connect(self.dsn, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO chat_message (chat_id, search_query)
                    VALUES (%s, %s)
                    RETURNING chat_history_id, created_at
                    """,
                    (chat_id, search_query),
                )
                row = cur.fetchone()
                if not row:
                    raise RuntimeError("Failed to create chat history entry")
                chat_history_id = row["chat_history_id"]
                created_at = row["created_at"]

                if papers:
                    insert_query = """
                        INSERT INTO search_results (chat_history_id, paper_id, score, rank)
                        VALUES (%s, %s, %s, %s)
                    """
                    values = []
                    for idx, paper in enumerate(papers):
                        paper_id = self._normalize_paper_id(paper)
                        if paper_id is None:
                            continue
                        values.append((chat_history_id, paper_id, None, idx + 1))
                    if values:
                        cur.executemany(insert_query, values)

                cur.execute(
                    "UPDATE chat SET updated_at = NOW() WHERE chat_id = %s",
                    (chat_id,),
                )
        return ChatMessage(search_query, created_at, papers)

    def get_chat_history(self, chat_id: int) -> List[ChatMessage]:
        with psycopg.connect(self.dsn, row_factory=dict_row) as conn:
            query = """
                WITH best_location AS (
                    SELECT
                        paper_id,
                        MAX(CASE WHEN link_type = 'best_oa_landing' THEN url END) AS best_oa_landing,
                        MAX(CASE WHEN link_type = 'best_oa_pdf' THEN url END) AS best_oa_pdf,
                        MAX(CASE WHEN link_type = 'primary_landing' THEN url END) AS primary_landing,
                        MAX(CASE WHEN link_type = 'primary_pdf' THEN url END) AS primary_pdf
                    FROM locations
                    GROUP BY paper_id
                )
                SELECT
                    cm.chat_history_id,
                    cm.search_query,
                    cm.created_at,
                    sr.rank,
                    p.paper_id,
                    p.title,
                    p.abstract,
                    p.year,
                    COALESCE(best.best_oa_landing, best.best_oa_pdf, best.primary_landing, best.primary_pdf) AS best_oa_location
                FROM chat_message cm
                LEFT JOIN search_results sr ON sr.chat_history_id = cm.chat_history_id
                LEFT JOIN papers p ON p.paper_id = sr.paper_id
                LEFT JOIN best_location best ON best.paper_id = p.paper_id
                WHERE cm.chat_id = %s
                ORDER BY cm.created_at ASC, sr.rank ASC NULLS LAST, p.paper_id ASC
            """
            with conn.cursor() as cur:
                cur.execute(query, (chat_id,))
                rows = cur.fetchall()

        messages: dict[int, ChatMessage] = {}
        order: List[int] = []
        for row in rows:
            history_id = row["chat_history_id"]
            if history_id not in messages:
                messages[history_id] = ChatMessage(
                    row["search_query"],
                    row["created_at"],
                    [],
                )
                order.append(history_id)
            paper_id = row["paper_id"]
            if paper_id is None:
                continue
            paper = PaperModel(
                paper_id,
                Title=row.get("title") or "",
                Abstract=row.get("abstract") or "",
                Year=row.get("year") or 0,
                Best_oa_location=row.get("best_oa_location") or "",
            )
            messages[history_id].papers.append(paper)
        return [messages[history_id] for history_id in order]

    def get_user_chats(self, user_id: int) -> List[ChatModel]:
        with psycopg.connect(self.dsn, row_factory=dict_row) as conn:
            query = """
                SELECT chat_id, user_id, updated_at, title
                FROM chat
                WHERE user_id = %s
                ORDER BY updated_at DESC, chat_id DESC
            """
            with conn.cursor() as cur:
                cur.execute(query, (user_id,))
                rows = cur.fetchall()
        return [
            ChatModel(row["chat_id"], row["user_id"], row["updated_at"], row["title"])
            for row in rows
        ]

    @staticmethod
    def _normalize_paper_id(paper: PaperModel) -> Optional[int]:
        value = getattr(paper, "ID", None)
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
