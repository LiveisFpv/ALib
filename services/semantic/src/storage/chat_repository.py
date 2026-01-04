from __future__ import annotations

from typing import Any, Iterable, List, Optional
from domain.models.chat import ChatMessage,ChatModel

import psycopg
from psycopg.rows import dict_row

from src.config.config import DATABASE_SETTINGS

# ! TODO all methods
class ChatRepository:
    def __init__(self, *, dsn: str | None = None) -> None:
        self.dsn = dsn or DATABASE_SETTINGS.psycopg_dsn()
    def create_chat(self, user_id:int)->ChatModel:
        pass
    def update_chat(self, chat:ChatModel)->ChatModel:
        pass
    def get_chat_history(self, chat_id:int)->List[ChatMessage]:
        pass
    def get_user_chats(self,user_id:int)->List[ChatModel]:
        pass
