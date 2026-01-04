from __future__ import annotations

from typing import Any, Iterable, List, Optional
from src.domain.models.chat import ChatMessage,ChatModel
from src.storage.chat_repository import ChatRepository
from src.services.user_service import UserService

# ! TODO all methods
class ChatService:
    def __init__(self,repository:ChatRepository,user_service: UserService):
        self.repository=repository
        self.user_service=user_service

    def create_chat(self,user_id:int)->ChatModel:
        pass

    def update_chat(self, chat:ChatModel)->ChatModel:
        pass

    def get_chat_history(self, chat_id:int)->List[ChatMessage]:
        pass
    
    def get_user_chats(self,user_id:int)->List[ChatModel]:
        pass
