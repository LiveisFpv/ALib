from __future__ import annotations

from typing import Any, Iterable, List, Optional
from src.domain.models.user import UserModel
from src.storage.user_repository import UserRepository

# ! TODO all methods
class UserService:
    def __init__(self, repository:UserRepository):
        self.repository=repository
    
    def create_user(self, user:UserModel)->str:
        pass

    def get_user(self, user_id:int)->UserModel|str:
        pass