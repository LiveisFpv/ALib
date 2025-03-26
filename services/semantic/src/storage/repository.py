from typing import Protocol

class Repository(Protocol):
    def get_papers(prompt:str)->str:
        ...
    def add_paper()->str:
        ...