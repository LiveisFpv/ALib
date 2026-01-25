from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class InstitutionReq(_message.Message):
    __slots__ = ("Query",)
    QUERY_FIELD_NUMBER: _ClassVar[int]
    Query: str
    def __init__(self, Query: _Optional[str] = ...) -> None: ...

class InstitutionsResp(_message.Message):
    __slots__ = ("Institutions",)
    INSTITUTIONS_FIELD_NUMBER: _ClassVar[int]
    Institutions: _containers.RepeatedCompositeFieldContainer[Institution]
    def __init__(self, Institutions: _Optional[_Iterable[_Union[Institution, _Mapping]]] = ...) -> None: ...

class Institution(_message.Message):
    __slots__ = ("Institution_id", "Name", "Country", "Ror_id", "Grid_id")
    INSTITUTION_ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    COUNTRY_FIELD_NUMBER: _ClassVar[int]
    ROR_ID_FIELD_NUMBER: _ClassVar[int]
    GRID_ID_FIELD_NUMBER: _ClassVar[int]
    Institution_id: int
    Name: str
    Country: str
    Ror_id: str
    Grid_id: str
    def __init__(self, Institution_id: _Optional[int] = ..., Name: _Optional[str] = ..., Country: _Optional[str] = ..., Ror_id: _Optional[str] = ..., Grid_id: _Optional[str] = ...) -> None: ...

class AuthorReq(_message.Message):
    __slots__ = ("Query",)
    QUERY_FIELD_NUMBER: _ClassVar[int]
    Query: str
    def __init__(self, Query: _Optional[str] = ...) -> None: ...

class AuthorsResp(_message.Message):
    __slots__ = ("Authors",)
    AUTHORS_FIELD_NUMBER: _ClassVar[int]
    Authors: _containers.RepeatedCompositeFieldContainer[Author]
    def __init__(self, Authors: _Optional[_Iterable[_Union[Author, _Mapping]]] = ...) -> None: ...

class Author(_message.Message):
    __slots__ = ("First_name", "Last_name", "Middle_name", "Orcid", "Author_id")
    FIRST_NAME_FIELD_NUMBER: _ClassVar[int]
    LAST_NAME_FIELD_NUMBER: _ClassVar[int]
    MIDDLE_NAME_FIELD_NUMBER: _ClassVar[int]
    ORCID_FIELD_NUMBER: _ClassVar[int]
    AUTHOR_ID_FIELD_NUMBER: _ClassVar[int]
    First_name: str
    Last_name: str
    Middle_name: str
    Orcid: str
    Author_id: int
    def __init__(self, First_name: _Optional[str] = ..., Last_name: _Optional[str] = ..., Middle_name: _Optional[str] = ..., Orcid: _Optional[str] = ..., Author_id: _Optional[int] = ...) -> None: ...

class HistoryReq(_message.Message):
    __slots__ = ("Chat_id",)
    CHAT_ID_FIELD_NUMBER: _ClassVar[int]
    Chat_id: int
    def __init__(self, Chat_id: _Optional[int] = ...) -> None: ...

class HistoryResp(_message.Message):
    __slots__ = ("ChatMessages",)
    CHATMESSAGES_FIELD_NUMBER: _ClassVar[int]
    ChatMessages: _containers.RepeatedCompositeFieldContainer[ChatMessage]
    def __init__(self, ChatMessages: _Optional[_Iterable[_Union[ChatMessage, _Mapping]]] = ...) -> None: ...

class ChatMessage(_message.Message):
    __slots__ = ("Search_query", "Created_at", "papers")
    SEARCH_QUERY_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    PAPERS_FIELD_NUMBER: _ClassVar[int]
    Search_query: str
    Created_at: str
    papers: PapersResponse
    def __init__(self, Search_query: _Optional[str] = ..., Created_at: _Optional[str] = ..., papers: _Optional[_Union[PapersResponse, _Mapping]] = ...) -> None: ...

class UserChatsReq(_message.Message):
    __slots__ = ("User_id",)
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    User_id: int
    def __init__(self, User_id: _Optional[int] = ...) -> None: ...

class DeleteChatReq(_message.Message):
    __slots__ = ("Chat_id",)
    CHAT_ID_FIELD_NUMBER: _ClassVar[int]
    Chat_id: int
    def __init__(self, Chat_id: _Optional[int] = ...) -> None: ...

class Chat(_message.Message):
    __slots__ = ("Chat_id", "User_id", "Updated_at", "Title")
    CHAT_ID_FIELD_NUMBER: _ClassVar[int]
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    UPDATED_AT_FIELD_NUMBER: _ClassVar[int]
    TITLE_FIELD_NUMBER: _ClassVar[int]
    Chat_id: int
    User_id: int
    Updated_at: str
    Title: str
    def __init__(self, Chat_id: _Optional[int] = ..., User_id: _Optional[int] = ..., Updated_at: _Optional[str] = ..., Title: _Optional[str] = ...) -> None: ...

class UpdateChatReq(_message.Message):
    __slots__ = ("Chat_id", "Title")
    CHAT_ID_FIELD_NUMBER: _ClassVar[int]
    TITLE_FIELD_NUMBER: _ClassVar[int]
    Chat_id: int
    Title: str
    def __init__(self, Chat_id: _Optional[int] = ..., Title: _Optional[str] = ...) -> None: ...

class ChatsResp(_message.Message):
    __slots__ = ("Chats",)
    CHATS_FIELD_NUMBER: _ClassVar[int]
    Chats: _containers.RepeatedCompositeFieldContainer[Chat]
    def __init__(self, Chats: _Optional[_Iterable[_Union[Chat, _Mapping]]] = ...) -> None: ...

class ChatResp(_message.Message):
    __slots__ = ("Chat",)
    CHAT_FIELD_NUMBER: _ClassVar[int]
    Chat: Chat
    def __init__(self, Chat: _Optional[_Union[Chat, _Mapping]] = ...) -> None: ...

class SearchRequest(_message.Message):
    __slots__ = ("Input_data", "Chat_id")
    INPUT_DATA_FIELD_NUMBER: _ClassVar[int]
    CHAT_ID_FIELD_NUMBER: _ClassVar[int]
    Input_data: str
    Chat_id: int
    def __init__(self, Input_data: _Optional[str] = ..., Chat_id: _Optional[int] = ...) -> None: ...

class AuthorPaperReq(_message.Message):
    __slots__ = ("Author_ID", "State")
    AUTHOR_ID_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    Author_ID: int
    State: str
    def __init__(self, Author_ID: _Optional[int] = ..., State: _Optional[str] = ...) -> None: ...

class PaperResponse(_message.Message):
    __slots__ = ("ID", "Title", "Abstract", "Year", "Best_oa_location")
    ID_FIELD_NUMBER: _ClassVar[int]
    TITLE_FIELD_NUMBER: _ClassVar[int]
    ABSTRACT_FIELD_NUMBER: _ClassVar[int]
    YEAR_FIELD_NUMBER: _ClassVar[int]
    BEST_OA_LOCATION_FIELD_NUMBER: _ClassVar[int]
    ID: str
    Title: str
    Abstract: str
    Year: int
    Best_oa_location: str
    def __init__(self, ID: _Optional[str] = ..., Title: _Optional[str] = ..., Abstract: _Optional[str] = ..., Year: _Optional[int] = ..., Best_oa_location: _Optional[str] = ...) -> None: ...

class PapersResponse(_message.Message):
    __slots__ = ("Papers",)
    PAPERS_FIELD_NUMBER: _ClassVar[int]
    Papers: _containers.RepeatedCompositeFieldContainer[PaperResponse]
    def __init__(self, Papers: _Optional[_Iterable[_Union[PaperResponse, _Mapping]]] = ...) -> None: ...

class AddRequest(_message.Message):
    __slots__ = ("ID", "Title", "Abstract", "Year", "Best_oa_location", "Referenced_works", "Related_works", "State")
    ID_FIELD_NUMBER: _ClassVar[int]
    TITLE_FIELD_NUMBER: _ClassVar[int]
    ABSTRACT_FIELD_NUMBER: _ClassVar[int]
    YEAR_FIELD_NUMBER: _ClassVar[int]
    BEST_OA_LOCATION_FIELD_NUMBER: _ClassVar[int]
    REFERENCED_WORKS_FIELD_NUMBER: _ClassVar[int]
    RELATED_WORKS_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    ID: str
    Title: str
    Abstract: str
    Year: int
    Best_oa_location: str
    Referenced_works: _containers.RepeatedCompositeFieldContainer[Referenced_works]
    Related_works: _containers.RepeatedCompositeFieldContainer[Related_works]
    State: str
    def __init__(self, ID: _Optional[str] = ..., Title: _Optional[str] = ..., Abstract: _Optional[str] = ..., Year: _Optional[int] = ..., Best_oa_location: _Optional[str] = ..., Referenced_works: _Optional[_Iterable[_Union[Referenced_works, _Mapping]]] = ..., Related_works: _Optional[_Iterable[_Union[Related_works, _Mapping]]] = ..., State: _Optional[str] = ...) -> None: ...

class Referenced_works(_message.Message):
    __slots__ = ("ID",)
    ID_FIELD_NUMBER: _ClassVar[int]
    ID: str
    def __init__(self, ID: _Optional[str] = ...) -> None: ...

class Related_works(_message.Message):
    __slots__ = ("ID",)
    ID_FIELD_NUMBER: _ClassVar[int]
    ID: str
    def __init__(self, ID: _Optional[str] = ...) -> None: ...

class ErrorResponse(_message.Message):
    __slots__ = ("Error",)
    ERROR_FIELD_NUMBER: _ClassVar[int]
    Error: str
    def __init__(self, Error: _Optional[str] = ...) -> None: ...
