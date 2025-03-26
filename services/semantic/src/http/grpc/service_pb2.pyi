from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class SearchRequest(_message.Message):
    __slots__ = ("Input_data",)
    INPUT_DATA_FIELD_NUMBER: _ClassVar[int]
    Input_data: str
    def __init__(self, Input_data: _Optional[str] = ...) -> None: ...

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
    __slots__ = ("ID", "Title", "Abstract", "Year", "Best_oa_location", "Referenced_works", "Related_works")
    ID_FIELD_NUMBER: _ClassVar[int]
    TITLE_FIELD_NUMBER: _ClassVar[int]
    ABSTRACT_FIELD_NUMBER: _ClassVar[int]
    YEAR_FIELD_NUMBER: _ClassVar[int]
    BEST_OA_LOCATION_FIELD_NUMBER: _ClassVar[int]
    REFERENCED_WORKS_FIELD_NUMBER: _ClassVar[int]
    RELATED_WORKS_FIELD_NUMBER: _ClassVar[int]
    ID: str
    Title: str
    Abstract: str
    Year: int
    Best_oa_location: str
    Referenced_works: _containers.RepeatedCompositeFieldContainer[Referenced_works]
    Related_works: _containers.RepeatedCompositeFieldContainer[Related_works]
    def __init__(self, ID: _Optional[str] = ..., Title: _Optional[str] = ..., Abstract: _Optional[str] = ..., Year: _Optional[int] = ..., Best_oa_location: _Optional[str] = ..., Referenced_works: _Optional[_Iterable[_Union[Referenced_works, _Mapping]]] = ..., Related_works: _Optional[_Iterable[_Union[Related_works, _Mapping]]] = ...) -> None: ...

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
