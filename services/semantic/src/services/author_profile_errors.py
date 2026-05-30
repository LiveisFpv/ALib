class AuthorProfileError(RuntimeError):
    pass


class AuthorProfileConflictError(AuthorProfileError):
    pass


class AuthorProfileValidationError(AuthorProfileError):
    pass
