"""Domain-level exceptions.

Keeping custom exceptions makes it easier to:
- map errors to HTTP status codes
- create stable error messages for UI clients
- add monitoring/alerts in the future
"""


class AppError(RuntimeError):
    """Base application error."""


class ValidationError(AppError):
    pass


class ExternalServiceError(AppError):
    """Raised when an external dependency (Endee, OpenAI) fails."""


class NotFoundError(AppError):
    pass
