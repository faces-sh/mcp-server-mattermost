"""Custom exceptions for Mattermost MCP server."""


class MattermostMCPError(Exception):
    """Base exception for all Mattermost MCP errors."""


class ConfigurationError(MattermostMCPError):
    """Raised when configuration is invalid or missing."""


class MattermostAPIError(MattermostMCPError):
    """Raised when Mattermost API returns an error.

    ``reason_phrase`` and ``body`` are the EVIDENCE the failure envelope reproduces: the
    literal status line and whatever the server sent back, uninterpreted. They are set only
    when there really was an HTTP response, which is what tells a 401 off the wire apart from
    a missing local credential (see ``envelope.py``).

    Attributes:
        status_code: HTTP status code from API
        error_id: Mattermost error identifier
        reason_phrase: HTTP reason phrase from the response, when there was a response
        body: The response body, byte for byte, when there was a response
    """

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        error_id: str | None = None,
        *,
        reason_phrase: str | None = None,
        body: str | None = None,
    ) -> None:
        """Initialize API error.

        Args:
            message: Error message
            status_code: HTTP status code
            error_id: Mattermost error ID
            reason_phrase: HTTP reason phrase from the response
            body: Verbatim response body
        """
        super().__init__(message)
        self.status_code = status_code
        self.error_id = error_id
        self.reason_phrase = reason_phrase
        self.body = body

    @property
    def had_http_response(self) -> bool:
        """Whether this failure came from a real HTTP response (so a status line is honest)."""
        return self.status_code is not None and self.reason_phrase is not None

    def __str__(self) -> str:
        """Format error with status code and error_id."""
        parts = [super().__str__()]
        if self.status_code is not None:
            parts.append(f"status={self.status_code}")
        if self.error_id is not None:
            parts.append(f"error_id={self.error_id}")
        return " ".join(parts)


class RateLimitError(MattermostAPIError):
    """Raised when API rate limit is exceeded (429)."""

    def __init__(
        self,
        retry_after: int | None = None,
        *,
        reason_phrase: str | None = None,
        body: str | None = None,
    ) -> None:
        """Initialize rate limit error.

        Args:
            retry_after: Seconds to wait before retrying
            reason_phrase: HTTP reason phrase from the response
            body: Verbatim response body
        """
        super().__init__("Rate limit exceeded", status_code=429, reason_phrase=reason_phrase, body=body)
        self.retry_after = retry_after


class AuthenticationError(MattermostAPIError):
    """Raised when authentication fails (401)."""

    def __init__(
        self,
        message: str = "Authentication failed",
        *,
        reason_phrase: str | None = None,
        body: str | None = None,
    ) -> None:
        """Initialize authentication error.

        Args:
            message: Error message
            reason_phrase: HTTP reason phrase from the response
            body: Verbatim response body
        """
        super().__init__(message, status_code=401, reason_phrase=reason_phrase, body=body)


class NotFoundError(MattermostAPIError):
    """Raised when requested resource is not found (404)."""

    def __init__(
        self,
        message: str = "Resource not found",
        *,
        error_id: str | None = None,
        reason_phrase: str | None = None,
        body: str | None = None,
    ) -> None:
        """Initialize not found error.

        Args:
            message: Error message (from API response or default)
            error_id: Mattermost error identifier from API response
            reason_phrase: HTTP reason phrase from the response
            body: Verbatim response body
        """
        super().__init__(message, status_code=404, error_id=error_id, reason_phrase=reason_phrase, body=body)


class ValidationError(MattermostMCPError):
    """Raised when input validation fails."""


class FileValidationError(ValidationError):
    """Raised when file path validation fails.

    Attributes:
        file_path: The invalid file path
    """

    def __init__(self, file_path: str, message: str) -> None:
        """Initialize file validation error.

        Args:
            file_path: The file path that failed validation
            message: Description of the validation failure
        """
        super().__init__(f"{message}: {file_path}")
        self.file_path = file_path


class TransportError(MattermostMCPError):
    """Raised when the request never reached Mattermost: refused, unresolved, untrusted, or timed out.

    There is no HTTP response behind this, so the envelope carries a snake_case code and NO
    status line. ``code`` says which of the four it was; ``detail`` is the underlying
    exception's own text, reproduced rather than paraphrased.

    Attributes:
        code: Envelope code, e.g. ``connection_refused``
        detail: The transport library's own description of the failure
    """

    def __init__(self, code: str, detail: str) -> None:
        """Initialize transport error.

        Args:
            code: Envelope code (snake_case)
            detail: Underlying failure text
        """
        super().__init__(detail)
        self.code = code
        self.detail = detail
