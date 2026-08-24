"""The uniform failure envelope every tool result carries when something goes wrong.

Contract (Maestro's ``docs/MCP_FAILURE_ENVELOPE.md``)::

    [<code>] <one plain sentence: what did not happen>
    HTTP <status> <reason phrase>
    <the provider's response body, verbatim>

Line 1 is for a person and for the model. Lines 2 and 3 are the evidence. The status line
appears only when there really was an HTTP response; a transport failure carries a
snake_case code and no status line, because inventing a status would be a lie. The body is
whatever Mattermost returned, byte for byte, with credentials struck out and nothing else
touched: deciding what a 403 MEANS is the caller's job, and every bug this shape exists to
prevent came from a layer that decided early.
"""

import re
from http import HTTPStatus

from fastmcp.exceptions import ValidationError as FastMCPValidationError
from pydantic import ValidationError as PydanticValidationError


BODY_LIMIT = 4000
# How far down an exception chain to look for the failure that knows what happened.
_CAUSE_CHAIN_LIMIT = 12
TRUNCATION_SUFFIX = " ...[truncated]"
REDACTED = "<redacted>"

# Keys whose VALUE is a credential wherever it appears (rule 8). ``token`` is on the list
# because a Mattermost session object carries the personal access token under exactly that
# name, so an echoed session would otherwise leak it.
_SECRET_KEYS = (
    "access_token",
    "refresh_token",
    "session_token",
    "id_token",
    "client_secret",
    "api_key",
    "apikey",
    "password",
    "token",
)
_SECRET_HEADERS = (
    "authorization",
    "proxy-authorization",
    "cookie",
    "set-cookie",
    "x-auth-token",
    "x-requested-with-token",
)

_KEY_ALT = "|".join(re.escape(key) for key in _SECRET_KEYS)
_HEADER_ALT = "|".join(re.escape(header) for header in _SECRET_HEADERS)

# ``Authorization: Bearer x`` on its own line (a raw header dump).
_HEADER_LINE_RE = re.compile(rf"^([ \t]*(?:{_HEADER_ALT})[ \t]*:[ \t]*).*$", re.IGNORECASE | re.MULTILINE)
# ``"Authorization": "Bearer x"`` inside a JSON object (an echoed request).
_HEADER_JSON_RE = re.compile(rf'("(?:{_HEADER_ALT})"\s*:\s*)"(?:[^"\\]|\\.)*"', re.IGNORECASE)
# ``"access_token": "x"`` and the single-quoted Python-repr spelling of the same thing.
_JSON_SECRET_RE = re.compile(rf'("(?:{_KEY_ALT})"\s*:\s*)"(?:[^"\\]|\\.)*"', re.IGNORECASE)
_REPR_SECRET_RE = re.compile(rf"('(?:{_KEY_ALT})'\s*:\s*)'(?:[^'\\]|\\.)*'", re.IGNORECASE)
# ``access_token=x`` in a query string or form body.
_QUERY_SECRET_RE = re.compile(rf"\b((?:{_KEY_ALT})=)[^&\s\"'<>]+", re.IGNORECASE)
# A bare credential after its scheme, anywhere.
_SCHEME_RE = re.compile(r"\b(Bearer|Basic|Token)\s+[A-Za-z0-9\-._~+/=]{8,}", re.IGNORECASE)


def redact(text: str) -> str:
    """Strike credentials out of text that is otherwise reproduced verbatim.

    Args:
        text: Any text about to be echoed back to the caller.

    Returns:
        The same text with every credential value replaced by ``<redacted>``.
    """
    if not text:
        return text
    text = _HEADER_LINE_RE.sub(rf"\g<1>{REDACTED}", text)
    text = _HEADER_JSON_RE.sub(rf'\g<1>"{REDACTED}"', text)
    text = _JSON_SECRET_RE.sub(rf'\g<1>"{REDACTED}"', text)
    text = _REPR_SECRET_RE.sub(rf"\g<1>'{REDACTED}'", text)
    text = _QUERY_SECRET_RE.sub(rf"\g<1>{REDACTED}", text)
    return _SCHEME_RE.sub(rf"\g<1> {REDACTED}", text)


def truncate(text: str) -> str:
    """Cap an echoed body at the contract's 4000 characters.

    Args:
        text: Body text, already redacted.

    Returns:
        The text, or its first 4000 characters followed by a truncation marker.
    """
    if len(text) <= BODY_LIMIT:
        return text
    return text[:BODY_LIMIT] + TRUNCATION_SUFFIX


def one_line(text: str) -> str:
    """Flatten a sentence so line 1 stays one line.

    The code has to LEAD the text, and a caller matching on it reads the first line. A
    multi-line first line splits the sentence across the status-line slot and breaks that.

    Args:
        text: The sentence, possibly containing newlines.

    Returns:
        The same words on one line.
    """
    return " ".join(text.split())


def status_line(status_code: int, reason_phrase: str | None) -> str:
    """Build the literal status line for a real HTTP response.

    Args:
        status_code: The status the server sent.
        reason_phrase: The reason phrase the server sent, if any.

    Returns:
        A line of the form ``HTTP 403 Forbidden``.
    """
    reason = (reason_phrase or "").strip()
    if not reason:
        try:
            reason = HTTPStatus(status_code).phrase
        except ValueError:
            reason = ""
    return f"HTTP {status_code} {reason}".rstrip()


def format_envelope(
    code: str,
    sentence: str,
    line: str | None = None,
    body: str | None = None,
) -> str:
    """Assemble the three-line envelope.

    Args:
        code: The bracketed code, ``http_<status>`` or snake_case.
        sentence: One plain sentence saying what did not happen.
        line: The literal HTTP status line, or None when the failure was not HTTP.
        body: The provider's response body, or None when there was none.

    Returns:
        The envelope text.
    """
    lines = [f"[{code}] {one_line(sentence)}"]
    if line:
        lines.append(line)
    if body:
        evidence = truncate(redact(body))
        if evidence:
            lines.append(evidence)
    return "\n".join(lines)


# What the four-hundreds and five-hundreds MEAN in plain words. Descriptions of what happened,
# never advice about what to do next: this server does not know whether the caller can fix it.
_HTTP_REASONS = {
    301: "the server redirected the request, and this client does not follow redirects.",
    302: "the server redirected the request, and this client does not follow redirects.",
    303: "the server redirected the request, and this client does not follow redirects.",
    307: "the server redirected the request, and this client does not follow redirects.",
    308: "the server redirected the request, and this client does not follow redirects.",
    400: "the server rejected the request as malformed.",
    401: "the server rejected the access token.",
    403: "the account does not have permission.",
    404: "the server could not find it.",
    405: "the server does not allow that on this resource.",
    408: "the server timed out waiting for the request.",
    409: "the request conflicts with the current state.",
    410: "it is gone from the server.",
    413: "the request was too large for the server.",
    422: "the server could not process the request.",
    429: "the server is rate limiting this account.",
    501: "the server does not implement it.",
    502: "the server got a bad answer from upstream.",
    503: "the server is unavailable.",
    504: "the server timed out upstream.",
}

# Transport failures, where there is no status line to lean on.
_TRANSPORT_REASONS = {
    "connection_refused": "nothing accepted a connection at the server address.",
    "connection_failed": "the connection to the server could not be established.",
    "dns_failure": "the server address did not resolve.",
    "tls_error": "the TLS connection to the server could not be established.",
    "timeout": "the server did not answer in time.",
    "too_many_redirects": "the server redirected more times than the client follows.",
    "network_error": "the request never reached the server.",
}


def _http_reason(status_code: int) -> str:
    """Plain-words reason for an HTTP status.

    Args:
        status_code: The status the server sent.

    Returns:
        One clause completing "Could not <action>: ...".
    """
    known = _HTTP_REASONS.get(status_code)
    if known is not None:
        return known
    if status_code >= HTTPStatus.INTERNAL_SERVER_ERROR:
        return "the server failed."
    if status_code < HTTPStatus.BAD_REQUEST:
        return "the server answered with something the client cannot use."
    return "the server rejected the request."


def action_phrase(tool_name: str) -> str:
    """Turn a tool name into words a person reads.

    Args:
        tool_name: The MCP tool that failed, e.g. ``post_message``.

    Returns:
        The same act in plain words, e.g. ``post message``.
    """
    cleaned = (tool_name or "").strip().replace("_", " ").strip()
    return cleaned or "do that"


def _causes(exc: BaseException) -> list[BaseException]:
    """Walk an exception chain outermost first.

    FastMCP re-raises a tool's failure wrapped in its own ToolError, so the exception that
    knows what happened is never the one handed to the middleware.

    Args:
        exc: The exception caught at the boundary.

    Returns:
        The exception and its causes, in order, bounded so a cycle cannot hang.
    """
    chain: list[BaseException] = []
    cursor: BaseException | None = exc
    while cursor is not None and len(chain) < _CAUSE_CHAIN_LIMIT:
        if any(cursor is seen for seen in chain):
            break
        chain.append(cursor)
        cursor = cursor.__cause__ or cursor.__context__
    return chain


def envelope_for(exc: BaseException, tool_name: str) -> str:  # noqa: PLR0911 - one return per failure kind
    """Render any failure as the envelope.

    Args:
        exc: The exception that ended the tool call.
        tool_name: The tool the caller asked for.

    Returns:
        The envelope text, ready to be the tool result with ``isError`` set.
    """
    # Imported here because exceptions.py is the lower layer: envelope.py must not be part of
    # its import cycle.
    from .circuit_buffer import CircuitError  # noqa: PLC0415
    from .exceptions import ConfigurationError, MattermostAPIError, TransportError, ValidationError  # noqa: PLC0415

    action = action_phrase(tool_name)

    for cause in _causes(exc):
        if isinstance(cause, MattermostAPIError):
            status = cause.status_code
            if cause.had_http_response and status is not None:
                return format_envelope(
                    f"http_{status}",
                    f"Could not {action}: {_http_reason(status)}",
                    status_line(status, cause.reason_phrase),
                    cause.body,
                )
            # A 401 raised WITHOUT a response is a missing local credential, not a refusal
            # from the server. Rule 4: no status line, because there was no status.
            code = "no_credentials" if status == HTTPStatus.UNAUTHORIZED else "bad_request"
            return format_envelope(code, f"Could not {action}: {one_line(str(cause))}")

        if isinstance(cause, TransportError):
            reason = _TRANSPORT_REASONS.get(cause.code, "the request never reached the server.")
            return format_envelope(
                cause.code,
                f"Could not {action}: {reason}",
                None,
                cause.detail,
            )

        if isinstance(cause, CircuitError):
            return format_envelope("handle_expired", f"Could not {action}: {one_line(str(cause))}")

        if isinstance(cause, ConfigurationError):
            return format_envelope("configuration_error", f"Could not {action}: {one_line(str(cause))}")

        if isinstance(cause, ValidationError):
            return format_envelope("bad_request", f"Could not {action}: {one_line(str(cause))}")

        if isinstance(cause, FastMCPValidationError):
            return format_envelope(
                "bad_request",
                f"Could not {action}: the arguments were rejected.",
                None,
                str(cause),
            )

        if isinstance(cause, PydanticValidationError):
            return format_envelope(
                "invalid_response",
                f"Could not {action}: the server's answer did not match the expected shape.",
                None,
                str(cause),
            )

    return format_envelope("internal_error", f"Could not {action}: {_unexpected(exc)}")


def _unexpected(exc: BaseException) -> str:
    """Describe a failure nothing else recognised, without leaking a stack trace.

    Args:
        exc: The unrecognised exception.

    Returns:
        A short clause naming what happened.
    """
    text = one_line(str(exc))
    return text or f"the server hit an unexpected {type(exc).__name__}."
