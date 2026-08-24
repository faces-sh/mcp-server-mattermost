"""The uniform failure envelope: shape, evidence, redaction, and the paths that used to swallow.

Every assertion here is written against the contract in Maestro's docs/MCP_FAILURE_ENVELOPE.md:
isError is always set, the code always leads, an HTTP failure always carries the literal status
line AND the server's body byte for byte, a transport failure never invents one, and no credential
survives the round trip.
"""

import socket
import ssl

import httpx
import pytest
import respx

from mcp_server_mattermost import envelope
from mcp_server_mattermost.client import MattermostClient, _transport_error
from mcp_server_mattermost.exceptions import AuthenticationError, TransportError


BASE = "https://test.mattermost.com/api/v4"


def _chain(outer: BaseException, inner: BaseException) -> BaseException:
    """Build the cause chain httpx really produces (mapped error raised FROM the socket error)."""
    outer.__cause__ = inner
    outer.__context__ = inner
    return outer


def _free_port() -> int:
    """Return a port nothing is listening on."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


async def _call(tool: str, args: dict) -> tuple[bool, str]:
    """Call a tool over the real MCP wire and return (is_error, text)."""
    from fastmcp import Client

    from mcp_server_mattermost.server import mcp

    async with Client(mcp) as client:
        result = await client.call_tool(tool, args, raise_on_error=False)
        text = "".join(getattr(block, "text", "") for block in result.content)
        return bool(result.is_error), text


class TestRedaction:
    """Rule 8: no secret survives, and nothing else is touched."""

    def test_authorization_header_line(self):
        text = "GET /api/v4/users/me\nAuthorization: Bearer abcdef0123456789\nAccept: */*"
        out = envelope.redact(text)
        assert "abcdef0123456789" not in out
        assert "Authorization: <redacted>" in out
        assert "Accept: */*" in out

    def test_authorization_inside_json(self):
        out = envelope.redact('{"headers":{"Authorization":"Bearer abcdef0123456789"},"path":"/posts"}')
        assert "abcdef0123456789" not in out
        assert '"Authorization": "<redacted>"' in out or '"Authorization":"<redacted>"' in out
        assert '"path":"/posts"' in out

    def test_cookies(self):
        out = envelope.redact("Set-Cookie: MMAUTHTOKEN=zzzzzzzzzzzz; Path=/\nCookie: MMCSRF=yyyyyyyy")
        assert "MMAUTHTOKEN" not in out
        assert "MMCSRF" not in out

    def test_token_values_in_json_body(self):
        body = '{"token":"xoxb-1234567890","access_token":"aaaa","refresh_token":"bbbb","client_secret":"cccc"}'
        out = envelope.redact(body)
        for secret in ("xoxb-1234567890", "aaaa", "bbbb", "cccc"):
            assert secret not in out
        assert out.count("<redacted>") == 4

    def test_query_string_credential(self):
        out = envelope.redact("POST /oauth/access_token?access_token=abcd1234&scope=read")
        assert "abcd1234" not in out
        assert "scope=read" in out

    def test_bare_bearer_anywhere(self):
        out = envelope.redact("upstream said: Bearer qqqqqqqqqqqqqqqq is not valid")
        assert "qqqqqqqqqqqqqqqq" not in out
        assert "is not valid" in out

    def test_ordinary_body_is_untouched(self):
        body = '{"id":"api.context.permissions.app_error","message":"You do not have the appropriate permissions."}'
        assert envelope.redact(body) == body


class TestShape:
    """Rules 2 and 4: the code leads; the status line is literal or absent."""

    def test_http_failure_carries_status_line(self):
        text = envelope.format_envelope("http_403", "Could not edit it: no permission.", "HTTP 403 Forbidden", "{}")
        assert text.splitlines()[0] == "[http_403] Could not edit it: no permission."
        assert text.splitlines()[1] == "HTTP 403 Forbidden"

    def test_non_http_failure_has_no_status_line(self):
        text = envelope.format_envelope("timeout", "Could not list teams: it did not answer.")
        assert "HTTP " not in text

    def test_body_is_capped_with_a_marker(self):
        text = envelope.format_envelope("http_500", "Could not do it: the server failed.", "HTTP 500 x", "y" * 5000)
        body = text.split("\n", 2)[2]
        assert body.endswith(" ...[truncated]")
        assert len(body) == envelope.BODY_LIMIT + len(envelope.TRUNCATION_SUFFIX)

    def test_first_line_stays_one_line(self):
        text = envelope.format_envelope("bad_request", "Could not do it:\nsomething\nwrapped.")
        assert text == "[bad_request] Could not do it: something wrapped."


class TestTransportClassification:
    """Rule 4 again, from the other side: which non-HTTP failure was it."""

    @pytest.mark.parametrize(
        ("inner", "expected"),
        [
            (ConnectionRefusedError(61, "Connection refused"), "connection_refused"),
            (socket.gaierror(8, "nodename nor servname provided"), "dns_failure"),
            (ssl.SSLCertVerificationError("certificate verify failed"), "tls_error"),
        ],
    )
    def test_connect_error_causes(self, inner, expected):
        exc = _chain(httpx.ConnectError("All connection attempts failed"), inner)
        assert _transport_error(exc).code == expected

    def test_timeout(self):
        assert _transport_error(httpx.ReadTimeout("timed out")).code == "timeout"

    def test_too_many_redirects(self):
        assert _transport_error(httpx.TooManyRedirects("exceeded max redirects")).code == "too_many_redirects"

    def test_transport_envelope_has_no_status_line(self):
        failure = TransportError("connection_refused", "All connection attempts failed")
        text = envelope.envelope_for(failure, "list_teams")
        assert text.startswith("[connection_refused] Could not list teams:")
        assert "HTTP " not in text
        assert "All connection attempts failed" in text


class TestMissingCredentialIsNotAnHttpFailure:
    """A 401 raised locally never had a status, so rule 4 forbids inventing one."""

    def test_no_credentials(self):
        exc = AuthenticationError("Mattermost token is required for this auth mode")
        text = envelope.envelope_for(exc, "get_me")
        assert text.startswith("[no_credentials] Could not get me:")
        assert "HTTP " not in text


@pytest.mark.asyncio
class TestOverTheWire:
    """The whole path, as a client sees it."""

    async def test_401_carries_status_line_and_verbatim_body(self, mock_settings):
        body = (
            '{"id":"api.context.session_expired.app_error",'
            '"message":"Invalid or expired session, please login again.","status_code":401}'
        )
        with respx.mock(assert_all_called=False) as rx:
            rx.get(f"{BASE}/users/me").mock(return_value=httpx.Response(401, text=body))
            is_error, text = await _call("get_me", {})
        assert is_error is True
        lines = text.splitlines()
        assert lines[0].startswith("[http_401] ")
        assert lines[1] == "HTTP 401 Unauthorized"
        assert body in text

    async def test_403_body_is_not_summarised(self, mock_settings):
        body = '{"id":"api.context.permissions.app_error","message":"You do not have the appropriate permissions."}'
        with respx.mock(assert_all_called=False) as rx:
            rx.get(f"{BASE}/channels/{'c' * 26}").mock(return_value=httpx.Response(403, text=body))
            is_error, text = await _call("get_channel", {"channel_id": "c" * 26})
        assert is_error is True
        assert text.splitlines()[0].startswith("[http_403] ")
        assert body in text

    async def test_echoed_credentials_are_redacted(self, mock_settings):
        leaky = 'upstream request was {"Authorization": "Bearer abcdef0123456789", "access_token": "zzzz1111"}'
        with respx.mock(assert_all_called=False) as rx:
            rx.post(f"{BASE}/posts").mock(return_value=httpx.Response(500, text=leaky))
            is_error, text = await _call("post_message", {"channel_id": "c" * 26, "message": "hi"})
        assert is_error is True
        assert "abcdef0123456789" not in text
        assert "zzzz1111" not in text
        assert "<redacted>" in text

    async def test_connection_refused_is_real_and_has_no_status_line(self, monkeypatch):
        """Not a mock: a genuinely closed port, so the whole httpx cause chain is the real one."""
        from mcp_server_mattermost.config import get_settings

        get_settings.cache_clear()
        monkeypatch.setenv("MATTERMOST_URL", f"http://127.0.0.1:{_free_port()}")
        monkeypatch.setenv("MATTERMOST_TOKEN", "test-token-12345")
        try:
            is_error, text = await _call("list_teams", {})
        finally:
            get_settings.cache_clear()
        assert is_error is True
        assert text.startswith("[connection_refused] ")
        assert "HTTP " not in text

    async def test_bad_arguments_are_a_bad_request(self, mock_settings):
        is_error, text = await _call("post_message", {"channel_id": "not-an-id", "message": "hi"})
        assert is_error is True
        assert text.startswith("[bad_request] ")
        assert "HTTP " not in text

    async def test_local_validation_failure(self, mock_settings):
        is_error, text = await _call(
            "create_bookmark",
            {"channel_id": "c" * 26, "display_name": "x", "bookmark_type": "link"},
        )
        assert is_error is True
        assert text.startswith("[bad_request] ")

    async def test_success_is_untouched(self, mock_settings):
        with respx.mock(assert_all_called=False) as rx:
            rx.get(f"{BASE}/users/me/teams").mock(return_value=httpx.Response(200, json=[]))
            is_error, text = await _call("list_teams", {})
        assert is_error is False
        assert not text.startswith("[")


@pytest.mark.asyncio
class TestUnresolvableHandleIsNotSwallowed:
    """Rule 6, on the one path in this server that silently succeeded on nonsense."""

    async def test_circuit_error_surfaces_as_a_failure(self, mock_settings, monkeypatch):
        from mcp_server_mattermost import circuit_buffer

        def _boom(_args):
            msg = "Unknown or expired circuit slug @@h3@@; it is no longer cached, re-fetch it."
            raise circuit_buffer.CircuitError(msg)

        monkeypatch.setattr(circuit_buffer, "resolve_args", _boom)
        with respx.mock(assert_all_called=False) as rx:
            route = rx.post(f"{BASE}/posts").mock(return_value=httpx.Response(200, json={}))
            is_error, text = await _call("post_message", {"channel_id": "c" * 26, "message": "@@h3@@"})
        assert is_error is True
        assert text.startswith("[handle_expired] ")
        assert not route.called, "the tool ran with an unresolved handle instead of failing"


class TestClientKeepsTheEvidence:
    """The client must not throw the body away before anyone can read it."""

    @pytest.mark.parametrize("status", [401, 403, 404, 429, 500])
    def test_every_error_status_keeps_body_and_reason(self, mock_settings, status):
        from mcp_server_mattermost.config import get_settings
        from mcp_server_mattermost.exceptions import MattermostAPIError

        client = MattermostClient(get_settings())
        body = f'{{"message":"failure {status}"}}'
        with pytest.raises(MattermostAPIError) as caught:
            client._handle_response(httpx.Response(status, text=body))
        assert caught.value.body == body
        assert caught.value.had_http_response is True


@pytest.mark.asyncio
class TestRedirectIsNotAnEmptyAnswer:
    """Rule 6: an unfollowed redirect used to come back as "no results"."""

    async def test_redirect_to_login_page_fails(self, mock_settings):
        with respx.mock(assert_all_called=False) as rx:
            rx.get(f"{BASE}/users/me/teams").mock(
                return_value=httpx.Response(302, headers={"Location": "https://sso.example.com/login"}),
            )
            is_error, text = await _call("list_teams", {})
        assert is_error is True, "a redirect reported an empty team list instead of failing"
        assert text.startswith("[http_302] ")
        assert text.splitlines()[1].startswith("HTTP 302 ")
