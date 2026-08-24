"""Middleware for structured logging of tool calls."""

import time
from collections.abc import Awaitable, Callable
from typing import Any

from fastmcp.exceptions import DisabledError, NotFoundError
from fastmcp.server.dependencies import get_http_request
from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp.tools.base import ToolResult
from mcp.types import TextContent

from .envelope import envelope_for
from .logging import logger, request_id_var
from . import circuit_buffer   # Maestro handle bus (docs/reqs/007); no-op outside Maestro


class LoggingMiddleware(Middleware):
    """Middleware that logs all tool calls with request correlation."""

    WHITELISTED_PARAMS: frozenset[str] = frozenset(
        {
            # IDs
            "team_id",
            "channel_id",
            "user_id",
            "post_id",
            "file_id",
            "bookmark_id",
            "root_id",
            # Pagination
            "page",
            "per_page",
            "limit",
            "offset",
            # Flags
            "include_deleted",
            "exclude_bots",
            "is_or_search",
            # Bookmarks
            "bookmarks_since",
            "new_sort_order",
            "bookmark_type",
        },
    )

    async def on_call_tool(
        self,
        context: MiddlewareContext,
        call_next: Callable[[MiddlewareContext], Awaitable[Any]],
    ) -> Any:  # noqa: ANN401
        """Log tool call start, success, and error events.

        Args:
            context: Middleware context with message and FastMCP context
            call_next: Next middleware or tool handler

        Returns:
            Tool result

        Raises:
            Exception: Re-raises any exception from tool after logging
        """
        request_id = self._get_request_id(context)
        request_id_var.set(request_id)

        tool_name = context.message.name
        params = self._redact_params(context.message.arguments or {})

        start_time = time.monotonic()

        logger.info(
            "Tool call started",
            extra={
                "event": "tool_call_start",
                "request_id": request_id,
                "tool": tool_name,
                "params": params,
            },
        )

        try:
            result = await call_next(context)
        except Exception as e:
            duration_ms = (time.monotonic() - start_time) * 1000

            logger.error(
                "Tool call failed",
                extra={
                    "event": "tool_call_error",
                    "request_id": request_id,
                    "tool": tool_name,
                    "duration_ms": round(duration_ms, 2),
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                },
            )

            raise
        else:
            duration_ms = (time.monotonic() - start_time) * 1000

            logger.info(
                "Tool call completed",
                extra={
                    "event": "tool_call_success",
                    "request_id": request_id,
                    "tool": tool_name,
                    "duration_ms": round(duration_ms, 2),
                },
            )

            return result

    def _get_request_id(self, context: MiddlewareContext) -> str:
        """Get request ID from HTTP header or FastMCP context.

        Tries X-Request-ID header first (HTTP transport),
        falls back to FastMCP's request_id (stdio transport).

        Args:
            context: Middleware context

        Returns:
            Request ID string
        """
        try:
            http_request = get_http_request()
            client_request_id = http_request.headers.get("X-Request-ID")
            if client_request_id:
                return client_request_id
        except Exception:  # noqa: BLE001, S110
            # stdio transport has no HTTP request - fall back to FastMCP request_id
            pass

        # Fall back to FastMCP context request_id, or generate one if not available
        if context.fastmcp_context and hasattr(context.fastmcp_context, "request_id"):
            return context.fastmcp_context.request_id
        return "unknown"

    def _redact_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Filter params to only include whitelisted keys.

        Args:
            params: Original tool parameters

        Returns:
            Filtered dict with only safe params
        """
        return {k: v for k, v in params.items() if k in self.WHITELISTED_PARAMS}


class CircuitMiddleware(Middleware):
    """Maestro handle bus (docs/reqs/007): expand any @@hN@@ handle in a tool's arguments before it runs, and
    park a large text result behind a handle on the way out so it can flow into the next tool by reference.
    No-op when the circuit env is absent (server run outside Maestro), so the server still works standalone.
    """

    async def on_call_tool(
        self,
        context: MiddlewareContext,
        call_next: Callable[[MiddlewareContext], Awaitable[Any]],
    ) -> Any:  # noqa: ANN401
        try:
            if context.message is not None and getattr(context.message, "arguments", None):
                context.message.arguments = circuit_buffer.resolve_args(context.message.arguments)
        except circuit_buffer.CircuitError:
            # A handle we cannot resolve is a FAILURE, not a hiccup. Swallowing it here ran the
            # tool with the literal "@@h3@@" in place of the payload the caller meant, and the
            # tool then succeeded on nonsense. circuit_buffer says so in as many words: "fails
            # loud, never silently passes the token".
            raise
        except Exception:  # noqa: BLE001 - any other circuit hiccup must never break a tool call
            pass
        result = await call_next(context)
        try:
            content = getattr(result, "content", None)
            if content and getattr(content[0], "type", None) == "text":
                content[0].text = circuit_buffer.wrap_result(content[0].text)
        except Exception:  # noqa: BLE001
            pass
        return result


class EnvelopeMiddleware(Middleware):
    """Turn every failure into the one shape Maestro parses (docs/MCP_FAILURE_ENVELOPE.md).

    OUTERMOST on purpose, and registered first so it is. By the time an exception reaches here
    FastMCP has already wrapped it as ``Error calling tool 'x': ...``, which buries the code the
    caller matches on and throws the provider's own words away. This rebuilds the result from the
    exception CHAIN instead, so the code leads the text and the server's response body is still
    attached, and the inner middleware still sees the raw exception it always saw.

    "Unknown tool" and "disabled tool" pass straight through: they say the CALL was wrong, not
    that a tool failed, and FastMCP already answers them precisely.
    """

    async def on_call_tool(
        self,
        context: MiddlewareContext,
        call_next: Callable[[MiddlewareContext], Awaitable[Any]],
    ) -> Any:  # noqa: ANN401
        """Run the tool and, on any failure, answer with the envelope.

        Args:
            context: Middleware context with message and FastMCP context
            call_next: Next middleware or tool handler

        Returns:
            The tool result, or an error result carrying the envelope.

        Raises:
            NotFoundError: If the tool does not exist
            DisabledError: If the tool is disabled
        """
        try:
            return await call_next(context)
        except (NotFoundError, DisabledError):
            raise
        except Exception as exc:  # noqa: BLE001 - every failure leaves as one shape
            tool_name = getattr(context.message, "name", "") or ""
            text = envelope_for(exc, tool_name)
            return ToolResult(content=[TextContent(type="text", text=text)], is_error=True)
