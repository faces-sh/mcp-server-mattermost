"""Logging configuration for MCP server.

All logs go to stderr per MCP specification, and NO write to stderr may ever block the event loop.

WHY THAT SECOND HALF EXISTS. stderr is a PIPE when this server runs the way it ships, as a stdio
child. A pipe holds 64KB on macOS, and when it is full `write()` does not fail, it BLOCKS. A parent
that stops reading stderr, or never reads it, therefore stops the SERVER: logging is synchronous, it
runs on the event loop, and the loop is what answers tool calls. Nothing recovers, because nothing
times out.

Measured, not theorised. Pointed at a dead endpoint with stderr piped and undrained, this server
answered 28 tool calls at 0.09s each and then hung forever on the 29th, inside `logging`, with all
ten later calls starved behind it. Draining stderr in the same run: 38/38 answered and 131KB of logs
came out. The pipe holds 64.

The fix is one seam, `install_nonblocking_stderr`, installed before anything else runs. It has to be
the STREAM rather than our handler, because our logger is not the only writer: the first attempt
replaced this module's handler and the server hung in exactly the same place, one frame further up,
inside FastMCP's own logger. Anything that writes to stderr now goes through a writer thread, so a
parent that stops reading blocks that thread and nothing else.
"""

import atexit
import contextlib
import io
import json
import logging
import queue
import sys
import threading
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, ClassVar, Final


LOGGER_NAME: Final = "mcp-server-mattermost"

# How many characters may be waiting on the writer thread. Reached only when the reader on the other
# end of stderr has stopped, and at that point the choice is drop or die. One full sweep of every
# tool produces about 131KB, so this holds a whole session's worth of a stalled reader.
STDERR_BUFFER_LIMIT: Final = 1_048_576

# How long a flush or shutdown waits for the writer. Short on purpose: waiting on a write that is
# already blocked would put the hang back, at exit instead of mid-session.
_WRITER_JOIN_TIMEOUT: Final = 1.0

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

_SHUTDOWN = object()


class JSONFormatter(logging.Formatter):
    """Format log records as JSON lines."""

    # Fields that are part of LogRecord but not useful in JSON output
    EXCLUDE_FIELDS: ClassVar[set[str]] = {
        "name",
        "msg",
        "args",
        "created",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "exc_info",
        "exc_text",
        "thread",
        "threadName",
        "taskName",
        "message",
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format record as JSON line.

        Args:
            record: Log record to format

        Returns:
            JSON string
        """
        log_data: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "level": record.levelname,
            "message": record.getMessage(),
        }

        # Add extra fields from record
        log_data.update(
            {
                key: value
                for key, value in record.__dict__.items()
                if key not in self.EXCLUDE_FIELDS and not key.startswith("_")
            },
        )

        return json.dumps(log_data, default=str)


class NonBlockingStderr(io.TextIOBase):
    """stderr that hands the bytes to a writer thread instead of waiting for the pipe.

    Bounded, so a reader that never comes back cannot turn into unbounded memory either. Past the
    ceiling text is DROPPED and counted, and the count is reported the moment writing succeeds
    again, so a hole in the log is never mistaken for a quiet period. That is the honest trade: a
    dropped log line is recoverable, an extension frozen in the middle of somebody's turn is not.

    Attributes:
        dropped_chars: Characters discarded because the reader stopped. Non-zero means exactly that.
    """

    def __init__(self, stream: Any, limit: int = STDERR_BUFFER_LIMIT) -> None:  # noqa: ANN401
        """Wrap a stream and start its writer thread.

        Args:
            stream: The real stderr to write through.
            limit: Characters that may be queued before text is dropped.
        """
        super().__init__()
        self._stream = stream
        self._limit = limit
        self._queue: queue.Queue[Any] = queue.Queue()
        self._pending = 0
        self._lock = threading.Lock()
        self.dropped_chars = 0
        # Daemon, so a writer wedged in a blocked write can never hold up process exit.
        self._writer = threading.Thread(target=self._drain, name="stderr-writer", daemon=True)
        self._writer.start()

    def write(self, text: str) -> int:
        """Queue text for the writer thread. Never blocks.

        Args:
            text: Text to write.

        Returns:
            The number of characters accepted, which is all of them: a caller that believed a short
            write meant "retry the rest" would spin here.
        """
        if not text:
            return 0
        with self._lock:
            if self._pending + len(text) > self._limit:
                self.dropped_chars += len(text)
                return len(text)
            if self.dropped_chars:
                note = f"[stderr-writer dropped {self.dropped_chars} characters while the reader was stalled]\n"
                self.dropped_chars = 0
                text = note + text
            self._pending += len(text)
        self._queue.put(text)
        return len(text)

    def flush(self) -> None:
        """Do nothing, deliberately.

        `logging.StreamHandler.emit` flushes after EVERY record, so this is on the hot path and must
        never wait. A flush that waited even a little put the hang straight back: with the reader
        stalled, each log line cost the event loop a full second and the server died on call 6
        instead of call 29. The writer thread flushes the real stream after each chunk, so nothing
        is lost by returning here; only `close` waits, and only with a ceiling.
        """

    def isatty(self) -> bool:
        """Report the wrapped stream's tty-ness, which is what colour decisions read."""
        return bool(getattr(self._stream, "isatty", lambda: False)())

    def fileno(self) -> int:
        """Return the wrapped stream's descriptor."""
        return int(self._stream.fileno())

    @property
    def encoding(self) -> str:  # type: ignore[override]  # TextIOBase declares this writeable; ours mirrors the wrapped stream
        """Return the wrapped stream's encoding."""
        return str(getattr(self._stream, "encoding", "utf-8"))

    def _drain(self) -> None:
        """Write queued text, off whatever thread produced it. Blocking here is the whole point."""
        while True:
            chunk = self._queue.get()
            if chunk is _SHUTDOWN:
                return
            with contextlib.suppress(Exception):
                self._stream.write(chunk)
                self._stream.flush()
            with self._lock:
                self._pending -= len(chunk)

    def drain(self, timeout: float = _WRITER_JOIN_TIMEOUT) -> bool:
        """Wait, with a ceiling, for the writer to catch up.

        Not called from the logging path. It exists for shutdown and for tests that need to observe
        what was written.

        Args:
            timeout: Seconds to wait before giving up.

        Returns:
            True if the queue emptied, False if the writer is still behind (or blocked).
        """
        idle = threading.Event()
        step = 0.01
        waited = 0.0
        while waited < timeout:
            with self._lock:
                if self._pending == 0:
                    return True
            idle.wait(step)
            waited += step
        return False

    def close(self) -> None:
        """Stop the writer thread, without waiting on a write that may already be blocked."""
        self.drain()
        self._queue.put(_SHUTDOWN)
        self._writer.join(timeout=_WRITER_JOIN_TIMEOUT)


def install_nonblocking_stderr() -> NonBlockingStderr | None:
    """Replace sys.stderr with a writer that cannot block the caller.

    Call this BEFORE the server is imported, so every logger in the process, ours and FastMCP's and
    anything either of them pulls in, is writing through it. Installing it on one logger's handler is
    not enough and was measured not to be.

    Returns:
        The installed stream, or None if one is already installed.
    """
    if isinstance(sys.stderr, NonBlockingStderr):
        return None
    stream = NonBlockingStderr(sys.stderr)
    sys.stderr = stream
    atexit.register(stream.close)
    return stream


def setup_logging(level: str = "INFO", log_format: str = "json") -> logging.Logger:
    """Configure logging to stderr for MCP compliance.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Output format ('json' for production, 'text' for development)

    Returns:
        Configured logger instance

    Raises:
        ValueError: If level is not a valid logging level
    """
    log = logging.getLogger(LOGGER_NAME)
    log.propagate = False

    level_upper = level.upper()
    level_value = getattr(logging, level_upper, None)
    if level_value is None or not isinstance(level_value, int):
        valid_levels = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
        msg = f"Invalid logging level: {level}. Must be one of {valid_levels}"
        raise ValueError(msg)
    log.setLevel(level_value)

    if not log.handlers:
        handler = logging.StreamHandler(sys.stderr)
        if log_format == "json":
            handler.setFormatter(JSONFormatter())
        else:
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                ),
            )
        log.addHandler(handler)

    return log


logger = logging.getLogger(LOGGER_NAME)
