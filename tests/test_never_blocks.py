"""The server must keep answering. THE TIMEOUT IS THE ASSERTION in every test here.

A test that waits for a reply cannot catch a hang, it just becomes one. So every check below reads
replies on a separate thread and fails on a DEADLINE, never on a blocking read.

What these protect, both measured before they were written:

1. stderr is a pipe when this server runs as a stdio child, it holds 64KB, and a full pipe makes
   `write()` BLOCK rather than fail. Logging is synchronous and sits on the event loop, so a parent
   that stops reading stderr used to stop the server: 28 tool calls at 0.09s, then a permanent hang
   on the 29th with every later call starved behind it.
2. circuit_buffer talks to Maestro's broker with synchronous urllib on a 10 second timeout. Called
   from the middleware directly, one tool call froze the whole loop for 10.1 seconds.
"""

import json
import subprocess
import sys
import threading
import time

import pytest


# Generous enough that a slow machine is not a failure, far under the FOREVER these catch.
CALL_DEADLINE = 5.0
STARTUP_DEADLINE = 20.0
NOTHING = "zzz-does-not-exist"


class StdioServer:
    """The real server over real stdio, with stderr piped and DELIBERATELY never drained.

    Not draining is the point. It is what a parent that only cares about stdout does, and it is the
    condition that used to wedge the server. A test that drained stderr would pass on the bug.
    """

    def __init__(self) -> None:
        """Start the server and the stdout reader thread."""
        self.proc = subprocess.Popen(  # noqa: S603
            [sys.executable, "-m", "mcp_server_mattermost"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,  # piped and never read: that is the hazard being tested
            env={
                "PATH": "/usr/bin:/bin",
                "MATTERMOST_URL": "http://127.0.0.1:9",  # nothing listening, so every call fails fast
                "MATTERMOST_TOKEN": "zzz",
                "PYTHONPATH": "src",
                "PYTHONUNBUFFERED": "1",
            },
            text=True,
            bufsize=1,
        )
        self._seen: dict[int, dict] = {}
        self._id = 0
        # A READER THREAD, because readline() blocks and a deadline checked between blocking reads
        # is not a deadline at all.
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        for raw in self.proc.stdout:
            line = raw.strip()
            if line.startswith("{"):
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if message.get("id") is not None:
                    self._seen[message["id"]] = message

    def rpc(self, method: str, params: dict, deadline: float = CALL_DEADLINE) -> dict | None:
        """Send a request and wait no longer than the deadline for its reply.

        Args:
            method: JSON-RPC method.
            params: JSON-RPC params.
            deadline: Seconds to wait before giving up.

        Returns:
            The reply, or None if the deadline passed, which means the server stopped answering.
        """
        self._id += 1
        mine = self._id
        self.proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": mine, "method": method, "params": params}) + "\n")
        self.proc.stdin.flush()
        end = time.monotonic() + deadline
        while time.monotonic() < end:
            if mine in self._seen:
                return self._seen.pop(mine)
            if self.proc.poll() is not None:
                msg = "server exited"
                raise RuntimeError(msg)
            time.sleep(0.02)
        return None

    def notify(self, method: str) -> None:
        """Send a notification."""
        self.proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": method, "params": {}}) + "\n")
        self.proc.stdin.flush()

    def close(self) -> None:
        """Stop the server."""
        self.proc.kill()
        self.proc.wait(timeout=10)


def _minimal_args(schema: dict) -> dict:
    """Fill only the required properties, with harmless values of the declared type."""
    props = schema.get("properties") or {}
    args: dict = {}
    for name in schema.get("required") or []:
        spec = props.get(name) or {}
        kind = spec.get("type")
        if spec.get("enum"):
            args[name] = spec["enum"][0]
        elif kind in {"integer", "number"}:
            args[name] = 1
        elif kind == "boolean":
            args[name] = False
        elif kind == "array":
            args[name] = []
        elif kind == "object":
            args[name] = {}
        else:
            args[name] = NOTHING
    return args


@pytest.fixture
def server():
    """A running stdio server whose stderr nobody reads."""
    running = StdioServer()
    try:
        assert running.rpc(
            "initialize",
            {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}},
            deadline=STARTUP_DEADLINE,
        ) is not None, "server never finished initialize"
        running.notify("notifications/initialized")
        yield running
    finally:
        running.close()


class TestStderrCannotStopTheServer:
    """The regression this file was written for."""

    def test_every_tool_answers_with_nobody_reading_stderr(self, server):
        """Walk every tool. Any call that misses its deadline fails the test as a HANG.

        The bug this catches is not "a wrong answer", it is "no answer, ever". Before the fix this
        walk answered 28 calls and then stopped, so the assertion has to be the clock.
        """
        listed = server.rpc("tools/list", {}, deadline=STARTUP_DEADLINE)
        assert listed is not None, "tools/list never answered"
        tools = listed["result"]["tools"]
        assert tools, "server advertises no tools"

        answered: list[str] = []
        for tool in tools:
            reply = server.rpc("tools/call", {"name": tool["name"], "arguments": _minimal_args(tool["inputSchema"])})
            if reply is None:
                pytest.fail(
                    f"{tool['name']} never answered within {CALL_DEADLINE}s "
                    f"(it was call {len(answered) + 1} of {len(tools)}; "
                    f"the {len(answered)} before it answered). The server is wedged, and every call "
                    f"queued behind this one is starved too.",
                )
            answered.append(tool["name"])

        assert len(answered) == len(tools)

    def test_still_answers_after_more_logs_than_a_pipe_holds(self, server):
        """A pipe holds 64KB. Keep calling well past that, then check it is still alive.

        The previous test walks each tool once, which is what first exposed this. This one keeps
        going, so the test does not quietly stop being a test if the tool count ever shrinks.
        """
        for _ in range(120):
            assert server.rpc("tools/call", {"name": "get_me", "arguments": {}}) is not None, (
                "the server stopped answering while its stderr went unread"
            )


class TestNonBlockingStderr:
    """The stream itself, without a server around it."""

    def test_write_returns_promptly_when_the_reader_is_gone(self):
        """A stalled reader must cost the writer time, never the caller."""
        from mcp_server_mattermost.logging import NonBlockingStderr

        class Stalled:
            """A stream that never returns from write, like a full pipe."""

            def write(self, _text):
                threading.Event().wait()

            def flush(self):
                pass

        stream = NonBlockingStderr(Stalled(), limit=1000)
        start = time.monotonic()
        for _ in range(500):
            stream.write("x" * 100)
        elapsed = time.monotonic() - start

        assert elapsed < 1.0, f"writing to a stalled stream took {elapsed:.1f}s; it must not wait at all"
        # flush() is on logging's hot path and must be free too, stalled reader or not.
        flush_start = time.monotonic()
        stream.flush()
        assert time.monotonic() - flush_start < 0.1, "flush() waited; that is the hang wearing a different hat"
        assert stream.dropped_chars > 0, "past the ceiling, text must be dropped rather than buffered forever"

    def test_a_dropped_gap_is_reported_not_hidden(self):
        """Dropping is acceptable. Dropping SILENTLY is not."""
        from mcp_server_mattermost.logging import NonBlockingStderr

        written: list[str] = []

        class Collecting:
            def write(self, text):
                written.append(text)

            def flush(self):
                pass

        stream = NonBlockingStderr(Collecting(), limit=10)
        stream.write("y" * 50)  # over the ceiling: dropped
        assert stream.dropped_chars == 50
        stream.write("next\n")
        stream.drain()

        assert any("dropped 50 characters" in chunk for chunk in written), (
            f"the gap was never reported: {written!r}"
        )

    def test_passes_text_through_when_the_reader_is_healthy(self):
        """The ordinary case still has to work."""
        from mcp_server_mattermost.logging import NonBlockingStderr

        written: list[str] = []

        class Collecting:
            def write(self, text):
                written.append(text)

            def flush(self):
                pass

        stream = NonBlockingStderr(Collecting())
        stream.write("hello\n")
        assert stream.drain() is True
        assert "".join(written) == "hello\n"


@pytest.mark.asyncio
class TestCircuitDoesNotBlockTheLoop:
    """circuit_buffer is synchronous and talks to the network. It must not do that on the loop."""

    async def test_a_slow_circuit_leaves_the_loop_free(self, mock_settings, monkeypatch):
        """A slow circuit call must cost a worker thread, not every other coroutine.

        urllib's timeout is 10 seconds, so on the loop one tool call froze the whole server for 10.1
        seconds: a heartbeat coroutine ticked ONCE instead of a hundred times.
        """
        import asyncio

        from fastmcp import Client

        from mcp_server_mattermost import circuit_buffer
        from mcp_server_mattermost.server import mcp

        def slow_resolve(args):
            time.sleep(0.5)  # what a synchronous urllib call looks like from the loop's point of view
            return args

        monkeypatch.setattr(circuit_buffer, "resolve_args", slow_resolve)

        beats = 0

        async def heartbeat():
            nonlocal beats
            while True:
                await asyncio.sleep(0.01)
                beats += 1

        async with Client(mcp) as client:
            task = asyncio.create_task(heartbeat())
            await client.call_tool("get_channel", {"channel_id": "c" * 26}, raise_on_error=False)
            task.cancel()

        assert beats > 10, (
            f"the event loop ticked {beats} times during a 0.5s circuit call; it was blocked, "
            f"so every other request in flight was blocked too"
        )
