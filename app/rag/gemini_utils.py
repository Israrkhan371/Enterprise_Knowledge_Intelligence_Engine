"""
Shared helper for enforcing a hard timeout around blocking Gemini SDK calls.

The google-genai SDK accepts an `http_options={"timeout": ...}` setting, but
several known SDK issues (e.g. googleapis/python-genai#911, #1893, #1330)
show it does not reliably enforce request timeouts across versions - calls
can still hang indefinitely even with a timeout configured. To guarantee a
deterministic timeout regardless of SDK/httpx behavior, blocking Gemini
calls are run in a worker thread with an explicit deadline instead.

Note: Python threads cannot be forcibly killed, so a timed-out call's
underlying network request keeps running in the background until it
finishes or errors on its own; only the caller stops waiting for it. This
is a standard limitation of thread-based timeouts and does not affect
correctness of the timeout behavior seen by callers.
"""
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Small dedicated pool so a burst of slow/hung Gemini calls can't starve
# other thread-based work in the process.
_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="gemini-call")


def call_with_timeout(fn: Callable[..., T], *args, timeout_seconds: float, **kwargs) -> T:
    """
    Runs fn(*args, **kwargs) with a hard timeout.

    Raises TimeoutError if fn doesn't complete within timeout_seconds.
    Any other exception raised by fn propagates unchanged.
    """
    future = _executor.submit(fn, *args, **kwargs)
    try:
        return future.result(timeout=timeout_seconds)
    except FutureTimeoutError:
        logger.error("Gemini call timed out after %.1fs (fn=%s)", timeout_seconds, getattr(fn, "__qualname__", fn))
        raise TimeoutError(f"Gemini request timed out after {timeout_seconds}s")
