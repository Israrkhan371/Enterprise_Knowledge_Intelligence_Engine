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
import random
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Callable, TypeVar

from google.genai import errors as genai_errors

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


def _is_transient(exc: Exception) -> bool:
    """
    503 ("high demand", explicitly described by Google as usually temporary)
    and 429 (rate limit) are worth retrying - the request itself is fine,
    the service is just momentarily overloaded. A 404 (e.g. a dead/renamed
    model string) or 400 (malformed request) is not transient: retrying
    those wastes time and quota on something that will never succeed
    without a code change, and a real outage/misconfiguration would look
    identical to "still loading" if we retried everything indiscriminately.
    """
    if isinstance(exc, TimeoutError):
        return True
    if isinstance(exc, genai_errors.ServerError):
        return getattr(exc, "status_code", None) in (503, 429)
    if isinstance(exc, genai_errors.ClientError):
        return getattr(exc, "status_code", None) == 429
    return False


def call_with_retry(
    fn: Callable[..., T],
    *args,
    timeout_seconds: float,
    max_attempts: int = 4,
    base_delay_seconds: float = 2.0,
    **kwargs,
) -> T:
    """
    Wraps call_with_timeout() with retries for transient errors only
    (503 "high demand", 429 rate-limited, and hard timeouts) using
    exponential backoff with jitter. Non-transient errors (404 dead model,
    400 bad request, auth failures) propagate immediately on the first
    attempt - see _is_transient() for why those aren't retried.

    Added after a citation-accuracy eval run showed 27/40 queries failing
    outright with 503s that a single retry would very likely have cleared,
    since Google's own error message describes these spikes as temporary.
    Without this, every transient hiccup permanently drops that query from
    the evaluation instead of just slowing it down.
    """
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return call_with_timeout(fn, *args, timeout_seconds=timeout_seconds, **kwargs)
        except Exception as exc:
            last_exc = exc
            if not _is_transient(exc) or attempt == max_attempts:
                raise
            delay = base_delay_seconds * (2 ** (attempt - 1)) + random.uniform(0, 1)
            logger.warning(
                "Transient Gemini error (attempt %d/%d): %s - retrying in %.1fs",
                attempt, max_attempts, exc, delay,
            )
            time.sleep(delay)
    raise last_exc  # pragma: no cover - loop always returns or raises above
