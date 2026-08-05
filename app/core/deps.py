import uuid

from fastapi import Depends, Header, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.models import User


def ensure_valid_uuid(value: str, *, status_code: int = 404, detail: str | None = None) -> None:
    """Raise before a malformed id reaches the DB layer.

    Several UUID primary/foreign key columns (users.id, usage_logs.id, ...)
    are looked up or inserted using caller-supplied strings (path params,
    request bodies) with no format check. Postgres's UUID column rejects
    non-UUID text with an unhandled DataError, which surfaces as a raw 500
    instead of a clean 404/400. Call this first wherever such a value is
    about to hit db.get()/an insert.
    """
    try:
        uuid.UUID(value)
    except ValueError:
        raise HTTPException(status_code=status_code, detail=detail or f"not a valid id: {value}")


def require_admin(
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
    db: Session = Depends(get_db),
) -> User:
    """Role gate for the /admin router.

    EKIE has no session/token auth layer yet — document upload/approval,
    AI-answer review, and analytics were previously reachable by anyone
    who found the URL, and review actions were attributed to a free-typed
    `reviewer` string nobody verified. This checks the caller-supplied
    X-User-Id against users.role so those admin-only actions (per the
    case study's "Administration" section) actually require an admin
    user, and lets callers derive the acting admin's identity instead of
    trusting client-supplied names.

    This is NOT a substitute for real authentication — nothing here
    verifies the caller actually *is* the user behind that id, only that
    such a user exists and is an admin. Replace with proper session/JWT
    auth in front of this same role check before this is exposed beyond
    trusted internal callers.
    """
    if not x_user_id:
        raise HTTPException(status_code=401, detail="X-User-Id header is required for admin endpoints")

    # users.id is a Postgres UUID column — without this check, a malformed
    # header (e.g. "admin", empty, a typo) reaches db.get() and psycopg2
    # raises an unhandled DataError, surfacing as a raw 500 with a stack
    # trace instead of a clean 401.
    ensure_valid_uuid(x_user_id, status_code=401, detail="X-User-Id must be a valid UUID")

    user = db.get(User, x_user_id)
    if not user:
        raise HTTPException(status_code=401, detail="unknown user")
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="admin role required")
    return user
