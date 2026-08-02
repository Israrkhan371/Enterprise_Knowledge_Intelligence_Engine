# Ezitech Python Coding Standards

**Document type:** Coding Standard
**Owner:** Ezitech Engineering
**Applies to:** All Python code in Ezitech and intern-built repositories.

## Naming conventions

- **Variables and functions:** `snake_case` (e.g. `chunk_text`,
  `document_id`).
- **Classes:** `PascalCase` (e.g. `GraphStore`, `DocumentChunk`).
- **Constants:** `UPPER_SNAKE_CASE` (e.g. `MAX_CONFIDENCE`,
  `SOURCE_LOADERS`).
- **Private/internal helpers:** prefix with a single underscore
  (e.g. `_chunk_embed_store`, `_populate_graph`).
- Names should describe intent, not implementation — `retrieved_ids` not
  `lst2`.

## Type hints and docstrings

- All public functions must have type hints on parameters and return
  values.
- All public functions must have a docstring explaining *why*, not just
  *what* — restating the signature in prose adds nothing; explaining a
  non-obvious tradeoff or gotcha does.
- Private helpers (leading underscore) should still have type hints;
  docstrings are encouraged but not mandatory if the function is short
  and the name is self-explanatory.

## Formatting

- Line length: 100 characters.
- Imports grouped: standard library, then third-party, then local
  (`app.*`), each group separated by a blank line.
- No wildcard imports (`from x import *`).

## Error handling

- Catch specific exceptions, not bare `except:`.
- Don't swallow exceptions silently — log them, even in a best-effort
  path (see `_populate_graph`'s `except Exception: logger.exception(...)`
  pattern for how a non-critical failure should be handled without
  crashing the primary operation).

## Testing

- New functionality needs test coverage before merge (see SOP - Code
  Review Process).
- Tests that need a live external service (Postgres, Neo4j, ChromaDB)
  must skip gracefully with `pytest.skip(...)` when that service isn't
  reachable, rather than failing the whole suite in environments without
  it running.

## Related documents

- SOP - Code Review Process
