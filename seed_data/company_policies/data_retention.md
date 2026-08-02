# Company Policy - Data Retention

**Document type:** Company Policy
**Owner:** Ezitech Legal & Security

## Scope

Applies to all documents uploaded to Ezitech knowledge platforms,
including EKIE, and to any raw text, embeddings, or derived knowledge
(e.g. graph entities) generated from them.

## Retention periods

- **Approved knowledge documents:** retained indefinitely while
  actively referenced; reviewed for staleness on the cadence defined by
  each platform's outdated-knowledge detection (e.g. EKIE's
  `detect_outdated` staleness heuristic).
- **Pending/unapproved uploads:** retained for 90 days. If not approved
  or rejected within that window, they're automatically flagged for
  admin review and may be purged.
- **Rejected documents:** retained for 30 days for audit purposes, then
  deleted, including their vector embeddings and graph nodes.
- **Usage logs (queries, answers, retrieval scores):** retained for 12
  months for evaluation and quality-improvement purposes, then
  aggregated into anonymized metrics and the raw logs deleted.

## Deletion requests

Any employee or intern can request deletion of a document they uploaded.
Deletion removes the Postgres record, the associated vector-store
embeddings, and any graph nodes/edges whose only supporting evidence was
that document.

## Data classification

Documents containing customer data or credentials must never be uploaded
to the knowledge platform — see Company Policy - Third-Party AI Usage
for related restrictions on what can be sent to external AI providers.

## Related documents

- Company Policy - Third-Party AI Usage
