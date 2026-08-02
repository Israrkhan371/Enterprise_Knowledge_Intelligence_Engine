# Company Policy - Third-Party AI Usage

**Document type:** Company Policy
**Owner:** Ezitech Legal & Security

## Scope

Applies to any use of third-party AI APIs (e.g. Google Gemini, OpenAI,
Anthropic) within Ezitech products or intern case studies, including
EKIE's use of the Gemini API for summarization, comparison, and answer
generation.

## What can be sent to a third-party AI API

- Internal engineering documentation, SOPs, coding standards, and
  similar knowledge-base content: **permitted**.
- Customer personal data (names, emails, financial information) or
  credentials: **never permitted**, regardless of the provider's stated
  data-handling terms.
- Source code from customer-facing production systems: requires sign-off
  from Legal & Security before use in any prompt sent externally.

## Provider requirements

- Only providers with a signed data-processing agreement with Ezitech
  may be used in production. For internship case studies using a free
  tier (e.g. Gemini's free tier), this restriction is relaxed as long as
  no restricted data (see above) is ever sent — case-study projects
  should treat every third-party call as if it could be logged by the
  provider.
- API keys must never be committed to source control. Use environment
  variables or a secrets manager.

## Timeouts and failure handling

Any code path calling a third-party AI API must have an explicit timeout
and a defined failure behavior (return an error to the caller, don't
hang indefinitely) — see `call_with_timeout()` in EKIE's
`app/rag/gemini_utils.py` for the reference implementation used across
all Gemini call sites in that project.

## Related documents

- Company Policy - Data Retention
