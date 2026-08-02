# SOP - Code Review Process

**Document type:** Standard Operating Procedure
**Owner:** Ezitech Engineering
**Applies to:** All pull requests across Ezitech and intern-built
repositories, including case-study projects.

## Purpose

A predictable review process so pull requests get consistent scrutiny
regardless of who reviews them, and interns get actionable feedback
instead of a rubber stamp or radio silence.

## Before opening a pull request

- The branch must be up to date with `main`.
- All new code must follow Ezitech Python Coding Standards (naming,
  type hints, docstrings).
- Automated tests must pass locally before requesting review.
- The PR description must state what changed and why, not just what.

## Review requirements

1. At least **one** approval is required before merging to `main`.
   Changes touching authentication, payments, or production
   infrastructure require **two** approvals.
2. Reviewers check for: correctness, test coverage, adherence to the
   coding standard, and whether the change matches its stated intent.
3. Reviewers should leave specific, actionable comments — "this could
   race under concurrent writes" rather than "looks off."
4. The PR author addresses every comment (either with a code change or a
   reply explaining why not) before re-requesting review.

## Approval criteria

A PR is approved when:
- CI is green (tests + lint).
- No unresolved review comments remain.
- The diff matches Ezitech Python Coding Standards, including type hints
  and docstrings on public functions.

## Merging

- Squash-merge by default to keep `main` history readable.
- The author merges their own PR once approved, unless the reviewer asks
  to merge it themselves.

## Related documents

- Ezitech Python Coding Standards
- SOP - Mentor Onboarding
