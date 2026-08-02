# SOP - Incident Response

**Document type:** Standard Operating Procedure
**Owner:** Ezitech Platform Engineering
**Applies to:** Any production incident affecting an Ezitech product,
including intern-built systems once they reach staging/production.

## Purpose

A consistent process for reporting, triaging, and resolving production
incidents so response time and communication don't depend on who happens
to notice the problem first.

## Severity levels

| Level | Definition | Initial response target |
|-------|------------|--------------------------|
| SEV-1 | Full outage or data loss affecting all users | 15 minutes |
| SEV-2 | Major feature broken, partial outage | 1 hour |
| SEV-3 | Degraded performance, workaround available | 4 hours |
| SEV-4 | Minor bug, no user-facing impact yet | Next business day |

## Reporting an incident

1. Post in the `#incidents` channel with severity, affected system, and a
   one-line summary.
2. Open an incident ticket using the incident template (severity, start
   time, current impact, who's investigating).
3. For SEV-1/SEV-2, page the on-call engineer immediately — don't wait
   for a ticket to be filed first.

## During the incident

- Designate one incident commander who owns communication and
  coordination (not necessarily the person fixing the bug).
- Post status updates in `#incidents` at least every 30 minutes for
  SEV-1/SEV-2.
- Do not deploy unrelated changes to the affected system until the
  incident is resolved.

## After the incident

- Write a blameless postmortem within 3 business days for SEV-1/SEV-2
  incidents: timeline, root cause, what caught it, what didn't, and
  concrete follow-up actions with owners.
- Postmortems are reviewed in the next engineering sync, not filed away
  unread.

## Related documents

- SOP - Escalation Procedures
