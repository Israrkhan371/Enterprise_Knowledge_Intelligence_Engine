# Fixing the Gemini API quota block

Symptom: `/ask` (and anything that calls `generate_answer()`) returns `429` with
`"LLM provider quota exceeded"`. The underlying Gemini error is
`RESOURCE_EXHAUSTED` on `generativelanguage.googleapis.com/generate_content_free_tier_requests`.

The current key is on Gemini's **free tier**, which enforces a small daily request cap
(RPD — requests per day) per Google Cloud project, in addition to per-minute limits (RPM/TPM).
Once RPD is hit, every call fails until the daily reset — there is no backoff or retry that fixes
this; only time or a plan change does.

## Option A — stay on the free tier (no cost)

**The quota resets once every 24 hours, at midnight Pacific Time (00:00 PT / 08:00 UTC)** — not on
a rolling window from your first request. If you hit the cap, the fix is to wait for that reset,
not to retry sooner.

While on the free tier:
- **Batch your testing.** Running the full 40-query eval set and the citation-accuracy check in the
  same session (as we did) burns the daily allowance fast. Run one, not both, per day if you're
  still on free.
- **Google can change free-tier limits without notice** — it cut them 50–80% in December 2025 and
  moved the Pro model family behind billing entirely in April 2026. The live, authoritative number
  for your specific project is in Google AI Studio, not any blog post (including this one):
  [aistudio.google.com](https://aistudio.google.com) → **Settings → Plan information**.
- Creating additional API keys does **not** add quota — limits are per Cloud *project*, not per
  key, so a second key under the same project shares the same cap.
- If your app's model choice has an unusually low free-tier RPD compared to Google's other Flash
  models, check whether switching the `GEMINI_MODEL` setting to a different currently-free Flash
  variant gives more daily headroom — worth a quick check in AI Studio before assuming the only fix
  is paying.

This is fine for continued development, but is not viable for the demo/grading week if the
evaluation work needs more than one full run per day.

## Option B — enable billing (paid tier)

1. Open **[Google AI Studio](https://aistudio.google.com)** → **API keys** (or **Settings → Plan
   information**).
2. Find the project behind the key `EKIE` uses, click **Set up billing** (or **Upgrade**, if shown).
3. Follow the Cloud Billing dialog: create or link a Google Cloud Billing account, add a payment
   method, and **prepay a minimum of $10** in credits (Google's current model — "Prepay Billing" —
   as of 2026; you top up a balance rather than being billed after the fact).
4. Google auto-verifies project eligibility (a few seconds) and upgrades it.
5. Usage is now visible under **AI Studio → Dashboard → Usage**, and billing/spend under **Cloud
   Billing**.

Official steps: <https://ai.google.dev/gemini-api/docs/billing>

**Two things worth knowing before doing this:**

- **Enabling billing removes the free tier from that project entirely** — every call becomes
  billable from the first token, including calls that would have fit inside the old free quota.
  If you want to keep a separate always-free key for casual local testing, create it under a
  *different* Google Cloud project before enabling billing on the one EKIE uses.
- **Paid tiers have their own spend caps** (roughly $250/month at the lowest paid tier), which
  pause requests once hit — paying doesn't mean unlimited, it means a much higher, and
  cost-tracked, ceiling. Set a budget alert in Cloud Billing so you're not surprised.

Given EKIE is a bounded internship project rather than a production service, Option A (wait for
reset, budget your daily test runs) is the reasonable default; only move to Option B if the
remaining Week 4 timeline genuinely can't absorb one evaluation run per day.
