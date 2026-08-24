# EKIE — Free Public Deployment Guide

Referenced from `docker-compose.prod.yml` and `docker/Caddyfile` — this
file fills in the steps those comments point to. See
`docs/Deployment_Guide.md` for local development; this guide covers
taking that same stack and putting it on the public internet for $0.

Steps are numbered continuously across phases (not restarted per phase),
so a cross-reference like "Phase 3 step 19" points at one specific step
below.

## Phase 1 — Get a free server

This stack (Postgres, Neo4j, ChromaDB, MLflow, Prometheus, plus an `api`
image that loads PyTorch, spaCy, sentence-transformers, and a 217MB
layout-detection model) needs more RAM than a typical "free web service"
tier offers (Render's free tier, for example, is 512MB — nowhere near
enough just to import those libraries). Oracle Cloud's Always Free tier
is the one free option with enough headroom to run the existing
`docker-compose.yml` largely unmodified.

1. Go to <https://www.oracle.com/cloud/free/> and start a free account.
   A card is required for identity verification but is not charged for
   Always Free resources.
2. Log in to the Oracle Cloud console.
3. Open the menu (☰) → **Compute** → **Instances**.
4. Click **Create Instance**.
5. Under "Image and shape," click **Edit** next to shape, choose
   **Ampere** → **VM.Standard.A1.Flex**.
6. Set **2 OCPUs** / **12 GB memory** — this is the current Always Free
   allocation for card-only (non-billing) accounts as of mid-2026; it was
   previously 4 OCPU/24GB, so older guides may show a larger number.
7. Choose **Ubuntu** as the OS, click **Create**, and wait for the
   instance to show "Running."
8. Note the instance's **Public IP address** — needed in step 9 and
   again in step 19.
9. On the instance's page, click the **Subnet** link → **Default
   Security List** → **Add Ingress Rules**. Add two rules, both with
   Source CIDR `0.0.0.0/0`: destination port **80**, and destination
   port **443**. (Not 8000 — see Phase 4 for why.)

## Phase 2 — Install Docker and get the code onto the server

10. From your own computer: `ssh ubuntu@<your-server-ip>` (using
    whichever key-based login Oracle showed you when the instance was
    created).
11. `sudo apt update`
12. `sudo apt install -y docker.io docker-compose-plugin git`
13. `git clone <your-repo-url>`
14. `cd Enterprise_Knowledge_Intelligence_Engine`

## Phase 3 — Get a free domain name

Caddy (see Phase 4) automatically obtains free HTTPS certificates, but
Let's Encrypt cannot issue a certificate for a bare IP address — a real
hostname is required.

15. Go to <https://www.duckdns.org>.
16. Sign in with an existing GitHub or Google account (no new signup
    needed).
17. Enter a subdomain of your choice, e.g. `yourname`, giving you
    `yourname.duckdns.org`.
18. This becomes the value you'll set as `EKIE_DOMAIN` in Phase 4.
19. Paste the server's public IP address (from step 8) into the "current
    ip" box on the DuckDNS page and click **update ip**.

## Phase 4 — Configure and start production mode

`docker-compose.prod.yml` is a Compose *override* layered on top of the
existing `docker-compose.yml`, not a replacement for it. Its own comments
already explain each change in detail (removing public ports from every
backend service, adding a `caddy` container as the only public-facing
one); the short version: only Caddy (ports 80/443) is reachable from the
internet, everything else — `api` included — talks only over Docker's
internal network. This is why step 9 opened 80/443 instead of 8000.

20. `cp .env.production.example .env`
21. `nano .env` and fill in:
    - `EKIE_DOMAIN=yourname.duckdns.org` (from step 17)
    - `POSTGRES_PASSWORD` and `NEO4J_PASSWORD` — real values, not the
      `CHANGE_THIS_PASSWORD` placeholder
    - `GOOGLE_API_KEY` — free at <https://aistudio.google.com/apikey>

    Save with `Ctrl+O`, `Enter`, then exit with `Ctrl+X`.
22. Start everything, layering both Compose files:
    ```
    docker compose -f docker-compose.yml -f docker-compose.prod.yml up --build -d
    ```
23. Confirm every service is up:
    ```
    docker compose -f docker-compose.yml -f docker-compose.prod.yml ps
    ```
24. Visit `https://yourname.duckdns.org`. Caddy needs a minute or two on
    first start to obtain the certificate from Let's Encrypt.

## Troubleshooting

- **Certificate never issues / Caddy keeps restarting:** confirm DNS has
  actually propagated (`nslookup yourname.duckdns.org` should return the
  server's IP) before Caddy tries — DuckDNS updates are usually near
  -instant, but occasionally take a few minutes.
- **502 from Caddy:** `docker compose -f docker-compose.yml -f
  docker-compose.prod.yml logs api` — the `api` container is what Caddy
  is proxying to; if it's still starting up (loading models) or crashed
  on a bad `.env` value, Caddy will 502 until it's healthy.
- **PyTorch install fails during `--build`** (only relevant on ARM
  shapes like Ampere A1): the Dockerfile installs `torch==2.5.1` from
  `https://download.pytorch.org/whl/cpu`, which may not carry an aarch64
  build for that exact pin. If this happens, dropping the `--index-url`
  flag for that one line and installing from plain PyPI instead is the
  first thing to try.

## Rollback / recovery

Same procedures as `docs/Deployment_Guide.md` section 4 — nothing about
production mode changes how migrations, volumes, or rebuilds work, only
which ports are public and what's fronting the API.
