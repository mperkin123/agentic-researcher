# Listing Form Filler (MVP)

Automates filling out web forms on business listing pages (NDA requests, broker contact, deal room access, messaging sellers, etc.).

## Safety / boundaries

- **Does not solve CAPTCHAs.** When a captcha is detected, it pauses and asks a human to solve it in the visible browser, then resumes.
- Skips optional fields when unsure.
- Skips anything that opts into marketing/newsletters.
- Anti-bot: runs headed (visible) with human-like pacing and jitter.

## Requirements

- Node 18+
- A stable IP/proxy that won’t get blocked (configure externally). Some sites block datacenter IPs.

## Install

```bash
cd form-filler
npm i
npx playwright install --with-deps chromium
```

## Configure

Set env vars:

- `OPENAI_API_KEY` (optional) — only needed for judgment calls on unknown fields.
- `FORMFILL_SLOWMO_MS` (optional) — base delay per action (default 250).

Buyer profile is currently in `src/buyer.js`.

## Run

```bash
node src/index.js --urls urls.txt
# or
node src/index.js --url "https://example.com/listing"
```

Outputs:

- `screenshots/<domain>/<timestamp>__submitted.png`
- `logs/runs/<timestamp>.jsonl`

## Notes

This is an MVP with heuristics. Expect site-by-site tweaks.
