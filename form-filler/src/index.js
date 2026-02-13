import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import readline from "node:readline";

import { chromium } from "playwright";

import { buyer } from "./buyer.js";
import { buildRunLogger } from "./logger.js";
import {
  dismissCommonPopups,
  findCandidateForm,
  fillFormHeuristically,
  detectCaptcha,
  clickLikelyOpenFormButtons,
  submitForm,
} from "./runner.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

function parseArgs(argv) {
  const args = { urls: [], url: null, urlsFile: null };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--url") args.url = argv[++i];
    else if (a === "--urls") args.urlsFile = argv[++i];
  }
  if (args.url) args.urls.push(args.url);
  if (args.urlsFile) {
    const raw = fs.readFileSync(args.urlsFile, "utf8");
    for (const line of raw.split(/\r?\n/)) {
      const u = line.trim();
      if (!u || u.startsWith("#")) continue;
      args.urls.push(u);
    }
  }
  return args;
}

async function promptEnter(message) {
  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
  await new Promise((resolve) => rl.question(message, () => resolve()));
  rl.close();
}

function domainFromUrl(u) {
  try {
    return new URL(u).hostname.replace(/^www\./, "");
  } catch {
    return "unknown-domain";
  }
}

async function main() {
  const { urls } = parseArgs(process.argv.slice(2));
  if (!urls.length) {
    console.error("Usage: node src/index.js --url <url> | --urls <file>");
    process.exit(1);
  }

  const slowMo = Number(process.env.FORMFILL_SLOWMO_MS || 250);

  const runLogger = buildRunLogger({
    runDir: path.join(__dirname, "..", "logs", "runs"),
  });

  // Headed browser: user can solve captchas.
  const browser = await chromium.launch({
    headless: false,
    slowMo,
    args: [
      "--disable-blink-features=AutomationControlled",
      "--disable-dev-shm-usage",
    ],
  });

  const context = await browser.newContext({
    viewport: { width: 1280, height: 800 },
    locale: "en-US",
    userAgent:
      "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
  });

  // Light anti-bot patches (won't bypass real bot detection, but avoids the obvious flags).
  await context.addInitScript(() => {
    Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
  });

  const page = await context.newPage();

  for (const url of urls) {
    const domain = domainFromUrl(url);
    const startedAt = new Date().toISOString();

    await runLogger.log({ type: "start", url, domain, startedAt });

    try {
      await page.goto(url, { waitUntil: "domcontentloaded", timeout: 60_000 });
      await page.waitForTimeout(1200 + Math.random() * 1000);

      await dismissCommonPopups(page);

      // Some listing pages require clicking a CTA to reveal the contact form.
      await clickLikelyOpenFormButtons(page);
      await page.waitForTimeout(1000 + Math.random() * 1000);
      await dismissCommonPopups(page);

      // Find a candidate form.
      let form = await findCandidateForm(page);
      if (!form) {
        // Try a second pass after more CTA clicks.
        await clickLikelyOpenFormButtons(page, { aggressive: true });
        await page.waitForTimeout(1000 + Math.random() * 1000);
        form = await findCandidateForm(page);
      }
      if (!form) throw new Error("No form found on page");

      // Captcha gate: pause for human.
      if (await detectCaptcha(page)) {
        await runLogger.log({ type: "captcha", url, domain, at: new Date().toISOString() });
        await promptEnter(
          `Captcha detected for ${domain}. Solve it in the opened browser, then press ENTER here to continue... `
        );
      }

      await fillFormHeuristically(page, form, buyer, { runLogger, url, domain });

      const submitResult1 = await submitForm(page, form, { runLogger, url, domain });
      if (!submitResult1.ok) {
        await page.waitForTimeout(1500 + Math.random() * 1200);
        await dismissCommonPopups(page);
        const submitResult2 = await submitForm(page, form, { runLogger, url, domain });
        if (!submitResult2.ok) {
          throw new Error(`Submit failed twice: ${submitResult2.reason || submitResult1.reason || "unknown"}`);
        }
      }

      // Screenshot
      const ts = new Date().toISOString().replace(/[:.]/g, "-");
      const outDir = path.join(__dirname, "..", "screenshots", domain);
      fs.mkdirSync(outDir, { recursive: true });
      const shotPath = path.join(outDir, `${ts}__submitted.png`);
      await page.screenshot({ path: shotPath, fullPage: true });

      await runLogger.log({ type: "submitted", url, domain, screenshot: shotPath, at: new Date().toISOString() });

      // Small cool-down between sites.
      await page.waitForTimeout(3500 + Math.random() * 3000);
    } catch (err) {
      await runLogger.log({
        type: "error",
        url,
        domain,
        at: new Date().toISOString(),
        error: { message: err?.message || String(err), stack: err?.stack },
      });
    }
  }

  await runLogger.close();
  await context.close();
  await browser.close();
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
