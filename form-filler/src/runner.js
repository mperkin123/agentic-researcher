import { guessFieldKey, isMarketingOptIn, jitter, labelForInput } from "./util.js";

export async function dismissCommonPopups(page) {
  // Cookie banners / modals.
  const selectors = [
    'button:has-text("Accept")',
    'button:has-text("I Accept")',
    'button:has-text("Agree")',
    'button:has-text("Got it")',
    'button:has-text("OK")',
    'button:has-text("Close")',
    '[aria-label="close"]',
    '[aria-label="Close"]',
    'button[title="Close"]',
  ];

  for (const sel of selectors) {
    try {
      const el = page.locator(sel).first();
      if (await el.isVisible({ timeout: 400 })) {
        await el.click({ timeout: 800 });
        await page.waitForTimeout(300 + Math.random() * 500);
      }
    } catch {
      // ignore
    }
  }
}

export async function clickLikelyOpenFormButtons(page, { aggressive = false } = {}) {
  const texts = [
    "Contact",
    "Request",
    "Inquire",
    "Enquire",
    "Email",
    "Message",
    "Get more information",
    "Get More Information",
    "Unlock",
    "View phone",
    "View Email",
    "NDA",
    "Sign NDA",
    "Request NDA",
  ];

  const candidates = [];
  for (const t of texts) {
    candidates.push(page.getByRole("button", { name: t }).first());
    candidates.push(page.getByRole("link", { name: t }).first());
    candidates.push(page.locator(`button:has-text("${t}")`).first());
    candidates.push(page.locator(`a:has-text("${t}")`).first());
  }

  // In aggressive mode, also click any sticky CTA.
  if (aggressive) {
    candidates.push(page.locator("[class*=cta i] button").first());
    candidates.push(page.locator("[class*=contact i] button").first());
  }

  for (const loc of candidates) {
    try {
      if (await loc.isVisible({ timeout: 300 })) {
        await loc.click({ timeout: 1200 });
        await page.waitForTimeout(700 + Math.random() * 900);
      }
    } catch {
      // ignore
    }
  }
}

export async function findCandidateForm(page) {
  // Prefer visible forms with text inputs/textarea.
  const forms = page.locator("form");
  const count = await forms.count();
  for (let i = 0; i < Math.min(count, 20); i++) {
    const form = forms.nth(i);
    try {
      if (!(await form.isVisible({ timeout: 300 }))) continue;
      const inputs = form.locator('input:not([type="hidden"]), textarea, select');
      const n = await inputs.count();
      if (n >= 3) return form;
    } catch {
      // ignore
    }
  }

  // Fallback: any visible input cluster (some sites don't use <form>)
  const cluster = page.locator(
    'input[type="text"], input[type="email"], input[type="tel"], textarea'
  );
  if (await cluster.first().isVisible({ timeout: 300 })) {
    // return page root; downstream uses locator scoping if it's a form.
    return page.locator("body");
  }
  return null;
}

export async function detectCaptcha(page) {
  // reCAPTCHA / hCaptcha / generic captcha
  const captchaSelectors = [
    'iframe[src*="recaptcha"]',
    'iframe[src*="hcaptcha"]',
    '[class*="recaptcha" i]',
    '[class*="hcaptcha" i]',
    'input[name*="captcha" i]',
    'img[src*="captcha" i]',
    'div:has-text("I am not a robot")',
  ];

  for (const sel of captchaSelectors) {
    try {
      if (await page.locator(sel).first().isVisible({ timeout: 250 })) return true;
    } catch {}
  }
  return false;
}

export async function fillFormHeuristically(page, formLocator, buyer, { runLogger, url, domain } = {}) {
  const scope = formLocator;

  // Collect fillable elements.
  const inputs = scope.locator('input:not([type="hidden"]), textarea, select');
  const n = await inputs.count();

  for (let i = 0; i < n; i++) {
    const el = inputs.nth(i);

    try {
      if (!(await el.isVisible({ timeout: 200 }))) continue;
      if (await el.isDisabled({ timeout: 200 })) continue;

      const tag = await el.evaluate((node) => node.tagName.toLowerCase());
      const type = tag === "input" ? await el.getAttribute("type") : tag;
      const label = await labelForInput(page, el);

      // Skip marketing opt-ins.
      if (isMarketingOptIn(label)) {
        // If it's a checkbox, try to ensure unchecked.
        if ((type || "").toLowerCase() === "checkbox") {
          const checked = await el.isChecked().catch(() => false);
          if (checked) await el.click({ delay: 50 });
        }
        continue;
      }

      const key = guessFieldKey(label, type);
      const value = valueForKey(key, buyer);

      if (!value) {
        // Skip optional/unknown fields by default.
        continue;
      }

      await page.waitForTimeout(jitter(120, 380));

      if (tag === "select") {
        // Choose matching option by text (best effort)
        await el.selectOption({ label: value }).catch(async () => {
          // fallback: first non-empty
          const options = el.locator("option");
          const on = await options.count();
          for (let j = 0; j < on; j++) {
            const opt = options.nth(j);
            const txt = (await opt.textContent())?.trim();
            const val = await opt.getAttribute("value");
            if (txt && val) {
              await el.selectOption(val);
              break;
            }
          }
        });
      } else if ((type || "").toLowerCase() === "checkbox") {
        // Only check if it's required and not marketing; otherwise leave.
        // We don't have strong required detection; so default to leave unchecked.
        continue;
      } else {
        await el.click({ timeout: 1000 }).catch(() => {});
        await el.fill("");
        await el.type(value, { delay: jitter(40, 110) });
      }

      if (runLogger) {
        await runLogger.log({
          type: "fill",
          url,
          domain,
          field: { label, key, type },
          at: new Date().toISOString(),
        });
      }
    } catch {
      // ignore field
    }
  }
}

function valueForKey(key, buyer) {
  switch (key) {
    case "name":
      return buyer.name;
    case "email":
      return buyer.email;
    case "address":
      return buyer.address;
    case "businessAddress":
      return buyer.businessAddress;
    case "message":
    case "notes":
    case "comment":
      return buyer.context;
    default:
      return null;
  }
}

export async function submitForm(page, formLocator, { runLogger, url, domain } = {}) {
  const scope = formLocator;

  const submitCandidates = [
    scope.locator('button[type="submit"]').first(),
    scope.locator('input[type="submit"]').first(),
    scope.locator('button:has-text("Submit")').first(),
    scope.locator('button:has-text("Send")').first(),
    scope.locator('button:has-text("Request")').first(),
    scope.locator('button:has-text("Contact")').first(),
    scope.locator('button:has-text("Inquire")').first(),
    scope.locator('button:has-text("Continue")').first(),
  ];

  for (const btn of submitCandidates) {
    try {
      if (await btn.isVisible({ timeout: 300 })) {
        await page.waitForTimeout(jitter(200, 600));
        await btn.click({ timeout: 4000 });

        // Wait briefly for either navigation or success text.
        await Promise.race([
          page.waitForLoadState("domcontentloaded", { timeout: 8000 }).catch(() => null),
          page.waitForTimeout(2500),
        ]);

        const ok = await looksLikeSuccess(page);
        if (runLogger) {
          await runLogger.log({ type: "submit", url, domain, ok, at: new Date().toISOString() });
        }
        return ok ? { ok: true } : { ok: false, reason: "no-success-signal" };
      }
    } catch {
      // try next candidate
    }
  }

  return { ok: false, reason: "no-submit-button" };
}

async function looksLikeSuccess(page) {
  const successSnippets = [
    "thank you",
    "thanks",
    "received",
    "we will be in touch",
    "message has been sent",
    "request submitted",
    "submission successful",
    "check your email",
  ];

  const bodyText = (await page.locator("body").innerText().catch(() => ""))
    .toLowerCase()
    .slice(0, 20_000);

  return successSnippets.some((s) => bodyText.includes(s));
}
