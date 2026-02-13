export function jitter(min, max) {
  return Math.floor(min + Math.random() * (max - min));
}

export function isMarketingOptIn(label = "") {
  const s = (label || "").toLowerCase();
  return [
    "newsletter",
    "marketing",
    "promotions",
    "promotional",
    "special offers",
    "updates",
    "subscribe",
    "subscription",
    "sms",
    "text me",
    "email me",
    "partners",
  ].some((k) => s.includes(k));
}

export function guessFieldKey(label = "", type = "") {
  const s = (label || "").toLowerCase();
  const t = (type || "").toLowerCase();

  if (t === "email" || s.includes("email")) return "email";
  if (s.includes("full name") || s === "name" || s.includes("your name")) return "name";
  if (s.includes("first name") || s.includes("last name")) return "name";
  if (s.includes("company") || s.includes("business")) return "businessAddress";
  if (s.includes("address")) return "address";
  if (s.includes("message") || s.includes("comments") || s.includes("notes") || t === "textarea")
    return "message";

  return "unknown";
}

export async function labelForInput(page, locator) {
  // Best-effort label extraction.
  try {
    const id = await locator.getAttribute("id");
    if (id) {
      const l = page.locator(`label[for="${cssEscape(id)}"]`).first();
      const txt = (await l.textContent().catch(() => ""))?.trim();
      if (txt) return txt;
    }
  } catch {}

  try {
    const aria = await locator.getAttribute("aria-label");
    if (aria) return aria.trim();
  } catch {}

  try {
    const ph = await locator.getAttribute("placeholder");
    if (ph) return ph.trim();
  } catch {}

  // Walk up for nearby label-like text
  try {
    const txt = await locator.evaluate((node) => {
      const container = node.closest("div,fieldset,section") || node.parentElement;
      const label = container?.querySelector("label");
      return label?.textContent || "";
    });
    if (txt && txt.trim()) return txt.trim();
  } catch {}

  return "";
}

function cssEscape(s) {
  return s.replace(/[^a-zA-Z0-9_-]/g, (m) => `\\${m}`);
}
