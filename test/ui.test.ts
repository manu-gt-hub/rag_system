/** ui.test.ts — The chat UI is fully translated (English and Spanish). */

import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

const html = readFileSync(new URL("../public/index.html", import.meta.url), "utf-8");

/** Top-level keys of one language block of the I18N object. */
function keysOf(lang: "en" | "es"): Set<string> {
  const start = html.indexOf(`            ${lang}: {`);
  const end = lang === "en" ? html.indexOf("            es: {") : html.indexOf("        };", start);
  const block = html.slice(start, end);
  return new Set([...block.matchAll(/^ {16}(\w+):/gm)].map((m) => m[1]));
}

describe("UI translations", () => {
  const en = keysOf("en");
  const es = keysOf("es");

  it("has the same keys in English and Spanish", () => {
    expect(en.size).toBeGreaterThan(10);
    expect([...es].sort()).toEqual([...en].sort());
  });

  it("translates every key used by the markup", () => {
    const used = [...html.matchAll(/data-i18n(?:-placeholder|-title)?="(\w+)"/g)].map((m) => m[1]);
    expect(used.length).toBeGreaterThan(5);
    for (const key of used) expect(en.has(key) && es.has(key), `missing translation: ${key}`).toBe(true);
  });

  it("translates every API error code", () => {
    const codes = ["rate_limited", "quota_exceeded", "message_too_long", "internal_error"];
    for (const code of codes) {
      expect(html.match(new RegExp(`\\b${code}:`, "g"))?.length, `error code ${code}`).toBe(2);
    }
  });

  it("sends the UI language to the API", () => {
    expect(html).toMatch(/history,\s*category: [^\n]+\n\s*lang,/);
  });
});
