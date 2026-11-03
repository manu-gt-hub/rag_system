// @vitest-environment happy-dom
/**
 * ui-dom.test.ts — Behaviour of the chat UI (public/index.html) in a DOM:
 * the page's own script runs against a fake API, and the tests check what a
 * visitor sees: header, language switcher, chat round trip, escaping, errors.
 */

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, it, vi } from "vitest";

// Node path helpers: in this environment `URL` is happy-dom's, not Node's
const HTML = readFileSync(resolve(dirname(fileURLToPath(import.meta.url)), "../public/index.html"), "utf-8");
const SCRIPT = HTML.match(/<script>([\s\S]*?)<\/script>/)![1];

// Browsers provide `new Option(text, value)`; happy-dom does not
// eslint-disable-next-line @typescript-eslint/no-explicit-any
(globalThis as any).Option ??= function (text: string, value: string) {
  const option = document.createElement("option");
  option.textContent = text;
  option.value = value;
  return option;
};

const CONFIG = {
  provider: "workers-ai",
  model: "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
  embedding_model: "@cf/baai/bge-m3",
};

type Handler = (body: Record<string, unknown>) => { status?: number; json: unknown };

/** Load the page with a fake API; returns the recorded /api/chat request bodies. */
async function loadPage({ lang, browserLang = "en-US", config = CONFIG, chat }: {
  lang?: string;
  browserLang?: string;
  config?: unknown;
  chat?: Handler;
} = {}) {
  localStorage.clear();
  if (lang) localStorage.setItem("lang", lang);
  Object.defineProperty(navigator, "language", { value: browserLang, configurable: true });

  const requests: Record<string, unknown>[] = [];
  globalThis.fetch = vi.fn(async (url: string, init?: RequestInit) => {
    if (url === "/api/config") return new Response(JSON.stringify(config));
    const body = JSON.parse(String(init?.body ?? "{}"));
    requests.push(body);
    const { status = 200, json } = chat?.(body) ?? { json: { answer: "ok", docs: [] } };
    return new Response(JSON.stringify(json), { status });
  }) as typeof fetch;

  document.documentElement.innerHTML = HTML.replace(/^[\s\S]*?<html[^>]*>/, "").replace(/<\/html>\s*$/, "");
  (0, eval)(SCRIPT); // run the page script in the global scope, like the browser does
  await settle();
  return requests;
}

const settle = async () => {
  for (let i = 0; i < 5; i++) await new Promise((r) => setTimeout(r, 0));
};

async function sendMessage(text: string) {
  (document.getElementById("input") as HTMLTextAreaElement).value = text;
  document.getElementById("form")!.dispatchEvent(new Event("submit", { cancelable: true }));
  await settle();
}

const $ = (sel: string) => document.querySelector(sel)!;
const lastBotMessage = () => [...document.querySelectorAll(".msg.bot")].pop()!;

afterEach(() => vi.restoreAllMocks());

describe("header", () => {
  it("shows the model line with bold labels", async () => {
    await loadPage();
    const info = $("#info");
    expect([...info.querySelectorAll("strong")].map((s) => s.textContent)).toEqual(["LLM:", "embeddings:", "Storage:"]);
    expect(info.textContent).toContain("@cf/meta/llama-3.3-70b-instruct-fp8-fast");
    expect(info.textContent).toContain("Vectorize + D1");
  });

  it("escapes values coming from the API", async () => {
    await loadPage({ config: { ...CONFIG, model: '<img src=x onerror="alert(1)">' } });
    expect($("#info").querySelector("img")).toBeNull();
    expect($("#info").textContent).toContain("<img src=x");
  });
});

describe("language", () => {
  it("defaults to English and follows a Spanish browser", async () => {
    await loadPage();
    expect($("h1").textContent).toContain("Your media catalog");
    await loadPage({ browserLang: "es-ES" });
    expect($("h1").textContent).toContain("Tu catálogo multimedia");
    expect(document.documentElement.lang).toBe("es");
  });

  it("switches the whole UI and remembers the choice", async () => {
    await loadPage();
    ($('.lang [data-lang="es"]') as HTMLButtonElement).click();
    expect($("h1").textContent).toContain("Tu catálogo multimedia");
    expect(($("#input") as HTMLTextAreaElement).placeholder).toMatch(/^Escribe tu mensaje/);
    expect($("#send").textContent).toBe("Enviar");
    expect([...document.querySelectorAll("#category option")].map((o) => o.textContent)).toContain("Películas");
    expect($('.lang [data-lang="es"]').getAttribute("aria-pressed")).toBe("true");
    expect(localStorage.getItem("lang")).toBe("es");
  });

  it("keeps the selected category when switching language", async () => {
    await loadPage();
    ($("#category") as HTMLSelectElement).value = "libros";
    ($('.lang [data-lang="es"]') as HTMLButtonElement).click();
    expect(($("#category") as HTMLSelectElement).value).toBe("libros");
  });
});

describe("chat", () => {
  it("sends message, history, category and UI language, and keeps the history", async () => {
    const requests = await loadPage({
      lang: "es",
      chat: () => ({ json: { answer: "Te recomiendo **Alien**.", docs: [], provider: "workers-ai", elapsed_seconds: 1 } }),
    });
    ($("#category") as HTMLSelectElement).value = "peliculas";
    await sendMessage("una de terror");
    await sendMessage("¿y más recientes?");

    expect(requests[0]).toMatchObject({ message: "una de terror", history: [], category: "peliculas", lang: "es" });
    expect(requests[1].history).toEqual([
      { role: "user", content: "una de terror" },
      { role: "assistant", content: "Te recomiendo **Alien**." },
    ]);
  });

  it("renders markdown but escapes HTML in answers and sources", async () => {
    await loadPage({
      chat: () => ({
        json: {
          answer: 'I recommend **Alien** <img src=x onerror="alert(1)">',
          docs: [{ title: "<b>Alien</b>", category: "peliculas", score: 0.8, year: 1979 }],
          provider: "workers-ai",
          elapsed_seconds: 1,
        },
      }),
    });
    await sendMessage("something scary");
    const msg = lastBotMessage();
    expect(msg.querySelector("strong")?.textContent).toBe("Alien");
    expect(msg.querySelector("img")).toBeNull();
    expect(msg.querySelector(".doc .t")?.textContent).toBe("<b>Alien</b>");
    expect(msg.querySelector("summary")?.textContent).toBe("Catalog sources (1)");
  });

  it("shows API errors in the UI language using the error code", async () => {
    await loadPage({
      lang: "es",
      chat: () => ({ status: 429, json: { error: "Too many requests, please slow down.", code: "rate_limited" } }),
    });
    await sendMessage("hola");
    expect(lastBotMessage().classList.contains("error")).toBe(true);
    expect(lastBotMessage().textContent).toBe("Demasiadas preguntas seguidas, espera un minuto.");
  });

  it("falls back to the API message for unknown error codes", async () => {
    await loadPage({ chat: () => ({ status: 400, json: { error: "Unknown category: comics", code: "unknown_category" } }) });
    await sendMessage("x");
    expect(lastBotMessage().textContent).toBe("Unknown category: comics");
  });

  it("starts a new conversation without history", async () => {
    const requests = await loadPage({ chat: () => ({ json: { answer: "ok", docs: [] } }) });
    await sendMessage("first");
    ($("#reset") as HTMLButtonElement).click();
    await sendMessage("second");
    expect(requests[1].history).toEqual([]);
  });
});
