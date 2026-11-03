/**
 * wrangler.ts — Run the local wrangler CLI (same behaviour on Windows, macOS,
 * Linux and CI, no shell quoting issues) and shared resource names.
 */

import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";

export const D1_NAME = "rag-catalog";
export const VECTORIZE_NAME = "rag-catalog";
export const EMBEDDING_DIMENSIONS = 1024; // @cf/baai/bge-m3

/** Path of the wrangler CLI entry point declared in its package.json "bin". */
function wranglerBin(): string {
  const require = createRequire(import.meta.url);
  const pkgPath = require.resolve("wrangler/package.json");
  const pkg = require(pkgPath) as { bin: Record<string, string> };
  return join(dirname(pkgPath), pkg.bin.wrangler);
}

/** Run wrangler; inherit output, or capture stdout when `capture` is set. */
export function wrangler(args: string[], { capture = false, allowFailure = false } = {}): string {
  const result = spawnSync(process.execPath, [wranglerBin(), ...args], {
    stdio: capture ? ["ignore", "pipe", "pipe"] : "inherit",
    encoding: "utf-8",
    maxBuffer: 64 * 1024 * 1024,
  });
  if (result.status !== 0 && !allowFailure) {
    throw new Error(`wrangler ${args.join(" ")} failed\n${result.stderr ?? ""}${result.stdout ?? ""}`);
  }
  return result.status === 0 ? (result.stdout ?? "") : "";
}

/** Load .env (CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_API_TOKEN) when present. */
export function loadEnv(): void {
  if (existsSync(".env")) process.loadEnvFile(".env");
}
