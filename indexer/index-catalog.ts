/**
 * index-catalog.ts — Build the catalog documents, embed them with Workers AI
 * (bge-m3) and load them into Vectorize (vectors) and D1 (documents + FTS5).
 *
 *   npm run index                  build, embed and load everything
 *   npm run index -- --if-changed  skip when D1 already holds this exact catalog
 *                                  (used by the deploy workflow on every push)
 *   npm run index -- --build-only  only write build/d1-data.sql; no API calls
 *                                  (used by the tests)
 *
 * Needs CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN (env or .env).
 */

import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { D1_NAME, VECTORIZE_NAME, loadEnv, wrangler } from "../scripts/wrangler";
import { EMBEDDING_MODEL } from "../src/retriever";
import { normalize } from "../src/text";
import { type CatalogDocument, DOCUMENT_VERSION, loadCatalog } from "./documents";

const ROOT = new URL("../", import.meta.url);
const BUILD = new URL("build/", ROOT);
const MAX_EMBED_CHARS = 4000;
const BATCH_SIZE = 50;

const args = new Set(process.argv.slice(2));

// ---------------------------------------------------------------------------
// Signature: identifies catalog content + document format + embedding model
// ---------------------------------------------------------------------------

function signatureOf(docs: CatalogDocument[]): string {
  const hash = createHash("sha256");
  hash.update(`${EMBEDDING_MODEL}|v${DOCUMENT_VERSION}|`);
  for (const d of docs) hash.update(JSON.stringify(d));
  return hash.digest("hex").slice(0, 16);
}

function remoteIndexState(): { signature: string; documents: number } | null {
  const out = wrangler(
    ["d1", "execute", D1_NAME, "--remote", "--json", "--command", "SELECT signature, documents FROM index_meta"],
    { capture: true, allowFailure: true }, // fails when the table does not exist yet
  );
  try {
    const row = JSON.parse(out)[0]?.results?.[0];
    return row ? { signature: row.signature, documents: Number(row.documents) } : null;
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// Embeddings (Workers AI REST API)
// ---------------------------------------------------------------------------

async function embedBatch(texts: string[]): Promise<number[][]> {
  const { CLOUDFLARE_ACCOUNT_ID: account, CLOUDFLARE_API_TOKEN: token } = process.env;
  if (!account || !token) throw new Error("Set CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN (env or .env)");
  const url = `https://api.cloudflare.com/client/v4/accounts/${account}/ai/run/${EMBEDDING_MODEL}`;
  for (let attempt = 1; ; attempt++) {
    const resp = await fetch(url, {
      method: "POST",
      headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
      body: JSON.stringify({ text: texts }),
    });
    if (resp.ok) return ((await resp.json()) as { result: { data: number[][] } }).result.data;
    const detail = await resp.text();
    // 409 "Cog prediction failed" (code 8007) is a transient inference error,
    // seen on the first calls while the model warms up; retry it like 429/5xx
    const retryable = resp.status === 409 || resp.status === 429 || resp.status >= 500;
    if (attempt >= 5 || !retryable) throw new Error(`Workers AI ${resp.status}: ${detail}`);
    console.warn(`Workers AI ${resp.status}, retrying (${attempt}/4)...`);
    await new Promise((r) => setTimeout(r, 3000 * attempt));
  }
}

/** Embeddings are cached per signature in build/, so a failed load can be retried for free. */
async function writeVectors(docs: CatalogDocument[], signature: string): Promise<URL> {
  const file = new URL(`vectors-${signature}.ndjson`, BUILD);
  if (existsSync(file)) {
    console.log(`Reusing cached embeddings ${fileURLToPath(file)}`);
    return file;
  }
  const lines: string[] = [];
  for (let i = 0; i < docs.length; i += BATCH_SIZE) {
    const batch = docs.slice(i, i + BATCH_SIZE);
    const vectors = await embedBatch(batch.map((d) => d.content.slice(0, MAX_EMBED_CHARS)));
    batch.forEach((doc, j) =>
      lines.push(JSON.stringify({ id: String(doc.id), values: vectors[j], metadata: { category: doc.category } })),
    );
    console.log(`Embedded ${Math.min(i + BATCH_SIZE, docs.length)} / ${docs.length}`);
  }
  writeFileSync(file, lines.join("\n") + "\n");
  return file;
}

// ---------------------------------------------------------------------------
// D1 load file: schema + documents + full-text index + signature (last)
// ---------------------------------------------------------------------------

const sql = (v: unknown) => (v === null || v === undefined ? "NULL" : `'${String(v).replaceAll("'", "''")}'`);

function writeD1(docs: CatalogDocument[], signature: string): URL {
  const lines = [readFileSync(new URL("schema.sql", ROOT), "utf-8")];
  for (const d of docs) {
    lines.push(
      `INSERT INTO docs (id, category, title, content, metadata, title_key) VALUES (${d.id}, ${sql(d.category)}, ` +
        `${sql(d.title)}, ${sql(d.content)}, ${sql(JSON.stringify(d.metadata))}, ${sql(d.titleKey)});`,
      `INSERT INTO docs_fts (rowid, title, keywords, content) VALUES (${d.id}, ${sql(normalize(d.title))}, ` +
        `${sql(normalize(d.keywords))}, ${sql(normalize(d.content))});`,
    );
  }
  lines.push(`INSERT INTO index_meta (signature, documents) VALUES (${sql(signature)}, ${docs.length});`);
  const file = new URL("d1-data.sql", BUILD);
  writeFileSync(file, lines.join("\n") + "\n");
  return file;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

loadEnv();
mkdirSync(BUILD, { recursive: true });

const docs = loadCatalog(new URL("resources/indexes/", ROOT));
const signature = signatureOf(docs);
const counts = docs.reduce<Record<string, number>>((acc, d) => ({ ...acc, [d.category]: (acc[d.category] ?? 0) + 1 }), {});
console.log(`Catalog: ${docs.length} documents ${JSON.stringify(counts)}, signature ${signature}`);

if (args.has("--build-only")) {
  writeD1(docs, signature);
  console.log("Wrote build/d1-data.sql");
  process.exit(0);
}

const remote = remoteIndexState();
if (args.has("--if-changed") && remote?.signature === signature) {
  console.log("Index is up to date, nothing to do.");
  process.exit(0);
}
console.log(`Remote index: ${remote ? `${remote.documents} documents, ${remote.signature}` : "none"}`);

const vectorsFile = await writeVectors(docs, signature);
const d1File = writeD1(docs, signature);

console.log("Loading vectors into Vectorize...");
wrangler(["vectorize", "upsert", VECTORIZE_NAME, `--file=${fileURLToPath(vectorsFile)}`]);
if (remote && remote.documents > docs.length) {
  const stale = Array.from({ length: remote.documents - docs.length }, (_, i) => String(docs.length + i + 1));
  for (let i = 0; i < stale.length; i += 100) {
    wrangler(["vectorize", "delete-vectors", VECTORIZE_NAME, "--ids", ...stale.slice(i, i + 100)]);
  }
}

console.log("Loading documents into D1...");
wrangler(["d1", "execute", D1_NAME, "--remote", "--yes", `--file=${fileURLToPath(d1File)}`]);
console.log(`Index loaded: ${docs.length} documents, signature ${signature}`);
