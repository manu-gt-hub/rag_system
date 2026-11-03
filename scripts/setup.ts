/**
 * setup.ts — One-time, idempotent creation of the Cloudflare resources:
 *   - D1 database "rag-catalog" (its id is written into wrangler.jsonc)
 *   - Vectorize index "rag-catalog" (1024 d, cosine) + "category" metadata index
 *
 *   npx wrangler login   (or CLOUDFLARE_API_TOKEN / CLOUDFLARE_ACCOUNT_ID in .env)
 *   npm run setup
 */

import { readFileSync, writeFileSync } from "node:fs";

import { D1_NAME, EMBEDDING_DIMENSIONS, VECTORIZE_NAME, loadEnv, wrangler } from "./wrangler";

const CONFIG = new URL("../wrangler.jsonc", import.meta.url);

function d1Id(): string | undefined {
  const list = JSON.parse(wrangler(["d1", "list", "--json"], { capture: true })) as { name: string; uuid: string }[];
  return list.find((db) => db.name === D1_NAME)?.uuid;
}

function vectorizeExists(): boolean {
  const list = JSON.parse(wrangler(["vectorize", "list", "--json"], { capture: true })) as { name: string }[];
  return list.some((index) => index.name === VECTORIZE_NAME);
}

loadEnv();

let id = d1Id();
if (id) {
  console.log(`D1 database ${D1_NAME} exists (${id})`);
} else {
  wrangler(["d1", "create", D1_NAME]);
  id = d1Id();
  if (!id) throw new Error("D1 database was not created");
}

const config = readFileSync(CONFIG, "utf-8");
const updated = config.replace(/("database_id":\s*")[^"]*(")/, `$1${id}$2`);
if (updated !== config) {
  writeFileSync(CONFIG, updated);
  console.log(`Wrote database_id ${id} to wrangler.jsonc (commit this change)`);
}

if (vectorizeExists()) {
  console.log(`Vectorize index ${VECTORIZE_NAME} exists`);
} else {
  wrangler(["vectorize", "create", VECTORIZE_NAME, `--dimensions=${EMBEDDING_DIMENSIONS}`, "--metric=cosine"]);
  // Must exist before vectors are inserted, to filter by category
  wrangler([
    "vectorize", "create-metadata-index", VECTORIZE_NAME, "--property-name=category", "--type=string",
  ]);
}

console.log("Resources ready. Next: npm run index && npm run deploy");
