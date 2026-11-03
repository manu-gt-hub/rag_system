/**
 * documents.ts — Turns the catalog JSON indexes into semantic documents.
 *
 * What gets embedded is text describing each item (title, year, genre,
 * director/author, cast, synopsis, rating), not file names. Items that only
 * make sense as a group are aggregated first:
 *   - series: one document per show (not per episode).
 *   - music: one document per artist/album folder (tracks have no metadata).
 *   - movies / games / books: duplicates are merged, keeping the richest one.
 */

import { readFileSync } from "node:fs";

import { CATEGORIES, normalize, titleKey } from "../src/text";

// Bump when document construction changes, so the next deploy re-indexes
export const DOCUMENT_VERSION = 6;

const MAX_SYNOPSIS_CHARS = 1500;
const MAX_TRACKS_LISTED = 40;
const MAX_EPISODES_LISTED = 12;

const DOC_LABELS: Record<string, string> = {
  peliculas: "MOVIE",
  libros: "BOOK",
  series: "TV SERIES",
  musica: "MUSIC",
  videojuegos: "VIDEO GAME",
  documentales: "DOCUMENTARY",
};

// Catalog genres are in Spanish; the English term is added so that English
// questions ("a horror movie") match both semantically and by keyword.
const GENRES_EN: Record<string, string> = {
  terror: "horror", suspense: "thriller", "ciencia ficcion": "science fiction",
  accion: "action", aventura: "adventure", aventuras: "adventure",
  belica: "war", crimen: "crime", comedia: "comedy", drama: "drama",
  animacion: "animation", documental: "documentary", misterio: "mystery",
  fantasia: "fantasy", fantastico: "fantasy", romance: "romance",
  historia: "history", historico: "historical", familia: "family",
  musica: "music", western: "western", novela: "novel", ensayo: "essay",
  filosofia: "philosophy", "ciencias sociales": "social sciences",
  "ciencias naturales": "natural sciences", "ciencias exactas": "exact sciences",
  cronica: "chronicle", teatro: "theatre", relato: "short story",
  divulgacion: "popular science", biografia: "biography", memorias: "memoir",
  autoayuda: "self-help", policial: "detective", intriga: "mystery",
  satira: "satire", juvenil: "young adult", interactivo: "interactive",
  otros: "other", "libros supervivencia": "survival", poesia: "poetry",
  filosofico: "philosophical", espiritualidad: "spirituality",
  "pelicula de tv": "tv movie",
};

// Video game genres (Libretro) are in English; the Spanish term is added so
// that Spanish questions ("un juego de rol") match too.
const GAME_GENRES_ES: Record<string, string> = {
  action: "acción", adventure: "aventura", platform: "plataformas",
  sports: "deportes", "role-playing (rpg)": "rol", racing: "carreras",
  strategy: "estrategia", puzzle: "puzle", fighting: "lucha",
  shooter: "disparos", simulation: "simulación", educational: "educativo",
  board: "juego de mesa", "shoot'em up": "matamarcianos", "beat'em up": "beat'em up (peleas)",
  music: "música", "card & board game": "cartas y mesa", pinball: "pinball",
  "quiz/trivia": "preguntas", compilation: "recopilación",
};

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Item = Record<string, any>;

export interface CatalogDocument {
  id: number;
  category: string;
  title: string;
  /** Text that is embedded and given to the LLM. */
  content: string;
  /** People, genres... (weighted above content in keyword search). */
  keywords: string;
  titleKey: string;
  metadata: Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Collapse whitespace of a scalar value; empty string for null. */
export function clean(value: unknown): string {
  if (value === null || value === undefined) return "";
  return String(value).replace(/\s+/g, " ").trim();
}

export function formatRating(rating: unknown, votes?: unknown): string {
  const value = Number(rating);
  if (rating === null || rating === undefined || Number.isNaN(value) || value <= 0) return "";
  let text = value.toFixed(2).replace(/0+$/, "").replace(/\.$/, "") + "/10";
  if (votes) text += ` (${votes} votes)`;
  return text;
}

/** 'Terror, Suspense' -> 'Terror, Suspense / horror, thriller'. */
export function withEnglishGenre(genre: string | null | undefined): string {
  if (!genre) return "";
  const english: string[] = [];
  for (const part of genre.split(",")) {
    const en = GENRES_EN[normalize(part.trim())];
    if (en && en !== part.trim().toLowerCase() && !english.includes(en)) english.push(en);
  }
  return english.length ? `${genre} / ${english.join(", ")}` : genre;
}

export function gameGenre(genre: string | null | undefined): string {
  if (!genre) return "";
  const spanish = GAME_GENRES_ES[genre.trim().toLowerCase()];
  return spanish && spanish !== genre.toLowerCase() ? `${genre} / ${spanish}` : genre;
}

/** Number of non-empty fields; used to keep the best duplicate. */
function richness(item: Item): number {
  return Object.values(item).filter(
    (v) => !(v === null || v === undefined || v === "" || v === 0 || v === false || (Array.isArray(v) && !v.length)),
  ).length;
}

/** Keep the richest item for each key, preserving the original order. */
function dedupe(items: Item[], keyFn: (item: Item) => string): Item[] {
  const best = new Map<string, Item>();
  for (const item of items) {
    const key = keyFn(item);
    const current = best.get(key);
    if (!current || richness(item) > richness(current)) best.set(key, item);
  }
  return [...best.values()];
}

function makeDoc(
  category: string,
  title: string,
  fields: [string, unknown][],
  keywords: unknown[],
  metadata: Record<string, unknown>,
): Omit<CatalogDocument, "id"> {
  const lines = [`[${DOC_LABELS[category]}] ${title}`];
  for (const [label, value] of fields) {
    const text = clean(value);
    if (text) lines.push(`${label}: ${text}`);
  }
  return {
    category,
    title,
    content: lines.join("\n"),
    keywords: keywords.filter(Boolean).map(clean).join(" "),
    titleKey: titleKey(title),
    metadata,
  };
}

// ---------------------------------------------------------------------------
// Per-category builders
// ---------------------------------------------------------------------------

/** Movies and documentaries: one document per title. */
function videoDocs(items: Item[], category: string) {
  return dedupe(items, (i) => normalize(i.titulo_oficial || i.titulo)).map((item) => {
    const title = clean(item.titulo_oficial || item.titulo);
    const cast: string[] = (item.actores || []).slice(0, 8);
    return makeDoc(
      category,
      title,
      [
        ["Catalog title", item.titulo !== title ? item.titulo : ""],
        ["Year", item["año"]],
        ["Genre", withEnglishGenre(item.genero)],
        ["Director", item.director],
        ["Cast", cast.join(", ")],
        ["Synopsis", clean(item.sinopsis).slice(0, MAX_SYNOPSIS_CHARS)],
        ["Rating", formatRating(item.rating || item.vote_average, item.rating_votos || item.vote_count)],
      ],
      [item.titulo, item.director, ...cast, item.genero],
      {
        year: item["año"] ?? null,
        genre: item.genero ?? null,
        director: item.director ?? null,
        cast,
        rating: item.rating || item.vote_average || null,
        poster_url: item.poster_url ?? null,
      },
    );
  });
}

function bookDocs(items: Item[]) {
  return dedupe(items, (i) => `${normalize(i.titulo)}|${normalize(i.autor || "")}`).map((item) =>
    makeDoc(
      "libros",
      clean(item.titulo),
      [
        ["Author", item.autor],
        ["Year", item["año"]],
        ["Genre", withEnglishGenre(item.genero)],
        ["Publisher", item.editorial],
        ["Synopsis", clean(item.sinopsis).slice(0, MAX_SYNOPSIS_CHARS)],
        ["Rating", formatRating(item.rating, item.rating_votos)],
      ],
      [item.autor, item.genero],
      {
        year: item["año"] ?? null,
        genre: item.genero ?? null,
        author: item.autor ?? null,
        rating: item.rating ?? null,
        poster_url: item.portada_url ?? null,
      },
    ),
  );
}

/** One document per show, summarising the episodes available. */
function seriesDocs(items: Item[]) {
  const groups = new Map<string, Item[]>();
  for (const item of items) {
    const key = item.serie || item.titulo;
    groups.set(key, [...(groups.get(key) ?? []), item]);
  }
  return [...groups.entries()].map(([folder, episodes]) => {
    const ref = episodes.reduce((best, e) => (richness(e) > richness(best) ? e : best));
    const title = clean(ref.titulo_serie_oficial || folder);
    const cast: string[] = (ref.actores || []).slice(0, 8);
    const seasons = [...new Set(episodes.map((e) => e.temporada).filter(Boolean))].sort((a, b) => a - b);
    return makeDoc(
      "series",
      title,
      [
        ["Catalog name", normalize(folder) !== normalize(title) ? folder : ""],
        ["Year", ref["año"]],
        ["Genre", withEnglishGenre(ref.genero)],
        ["Creator/Director", ref.director],
        ["Cast", cast.join(", ")],
        ["Total seasons", ref.temporadas],
        ["Episodes in catalog", `${episodes.length}` + (seasons.length ? ` (seasons ${seasons.join(", ")})` : "")],
        ["Synopsis", clean(ref.sinopsis).slice(0, MAX_SYNOPSIS_CHARS)],
        ["Rating", formatRating(ref.rating, ref.rating_votos)],
        ["Some episodes", episodes.slice(0, MAX_EPISODES_LISTED).map((e) => clean(e.titulo)).join("; ")],
      ],
      [folder, ref.director, ...cast, ref.genero],
      {
        year: ref["año"] ?? null,
        genre: ref.genero ?? null,
        cast,
        rating: ref.rating ?? null,
        poster_url: ref.poster_url ?? null,
        episodes: episodes.length,
      },
    );
  });
}

const MUSIC_NOISE =
  /\[[^\]]*\]|\{[^}]*\}|⭐️?|www\.\S+|\b(mp3 songs|mp3|flac|320 ?kbps|kbps|cdrip|\d+bit\S*|\d{2}-\d{2,3}|hd|quality( album| songs)?)\b/gi;

export function cleanMusicName(name: string): string {
  const cleaned = name
    .replaceAll("_", " ")
    .replace(MUSIC_NOISE, " ")
    .replace(/\(\s*\)/g, " ")
    .replace(/\s+/g, " ")
    .replace(/^[\s\-_.]+|[\s\-_.]+$/g, "");
  return cleaned || "Varios";
}

/** One document per collection / sub-collection folder (two levels under Musica/). */
function musicDocs(items: Item[]) {
  const groups = new Map<string, { key: string[]; tracks: Item[] }>();
  for (const item of items) {
    const parts: string[] = String(item.archivo ?? "").split("/").slice(1, -1); // drop "Musica/" and file
    const key = parts.length ? parts.slice(0, 2) : [item.album || "Varios"];
    const id = key.join("\u0000");
    if (!groups.has(id)) groups.set(id, { key, tracks: [] });
    groups.get(id)!.tracks.push(item);
  }
  return [...groups.values()].map(({ key, tracks }) => {
    const names = key.map(cleanMusicName);
    const subfolders = [
      ...new Set(
        tracks
          .map((t) => String(t.archivo).split("/"))
          .filter((p) => p.length > 4)
          .map((p) => cleanMusicName(p[3])),
      ),
    ].sort();
    const artists = [...new Set(tracks.filter((t) => t.artista).map((t) => clean(t.artista)))].sort();
    const years = [...new Set(tracks.map((t) => t["año"]).filter(Boolean))].sort((a, b) => a - b);
    return makeDoc(
      "musica",
      names.join(" / "),
      [
        ["Collection", names[0]],
        ["Artist/Album", names.slice(1).join(" / ")],
        ["Artists", artists.join(", ")],
        ["Years", years.join(", ")],
        ["Track count", tracks.length],
        ["Discs/subfolders", subfolders.slice(0, 30).join("; ")],
        ["Tracks", tracks.slice(0, MAX_TRACKS_LISTED).map((t) => clean(t.titulo).replace(/^\d+\s*/, "")).join("; ")],
      ],
      [...names, ...artists],
      { tracks: tracks.length },
    );
  });
}

function gameDocs(items: Item[]) {
  return dedupe(items, (i) => `${normalize(i.titulo)}|${i.plataforma}`).map((item) => {
    const genre = gameGenre(item.genero);
    return makeDoc(
      "videojuegos",
      clean(item.titulo),
      [
        ["Platform", item.plataforma],
        ["Year", item["año"]],
        ["Genre", genre],
        ["Rating", formatRating(item.rating)],
      ],
      [item.plataforma, genre],
      {
        year: item["año"] ?? null,
        genre: item.genero ?? null,
        platform: item.plataforma ?? null,
        rating: item.rating ?? null,
      },
    );
  });
}

export function buildDocuments(category: string, items: Item[]): Omit<CatalogDocument, "id">[] {
  switch (category) {
    case "peliculas":
    case "documentales":
      return videoDocs(items, category);
    case "libros":
      return bookDocs(items);
    case "series":
      return seriesDocs(items);
    case "musica":
      return musicDocs(items);
    case "videojuegos":
      return gameDocs(items);
    default:
      throw new Error(`Unknown category: ${category}`);
  }
}

/** Load every resources/indexes/<category>/index.json and build all documents. */
export function loadCatalog(indexesDir: URL): CatalogDocument[] {
  const docs: CatalogDocument[] = [];
  for (const category of CATEGORIES) {
    const data = JSON.parse(readFileSync(new URL(`${category}/index.json`, indexesDir), "utf-8"));
    const items: Item[] = data[category] ?? Object.values(data)[0] ?? [];
    for (const doc of buildDocuments(category, items)) docs.push({ id: docs.length + 1, ...doc });
  }
  return docs;
}
