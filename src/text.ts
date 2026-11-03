/**
 * text.ts — Text rules shared by the Worker and the indexer: normalization,
 * meaningful query terms and category detection. Questions may come in
 * English or Spanish; the catalog data is in Spanish.
 */

export const CATEGORY_LABELS: Record<string, string> = {
  peliculas: "Movies",
  libros: "Books",
  series: "TV series",
  musica: "Music",
  videojuegos: "Video games",
  documentales: "Documentaries",
};
export const CATEGORIES = Object.keys(CATEGORY_LABELS);

// Words that carry no meaning for catalog search: stopwords plus the
// conversational filler typical of recommendation questions.
const STOPWORDS = new Set([
  // Spanish
  "a", "al", "algo", "alguna", "alguno", "algunas", "algunos", "ante", "como",
  "con", "cual", "cuales", "cuando", "de", "del", "desde", "donde", "el", "ella",
  "en", "entre", "era", "es", "esta", "este", "esto", "hay", "la", "las", "le",
  "les", "lo", "los", "mas", "me", "mi", "mis", "muy", "no", "nos", "o", "otra",
  "otro", "para", "pero", "por", "que", "quien", "se", "sea", "segun", "si",
  "sin", "sobre", "su", "sus", "tal", "te", "ti", "tu", "tus", "u", "un", "una",
  "unas", "uno", "unos", "y", "ya", "yo",
  "tengo", "tienes", "tiene", "tenemos", "dame", "dime", "quiero", "quisiera",
  "puedes", "podrias", "recomienda", "recomiendas", "recomiendame",
  "recomendacion", "recomendaciones", "sugiere", "sugiereme", "busco", "buscar",
  "ver", "contenido", "catalogo", "coleccion", "trata", "tratan", "parecido",
  "parecida", "similar", "similares", "estilo", "tipo", "gusta", "gustan",
  "gustaria", "mejor", "mejores", "bueno", "buena", "buenas", "buenos",
  "favor", "hola", "gracias", "hoy", "noche", "ahora", "cosa", "cosas",
  // English
  "the", "and", "for", "with", "about", "what", "which", "who", "have", "has",
  "any", "some", "something", "anything", "recommend", "suggest", "want",
  "would", "like", "can", "could", "you", "your", "give", "show", "find",
  "good", "best", "please", "watch", "read", "listen", "play", "kind", "from",
  "content", "according", "does", "did", "there", "rating", "ratings",
  "movie", "movies", "book", "books", "shows", "song", "songs", "music", "game",
  "games", "videogame", "videogames", "documentary", "documentaries",
  // category nouns (handled by category detection, too common in documents)
  "pelicula", "peliculas", "peli", "pelis", "film", "libro", "libros", "serie",
  "series", "musica", "videojuego", "videojuegos", "juego", "juegos", "documental",
  "documentales",
]);

// Category trigger words (normalized). "juego" in singular is left out on
// purpose so that "Juego de tronos" is not taken as a video game.
const CATEGORY_KEYWORDS: Record<string, Set<string>> = {
  peliculas: new Set(["pelicula", "peliculas", "peli", "pelis", "film", "films", "cine", "movie", "movies"]),
  libros: new Set(["libro", "libros", "leer", "lectura", "novela", "novelas", "ensayo", "ensayos",
    "book", "books", "novel", "novels", "read", "reading"]),
  series: new Set(["serie", "series", "temporada", "temporadas", "episodio", "episodios", "capitulo",
    "capitulos", "tv", "season", "seasons", "episode", "episodes"]),
  musica: new Set(["musica", "cancion", "canciones", "album", "albumes", "disco", "discos", "escuchar",
    "grupo", "grupos", "cantante", "banda", "rap", "rock", "pop", "jazz", "reggae", "blues", "funk",
    "soul", "rumba", "ska", "punk", "hiphop", "chillout", "music", "song", "songs", "listen", "band",
    "singer"]),
  videojuegos: new Set(["videojuego", "videojuegos", "juegos", "jugar", "consola", "nintendo",
    "playstation", "snes", "gameboy", "switch", "videogame", "videogames", "games", "rpg"]),
  documentales: new Set(["documental", "documentales", "documentary", "documentaries"]),
};

/** Lowercase and strip accents (ñ becomes n). */
export function normalize(text: string): string {
  return (text ?? "").normalize("NFKD").replace(/\p{M}/gu, "").toLowerCase();
}

/** Normalized alphanumeric tokens. */
export function tokenize(text: string): string[] {
  return normalize(text).match(/[a-z0-9]+/g) ?? [];
}

/** Space-padded normalized tokens, used to find a title inside a question. */
export function titleKey(text: string): string {
  return ` ${tokenize(text).join(" ")} `;
}

/** Meaningful search terms of a user query (no stopwords, no duplicates). */
export function queryTerms(text: string): string[] {
  const terms: string[] = [];
  for (const tok of tokenize(text)) {
    if (STOPWORDS.has(tok) || (tok.length < 3 && !/^\d+$/.test(tok))) continue;
    if (!terms.includes(tok)) terms.push(tok);
  }
  return terms;
}

/** Category explicitly mentioned in the question, if exactly one is. */
export function detectCategory(text: string): string | null {
  const tokens = new Set(tokenize(text));
  if (tokens.has("game") && tokens.has("boy")) tokens.add("gameboy");
  const matches = Object.entries(CATEGORY_KEYWORDS)
    .filter(([, words]) => [...tokens].some((t) => words.has(t)))
    .map(([cat]) => cat);
  return matches.length === 1 ? matches[0] : null;
}

/**
 * FTS5 query for the terms. SQLite has no Spanish stemmer, so longer terms
 * become prefix queries without a plural "s" ("peliculas" -> "pelicula"*).
 */
export function ftsQuery(terms: string[], operator: "OR" | "AND"): string {
  return terms
    .map((t) => (t.length >= 4 ? `"${t.replace(/s$/, "")}"*` : `"${t}"`))
    .join(` ${operator} `);
}
