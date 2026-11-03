/** indexer.test.ts — Document construction from the catalog JSON. */

import { describe, expect, it } from "vitest";

import { buildDocuments, cleanMusicName, formatRating, loadCatalog, withEnglishGenre } from "../indexer/documents";

describe("documents", () => {
  it("builds a movie document with metadata", () => {
    const [doc] = buildDocuments("peliculas", [
      {
        titulo: "300", titulo_oficial: "300", "año": 2006, genero: "Acción", director: "Zack Snyder",
        actores: ["Gerard Butler"], sinopsis: "Leónidas y Esparta.", rating: 7.208, rating_votos: 100,
      },
    ]);
    expect(doc.content.startsWith("[MOVIE] 300")).toBe(true);
    expect(doc.content).toContain("Director: Zack Snyder");
    expect(doc.content).toContain("Gerard Butler");
    expect(doc.content).toContain("Rating: 7.21/10 (100 votes)");
    expect(doc.content).toContain("Genre: Acción / action");
  });

  it("merges duplicate movies keeping the richest", () => {
    const docs = buildDocuments("peliculas", [
      { titulo: "Conan" },
      { titulo: "Conan", titulo_oficial: "Conan", director: "X" },
    ]);
    expect(docs).toHaveLength(1);
    expect(docs[0].content).toContain("Director: X");
  });

  it("groups series episodes into one show", () => {
    const episodes = Array.from({ length: 5 }, (_, n) => ({
      titulo: `Ep ${n}`, serie: "Perdidos", temporada: 1, titulo_serie_oficial: "Perdidos", genero: "Misterio",
    }));
    const docs = buildDocuments("series", episodes);
    expect(docs).toHaveLength(1);
    expect(docs[0].title).toBe("Perdidos");
    expect(docs[0].content).toContain("Episodes in catalog: 5 (seasons 1)");
  });

  it("groups music tracks by artist/album folder", () => {
    const docs = buildDocuments("musica", [
      { titulo: "01 Song A", archivo: "Musica/_GENERO ESPAÑOL/Estopa - Discografia/Disco 1/01 a.mp3" },
      { titulo: "02 Song B", archivo: "Musica/_GENERO ESPAÑOL/Estopa - Discografia/Disco 2/02 b.mp3" },
      { titulo: "03 Song C", archivo: "Musica/ACDC/03 c.mp3" },
    ]);
    expect(docs.map((d) => d.title)).toEqual(["GENERO ESPAÑOL / Estopa - Discografia", "ACDC"]);
    expect(docs[0].content).toContain("Track count: 2");
    expect(docs[0].content).toContain("Discs/subfolders: Disco 1; Disco 2");
  });

  it("cleans release noise from music folder names", () => {
    expect(cleanMusicName("VA - 100 Greatest Hip Hop (2019) Mp3 320kbps [PMEDIA]")).toBe("VA - 100 Greatest Hip Hop (2019)");
    expect(cleanMusicName("100 Greatest Bob Marley Songs (2021) Mp3 320kbps [PMEDIA] ⭐️")).toBe(
      "100 Greatest Bob Marley Songs (2021)",
    );
  });

  it("deduplicates games per platform and translates genres", () => {
    const docs = buildDocuments("videojuegos", [
      { titulo: "Tetris", plataforma: "Game Boy" },
      { titulo: "Tetris", plataforma: "Game Boy", genero: "Puzzle" },
      { titulo: "Tetris", plataforma: "SNES" },
      { titulo: "Pokemon", plataforma: "Game Boy", genero: "Role-playing (RPG)" },
    ]);
    expect(docs).toHaveLength(3);
    expect(docs[2].content).toContain("Genre: Role-playing (RPG) / rol");
  });

  it("formats ratings and bilingual genres", () => {
    expect(formatRating(7.56, 4324)).toBe("7.56/10 (4324 votes)");
    expect(formatRating(0)).toBe("");
    expect(formatRating(null)).toBe("");
    expect(withEnglishGenre("Terror, Suspense, Ciencia ficción")).toBe(
      "Terror, Suspense, Ciencia ficción / horror, thriller, science fiction",
    );
  });

  it("loads the real catalog with grouping", () => {
    const docs = loadCatalog(new URL("../resources/indexes/", import.meta.url));
    const count = (cat: string) => docs.filter((d) => d.category === cat).length;
    expect(new Set(docs.map((d) => d.category)).size).toBe(6);
    expect(count("series")).toBeLessThan(50); // grouped by show, not 1,800 episodes
    expect(count("musica")).toBeLessThan(1000); // grouped by folder, not 12,900 tracks
    expect(docs.every((d, i) => d.id === i + 1 && d.title && d.content)).toBe(true);
  });
});
