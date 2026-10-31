"""
indexer.py — Loads JSON indexes, generates embeddings and stores them in pgvector.
Runnable as: python -m src.indexer

Each catalog item becomes one semantic document. Items that only make sense
as a group are aggregated first:
  - series: one document per show (not per episode).
  - musica: one document per artist/album folder (tracks have no metadata).
  - peliculas / videojuegos: duplicates (same title, regional ROMs) are merged.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import OrderedDict
from typing import Any, Dict, List

import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector

from src.config import (
    CONTENT_CATEGORIES,
    EMBEDDING_DIMENSION,
    EMBEDDING_PROVIDER,
    INDEXES_DIR,
    MAX_EMBED_CHARS,
    POSTGRES_DSN,
)
from src.embeddings import EmbeddingService
from src.text_utils import normalize, title_key

logger = logging.getLogger(__name__)

CATEGORY_LABELS = {
    "peliculas": "MOVIE",
    "libros": "BOOK",
    "series": "TV SERIES",
    "musica": "MUSIC",
    "videojuegos": "VIDEO GAME",
    "documentales": "DOCUMENTARY",
}

MAX_SYNOPSIS_CHARS = 1500
MAX_TRACKS_LISTED = 40
MAX_EPISODES_LISTED = 12

# Bump when document construction changes so `--if-needed` re-indexes
DOCUMENT_VERSION = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean(value: Any) -> str:
    """Collapse whitespace of a scalar value; empty string for None."""
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _format_rating(rating: Any, votes: Any = None) -> str:
    try:
        rating = float(rating)
    except (TypeError, ValueError):
        return ""
    if rating <= 0:
        return ""
    text = f"{rating:.2f}".rstrip("0").rstrip(".") + "/10"
    if votes:
        text += f" ({votes} votes)"
    return text


# Catalog genres are in Spanish; the English term is added so that English
# questions ("a horror movie") match both semantically and by keyword.
GENRES_EN = {
    "terror": "horror", "suspense": "thriller", "ciencia ficcion": "science fiction",
    "accion": "action", "aventura": "adventure", "aventuras": "adventure",
    "belica": "war", "crimen": "crime", "comedia": "comedy", "drama": "drama",
    "animacion": "animation", "documental": "documentary", "misterio": "mystery",
    "fantasia": "fantasy", "fantastico": "fantasy", "romance": "romance",
    "historia": "history", "historico": "historical", "familia": "family",
    "musica": "music", "western": "western", "novela": "novel", "ensayo": "essay",
    "filosofia": "philosophy", "ciencias sociales": "social sciences",
    "ciencias naturales": "natural sciences", "ciencias exactas": "exact sciences",
    "cronica": "chronicle", "teatro": "theatre", "relato": "short story",
    "divulgacion": "popular science", "biografia": "biography", "memorias": "memoir",
    "autoayuda": "self-help", "policial": "detective", "intriga": "mystery",
    "satira": "satire", "juvenil": "young adult", "interactivo": "interactive",
    "otros": "other", "libros supervivencia": "survival", "poesia": "poetry",
    "filosofico": "philosophical", "espiritualidad": "spirituality",
    "pelicula de tv": "tv movie",
}


def _with_english_genre(genre: str | None) -> str:
    """'Terror, Suspense' -> 'Terror, Suspense / horror, thriller'."""
    if not genre:
        return ""
    english = []
    for part in genre.split(","):
        en = GENRES_EN.get(normalize(part.strip()))
        if en and en != part.strip().lower() and en not in english:
            english.append(en)
    return f"{genre} / {', '.join(english)}" if english else genre


def _richness(item: Dict[str, Any]) -> int:
    """Number of non-empty fields; used to keep the best duplicate."""
    return sum(1 for v in item.values() if v not in (None, "", [], 0, 0.0))


def _make_doc(
    category: str,
    title: str,
    fields: List[tuple],
    keywords: List[str],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Assemble a document dict from (label, value) pairs."""
    lines = [f"[{CATEGORY_LABELS[category]}] {title}"]
    for label, value in fields:
        value = _clean(value)
        if value:
            lines.append(f"{label}: {value}")
    return {
        "category": category,
        "title": title,
        "text": "\n".join(lines),
        "keywords": " ".join(_clean(k) for k in keywords if k),
        "metadata": metadata,
    }


def _dedupe(items: List[Dict[str, Any]], key_fn) -> List[Dict[str, Any]]:
    """Keep the richest item for each key, preserving original order."""
    best: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    for item in items:
        key = key_fn(item)
        if key not in best or _richness(item) > _richness(best[key]):
            best[key] = item
    return list(best.values())


# ---------------------------------------------------------------------------
# Per-category document builders
# ---------------------------------------------------------------------------

def _build_video_docs(items: List[Dict[str, Any]], category: str) -> List[Dict[str, Any]]:
    """Movies and documentaries: one document per title."""
    items = _dedupe(items, lambda i: normalize(i.get("titulo_oficial") or i["titulo"]))
    docs = []
    for item in items:
        title = _clean(item.get("titulo_oficial") or item.get("titulo"))
        actores = item.get("actores") or []
        rating = _format_rating(item.get("rating") or item.get("vote_average"),
                                item.get("rating_votos") or item.get("vote_count"))
        sinopsis = _clean(item.get("sinopsis"))[:MAX_SYNOPSIS_CHARS]
        docs.append(_make_doc(
            category,
            title,
            [
                ("Catalog title", item.get("titulo") if item.get("titulo") != title else ""),
                ("Year", item.get("año")),
                ("Genre", _with_english_genre(item.get("genero"))),
                ("Director", item.get("director")),
                ("Cast", ", ".join(actores[:8])),
                ("Synopsis", sinopsis),
                ("Rating", rating),
            ],
            [item.get("titulo"), item.get("director"), *actores[:8], item.get("genero")],
            {
                "year": item.get("año"),
                "genre": item.get("genero"),
                "director": item.get("director"),
                "cast": actores[:8],
                "rating": item.get("rating") or item.get("vote_average"),
                "poster_url": item.get("poster_url"),
                "file": item.get("archivo"),
            },
        ))
    return docs


def _build_book_docs(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    items = _dedupe(items, lambda i: (normalize(i["titulo"]), normalize(i.get("autor") or "")))
    docs = []
    for item in items:
        title = _clean(item.get("titulo"))
        sinopsis = _clean(item.get("sinopsis"))[:MAX_SYNOPSIS_CHARS]
        docs.append(_make_doc(
            "libros",
            title,
            [
                ("Author", item.get("autor")),
                ("Year", item.get("año")),
                ("Genre", _with_english_genre(item.get("genero"))),
                ("Publisher", item.get("editorial")),
                ("Synopsis", sinopsis),
                ("Rating", _format_rating(item.get("rating"), item.get("rating_votos"))),
            ],
            [item.get("autor"), item.get("genero")],
            {
                "year": item.get("año"),
                "genre": item.get("genero"),
                "author": item.get("autor"),
                "rating": item.get("rating"),
                "poster_url": item.get("portada_url"),
                "file": item.get("archivo"),
            },
        ))
    return docs


def _build_series_docs(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One document per show, summarising the episodes available."""
    groups: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
    for item in items:
        groups.setdefault(item.get("serie") or item["titulo"], []).append(item)

    docs = []
    for folder, episodes in groups.items():
        ref = max(episodes, key=_richness)
        title = _clean(ref.get("titulo_serie_oficial") or folder)
        actores = ref.get("actores") or []
        seasons = sorted({e["temporada"] for e in episodes if e.get("temporada")})
        sample = [_clean(e["titulo"]) for e in episodes[:MAX_EPISODES_LISTED]]
        docs.append(_make_doc(
            "series",
            title,
            [
                ("Catalog name", folder if normalize(folder) != normalize(title) else ""),
                ("Year", ref.get("año")),
                ("Genre", _with_english_genre(ref.get("genero"))),
                ("Creator/Director", ref.get("director")),
                ("Cast", ", ".join(actores[:8])),
                ("Total seasons", ref.get("temporadas")),
                ("Episodes in catalog", f"{len(episodes)}"
                    + (f" (seasons {', '.join(map(str, seasons))})" if seasons else "")),
                ("Synopsis", _clean(ref.get("sinopsis"))[:MAX_SYNOPSIS_CHARS]),
                ("Rating", _format_rating(ref.get("rating"), ref.get("rating_votos"))),
                ("Some episodes", "; ".join(sample)),
            ],
            [folder, ref.get("director"), *actores[:8], ref.get("genero")],
            {
                "year": ref.get("año"),
                "genre": ref.get("genero"),
                "cast": actores[:8],
                "rating": ref.get("rating"),
                "poster_url": ref.get("poster_url"),
                "episodes": len(episodes),
                "file": f"Series/{folder}",
            },
        ))
    return docs


_MUSIC_NOISE = re.compile(
    r"\[[^\]]*\]|\{[^}]*\}|⭐️?|www\.\S+|\b(mp3 songs|mp3|flac|320 ?kbps|kbps|cdrip|\d+bit\S*|"
    r"\d{2}-\d{2,3}|hd|quality( album| songs)?)\b",
    re.IGNORECASE,
)


def _clean_music_name(name: str) -> str:
    name = _MUSIC_NOISE.sub(" ", name.replace("_", " "))
    name = re.sub(r"\(\s*\)", " ", name)
    name = re.sub(r"\s+", " ", name).strip(" -_.")
    return name or "Varios"


def _build_music_docs(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One document per collection/sub-collection folder (two levels under Musica/)."""
    groups: "OrderedDict[tuple, List[Dict[str, Any]]]" = OrderedDict()
    for item in items:
        parts = item.get("archivo", "").split("/")[1:-1]  # drop "Musica/" and filename
        key = tuple(parts[:2]) or (item.get("album") or "Varios",)
        groups.setdefault(key, []).append(item)

    docs = []
    for key, tracks in groups.items():
        names = [_clean_music_name(p) for p in key]
        title = " / ".join(names)
        subfolders = sorted({
            _clean_music_name(t["archivo"].split("/")[3])
            for t in tracks if len(t["archivo"].split("/")) > 4
        })
        track_titles = [
            re.sub(r"^\d+\s*", "", _clean(t["titulo"])) for t in tracks[:MAX_TRACKS_LISTED]
        ]
        artists = sorted({_clean(t["artista"]) for t in tracks if t.get("artista")})
        years = sorted({t["año"] for t in tracks if t.get("año")})
        docs.append(_make_doc(
            "musica",
            title,
            [
                ("Collection", names[0]),
                ("Artist/Album", " / ".join(names[1:])),
                ("Artists", ", ".join(artists)),
                ("Years", ", ".join(map(str, years))),
                ("Track count", len(tracks)),
                ("Discs/subfolders", "; ".join(subfolders[:30])),
                ("Tracks", "; ".join(track_titles)),
            ],
            [*names, *artists],
            {
                "tracks": len(tracks),
                "file": "Musica/" + "/".join(key),
            },
        ))
    return docs


# Libretro genres are in English; the Spanish term is added so that Spanish
# questions ("un juego de rol") match both semantically and by keyword.
GAME_GENRES_ES = {
    "action": "acción", "adventure": "aventura", "platform": "plataformas",
    "sports": "deportes", "role-playing (rpg)": "rol", "racing": "carreras",
    "strategy": "estrategia", "puzzle": "puzle", "fighting": "lucha",
    "shooter": "disparos", "simulation": "simulación", "educational": "educativo",
    "board": "juego de mesa", "shoot'em up": "matamarcianos", "beat'em up": "beat'em up (peleas)",
    "music": "música", "card & board game": "cartas y mesa", "pinball": "pinball",
    "quiz/trivia": "preguntas", "compilation": "recopilación",
}


def _game_genre(genre: str | None) -> str:
    if not genre:
        return ""
    spanish = GAME_GENRES_ES.get(genre.strip().lower())
    return f"{genre} / {spanish}" if spanish and spanish != genre.lower() else genre


def _build_game_docs(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    items = _dedupe(items, lambda i: (normalize(i["titulo"]), i.get("plataforma")))
    docs = []
    for item in items:
        title = _clean(item.get("titulo"))
        genre = _game_genre(item.get("genero"))
        docs.append(_make_doc(
            "videojuegos",
            title,
            [
                ("Plataforma", item.get("plataforma")),
                ("Year", item.get("año")),
                ("Genre", genre),
                ("Rating", _format_rating(item.get("rating"))),
            ],
            [item.get("plataforma"), genre],
            {
                "year": item.get("año"),
                "genre": item.get("genero"),
                "platform": item.get("plataforma"),
                "rating": item.get("rating"),
                "file": item.get("archivo"),
            },
        ))
    return docs


def build_documents(category: str, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Dispatch to the builder of each category."""
    if category in ("peliculas", "documentales"):
        return _build_video_docs(items, category)
    if category == "libros":
        return _build_book_docs(items)
    if category == "series":
        return _build_series_docs(items)
    if category == "musica":
        return _build_music_docs(items)
    if category == "videojuegos":
        return _build_game_docs(items)
    raise ValueError(f"Unknown category: {category}")


def load_indexes() -> List[Dict[str, Any]]:
    """Load all JSON indexes and return a list of {category, title, text, keywords, metadata}."""
    documents = []
    for category in CONTENT_CATEGORIES:
        index_path = INDEXES_DIR / category / "index.json"
        if not index_path.exists():
            logger.warning("Index not found at %s, skipping.", index_path)
            continue

        with open(index_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        items = data.get(category, data.get(list(data.keys())[0], []))
        docs = build_documents(category, items)
        logger.info("Loaded %d records from %s -> %d documents", len(items), category, len(docs))
        documents.extend(docs)

    logger.info("Total documents: %d", len(documents))
    return documents


# ---------------------------------------------------------------------------
# pgvector: table creation and insertion
# ---------------------------------------------------------------------------

def init_pgvector(conn) -> None:
    """(Re)create the documents table. Search indexes are built after loading."""
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("DROP TABLE IF EXISTS documents;")
        cur.execute(f"""
            CREATE TABLE documents (
                id          SERIAL PRIMARY KEY,
                category    TEXT NOT NULL,
                title       TEXT,
                content     TEXT NOT NULL,
                metadata    JSONB,
                title_key   TEXT,
                tsv         TSVECTOR,
                embedding   vector({EMBEDDING_DIMENSION})
            );
        """)
    conn.commit()


def create_search_indexes(conn) -> None:
    """HNSW index for cosine similarity + GIN index for full-text search."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE INDEX idx_documents_embedding
            ON documents USING hnsw (embedding vector_cosine_ops);
        """)
        cur.execute("CREATE INDEX idx_documents_tsv ON documents USING gin (tsv);")
        cur.execute("CREATE INDEX idx_documents_category ON documents (category);")
        cur.execute("ANALYZE documents;")
    conn.commit()


def current_index_signature() -> str:
    """Identifies the embedding space; a change requires re-indexing."""
    from src.config import LOCAL_EMBEDDING_MODEL, OPENAI_EMBEDDING_MODEL
    model = OPENAI_EMBEDDING_MODEL if EMBEDDING_PROVIDER == "openai" else LOCAL_EMBEDDING_MODEL
    return f"{EMBEDDING_PROVIDER}:{model}:{EMBEDDING_DIMENSION}:docs-v{DOCUMENT_VERSION}"


def read_index_signature(conn) -> str | None:
    """Signature stored by the last indexing run, or None if there is no index."""
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('index_meta') IS NOT NULL;")
        if not cur.fetchone()[0]:
            return None
        cur.execute("SELECT signature FROM index_meta LIMIT 1;")
        row = cur.fetchone()
    return row[0] if row else None


def write_index_signature(conn, documents: int) -> None:
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS index_meta (
                signature   TEXT NOT NULL,
                documents   INTEGER NOT NULL,
                indexed_at  TIMESTAMPTZ NOT NULL DEFAULT now()
            );
        """)
        cur.execute("TRUNCATE index_meta;")
        cur.execute(
            "INSERT INTO index_meta (signature, documents) VALUES (%s, %s);",
            (current_index_signature(), documents),
        )
    conn.commit()


def insert_documents(
    conn,
    documents: List[Dict[str, Any]],
    embeddings: List[List[float]],
) -> int:
    """Insert documents with their embeddings into pgvector. Returns count inserted."""
    rows = [
        (
            doc["category"],
            doc["title"],
            doc["text"],
            json.dumps(doc["metadata"], ensure_ascii=False),
            title_key(doc["title"]),
            normalize(doc["title"]),
            normalize(doc["keywords"]),
            normalize(doc["text"]),
            emb,
        )
        for doc, emb in zip(documents, embeddings)
    ]
    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO documents (category, title, content, metadata, title_key, tsv, embedding)
            SELECT category, title, content, metadata::jsonb, title_key,
                   setweight(to_tsvector('spanish', t), 'A')
                   || setweight(to_tsvector('spanish', k), 'B')
                   || setweight(to_tsvector('spanish', c), 'D'),
                   embedding
            FROM (VALUES %s) AS v(category, title, content, metadata, title_key, t, k, c, embedding)
            """,
            rows,
            template=f"(%s, %s, %s, %s, %s, %s, %s, %s, %s::vector({EMBEDDING_DIMENSION}))",
            page_size=200,
        )
    conn.commit()
    return len(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_indexing(if_needed: bool = False) -> None:
    """Full indexing pipeline.

    Args:
        if_needed: Skip when the database already holds an index built with
            the current embedding model (used by the Docker entrypoint).
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    for noisy in ("httpx", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    conn = psycopg2.connect(POSTGRES_DSN)
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()
    register_vector(conn)

    signature = current_index_signature()
    if if_needed:
        existing = read_index_signature(conn)
        if existing == signature:
            logger.info("Index is up to date (%s), skipping indexing.", signature)
            conn.close()
            return
        logger.info("Index missing or built with another model (%s), indexing...", existing)

    # 1. Load documents
    documents = load_indexes()
    if not documents:
        logger.error("No documents found to index.")
        conn.close()
        return

    # 2. Generate embeddings
    logger.info("Generating embeddings for %d documents with %s...", len(documents), signature)
    embedding_svc = EmbeddingService()
    texts = [doc["text"][:MAX_EMBED_CHARS] for doc in documents]
    embeddings = embedding_svc.embed_batch(texts)

    # 3. Recreate schema and insert
    logger.info("Recreating pgvector schema...")
    init_pgvector(conn)

    logger.info("Inserting documents...")
    inserted = insert_documents(conn, documents, embeddings)

    logger.info("Building HNSW and full-text indexes...")
    create_search_indexes(conn)
    write_index_signature(conn, inserted)
    logger.info("Indexing complete: %d documents inserted.", inserted)

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index the catalog into pgvector.")
    parser.add_argument("--if-needed", action="store_true",
                        help="Only index if there is no index for the current embedding model.")
    run_indexing(if_needed=parser.parse_args().if_needed)
