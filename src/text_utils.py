"""
text_utils.py — Text normalization helpers shared by indexer and retriever.
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Optional

# Words that carry no meaning for catalog search: Spanish stopwords plus
# conversational filler typical of recommendation questions.
STOPWORDS = {
    "a", "al", "algo", "alguna", "alguno", "algunas", "algunos", "ante", "como",
    "con", "cual", "cuales", "cuando", "de", "del", "desde", "donde", "el", "ella",
    "en", "entre", "era", "es", "esta", "este", "esto", "hay", "la", "las", "le",
    "les", "lo", "los", "mas", "me", "mi", "mis", "muy", "no", "nos", "o", "otra",
    "otro", "para", "pero", "por", "que", "quien", "se", "sea", "segun", "si",
    "sin", "sobre", "su", "sus", "tal", "te", "ti", "tu", "tus", "u", "un", "una",
    "unas", "uno", "unos", "y", "ya", "yo",
    # conversational filler
    "tengo", "tienes", "tiene", "tenemos", "dame", "dime", "quiero", "quisiera",
    "puedes", "podrias", "recomienda", "recomiendas", "recomiendame",
    "recomendacion", "recomendaciones", "sugiere", "sugiereme", "busco", "buscar",
    "ver", "contenido", "catalogo", "coleccion", "trata", "tratan", "parecido",
    "parecida", "similar", "similares", "estilo", "tipo", "gusta", "gustan",
    "gustaria", "mejor", "mejores", "bueno", "buena", "buenas", "buenos",
    "favor", "hola", "gracias", "hoy", "noche", "ahora", "cosa", "cosas",
    # English filler (the catalog is in Spanish, questions may not be)
    "the", "and", "for", "with", "about", "what", "which", "who", "have", "has",
    "any", "some", "something", "anything", "recommend", "suggest", "want",
    "would", "like", "can", "could", "you", "your", "give", "show", "find",
    "good", "best", "please", "watch", "read", "listen", "play", "kind", "from",
    "content", "according", "does", "did", "there", "rating", "ratings",
    "movie", "movies", "book", "books", "shows", "song", "songs", "music", "game",
    "games", "videogame", "videogames", "documentary", "documentaries",
    # category nouns (handled by category detection, too common in documents)
    "pelicula", "peliculas", "peli", "pelis", "film", "libro", "libros", "serie",
    "series", "musica", "videojuego", "videojuegos", "juego", "juegos", "documental",
    "documentales",
}

# Category trigger words (after normalization). "juego" in singular is left out
# on purpose so that "Juego de tronos" is not taken as a video game.
CATEGORY_KEYWORDS = {
    "peliculas": {"pelicula", "peliculas", "peli", "pelis", "film", "films", "cine",
                  "movie", "movies"},
    "libros": {"libro", "libros", "leer", "lectura", "novela", "novelas", "ensayo",
               "ensayos", "book", "books", "novel", "novels", "read", "reading"},
    "series": {"serie", "series", "temporada", "temporadas", "episodio",
               "episodios", "capitulo", "capitulos", "tv", "season", "seasons",
               "episode", "episodes"},
    "musica": {"musica", "cancion", "canciones", "album", "albumes", "disco",
               "discos", "escuchar", "grupo", "grupos", "cantante", "banda",
               "rap", "rock", "pop", "jazz", "reggae", "blues", "funk", "soul",
               "rumba", "ska", "punk", "hiphop", "chillout", "music", "song",
               "songs", "listen", "band", "singer"},
    "videojuegos": {"videojuego", "videojuegos", "juegos", "jugar", "consola",
                    "nintendo", "playstation", "snes", "gameboy", "switch",
                    "videogame", "videogames", "games", "rpg"},
    "documentales": {"documental", "documentales", "documentary", "documentaries"},
}


def normalize(text: str) -> str:
    """Lowercase and strip accents (ñ is kept as n)."""
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.lower()


def tokenize(text: str) -> List[str]:
    """Normalized alphanumeric tokens."""
    return re.findall(r"[a-z0-9]+", normalize(text))


def title_key(text: str) -> str:
    """Space-padded normalized tokens, used to find a title inside a question."""
    return f" {' '.join(tokenize(text))} "


def query_terms(text: str) -> List[str]:
    """Meaningful search terms of a user query (no stopwords, no duplicates)."""
    seen = []
    for tok in tokenize(text):
        if tok in STOPWORDS or (len(tok) < 3 and not tok.isdigit()):
            continue
        if tok not in seen:
            seen.append(tok)
    return seen


def detect_category(text: str) -> Optional[str]:
    """Return the category explicitly mentioned in the question, if exactly one is."""
    tokens = set(tokenize(text))
    if "game" in tokens and "boy" in tokens:
        tokens.add("gameboy")
    matches = [cat for cat, words in CATEGORY_KEYWORDS.items() if tokens & words]
    return matches[0] if len(matches) == 1 else None
