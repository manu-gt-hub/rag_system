"""
test_unit.py — Fast tests with no API keys or database required.

Runnable as:
    pytest tests/test_unit.py -v
"""

from __future__ import annotations

import src.web_app as web_app
from src.indexer import build_documents, load_indexes, _clean_music_name
from src.rag_pipeline import build_retrieval_query
from src.retriever import build_tsquery
from src.text_utils import detect_category, normalize, query_terms


# ---------------------------------------------------------------------------
# text_utils
# ---------------------------------------------------------------------------

class TestTextUtils:

    def test_normalize_strips_accents(self):
        assert normalize("Películas de Acción ÑÚ") == "peliculas de accion nu"

    def test_query_terms_drop_filler(self):
        assert query_terms("¿Qué contenido tengo de Danny Boyle?") == ["danny", "boyle"]

    def test_query_terms_keep_numbers(self):
        assert "300" in query_terms("¿Quién dirigió 300?")

    def test_detect_single_category(self):
        assert detect_category("¿Qué películas de terror tengo?") == "peliculas"
        assert detect_category("Recomiéndame un libro") == "libros"
        assert detect_category("algo para jugar en la Game Boy") == "videojuegos"

    def test_detect_category_english(self):
        assert detect_category("Recommend me a horror movie") == "peliculas"
        assert detect_category("Any TV show like Breaking Bad?") == "series"
        assert detect_category("A documentary about politics") == "documentales"

    def test_query_terms_english(self):
        assert query_terms("What do I have by Danny Boyle?") == ["danny", "boyle"]

    def test_detect_category_ambiguous_or_missing(self):
        assert detect_category("¿Qué contenido tengo de Danny Boyle?") is None
        assert detect_category("Una serie basada en un libro") is None

    def test_music_genre_detects_music(self):
        assert detect_category("rap en español") == "musica"

    def test_juego_de_tronos_is_not_a_videogame(self):
        assert detect_category("¿Tengo Juego de tronos?") is None

    def test_title_key_finds_title_in_question(self):
        from src.text_utils import title_key
        question = title_key("What rating does the movie Algunos hombres buenos have?")
        assert title_key("Algunos hombres buenos") in question
        assert title_key("Alien") not in title_key("Aliens and robots")

    def test_tsquery_is_or_of_safe_terms(self):
        assert build_tsquery("¿Qué hay de Nicolas Cage & robos?") == "nicolas | cage | robos"
        assert build_tsquery("¿Qué tengo?") == ""
        assert build_tsquery("Danny Boyle", "&") == "danny & boyle"


# ---------------------------------------------------------------------------
# Document construction
# ---------------------------------------------------------------------------

class TestDocuments:

    def test_movie_document_contains_metadata(self):
        docs = build_documents("peliculas", [{
            "titulo": "300", "titulo_oficial": "300", "año": 2006,
            "genero": "Acción", "director": "Zack Snyder",
            "actores": ["Gerard Butler"], "sinopsis": "Leónidas y Esparta.",
            "rating": 7.208, "rating_votos": 100,
        }])
        assert len(docs) == 1
        text = docs[0]["text"]
        assert text.startswith("[MOVIE] 300")
        assert "Director: Zack Snyder" in text
        assert "Gerard Butler" in text
        assert "7.21/10 (100 votes)" in text
        assert "Genre: Acción / action" in text

    def test_movies_are_deduplicated_keeping_richest(self):
        docs = build_documents("peliculas", [
            {"titulo": "Conan"},
            {"titulo": "Conan", "titulo_oficial": "Conan", "director": "X"},
        ])
        assert len(docs) == 1
        assert "Director: X" in docs[0]["text"]

    def test_series_grouped_by_show(self):
        episodes = [
            {"titulo": f"Ep {n}", "serie": "Perdidos", "temporada": 1,
             "titulo_serie_oficial": "Perdidos", "genero": "Misterio"}
            for n in range(5)
        ]
        docs = build_documents("series", episodes)
        assert len(docs) == 1
        assert docs[0]["title"] == "Perdidos"
        assert "Episodes in catalog: 5" in docs[0]["text"]

    def test_music_grouped_by_folder(self):
        tracks = [
            {"titulo": "01 Song A", "archivo": "Musica/_GENERO ESPAÑOL/Estopa - Discografia/Disco 1/01 a.mp3"},
            {"titulo": "02 Song B", "archivo": "Musica/_GENERO ESPAÑOL/Estopa - Discografia/Disco 2/02 b.mp3"},
            {"titulo": "03 Song C", "archivo": "Musica/ACDC/03 c.mp3"},
        ]
        docs = build_documents("musica", tracks)
        assert [d["title"] for d in docs] == ["GENERO ESPAÑOL / Estopa - Discografia", "ACDC"]
        assert "Track count: 2" in docs[0]["text"]

    def test_music_name_cleanup(self):
        assert _clean_music_name("VA - 100 Greatest Hip Hop (2019) Mp3 320kbps [PMEDIA]") \
            == "VA - 100 Greatest Hip Hop (2019)"

    def test_games_deduplicated_per_platform(self):
        docs = build_documents("videojuegos", [
            {"titulo": "Tetris", "plataforma": "Game Boy"},
            {"titulo": "Tetris", "plataforma": "Game Boy", "genero": "Puzzle"},
            {"titulo": "Tetris", "plataforma": "SNES"},
        ])
        assert len(docs) == 2

    def test_game_genres_translated(self):
        docs = build_documents("videojuegos", [
            {"titulo": "Pokemon", "plataforma": "Game Boy", "genero": "Role-playing (RPG)"},
        ])
        assert "Genre: Role-playing (RPG) / rol" in docs[0]["text"]

    def test_real_indexes_load(self):
        docs = load_indexes()
        by_cat = {}
        for d in docs:
            by_cat[d["category"]] = by_cat.get(d["category"], 0) + 1
        assert set(by_cat) == {"peliculas", "libros", "series", "musica", "videojuegos", "documentales"}
        assert by_cat["series"] < 50       # grouped by show, not 1800 episodes
        assert by_cat["musica"] < 1000     # grouped by folder, not 12900 tracks
        assert all(d["text"].strip() and d["title"] for d in docs)


# ---------------------------------------------------------------------------
# Conversation handling
# ---------------------------------------------------------------------------

class TestRetrievalQuery:

    def test_standalone_question_untouched(self):
        history = [{"role": "user", "content": "películas de terror"}]
        q = "Recomiéndame un libro de ciencia ficción con naves espaciales"
        assert build_retrieval_query(q, history) == q

    def test_short_follow_up_uses_previous_question(self):
        history = [
            {"role": "user", "content": "películas de terror"},
            {"role": "assistant", "content": "Te recomiendo..."},
        ]
        assert build_retrieval_query("¿y alguna más reciente?", history) \
            == "películas de terror ¿y alguna más reciente?"


# ---------------------------------------------------------------------------
# Web API (pipeline replaced by a fake: no keys, no database)
# ---------------------------------------------------------------------------

class _FakePipeline:
    def __init__(self):
        self.calls = []

    def query(self, question, top_k=None, category_filter=None, history=None):
        from src.rag_pipeline import RAGResponse
        self.calls.append((question, category_filter, history))
        return RAGResponse(query=question, answer="I recommend **Alien**.",
                           retrieved_docs=[], elapsed_seconds=0.1, provider="fake")


class TestWebApi:

    def setup_method(self):
        self.fake = _FakePipeline()
        web_app.pipeline = self.fake
        self.client = web_app.app.test_client()

    def teardown_method(self):
        web_app.pipeline = None

    def test_chat_passes_history_and_category(self):
        history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hello!"}]
        resp = self.client.post("/api/chat", json={
            "message": "a horror one", "history": history, "category": "peliculas"})
        assert resp.status_code == 200
        assert resp.get_json()["answer"] == "I recommend **Alien**."
        assert self.fake.calls == [("a horror one", "peliculas", history)]

    def test_query_requires_question(self):
        assert self.client.post("/api/query", json={}).status_code == 400

    def test_unknown_category_rejected(self):
        resp = self.client.post("/api/query", json={"question": "x", "category": "comics"})
        assert resp.status_code == 400

    def test_index_page_renders(self):
        resp = self.client.get("/")
        assert resp.status_code == 200
        assert "Your media catalog" in resp.get_data(as_text=True)
