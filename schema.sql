-- D1 schema: document store + full-text index (the vectors live in Vectorize).
-- The indexer prepends it to every load, so a load fully replaces the catalog.

DROP TABLE IF EXISTS docs_fts;
DROP TABLE IF EXISTS docs;
DROP TABLE IF EXISTS index_meta;

CREATE TABLE docs (
    id         INTEGER PRIMARY KEY,   -- same id as the Vectorize vector
    category   TEXT NOT NULL,
    title      TEXT NOT NULL,
    content    TEXT NOT NULL,         -- text given to the LLM
    metadata   TEXT,                  -- JSON: year, genre, rating, poster_url...
    title_key  TEXT NOT NULL          -- normalized title, matched inside questions
);
CREATE INDEX idx_docs_category ON docs (category);

-- Contentless FTS5 index over normalized (accent-free) text. Columns are
-- weighted at query time: title > people/genre keywords > content.
CREATE VIRTUAL TABLE docs_fts USING fts5 (
    title, keywords, content,
    content = '',
    tokenize = 'unicode61 remove_diacritics 2'
);

-- Which catalog/model/document format is loaded; the indexer compares it with
-- the repository to decide whether a deploy needs to re-index.
CREATE TABLE index_meta (
    signature  TEXT NOT NULL,
    documents  INTEGER NOT NULL,
    indexed_at TEXT NOT NULL DEFAULT (datetime('now'))
);
