BEGIN;

-- ==============================================================
-- Core entities
-- ==============================================================

-- Users
CREATE TABLE users (
  id            SERIAL PRIMARY KEY,
  email         TEXT   NOT NULL UNIQUE
);

-- Authors
CREATE TABLE authors (
  author_id    SERIAL PRIMARY KEY,
  first_name   TEXT NOT NULL,
  last_name    TEXT NOT NULL,
  middle_name  TEXT NULL,
  orcid        TEXT NULL UNIQUE
);

-- Papers
CREATE TABLE papers (
  paper_id            SERIAL PRIMARY KEY,
  created_by_user_id  INT NULL,
  title               TEXT NULL,
  abstract            TEXT NULL,
  year                INT  NULL CHECK (year BETWEEN 0 AND 9999),
  state               TEXT NULL,
  created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  CONSTRAINT fk_papers_user
    FOREIGN KEY (created_by_user_id) REFERENCES users(id)
    ON DELETE SET NULL ON UPDATE CASCADE
);

-- Institutions
CREATE TABLE institutions (
  institution_id  SERIAL PRIMARY KEY,
  name            TEXT NOT NULL,
  country         TEXT NULL,
  ror_id          TEXT NULL UNIQUE,
  grid_id         TEXT NULL UNIQUE
);

-- Paper ↔ Institutions (many-to-many)
CREATE TABLE paper_institutions (
  institution_id INT NOT NULL,
  paper_id       INT NOT NULL,
  PRIMARY KEY (institution_id, paper_id),
  FOREIGN KEY (institution_id) REFERENCES institutions(institution_id)
    ON DELETE CASCADE ON UPDATE CASCADE,
  FOREIGN KEY (paper_id) REFERENCES papers(paper_id)
    ON DELETE CASCADE ON UPDATE CASCADE
);

-- Identifier types (e.g., DOI, PMID, arXiv)
CREATE TABLE identifier_types (
  identifier_type_id SERIAL PRIMARY KEY,
  name               TEXT NOT NULL UNIQUE
);

-- Paper identifiers (type + value)
CREATE TABLE paper_identifiers (
  paper_id           INT  NOT NULL,
  identifier_type_id INT  NOT NULL,
  identifier         TEXT NOT NULL,
  PRIMARY KEY (paper_id, identifier_type_id, identifier),
  FOREIGN KEY (paper_id) REFERENCES papers(paper_id)
    ON DELETE CASCADE ON UPDATE CASCADE,
  FOREIGN KEY (identifier_type_id) REFERENCES identifier_types(identifier_type_id)
    ON DELETE RESTRICT ON UPDATE CASCADE
);

-- Paper ↔ Authors (many-to-many)
CREATE TABLE paper_authors (
  paper_id     INT NOT NULL,
  author_id    INT NOT NULL,
  author_order INT NULL,
  PRIMARY KEY (paper_id, author_id),
  FOREIGN KEY (paper_id)  REFERENCES papers(paper_id)
    ON DELETE CASCADE ON UPDATE CASCADE,
  FOREIGN KEY (author_id) REFERENCES authors(author_id)
    ON DELETE CASCADE ON UPDATE CASCADE
);

-- Paper ↔ Paper relations (e.g., related, references)
CREATE TABLE paper_relations (
  src_paper_id INT NOT NULL,
  dst_paper_id INT NOT NULL,
  PRIMARY KEY (src_paper_id, dst_paper_id),
  CONSTRAINT chk_paper_relations_no_self CHECK (src_paper_id <> dst_paper_id),
  FOREIGN KEY (src_paper_id) REFERENCES papers(paper_id)
    ON DELETE CASCADE ON UPDATE CASCADE,
  FOREIGN KEY (dst_paper_id) REFERENCES papers(paper_id)
    ON DELETE CASCADE ON UPDATE CASCADE
);

-- Locations of the paper (links)
CREATE TABLE locations (
  location_id  SERIAL PRIMARY KEY,
  paper_id     INT  NOT NULL,
  url          TEXT NULL,
  link_type    TEXT NULL,
  version      TEXT NOT NULL,
  FOREIGN KEY (paper_id) REFERENCES papers(paper_id)
    ON DELETE CASCADE ON UPDATE CASCADE
);

-- Chats and messages (search history)
CREATE TABLE chat (
  chat_id     SERIAL PRIMARY KEY,
  user_id     INT NOT NULL,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  title       TEXT NOT NULL,
  FOREIGN KEY (user_id) REFERENCES users(id)
    ON DELETE CASCADE ON UPDATE CASCADE
);

CREATE TABLE chat_message (
  chat_history_id  SERIAL PRIMARY KEY,
  chat_id          INT NOT NULL,
  search_query     TEXT NOT NULL,
  created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  FOREIGN KEY (chat_id) REFERENCES chat(chat_id)
    ON DELETE CASCADE ON UPDATE CASCADE
);

-- Search results for a message/query
CREATE TABLE search_results (
  chat_history_id  INT NOT NULL,
  paper_id         INT NOT NULL,
  score            DOUBLE PRECISION NULL,
  rank             INT NULL,
  created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (chat_history_id, paper_id),
  FOREIGN KEY (chat_history_id) REFERENCES chat_message(chat_history_id)
    ON DELETE CASCADE ON UPDATE CASCADE,
  FOREIGN KEY (paper_id)        REFERENCES papers(paper_id)
    ON DELETE CASCADE ON UPDATE CASCADE
);

-- ==============================================================
-- Indexes for performance (FKs and common queries)
-- ==============================================================

CREATE INDEX idx_chat_user_id                 ON chat(user_id);
CREATE INDEX idx_chat_message_chat_id         ON chat_message(chat_id);
CREATE INDEX idx_search_results_hist_id       ON search_results(chat_history_id);
CREATE INDEX idx_search_results_paper_id      ON search_results(paper_id);
CREATE INDEX idx_locations_paper_id           ON locations(paper_id);
CREATE INDEX idx_papers_year                  ON papers(year);
CREATE INDEX idx_paper_authors_paper_id       ON paper_authors(paper_id);
CREATE INDEX idx_paper_authors_author_id      ON paper_authors(author_id);
CREATE INDEX idx_paper_institutions_paper_id  ON paper_institutions(paper_id);
CREATE INDEX idx_paper_institutions_inst_id   ON paper_institutions(institution_id);
CREATE INDEX idx_paper_identifiers_paper_id   ON paper_identifiers(paper_id);
CREATE INDEX idx_paper_identifiers_type_id    ON paper_identifiers(identifier_type_id);
CREATE INDEX idx_paper_relations_src          ON paper_relations(src_paper_id);
CREATE INDEX idx_paper_relations_dst          ON paper_relations(dst_paper_id);

COMMIT;

