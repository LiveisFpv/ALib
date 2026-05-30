BEGIN;

CREATE TABLE user_orcid_links (
  user_id       INT PRIMARY KEY,
  orcid         TEXT NOT NULL UNIQUE,
  confirmed_at  TIMESTAMPTZ NOT NULL,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  CONSTRAINT chk_user_orcid_links_format
    CHECK (orcid ~ '^[0-9]{4}-[0-9]{4}-[0-9]{4}-[0-9]{3}[0-9X]$'),
  CONSTRAINT fk_user_orcid_links_user
    FOREIGN KEY (user_id) REFERENCES users(id)
      ON DELETE CASCADE ON UPDATE CASCADE
);

COMMIT;
