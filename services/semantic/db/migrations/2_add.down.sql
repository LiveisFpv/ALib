ALTER TABLE chat DROP COLUMN updated_at;
ALTER TABLE users ADD COLUMN email TEXT;
UPDATE users SET email = CONCAT('user_', id, '@local') WHERE email IS NULL;
ALTER TABLE users ALTER COLUMN email SET NOT NULL;
ALTER TABLE users ADD CONSTRAINT users_email_key UNIQUE (email);
