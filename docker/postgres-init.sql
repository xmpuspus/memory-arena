-- Enable pgvector extension on first DB initialization.
-- Mounted into /docker-entrypoint-initdb.d/ on the pgvector container.
CREATE EXTENSION IF NOT EXISTS vector;
