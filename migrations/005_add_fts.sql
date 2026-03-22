-- Full-text search: tsvector column on content (chunks) table
ALTER TABLE content ADD COLUMN tsv tsvector;

CREATE INDEX idx_content_tsv ON content USING GIN(tsv);

CREATE OR REPLACE FUNCTION content_tsv_trigger() RETURNS trigger AS $$
BEGIN
    NEW.tsv := to_tsvector('english', COALESCE(NEW.chunk_text, ''));
    RETURN NEW;
END
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_content_tsv
    BEFORE INSERT OR UPDATE OF chunk_text ON content
    FOR EACH ROW EXECUTE FUNCTION content_tsv_trigger();

-- Backfill existing rows
UPDATE content SET tsv = to_tsvector('english', COALESCE(chunk_text, ''));
