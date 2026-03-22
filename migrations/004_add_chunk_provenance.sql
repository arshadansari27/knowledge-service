ALTER TABLE provenance ADD COLUMN chunk_id UUID REFERENCES content(id) ON DELETE SET NULL;
CREATE INDEX idx_provenance_chunk_id ON provenance(chunk_id);
