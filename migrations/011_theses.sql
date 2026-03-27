CREATE TABLE theses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    owner TEXT,
    status TEXT NOT NULL DEFAULT 'draft',
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE thesis_claims (
    thesis_id UUID REFERENCES theses(id) ON DELETE CASCADE,
    triple_hash TEXT NOT NULL,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'supporting',
    added_at TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (thesis_id, triple_hash)
);

CREATE INDEX idx_thesis_claims_hash ON thesis_claims(triple_hash);
CREATE INDEX idx_theses_status ON theses(status);

CREATE TRIGGER update_theses_updated_at
    BEFORE UPDATE ON theses
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();
