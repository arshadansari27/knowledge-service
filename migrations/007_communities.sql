CREATE TABLE communities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    level INTEGER NOT NULL,
    label TEXT,
    summary TEXT,
    member_entities TEXT[] NOT NULL,
    member_count INTEGER NOT NULL,
    built_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_communities_level ON communities(level);
CREATE INDEX idx_communities_built_at ON communities(built_at);
