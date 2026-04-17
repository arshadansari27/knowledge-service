-- Drop tables whose feature code was removed in earlier refactors.
--   communities:   community-detection feature (never re-ingested after removal)
--   thesis_claims, theses: ThesisStore (deleted; README/CLAUDE scrubbed in PR #66)
DROP TABLE IF EXISTS communities;
DROP TABLE IF EXISTS thesis_claims;
DROP TABLE IF EXISTS theses;
