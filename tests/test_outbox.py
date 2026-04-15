# tests/test_outbox.py
from unittest.mock import AsyncMock
from knowledge_service.ingestion.outbox import OutboxStore


class TestOutboxStoreStage:
    async def test_stage_returns_inserted_id(self):
        conn = AsyncMock()
        conn.fetchval.return_value = 42
        store = OutboxStore()

        returned_id = await store.stage(
            conn,
            operation="insert",
            triple_hash="abc",
            subject="s",
            predicate="p",
            object_="o",
            confidence=0.8,
            knowledge_type="claim",
            graph="http://ks/graph/extracted",
        )

        assert returned_id == 42
        # One call into PG
        assert conn.fetchval.await_count == 1
        args, _ = conn.fetchval.call_args
        # First positional is the SQL string
        assert "INSERT INTO triple_outbox" in args[0]
        # Params follow — check a couple of them
        assert args[1] == "abc"  # triple_hash
        assert args[2] == "insert"  # operation

    async def test_stage_serialises_payload(self):
        conn = AsyncMock()
        conn.fetchval.return_value = 1
        store = OutboxStore()

        await store.stage(
            conn,
            operation="insert_inferred",
            triple_hash="xyz",
            subject="s",
            predicate="p",
            object_="o",
            graph="http://ks/graph/inferred",
            payload={"derived_from": ["h1"], "inference_method": "inverse"},
        )

        args, _ = conn.fetchval.call_args
        # payload is last positional; expect JSON-serialised string
        payload_param = args[-1]
        assert '"derived_from"' in payload_param
        assert '"inverse"' in payload_param
