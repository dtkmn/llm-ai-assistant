import pytest

from src.thread_store import DEFAULT_THREAD_TITLE, ThreadStore


def test_thread_store_persists_threads_messages_and_latest_payload(tmp_path):
    db_path = tmp_path / "threads.sqlite3"
    store = ThreadStore(db_path)
    thread = store.create_thread(title="Phoenix", thread_id="thread_phoenix")
    store.append_message("thread_phoenix", role="user", content="What happened?")
    store.append_turn(
        "thread_phoenix",
        user_content="What is next?",
        assistant_content="Project Phoenix launched.",
        thinking={"available": True, "content": "I checked the loop evidence."},
        loop_payload={"summary": {"final_decision": "not_verified"}},
    )
    store.close()

    restored = ThreadStore(db_path)
    restored_thread = restored.get_thread(thread.id)

    assert restored_thread is not None
    assert restored_thread.title == "Phoenix"
    assert restored_thread.message_count == 3
    assert [message.role for message in restored_thread.messages] == [
        "user",
        "user",
        "assistant",
    ]
    assert restored_thread.messages[2].thinking == {
        "available": True,
        "content": "I checked the loop evidence.",
    }
    assert restored_thread.latest == {
        "summary": {"final_decision": "not_verified"}
    }


def test_thread_store_ensures_and_clears_thread():
    store = ThreadStore.in_memory()

    ensured = store.ensure_thread("thread_local")
    store.append_message("thread_local", role="user", content="hello")
    cleared = store.clear_thread("thread_local")

    assert ensured.title == DEFAULT_THREAD_TITLE
    assert cleared.id == "thread_local"
    assert cleared.message_count == 0
    assert cleared.messages == ()


def test_thread_store_returns_recent_messages_in_original_order():
    store = ThreadStore.in_memory()
    store.create_thread(thread_id="thread_local")
    for index in range(5):
        role = "user" if index % 2 == 0 else "assistant"
        store.append_message(
            "thread_local",
            role=role,
            content=f"message {index}",
        )

    messages = store.recent_messages("thread_local", limit=3)

    assert [(message.role, message.content) for message in messages] == [
        ("user", "message 2"),
        ("assistant", "message 3"),
        ("user", "message 4"),
    ]


def test_thread_store_retrieves_semantic_memories_by_similarity():
    store = ThreadStore.in_memory()
    store.create_thread(thread_id="thread_local")
    relevant = store.append_message(
        "thread_local",
        role="user",
        content="Dynamic programming stores answers to subproblems.",
    )
    irrelevant = store.append_message(
        "thread_local",
        role="user",
        content="Banana bread needs ripe bananas.",
    )
    store.upsert_message_embedding(
        relevant,
        embedding_model="fake-memory",
        vector=[1.0, 0.0],
    )
    store.upsert_message_embedding(
        irrelevant,
        embedding_model="fake-memory",
        vector=[0.0, 1.0],
    )

    memories = store.semantic_memories(
        "thread_local",
        embedding_model="fake-memory",
        query_vector=[1.0, 0.0],
    )

    assert [memory.message_id for memory in memories] == [relevant.id]
    assert memories[0].content == "Dynamic programming stores answers to subproblems."
    assert memories[0].score == 1.0


def test_thread_store_semantic_memories_can_exclude_recent_messages():
    store = ThreadStore.in_memory()
    store.create_thread(thread_id="thread_local")
    old_message = store.append_message(
        "thread_local",
        role="user",
        content="Old relevant memory.",
    )
    recent_message = store.append_message(
        "thread_local",
        role="assistant",
        content="Recent duplicate memory.",
    )
    store.upsert_message_embedding(
        old_message,
        embedding_model="fake-memory",
        vector=[1.0, 0.0],
    )
    store.upsert_message_embedding(
        recent_message,
        embedding_model="fake-memory",
        vector=[1.0, 0.0],
    )

    memories = store.semantic_memories(
        "thread_local",
        embedding_model="fake-memory",
        query_vector=[1.0, 0.0],
        exclude_message_ids=(recent_message.id,),
    )

    assert [memory.message_id for memory in memories] == [old_message.id]


def test_thread_store_clear_removes_semantic_memories():
    store = ThreadStore.in_memory()
    store.create_thread(thread_id="thread_local")
    message = store.append_message(
        "thread_local",
        role="user",
        content="Remember this.",
    )
    store.upsert_message_embedding(
        message,
        embedding_model="fake-memory",
        vector=[1.0],
    )

    store.clear_thread("thread_local")

    assert not store.has_message_embeddings("thread_local", "fake-memory")
    assert store.semantic_memories(
        "thread_local",
        embedding_model="fake-memory",
        query_vector=[1.0],
    ) == ()


def test_thread_store_clear_missing_thread_does_not_recreate():
    store = ThreadStore.in_memory()
    store.create_thread(thread_id="thread_local")
    assert store.delete_thread("thread_local") is True

    cleared = store.clear_thread("thread_local")

    assert cleared is None
    assert store.get_thread("thread_local") is None


def test_thread_store_rejects_invalid_message_role():
    store = ThreadStore.in_memory()
    store.create_thread(thread_id="thread_local")

    with pytest.raises(ValueError, match="role"):
        store.append_message("thread_local", role="system", content="hidden")


def test_thread_store_append_turn_serialization_failure_leaves_no_partial_message():
    store = ThreadStore.in_memory()
    store.create_thread(thread_id="thread_local")

    with pytest.raises(TypeError):
        store.append_turn(
            "thread_local",
            user_content="hello",
            assistant_content="broken",
            thinking={"bad": object()},
        )

    assert store.get_thread("thread_local").messages == ()
    store.append_message("thread_local", role="user", content="later")
    messages = store.get_thread("thread_local").messages
    assert [(message.role, message.content) for message in messages] == [
        ("user", "later")
    ]


def test_thread_store_append_turn_skips_when_generation_changed():
    store = ThreadStore.in_memory()
    thread = store.create_thread(thread_id="thread_local")

    store.clear_thread("thread_local")
    result = store.append_turn(
        "thread_local",
        user_content="stale",
        assistant_content="stale answer",
        expected_generation=thread.generation,
        expected_instance_id=thread.instance_id,
    )

    assert result is None
    assert store.get_thread("thread_local").messages == ()


def test_thread_store_append_turn_skips_recreated_thread_with_same_id():
    store = ThreadStore.in_memory()
    old_thread = store.create_thread(thread_id="thread_aba")
    store.delete_thread("thread_aba")
    new_thread = store.create_thread(thread_id="thread_aba")

    result = store.append_turn(
        "thread_aba",
        user_content="stale",
        assistant_content="stale answer",
        expected_generation=old_thread.generation,
        expected_instance_id=old_thread.instance_id,
    )

    assert old_thread.instance_id != new_thread.instance_id
    assert result is None
    assert store.get_thread("thread_aba").messages == ()


def test_thread_store_append_turn_requires_complete_guard():
    store = ThreadStore.in_memory()
    thread = store.create_thread(thread_id="thread_guard")

    with pytest.raises(ValueError, match="must be supplied together"):
        store.append_turn(
            "thread_guard",
            user_content="stale",
            assistant_content="stale answer",
            expected_generation=thread.generation,
        )
