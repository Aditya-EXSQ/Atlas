"""
Quick test script to verify ORM implementation works correctly.
Tests basic database operations with the new SQLAlchemy-based repository.
"""

import numpy as np

from memory.persistent_store import PersistentRAGStore


def test_orm_implementation():
    """Test the ORM-based persistent store."""
    print("=" * 60)
    print("Testing ORM-based PersistentRAGStore")
    print("=" * 60)

    # Initialize store
    print("\n1. Initializing store...")
    store = PersistentRAGStore(
        embedding_dim=384,
        index_path="./test_faiss_index",
        index_file="test_index.faiss",
        db_file="test_db.db",
    )

    # Clear any existing data
    print("\n2. Clearing existing data...")
    store.clear()

    # Test adding memories
    print("\n3. Testing add operation...")
    embedding1 = np.random.randn(384)
    embedding2 = np.random.randn(384)

    id1 = store.add(embedding1, "Hello, this is a test memory", "user")
    print(f"   Added memory with ID: {id1}")

    id2 = store.add(embedding2, "This is another test memory", "assistant")
    print(f"   Added memory with ID: {id2}")

    # Test store size
    print("\n4. Testing store size...")
    size = store.get_store_size()
    print(f"   Store size: {size}")
    assert size == 2, f"Expected 2 memories, got {size}"

    # Test statistics
    print("\n5. Testing statistics...")
    stats = store.get_stats()
    print(f"   Stats: {stats}")
    assert stats["vector_count"] == 2, "Vector count mismatch"
    assert stats["db_count"] == 2, "DB count mismatch"

    # Test search
    print("\n6. Testing search...")
    query_embedding = np.random.randn(384)
    results = store.search(query_embedding, top_k=2)
    print(f"   Found {len(results)} results")

    for i, (memory, score) in enumerate(results):
        print(f"   Result {i + 1}:")
        print(f"      ID: {memory['id']}")
        print(f"      Text: {memory['text'][:50]}...")
        print(f"      Type: {memory['type']}")
        print(f"      Score: {score:.4f}")

    assert len(results) == 2, f"Expected 2 results, got {len(results)}"

    # Test save
    print("\n7. Testing save...")
    store.save()

    # Test clear
    print("\n8. Testing clear...")
    store.clear()
    size_after_clear = store.get_store_size()
    print(f"   Store size after clear: {size_after_clear}")
    assert size_after_clear == 0, "Store should be empty after clear"

    # Cleanup
    print("\n9. Cleaning up...")
    store.close()

    print("\n" + "=" * 60)
    print("✓ All tests passed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        test_orm_implementation()
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
