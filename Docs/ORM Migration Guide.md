# Quick Reference: ORM Usage Examples

## Before (Raw SQL) vs After (ORM)

### Initializing the Database

**Before:**
```python
self.db_conn = sqlite3.connect(self.db_file)
cursor = self.db_conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS memories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT NOT NULL,
        created_at TIMESTAMP NOT NULL,
        type TEXT NOT NULL
    )
""")
self.db_conn.commit()
```

**After:**
```python
self.repository = SyncMemoryRepository(self.db_file)
self.repository.initialize()
```

---

### Adding a Memory

**Before:**
```python
cursor = self.db_conn.cursor()
cursor.execute("""
    INSERT INTO memories (text, created_at, type)
    VALUES (?, ?, ?)
""", (text, created_at, memory_type))
self.db_conn.commit()
memory_id = cursor.lastrowid
```

**After:**
```python
memory = self.repository.add_memory(text, memory_type, created_at)
memory_id = memory.id
```

---

### Retrieving a Memory

**Before:**
```python
cursor.execute("""
    SELECT id, text, created_at, type
    FROM memories
    WHERE id = ?
""", (db_id,))
row = cursor.fetchone()
if row:
    memory = {
        "id": row["id"],
        "text": row["text"],
        "created_at": row["created_at"],
        "type": row["type"],
    }
```

**After:**
```python
memory_obj = self.repository.get_memory_by_id(db_id)
if memory_obj:
    memory = memory_obj.to_dict()
```

---

### Counting Memories

**Before:**
```python
cursor.execute("SELECT COUNT(*) FROM memories")
db_count = cursor.fetchone()[0]
```

**After:**
```python
db_count = self.repository.count_memories()
```

---

### Deleting All Memories

**Before:**
```python
cursor = self.db_conn.cursor()
cursor.execute("DELETE FROM memories")
self.db_conn.commit()
```

**After:**
```python
self.repository.delete_all_memories()
```

---

## Key Improvements

✅ **Less code** - Reduced boilerplate by ~60%  
✅ **Type safety** - IDE autocomplete and type checking  
✅ **SQL injection safe** - Automatic parameter handling  
✅ **Maintainable** - No raw SQL strings to manage  
✅ **Testable** - Easy to mock repository in tests  
✅ **Future-ready** - Async support built-in for scalability
