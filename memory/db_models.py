"""
SQLAlchemy ORM models and repository for persistent RAG storage.
Provides async database operations with a clean repository pattern.
"""

from datetime import datetime
from typing import Dict, List, Optional

from sqlalchemy import DateTime, Integer, String, Text, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    pass


class Session(Base):
    """
    Session model representing a chat session.

    Attributes:
        id: Primary key auto-incremented ID
        name: Name of the session
        created_at: Timestamp when the session was created
    """

    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at,
        }

    def __repr__(self) -> str:
        return f"<Session(id={self.id}, name={self.name})>"


class Memory(Base):
    """
    Memory model representing stored conversation memories.

    Attributes:
        id: Primary key auto-incremented ID
        text: The actual text content of the memory
        created_at: Timestamp when the memory was created
        type: Type of memory ('user' or 'assistant')
        session_id: Foreign key linking to a session
    """

    __tablename__ = "memories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    type: Mapped[str] = mapped_column(String, nullable=False)
    session_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    def to_dict(self) -> Dict:
        """
        Convert Memory instance to dictionary format.

        Returns:
            Dictionary with memory fields
        """
        return {
            "id": self.id,
            "text": self.text,
            "content": self.text,  # Alias for compatibility
            "created_at": self.created_at,
            "type": self.type,
            "session_id": self.session_id,
        }

    def __repr__(self) -> str:
        return f"<Memory(id={self.id}, type={self.type}, session_id={self.session_id})>"


class MemoryRepository:
    """
    Repository pattern for Memory database operations.
    Provides clean interface for CRUD operations with async support.
    """

    def __init__(self, db_path: str):
        """
        Initialize the repository with database connection.

        Args:
            db_path: Path to the SQLite database file
        """
        # Use aiosqlite for async SQLite support
        self.db_url = f"sqlite+aiosqlite:///{db_path}"
        self.engine = create_async_engine(self.db_url, echo=False)
        self.async_session = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

    async def initialize(self):
        """
        Initialize database schema.
        Creates tables if they don't exist.
        """
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def create_session(self, name: str) -> Session:
        """
        Create a new chat session.

        Args:
            name: Name of the session

        Returns:
            Created Session object
        """
        session = Session(name=name, created_at=datetime.now())
        async with self.async_session() as session_db:
            session_db.add(session)
            await session_db.commit()
            await session_db.refresh(session)
            return session

    async def get_all_sessions(self) -> List[Session]:
        """
        Get all chat sessions.

        Returns:
            List of all Session objects
        """
        async with self.async_session() as session:
            result = await session.execute(
                select(Session).order_by(Session.created_at.desc())
            )
            return list(result.scalars().all())

    async def add_memory(
        self,
        text: str,
        memory_type: str = "user",
        created_at: Optional[datetime] = None,
        session_id: Optional[int] = None,
    ) -> Memory:
        """
        Add a new memory to the database.

        Args:
            text: The text content of the memory
            memory_type: Type of memory ('user' or 'assistant')
            created_at: Timestamp (defaults to now)
            session_id: ID of the session this memory belongs to

        Returns:
            The created Memory instance
        """
        if created_at is None:
            created_at = datetime.now()

        memory = Memory(
            text=text, type=memory_type, created_at=created_at, session_id=session_id
        )

        async with self.async_session() as session:
            session.add(memory)
            await session.commit()
            await session.refresh(memory)

        return memory

    async def get_memory_by_id(self, memory_id: int) -> Optional[Memory]:
        """
        Retrieve a memory by its ID.

        Args:
            memory_id: The ID of the memory to retrieve

        Returns:
            Memory instance if found, None otherwise
        """
        async with self.async_session() as session:
            result = await session.get(Memory, memory_id)
            return result

    async def get_all_memories(self, session_id: Optional[int] = None) -> List[Memory]:
        """
        Retrieve all memories from the database, optionally filtered by session.

        Args:
            session_id: Optional session ID to filter by

        Returns:
            List of Memory instances
        """
        async with self.async_session() as session:
            stmt = select(Memory)
            if session_id is not None:
                stmt = stmt.where(Memory.session_id == session_id)

            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def count_memories(self) -> int:
        """
        Count total number of memories in the database.

        Returns:
            Total count of memories
        """
        async with self.async_session() as session:
            result = await session.execute(select(Memory))
            return len(list(result.scalars().all()))

    async def delete_all_memories(self):
        """
        Delete all memories from the database.
        """
        async with self.async_session() as session:
            await session.execute(Memory.__table__.delete())
            await session.commit()

    async def close(self):
        """
        Close database connections and dispose of the engine.
        """
        await self.engine.dispose()


# Synchronous wrapper for backward compatibility
class SyncMemoryRepository:
    """
    Synchronous wrapper around MemoryRepository for backward compatibility.
    Uses asyncio.run() to execute async operations synchronously.
    """

    def __init__(self, db_path: str):
        """
        Initialize the synchronous repository.

        Args:
            db_path: Path to the SQLite database file
        """
        import asyncio

        self.repo = MemoryRepository(db_path)
        self._loop = asyncio.new_event_loop()
        self._closed = False

    def initialize(self):
        """Initialize database schema synchronously."""
        if not self._closed:
            self._loop.run_until_complete(self.repo.initialize())

    def create_session(self, name: str) -> Session:
        """Create a session synchronously."""
        if self._closed:
            raise RuntimeError("Repository is closed")
        return self._loop.run_until_complete(self.repo.create_session(name))

    def get_all_sessions(self) -> List[Session]:
        """Get all sessions synchronously."""
        if self._closed:
            return []
        return self._loop.run_until_complete(self.repo.get_all_sessions())

    def add_memory(
        self,
        text: str,
        memory_type: str = "user",
        created_at: Optional[datetime] = None,
        session_id: Optional[int] = None,
    ) -> Memory:
        """Add a memory synchronously."""
        if self._closed:
            raise RuntimeError("Repository is closed")
        return self._loop.run_until_complete(
            self.repo.add_memory(text, memory_type, created_at, session_id)
        )

    def get_memory_by_id(self, memory_id: int) -> Optional[Memory]:
        """Get a memory by ID synchronously."""
        if self._closed:
            return None
        return self._loop.run_until_complete(self.repo.get_memory_by_id(memory_id))

    def get_all_memories(self, session_id: Optional[int] = None) -> List[Memory]:
        """Get all memories synchronously."""
        if self._closed:
            return []
        return self._loop.run_until_complete(self.repo.get_all_memories(session_id))

    def count_memories(self) -> int:
        """Count memories synchronously."""
        if self._closed:
            return 0
        return self._loop.run_until_complete(self.repo.count_memories())

    def delete_all_memories(self):
        """Delete all memories synchronously."""
        if not self._closed:
            self._loop.run_until_complete(self.repo.delete_all_memories())

    def close(self):
        """Close database connections synchronously."""
        if not self._closed:
            self._closed = True
            try:
                if not self._loop.is_closed():
                    self._loop.run_until_complete(self.repo.close())
                    self._loop.close()
            except RuntimeError:
                # Event loop already closed, just mark as closed
                pass

    def __del__(self):
        """Cleanup on deletion."""
        self.close()
