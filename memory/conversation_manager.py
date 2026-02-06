from typing import Dict, List, Optional

from utils.helpers import format_timestamp


class ConversationManager:
    """
    Manages active conversation history with a sliding window.
    Keeps the most recent N turns in memory.
    """

    def __init__(self, max_active_turns: int = 5):
        """
        Initialize conversation manager.

        Args:
            max_active_turns: Maximum number of turns to keep in active memory
        """
        self.max_active_turns = max_active_turns
        self.turns: List[Dict[str, str]] = []
        self.archived_turns: List[Dict[str, str]] = []

    def add_turn(self, role: str, content: str) -> Optional[Dict[str, str]]:
        """
        Add a new conversation turn.

        Args:
            role: Role of the speaker ("user" or "assistant")
            content: Content of the message

        Returns:
            Archived turn if window exceeded, None otherwise
        """
        turn = {"role": role, "content": content, "timestamp": format_timestamp()}

        self.turns.append(turn)

        # Check if we need to archive oldest turn
        if len(self.turns) > self.max_active_turns:
            archived = self.turns.pop(0)
            self.archived_turns.append(archived)
            return archived

        return None

    def get_active_turns(self) -> List[Dict[str, str]]:
        """
        Get all active turns in the conversation window.

        Returns:
            List of active conversation turns
        """
        return self.turns.copy()

    def get_messages_for_llm(self) -> List[Dict[str, str]]:
        """
        Format active turns for LLM input (without timestamps).

        Returns:
            List of messages in format [{"role": "user/assistant", "content": "..."}]
        """
        return [
            {"role": turn["role"], "content": turn["content"]} for turn in self.turns
        ]

    def format_turn_for_embedding(self, turn: Dict[str, str]) -> str:
        """
        Format a turn into a string suitable for embedding.

        Args:
            turn: Conversation turn dictionary

        Returns:
            Formatted string combining role and content
        """
        return f"{turn['role']}: {turn['content']}"

    def get_conversation_summary(self) -> str:
        """
        Get a summary of the current conversation state.

        Returns:
            String summary of conversation
        """
        active_count = len(self.turns)
        archived_count = len(self.archived_turns)
        return f"Active turns: {active_count}/{self.max_active_turns} | Archived: {archived_count}"

    def clear_history(self):
        """
        Clear all conversation history (active and archived).
        """
        self.turns.clear()
        self.archived_turns.clear()

    def get_context_window_size(self) -> int:
        """
        Get the current size of the active context window.

        Returns:
            Number of active turns
        """
        return len(self.turns)
