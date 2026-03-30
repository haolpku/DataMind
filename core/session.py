"""
Session 管理: 为每个 session_id 维护独立的 Memory 实例，实现并发隔离。
"""

from modules.memory.memory import create_memory


class SessionManager:
    def __init__(self):
        self._sessions: dict = {}

    def get_memory(self, session_id: str = "default"):
        if session_id not in self._sessions:
            self._sessions[session_id] = create_memory(session_id)
        return self._sessions[session_id]

    def clear(self, session_id: str):
        self._sessions.pop(session_id, None)

    def clear_all(self):
        self._sessions.clear()
