"""
记忆模块: 短期对话记忆 + 长期记忆

短期记忆: 保持最近对话的上下文 (FIFO, token 限制)
长期记忆: 从溢出的短期记忆中提取关键信息持久保存
"""

from llama_index.core.memory import Memory

from config import settings


def create_memory(session_id: str = "default") -> Memory:
    """创建带短期+长期记忆的 Memory 实例"""
    memory = Memory.from_defaults(
        session_id=session_id,
        token_limit=settings.memory_token_limit,
        chat_history_token_ratio=settings.chat_history_token_ratio,
    )
    return memory
