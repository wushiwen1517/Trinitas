# core/memory.py — 异步版本（aiosqlite）
import aiosqlite
import asyncio
from datetime import datetime
from typing import List, Tuple

from core.config import DB_NAME, SHORT_TERM_LIMIT, LONG_TERM_TRIGGER, MAX_LONG_TERM_CHARS, MODEL_ALPHA
from core.protocol import OllamaStreamExecutor


class MemoryManager:
    """异步分层记忆管理器"""

    def __init__(self):
        self.db = DB_NAME
        self._schema_ready = False
        self._lock = asyncio.Lock()

    async def _ensure_schema(self):
        if self._schema_ready:
            return
        async with aiosqlite.connect(self.db) as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS chats(
                    id TEXT PRIMARY KEY, title TEXT, long_term_summary TEXT
                )""")
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS messages(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT, role TEXT, content TEXT, created_at TEXT
                )""")
            await conn.commit()

            # 兼容旧表（补列）
            async with conn.execute("PRAGMA table_info(chats)") as cur:
                cols = [r[1] for r in await cur.fetchall()]
            if "title" not in cols:
                await conn.execute("ALTER TABLE chats ADD COLUMN title TEXT DEFAULT '新对话'")
            if "long_term_summary" not in cols:
                await conn.execute("ALTER TABLE chats ADD COLUMN long_term_summary TEXT DEFAULT ''")

            async with conn.execute("PRAGMA table_info(messages)") as cur:
                cols = [r[1] for r in await cur.fetchall()]
            if "created_at" not in cols:
                await conn.execute("ALTER TABLE messages ADD COLUMN created_at TEXT DEFAULT ''")
            await conn.commit()
        self._schema_ready = True

    def ensure_chat(self, chat_id: str):
        """同步兼容入口（orchestrator 调用）"""
        import sqlite3
        conn = sqlite3.connect(self.db)
        c = conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS chats(id TEXT PRIMARY KEY, title TEXT, long_term_summary TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS messages(id INTEGER PRIMARY KEY AUTOINCREMENT, chat_id TEXT, role TEXT, content TEXT, created_at TEXT)""")
        c.execute("SELECT id FROM chats WHERE id=?", (chat_id,))
        if not c.fetchone():
            c.execute("INSERT INTO chats(id, title, long_term_summary) VALUES (?, ?, ?)", (chat_id, "新对话", ""))
        conn.commit(); conn.close()

    def save_message(self, chat_id: str, role: str, content: str):
        """同步保存消息"""
        import sqlite3
        conn = sqlite3.connect(self.db)
        conn.execute("INSERT INTO messages(chat_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                     (chat_id, role, content, datetime.now().isoformat()))
        conn.commit(); conn.close()

    def get_all_messages(self, chat_id: str) -> List[Tuple[str, str, int]]:
        import sqlite3
        conn = sqlite3.connect(self.db)
        c = conn.cursor()
        c.execute("SELECT role, content, id FROM messages WHERE chat_id=? ORDER BY id ASC", (chat_id,))
        rows = c.fetchall(); conn.close()
        return rows

    def get_recent_messages(self, chat_id: str) -> List[Tuple[str, str]]:
        rows = self.get_all_messages(chat_id)
        return [(r, c) for r, c, _ in rows[-SHORT_TERM_LIMIT:]]

    def get_long_term_summary(self, chat_id: str) -> str:
        import sqlite3
        conn = sqlite3.connect(self.db)
        c = conn.cursor()
        c.execute("SELECT long_term_summary FROM chats WHERE id=?", (chat_id,))
        row = c.fetchone(); conn.close()
        return (row[0] or "") if row else ""

    async def update_long_term_summary(self, chat_id: str, summary: str):
        await self._ensure_schema()
        async with aiosqlite.connect(self.db) as conn:
            await conn.execute("UPDATE chats SET long_term_summary=? WHERE id=?", (summary, chat_id))
            await conn.commit()

    async def _summarize_text(self, text: str) -> str:
        executor = OllamaStreamExecutor()
        sys_prompt = "你是长期记忆压缩模块。请将对话压缩为高密度摘要：保留关键事实、约束条件、未完成任务、偏好，删除闲聊冗余。"
        result = await executor.generate_text(model=MODEL_ALPHA, prompt=text, system=sys_prompt)
        return result.strip()

    async def maybe_compress(self, chat_id: str) -> bool:
        rows = self.get_all_messages(chat_id)
        if not rows: return False
        all_text = "\n".join([f"{r}:{c}" for r, c, _ in rows])
        if len(all_text) < LONG_TERM_TRIGGER: return False

        if len(rows) <= SHORT_TERM_LIMIT:
            old_rows, keep_rows = rows, []
        else:
            old_rows, keep_rows = rows[:-SHORT_TERM_LIMIT], rows[-SHORT_TERM_LIMIT:]

        old_text = "\n".join([f"{r}:{c}" for r, c, _ in old_rows]).strip()
        if not old_text: return False

        existing = self.get_long_term_summary(chat_id)
        combine = f"已有长期记忆：\n{existing}\n\n新增历史：\n{old_text}\n\n请合并并重新输出摘要。"
        new_summary = await self._summarize_text(combine)
        if len(new_summary) > MAX_LONG_TERM_CHARS:
            new_summary = new_summary[:MAX_LONG_TERM_CHARS]
        await self.update_long_term_summary(chat_id, new_summary)

        # 异步删除旧消息
        await self._ensure_schema()
        async with aiosqlite.connect(self.db) as conn:
            if keep_rows:
                ids = [str(rid) for _, _, rid in keep_rows]
                ph = ",".join(["?"] * len(ids))
                await conn.execute(f"DELETE FROM messages WHERE chat_id=? AND id NOT IN ({ph})", (chat_id, *ids))
            else:
                await conn.execute("DELETE FROM messages WHERE chat_id=?", (chat_id,))
            await conn.commit()
        return True

    def build_context_text(self, chat_id: str) -> str:
        lt = self.get_long_term_summary(chat_id)
        recent = self.get_recent_messages(chat_id)
        rt = "\n".join([f"{r}:{c}" for r, c in recent])
        return f"【长期记忆】\n{lt}\n\n【近期对话】\n{rt}" if lt else rt
