# core/context_trimmer.py
MAX_PROMPT_CHARS = 20000


class ContextTrimmer:
    def __init__(self, max_chars=MAX_PROMPT_CHARS):
        self.max_chars = max_chars

    def trim(self, long_term: str, recent_text: str) -> str:
        """按消息边界截断：优先保留长期记忆，再尽量保留最近的完整消息"""
        combined = f"{long_term}\n{recent_text}"
        if len(combined) <= self.max_chars:
            return combined

        budget = self.max_chars - len(long_term) - 2  # 2 for \n
        if budget <= 0:
            return long_term[:self.max_chars]

        # 按消息边界（换行）从尾部保留
        messages = recent_text.split("\n")
        kept = []
        used = 0
        for msg in reversed(messages):
            cost = len(msg) + 1  # +1 for \n
            if used + cost > budget:
                break
            kept.append(msg)
            used += cost
        kept.reverse()
        trimmed_recent = "\n".join(kept)
        return f"{long_term}\n{trimmed_recent}"
