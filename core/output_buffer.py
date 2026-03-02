# core/output_buffer.py

class OutputBuffer:
    """
    工业输出缓冲器
    用于显式收集流式分块输出，避免逻辑散落在 orchestrator
    """

    def __init__(self):
        self._chunks = []

    def append(self, chunk: str):
        if chunk is None:
            return
        self._chunks.append(str(chunk))

    def get_all(self) -> str:
        return "".join(self._chunks)

    def clear(self):
        self._chunks = []

    def size(self) -> int:
        return len(self.get_all())