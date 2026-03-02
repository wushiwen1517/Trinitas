from __future__ import annotations

import asyncio
import json
import random
import re
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx


class ModelExecutionError(RuntimeError):
    pass


class OllamaStreamExecutor:
    """
    Ollama 执行器（协议层）
    - stream()：流式文本输出
    - generate_text()：非流式文本输出
    - vision_extract()：视觉识别
    - 连接池复用，减少连接开销
    - 流式结束时提取 token 统计
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        generate_path: str = "/api/generate",
        timeout_seconds: float = 1800.0,
        keep_alive: str = "60m",
        max_attempts: int = 3,
        retry_backoff_seconds: float = 1.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.generate_path = generate_path
        self.timeout_seconds = timeout_seconds
        self.keep_alive = keep_alive
        self.max_attempts = max(1, int(max_attempts))
        self.retry_backoff_seconds = max(0.0, float(retry_backoff_seconds))
        self._client: Optional[httpx.AsyncClient] = None
        self._client_lock = asyncio.Lock()

    async def _get_client(self) -> httpx.AsyncClient:
        async with self._client_lock:
            if self._client is None or self._client.is_closed:
                timeout = httpx.Timeout(
                    timeout=self.timeout_seconds, connect=30.0,
                    read=self.timeout_seconds, write=120.0, pool=30.0,
                )
                limits = httpx.Limits(
                    max_keepalive_connections=10, max_connections=20,
                    keepalive_expiry=300.0,
                )
                self._client = httpx.AsyncClient(timeout=timeout, limits=limits)
            return self._client

    async def close(self) -> None:
        async with self._client_lock:
            if self._client and not self._client.is_closed:
                await self._client.aclose()
                self._client = None

    @property
    def generate_url(self) -> str:
        return f"{self.base_url}{self.generate_path}"

    def _build_payload(self, model, prompt, *, stream, images=None, options=None, keep_alive=None, system=None):
        payload = {"model": model, "prompt": prompt, "stream": stream, "keep_alive": keep_alive or self.keep_alive}
        if system: payload["system"] = system
        if images: payload["images"] = images
        if options: payload["options"] = options
        return payload

    async def healthcheck(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=10.0) as c:
                r = await c.get(f"{self.base_url}/api/tags")
                return r.status_code == 200
        except Exception:
            return False

    async def generate_text(self, model, prompt, *, images=None, options=None, keep_alive=None, system=None) -> str:
        payload = self._build_payload(model=model, prompt=prompt, stream=False, images=images, options=options, keep_alive=keep_alive, system=system)
        last_error = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                client = await self._get_client()
                resp = await client.post(self.generate_url, json=payload)
                break
            except Exception as e:
                last_error = e
                if attempt < self.max_attempts:
                    await asyncio.sleep(self.retry_backoff_seconds * attempt + random.uniform(0, 0.3))
                    continue
                raise ModelExecutionError(f"无法连接 Ollama ({self.generate_url})：{e}") from e
        else:
            raise ModelExecutionError(f"无法连接 Ollama ({self.generate_url})：{last_error}")

        if resp.status_code != 200:
            raise ModelExecutionError(f"Ollama HTTP {resp.status_code}，模型={model}，详情：{resp.text}")
        try:
            data = resp.json()
        except Exception as e:
            raise ModelExecutionError(f"Ollama 返回非 JSON：{resp.text[:500]}") from e
        if not isinstance(data, dict):
            raise ModelExecutionError(f"Ollama 返回结构异常：{type(data)}")
        return str(data.get("response", "") or "")

    async def stream(self, model, prompt, *, images=None, options=None, keep_alive=None, system=None) -> AsyncIterator[str]:
        payload = self._build_payload(model=model, prompt=prompt, stream=True, images=images, options=options, keep_alive=keep_alive, system=system)
        last_error = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                client = await self._get_client()
                async with client.stream("POST", self.generate_url, json=payload) as resp:
                    if resp.status_code != 200:
                        raw = await resp.aread()
                        raise ModelExecutionError(f"Ollama HTTP {resp.status_code}，模型={model}，详情：{raw.decode('utf-8','ignore')}")
                    async for line in resp.aiter_lines():
                        if not line: continue
                        try: data = json.loads(line)
                        except Exception: continue
                        if not isinstance(data, dict): continue
                        chunk = data.get("response")
                        if chunk: yield str(chunk)
                        if data.get("done") is True:
                            stats = self._extract_stream_stats(data)
                            if stats: yield f"\n[[TOKEN_STATS:{json.dumps(stats)}]]\n"
                return
            except ModelExecutionError:
                raise
            except Exception as e:
                last_error = e
                if attempt >= self.max_attempts:
                    raise ModelExecutionError(f"流式调用失败（模型={model}）：{e}") from e
                await asyncio.sleep(self.retry_backoff_seconds * attempt + random.uniform(0, 0.3))
        raise ModelExecutionError(f"流式调用失败（模型={model}）：{last_error}")

    async def stream_text(self, model, prompt, *, images=None, options=None, keep_alive=None) -> AsyncIterator[str]:
        async for chunk in self.stream(model=model, prompt=prompt, images=images, options=options, keep_alive=keep_alive):
            yield chunk

    @staticmethod
    def _extract_stream_stats(data: Dict[str, Any]) -> Dict[str, Any]:
        stats: Dict[str, Any] = {}
        if "eval_count" in data: stats["eval_count"] = data["eval_count"]
        if "prompt_eval_count" in data: stats["prompt_eval_count"] = data["prompt_eval_count"]
        if "total_duration" in data: stats["total_duration_ns"] = data["total_duration"]
        if "eval_duration" in data: stats["eval_duration_ns"] = data["eval_duration"]
        if "prompt_eval_duration" in data: stats["prompt_eval_duration_ns"] = data["prompt_eval_duration"]
        if stats.get("eval_count") and stats.get("eval_duration_ns"):
            dur = stats["eval_duration_ns"] / 1e9
            if dur > 0: stats["tokens_per_second"] = round(stats["eval_count"] / dur, 2)
        return stats

    async def vision_extract(self, model, prompt, *, images, options=None, keep_alive=None, strict_vision_pack=True) -> str:
        raw = await self.generate_text(model=model, prompt=prompt, images=images, options=options, keep_alive=keep_alive)
        if not strict_vision_pack: return raw
        m = re.search(r"\[VisionPack\](.*?)\[/VisionPack\]", raw, re.S | re.I)
        return m.group(1).strip() if m else raw.strip()
