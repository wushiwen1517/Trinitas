import asyncio
import time
import uuid

import httpx


async def main() -> None:
    url = "http://127.0.0.1:8181/api/chat"
    headers = {"X-TRINITAS-KEY": "wushiwen5170", "Content-Type": "application/json"}
    payload = {
        "message": "解释一下幂等，并给 FastAPI 示例。",
        "chat_id": f"debug_{uuid.uuid4().hex[:8]}",
        "mode": "pro",
    }

    t0 = time.monotonic()
    async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
        async with client.stream("POST", url, headers=headers, json=payload) as resp:
            print("status:", resp.status_code)
            resp.raise_for_status()
            async for chunk in resp.aiter_text():
                dt = time.monotonic() - t0
                print(f"[{dt:7.2f}s] +{len(chunk)} chars:", repr(chunk[:120]))


if __name__ == "__main__":
    asyncio.run(main())
